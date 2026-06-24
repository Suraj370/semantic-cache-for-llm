package cache

import (
	"time"
)

// cacheState holds per-request working data shared between the lookup (read)
// path and the store (write) path.
//
// No mutex is needed: a single request runs lookup then store sequentially.
// The only concurrent access is the async store goroutine, which takes a
// snapshot of the fields it needs before launching.
type cacheState struct {
	// DirectCacheID is the deterministic UUIDv5 computed from
	// (provider, model, cacheKey, requestHash, paramsHash). Set by the direct
	// lookup path and reused as the storage ID when the direct path ran.
	DirectCacheID string

	// ParamsHash is the xxhash of the request parameters map. Set once and
	// shared by both lookup paths.
	ParamsHash string

	// Embeddings is the float32 vector computed for the request input. Set by
	// the semantic lookup path and reused as the storage embedding.
	Embeddings []float32

	// EmbeddingTokens is the number of tokens consumed by the embedding call.
	EmbeddingTokens int

	// FilteredInput is the memoized result of getInputForCaching(req): either
	// the messages slice or the prompt string, after system-prompt filtering.
	FilteredInput interface{}

	// ShortCircuited is set when the lookup returned a cached hit. The store
	// path checks this to skip writing on cache-hit replay chunks.
	ShortCircuited bool

	// CreatedAt records when the lookup started, for latency telemetry.
	CreatedAt time.Time
}

// cacheStateMaxAge is how long an orphaned state entry may live before being
// reaped by the background cleanup loop.
const cacheStateMaxAge = 60 * time.Minute

// cacheStateCleanupInterval is the ticker period for the state reaper.
const cacheStateCleanupInterval = 5 * time.Minute

// newCacheState allocates a fresh state for requestID, overwriting any prior.
func (c *Cache) newCacheState(requestID string) *cacheState {
	s := &cacheState{CreatedAt: time.Now()}
	c.cacheStates.Store(requestID, s)
	return s
}

// getCacheState returns the state for requestID, or nil if none exists.
func (c *Cache) getCacheState(requestID string) *cacheState {
	if v, ok := c.cacheStates.Load(requestID); ok {
		return v.(*cacheState)
	}
	return nil
}

// clearCacheState removes the state entry for requestID.
func (c *Cache) clearCacheState(requestID string) {
	c.cacheStates.Delete(requestID)
}

// runStateCleanupLoop reaps stale state entries on a ticker until stopCh is
// closed. Started by New, stopped by Close.
func (c *Cache) runStateCleanupLoop() {
	defer c.cleanupWg.Done()
	ticker := time.NewTicker(cacheStateCleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.cleanupOldCacheStates()
		}
	}
}

// cleanupOldCacheStates deletes states older than cacheStateMaxAge. These
// represent requests that never reached the store path (client disconnects,
// framework errors).
func (c *Cache) cleanupOldCacheStates() {
	cutoff := time.Now().Add(-cacheStateMaxAge)
	var stale []string
	c.cacheStates.Range(func(k, v any) bool {
		if v.(*cacheState).CreatedAt.Before(cutoff) {
			stale = append(stale, k.(string))
		}
		return true
	})
	for _, id := range stale {
		c.cacheStates.Delete(id)
	}
	if len(stale) > 0 {
		c.logger.Debug("reaped %d stale cache states", len(stale))
	}
}
