package cache

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// StreamAccumulator collects streaming chunks for a single request so they can
// be flushed as one cache entry when the final chunk arrives.
type StreamAccumulator struct {
	mu         sync.Mutex
	RequestID  string
	StorageID  string
	Chunks     []*streamChunk
	LastSeenAt time.Time
	IsComplete bool
	Embedding  []float32
	Metadata   map[string]any
	TTL        time.Duration
}

// streamAccumulatorMaxAge is how long an accumulator may live without
// receiving a final chunk before being reaped.
const streamAccumulatorMaxAge = 5 * time.Minute

// streamCleanupInterval is the ticker period for the accumulator reaper.
const streamCleanupInterval = 1 * time.Minute

// getOrCreateStreamAccumulator returns the accumulator for requestID, creating
// one if none exists. Thread-safe via sync.Map.LoadOrStore.
func (c *Cache) getOrCreateStreamAccumulator(requestID, storageID string, embedding []float32, metadata map[string]any, ttl time.Duration) *StreamAccumulator {
	if v, ok := c.streamAccumulators.Load(requestID); ok {
		return v.(*StreamAccumulator)
	}
	acc := &StreamAccumulator{
		RequestID:  requestID,
		StorageID:  storageID,
		Chunks:     make([]*streamChunk, 0),
		LastSeenAt: time.Now(),
		Embedding:  embedding,
		Metadata:   metadata,
		TTL:        ttl,
	}
	actual, _ := c.streamAccumulators.LoadOrStore(requestID, acc)
	return actual.(*StreamAccumulator)
}

// addStreamChunk appends a chunk to the accumulator and refreshes LastSeenAt.
func (c *Cache) addStreamChunk(requestID string, chunk *streamChunk) error {
	v, ok := c.streamAccumulators.Load(requestID)
	if !ok {
		return fmt.Errorf("no stream accumulator for request %s", requestID)
	}
	acc := v.(*StreamAccumulator)
	acc.mu.Lock()
	defer acc.mu.Unlock()
	acc.Chunks = append(acc.Chunks, chunk)
	acc.LastSeenAt = chunk.Timestamp
	return nil
}

// processAccumulatedStream sorts and flushes chunks to the vector store.
// Called once when the final chunk arrives (via storeStreamResponse).
func (c *Cache) processAccumulatedStream(ctx context.Context, requestID string) error {
	v, ok := c.streamAccumulators.Load(requestID)
	if !ok {
		return fmt.Errorf("no stream accumulator for request %s", requestID)
	}
	acc := v.(*StreamAccumulator)
	acc.mu.Lock()
	defer acc.mu.Unlock()
	defer c.streamAccumulators.Delete(requestID)

	sort.SliceStable(acc.Chunks, func(i, j int) bool {
		return acc.Chunks[i].Index < acc.Chunks[j].Index
	})

	encoded := make([]string, 0, len(acc.Chunks))
	for _, chunk := range acc.Chunks {
		if len(chunk.Data) == 0 {
			continue
		}
		encoded = append(encoded, string(chunk.Data))
	}
	if len(encoded) == 0 {
		c.logger.Warn("stream for request %s has no valid chunks, skipping cache write", requestID)
		return nil
	}

	meta := make(map[string]any, len(acc.Metadata)+1)
	for k, v := range acc.Metadata {
		meta[k] = v
	}
	meta["stream_chunks"] = encoded

	if err := c.store.Add(ctx, c.config.Namespace, acc.StorageID, acc.Embedding, meta); err != nil {
		return fmt.Errorf("failed to store streaming cache entry: %w", err)
	}
	c.logger.Debug("cached stream with %d chunks, storageID=%s", len(encoded), acc.StorageID)
	return nil
}

// runStreamCleanupLoop reaps abandoned accumulators on a ticker.
func (c *Cache) runStreamCleanupLoop() {
	defer c.cleanupWg.Done()
	ticker := time.NewTicker(streamCleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.cleanupOldStreamAccumulators()
		}
	}
}

// cleanupOldStreamAccumulators reaps accumulators whose most recent chunk is
// older than streamAccumulatorMaxAge.
func (c *Cache) cleanupOldStreamAccumulators() {
	cutoff := time.Now().Add(-streamAccumulatorMaxAge)
	var stale []string
	c.streamAccumulators.Range(func(k, v any) bool {
		acc := v.(*StreamAccumulator)
		acc.mu.Lock()
		old := acc.LastSeenAt.Before(cutoff)
		acc.mu.Unlock()
		if old {
			stale = append(stale, k.(string))
		}
		return true
	})
	for _, id := range stale {
		c.streamAccumulators.Delete(id)
	}
	if len(stale) > 0 {
		c.logger.Debug("reaped %d stale stream accumulators", len(stale))
	}
}
