package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/suraj370/semantic-cache/embedding"
	"github.com/suraj370/semantic-cache/logger"
	"github.com/suraj370/semantic-cache/vectorstore"
)

// Cache implements dual-path semantic caching over any VectorStore backend.
//
// Lookup runs a direct O(1) hash lookup and (when an Embedder is configured)
// an ANN semantic similarity search. On a miss it returns a MissHandle whose
// Store / StoreStream methods write the upstream response back to the cache.
type Cache struct {
	store    vectorstore.VectorStore
	config   *Config
	logger   logger.Logger
	embedder embedding.Embedder // nil in direct-only mode

	// streamAccumulators maps requestID → *StreamAccumulator for in-progress
	// streaming writes. Not used by the main Store/StoreStream path.
	streamAccumulators sync.Map

	// cacheStates maps requestID → *cacheState for the Lookup→Store span.
	cacheStates sync.Map

	// writersWg tracks async cache-write goroutines.
	writersWg sync.WaitGroup

	// cleanupWg tracks background reaper goroutines.
	cleanupWg sync.WaitGroup

	// stopCh is closed by Close to signal reapers to exit.
	stopCh chan struct{}

	// closeOnce prevents double-close of stopCh.
	closeOnce sync.Once

	// metrics receives observability events. Never nil (uses noop by default).
	metrics MetricsRecorder
}

// New constructs and initialises a Cache. It creates the vector store namespace
// (if not already present) and starts the background cleanup goroutines.
//
//   - store must be a connected VectorStore (call vectorstore.NewVectorStore first).
//   - embedder may be nil for direct-only mode (EmbeddingDimension = 1).
//   - log may be nil; a NoopLogger is used in that case.
func New(ctx context.Context, config *Config, store vectorstore.VectorStore, embedder embedding.Embedder, log logger.Logger, opts ...Option) (*Cache, error) {
	if config == nil {
		return nil, fmt.Errorf("config is required")
	}
	if store == nil {
		return nil, fmt.Errorf("store is required")
	}
	if log == nil {
		log = noopLogger{}
	}

	// Apply defaults.
	if config.Namespace == "" {
		config.Namespace = DefaultNamespace
	}
	if config.TTL == 0 {
		config.TTL = DefaultTTL
	}
	if config.Threshold == 0 {
		config.Threshold = DefaultThreshold
	}
	if config.ConversationHistoryThreshold == 0 {
		config.ConversationHistoryThreshold = DefaultConversationLimit
	}
	if config.EmbeddingDimension <= 0 {
		return nil, fmt.Errorf("EmbeddingDimension must be > 0 (use 1 for direct-only mode)")
	}
	if embedder != nil && config.EmbeddingDimension == 1 {
		log.Warn("embedder provided but EmbeddingDimension=1; semantic search is disabled (direct-only mode)")
		embedder = nil
	}

	// Convert local property definitions to vectorstore types.
	vsProps := make(map[string]vectorstore.VectorStoreProperties, len(VectorStoreProperties))
	for k, v := range VectorStoreProperties {
		vsProps[k] = vectorstore.VectorStoreProperties{
			DataType:    vectorstore.VectorStorePropertyType(v.DataType),
			Description: v.Description,
		}
	}

	createCtx, cancel := context.WithTimeout(ctx, CreateNamespaceTimeout)
	defer cancel()
	if err := store.CreateNamespace(createCtx, config.Namespace, config.EmbeddingDimension, vsProps); err != nil {
		return nil, fmt.Errorf("failed to create cache namespace: %w", err)
	}

	c := &Cache{
		store:    store,
		config:   config,
		logger:   log,
		embedder: embedder,
		stopCh:   make(chan struct{}),
		metrics:  noopMetricsRecorder{},
	}
	for _, opt := range opts {
		opt(c)
	}

	c.cleanupWg.Add(2)
	go c.runStateCleanupLoop()
	go c.runStreamCleanupLoop()

	return c, nil
}

// Lookup checks the cache for a matching response to req.
//
// Returns:
//   - (*LookupResult, nil, nil) — cache hit; use the returned result.
//   - (nil, *MissHandle, nil) — cache miss; call handle.Store or handle.StoreStream
//     once you have the upstream response.
//   - (nil, nil, error) — caching is not applicable (missing cache key, unsupported
//     request type) or a hard error occurred.
func (c *Cache) Lookup(ctx context.Context, requestID string, req Request, opts LookupOptions) (*LookupResult, *MissHandle, error) {
	start := time.Now()

	cacheKey := opts.CacheKey
	if cacheKey == "" {
		cacheKey = c.config.DefaultCacheKey
	}
	if cacheKey == "" {
		c.metrics.RecordLookup(LookupOutcomeSkipped, time.Since(start).Milliseconds(), 0)
		return nil, nil, nil
	}

	if requestID == "" {
		requestID = uuid.New().String()
	}

	state := c.newCacheState(requestID)

	// Skip long conversations (unlikely to hit cache and pollute the store).
	excludeSystem := c.config.ExcludeSystemPrompt != nil && *c.config.ExcludeSystemPrompt
	if c.config.ConversationHistoryThreshold > 0 {
		count := conversationMessageCount(req, excludeSystem)
		if count > c.config.ConversationHistoryThreshold {
			c.clearCacheState(requestID)
			c.metrics.RecordLookup(LookupOutcomeSkipped, time.Since(start).Milliseconds(), 0)
			return nil, nil, nil
		}
	}

	// Build params metadata + hash once, shared by both search paths.
	paramsMetadata := c.buildParamsMetadata(req)
	paramsHash, err := hashMap(paramsMetadata)
	if err != nil {
		c.clearCacheState(requestID)
		return nil, nil, fmt.Errorf("failed to compute params hash: %w", err)
	}
	state.ParamsHash = paramsHash

	ttl := opts.TTL
	if ttl <= 0 {
		ttl = c.config.TTL
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = c.config.Threshold
	}

	doDirect := opts.CacheType == "" || opts.CacheType == CacheTypeDirect
	doSemantic := opts.CacheType == "" || opts.CacheType == CacheTypeSemantic

	canSemantic := doSemantic && c.embedder != nil && req.Type != RequestTypeEmbedding

	lookupCtx, cancel := context.WithTimeout(ctx, CacheConnectionTimeout)
	defer cancel()

	// 1. Direct hash lookup.
	if doDirect {
		hit, err := c.performDirectSearch(lookupCtx, state, req, cacheKey, paramsMetadata, paramsHash)
		if err != nil {
			c.logger.Warn("direct search error (proceeding as miss): %v", err)
			c.metrics.RecordError("direct_search")
		} else if hit != nil {
			c.clearCacheState(requestID)
			c.metrics.RecordLookup(LookupOutcomeDirectHit, time.Since(start).Milliseconds(), 0)
			return hit, nil, nil
		}
	}

	// 2. Semantic similarity search.
	if canSemantic {
		hit, err := c.performSemanticSearch(lookupCtx, state, req, cacheKey, paramsHash, threshold)
		if err != nil {
			c.logger.Warn("semantic search error (proceeding as miss): %v", err)
			c.metrics.RecordError("semantic_search")
		} else if hit != nil {
			c.clearCacheState(requestID)
			c.metrics.RecordLookup(LookupOutcomeSemanticHit, time.Since(start).Milliseconds(), state.EmbeddingTokens)
			return hit, nil, nil
		}
	} else if !doDirect && doSemantic {
		// Semantic-only requested but not feasible; fall through.
		c.logger.Warn("semantic search requested but embedder is not configured or request type is not embeddable")
	}

	// Cache miss.
	c.metrics.RecordLookup(LookupOutcomeMiss, time.Since(start).Milliseconds(), state.EmbeddingTokens)

	// Determine the storage ID and embedding for the write path.
	storageID := requestID
	if state.DirectCacheID != "" {
		storageID = state.DirectCacheID
	}

	// For stores that require a vector on every entry, write a placeholder.
	if c.store.RequiresVectors() && state.Embeddings == nil && c.config.EmbeddingDimension > 0 {
		vec := make([]float32, c.config.EmbeddingDimension)
		vec[0] = 1.0
		state.Embeddings = vec
	}

	mo := missOpts{
		provider:   req.Provider,
		model:      req.Model,
		paramsHash: paramsHash,
		cacheKey:   cacheKey,
		embedding:  state.Embeddings,
		ttl:        ttl,
	}

	return nil, &MissHandle{
		cache:     c,
		requestID: requestID,
		storageID: storageID,
		opts:      mo,
		noStore:   opts.NoStore,
	}, nil
}

// storeResponse writes a non-streaming response to the cache.
func (c *Cache) storeResponse(m *MissHandle, response json.RawMessage) error {
	if len(response) == 0 {
		return fmt.Errorf("response is empty")
	}

	meta := buildUnifiedMetadata(m.opts.provider, m.opts.model, m.opts.paramsHash, m.opts.cacheKey, m.opts.ttl)
	meta["response"] = string(response)
	meta["stream_chunks"] = []string{}

	c.writersWg.Add(1)
	go func() {
		defer c.writersWg.Done()
		storeStart := time.Now()
		storeCtx, cancel := context.WithTimeout(context.Background(), CacheSetTimeout)
		defer cancel()
		err := c.store.Add(storeCtx, c.config.Namespace, m.storageID, m.opts.embedding, meta)
		c.metrics.RecordStore(time.Since(storeStart).Milliseconds(), err)
		if err != nil {
			c.logger.Warn("failed to cache response (id=%s): %v", m.storageID, err)
		} else {
			c.logger.Debug("cached response id=%s", m.storageID)
		}
	}()

	c.clearCacheState(m.requestID)
	return nil
}

// storeStreamResponse writes a completed streaming response (all chunks) to the cache.
func (c *Cache) storeStreamResponse(m *MissHandle, chunks []json.RawMessage) error {
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks to store")
	}

	encoded := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if len(chunk) == 0 {
			continue
		}
		encoded = append(encoded, string(chunk))
	}
	if len(encoded) == 0 {
		return fmt.Errorf("no non-empty chunks to store")
	}

	meta := buildUnifiedMetadata(m.opts.provider, m.opts.model, m.opts.paramsHash, m.opts.cacheKey, m.opts.ttl)
	meta["stream_chunks"] = encoded
	meta["response"] = ""

	c.writersWg.Add(1)
	go func() {
		defer c.writersWg.Done()
		storeStart := time.Now()
		storeCtx, cancel := context.WithTimeout(context.Background(), CacheSetTimeout)
		defer cancel()
		err := c.store.Add(storeCtx, c.config.Namespace, m.storageID, m.opts.embedding, meta)
		c.metrics.RecordStore(time.Since(storeStart).Milliseconds(), err)
		if err != nil {
			c.logger.Warn("failed to cache stream (id=%s): %v", m.storageID, err)
		} else {
			c.logger.Debug("cached stream with %d chunks id=%s", len(encoded), m.storageID)
		}
	}()

	c.clearCacheState(m.requestID)
	return nil
}

// Invalidate deletes all cache entries written under cacheKey.
func (c *Cache) Invalidate(ctx context.Context, cacheKey string) error {
	if cacheKey == "" {
		return fmt.Errorf("cacheKey is required")
	}
	queries := []vectorstore.Query{
		{Field: "cache_key", Operator: vectorstore.QueryOperatorEqual, Value: cacheKey},
		{Field: "from_semantic_cache", Operator: vectorstore.QueryOperatorEqual, Value: true},
	}
	deleteCtx, cancel := context.WithTimeout(ctx, CacheSetTimeout)
	defer cancel()
	results, err := c.store.DeleteAll(deleteCtx, c.config.Namespace, queries)
	if err != nil {
		return fmt.Errorf("failed to invalidate cache key %q: %w", cacheKey, err)
	}
	var firstErr error
	for _, r := range results {
		if r.Status == vectorstore.DeleteStatusError {
			c.logger.Warn("failed to delete entry %s: %s", r.ID, r.Error)
			if firstErr == nil {
				firstErr = fmt.Errorf("delete error for entry %s: %s", r.ID, r.Error)
			}
		}
	}
	c.logger.Debug("invalidated cache key %q (%d entries)", cacheKey, len(results))
	return firstErr
}

// InvalidateByID deletes a single cache entry by its storage ID.
func (c *Cache) InvalidateByID(ctx context.Context, cacheID string) error {
	if cacheID == "" {
		return fmt.Errorf("cacheID is required")
	}
	deleteCtx, cancel := context.WithTimeout(ctx, CacheSetTimeout)
	defer cancel()
	if err := c.store.Delete(deleteCtx, c.config.Namespace, cacheID); err != nil {
		return fmt.Errorf("failed to delete cache entry %s: %w", cacheID, err)
	}
	c.logger.Debug("invalidated cache entry %s", cacheID)
	return nil
}

// WaitForPendingOps blocks until all in-flight cache-write goroutines complete.
// Useful in tests to ensure entries are visible before asserting cache hits.
func (c *Cache) WaitForPendingOps() {
	c.writersWg.Wait()
}

// Close drains in-flight writes and stops the background cleanup goroutines.
func (c *Cache) Close() error {
	c.closeOnce.Do(func() {
		close(c.stopCh)
		c.writersWg.Wait()
		c.cleanupWg.Wait()
		c.cleanupOldStreamAccumulators()
	})
	return nil
}

// noopLogger silently discards all log messages.
type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
