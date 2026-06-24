package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/google/uuid"
	"github.com/suraj370/semantic-cache/vectorstore"
)

// directCacheNamespace is a fixed UUID used as the SHA-1 namespace when
// generating deterministic cache IDs. The bytes are arbitrary but must be
// stable across restarts.
var directCacheNamespace = uuid.MustParse("b1f3c2d4-e5a6-7890-abcd-ef1234567890")

// performDirectSearch does an O(1) point-fetch on the deterministic ID derived
// from (provider, model, cacheKey, requestHash, paramsHash). Returns nil, nil
// on a cache miss (including soft misses such as expired entries).
func (c *Cache) performDirectSearch(ctx context.Context, state *cacheState, req Request, cacheKey string, paramsMetadata map[string]interface{}, paramsHash string) (*LookupResult, error) {
	requestHash, err := c.generateRequestHash(req, paramsMetadata)
	if err != nil {
		return nil, fmt.Errorf("failed to generate request hash: %w", err)
	}

	directID, err := c.generateDirectCacheID(req.Provider, req.Model, cacheKey, requestHash, paramsHash)
	if err != nil {
		return nil, fmt.Errorf("failed to generate direct cache ID: %w", err)
	}
	state.DirectCacheID = directID

	result, err := c.store.GetChunk(ctx, c.config.Namespace, directID)
	if err != nil {
		errMsg := strings.ToLower(err.Error())
		if errors.Is(err, vectorstore.ErrNotFound) ||
			strings.Contains(errMsg, "not found") ||
			strings.Contains(errMsg, "status code: 404") {
			return nil, nil
		}
		return nil, fmt.Errorf("direct cache fetch failed: %w", err)
	}

	return c.buildResponseFromResult(ctx, result, req, CacheTypeDirect, nil, 0, state.CreatedAt)
}

// performSemanticSearch generates an embedding and queries the ANN index for
// the nearest cached response above the similarity threshold.
func (c *Cache) performSemanticSearch(ctx context.Context, state *cacheState, req Request, cacheKey string, paramsHash string, threshold float64) (*LookupResult, error) {
	excludeSystem := c.config.ExcludeSystemPrompt != nil && *c.config.ExcludeSystemPrompt
	text, err := extractTextForEmbedding(req, excludeSystem)
	if err != nil {
		return nil, fmt.Errorf("failed to extract text for embedding: %w", err)
	}

	embedStart := time.Now()
	embedding, tokens, err := c.embedder.Embed(ctx, text)
	c.metrics.RecordEmbedding(time.Since(embedStart).Milliseconds(), tokens, err)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	state.Embeddings = embedding
	state.EmbeddingTokens = tokens

	filters := []vectorstore.Query{
		{Field: "cache_key", Operator: vectorstore.QueryOperatorEqual, Value: cacheKey},
		{Field: "params_hash", Operator: vectorstore.QueryOperatorEqual, Value: paramsHash},
		{Field: "from_semantic_cache", Operator: vectorstore.QueryOperatorEqual, Value: true},
	}
	cacheByProvider := c.config.CacheByProvider == nil || *c.config.CacheByProvider
	cacheByModel := c.config.CacheByModel == nil || *c.config.CacheByModel
	if cacheByProvider && req.Provider != "" {
		filters = append(filters, vectorstore.Query{Field: "provider", Operator: vectorstore.QueryOperatorEqual, Value: req.Provider})
	}
	if cacheByModel && req.Model != "" {
		filters = append(filters, vectorstore.Query{Field: "model", Operator: vectorstore.QueryOperatorEqual, Value: req.Model})
	}

	fields := selectFieldsForRequest(req.Type)
	results, err := c.store.GetNearest(ctx, c.config.Namespace, embedding, filters, fields, threshold, 1)
	if err != nil {
		return nil, fmt.Errorf("semantic search failed: %w", err)
	}
	if len(results) == 0 {
		return nil, nil
	}

	return c.buildResponseFromResult(ctx, results[0], req, CacheTypeSemantic, results[0].Score, tokens, state.CreatedAt)
}

// selectFieldsForRequest returns the projection list for the given request type.
func selectFieldsForRequest(t RequestType) []string {
	if t.IsStream() {
		return selectFieldsStream
	}
	return selectFieldsNonStream
}

var (
	selectFieldsStream    = filterSelectFields("response")
	selectFieldsNonStream = filterSelectFields("stream_chunks")
)

func filterSelectFields(skip string) []string {
	out := make([]string, 0, len(selectFields))
	for _, f := range selectFields {
		if f != skip {
			out = append(out, f)
		}
	}
	return out
}

// buildResponseFromResult constructs a LookupResult from a raw vector store hit.
// Returns (nil, nil) for soft misses (expired entry, format mismatch).
func (c *Cache) buildResponseFromResult(ctx context.Context, result vectorstore.SearchResult, req Request, cacheType CacheType, score *float64, embeddingTokens int, lookupStart time.Time) (*LookupResult, error) {
	if result.Properties == nil {
		return nil, fmt.Errorf("cache entry has no properties")
	}

	if expired, parseFailed := isExpiredEntry(result.Properties); expired {
		c.metrics.RecordEviction()
		c.writersWg.Add(1)
		go func() {
			defer c.writersWg.Done()
			delCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := c.store.Delete(delCtx, c.config.Namespace, result.ID); err != nil {
				c.logger.Warn("failed to delete expired entry %s: %v", result.ID, err)
			}
		}()
		return nil, nil
	} else if parseFailed {
		return nil, nil
	}

	latency := time.Since(lookupStart).Milliseconds()

	if req.Type.IsStream() {
		chunks, err := parseStreamChunks(result.Properties["stream_chunks"])
		if err != nil || len(chunks) == 0 {
			c.logger.Warn("cache entry %s has no valid stream_chunks, treating as miss", result.ID)
			return nil, nil
		}
		ch := make(chan json.RawMessage, len(chunks))
		done := ctx.Done()
		go func() {
			defer close(ch)
			for _, chunk := range chunks {
				select {
				case ch <- json.RawMessage(chunk):
				case <-done:
					return
				}
			}
		}()
		return &LookupResult{
			HitType:         cacheType,
			CacheID:         result.ID,
			Stream:          ch,
			Latency:         latency,
			Similarity:      score,
			EmbeddingTokens: embeddingTokens,
		}, nil
	}

	responseRaw, ok := result.Properties["response"]
	if !ok || responseRaw == nil {
		c.logger.Warn("cache entry %s missing response field, treating as miss", result.ID)
		return nil, nil
	}
	responseStr, ok := responseRaw.(string)
	if !ok {
		return nil, fmt.Errorf("cached response is not a string")
	}
	return &LookupResult{
		HitType:         cacheType,
		CacheID:         result.ID,
		Response:        json.RawMessage(responseStr),
		Latency:         latency,
		Similarity:      score,
		EmbeddingTokens: embeddingTokens,
	}, nil
}

// generateRequestHash returns a deterministic hex xxhash of the normalized
// request input and its parameter metadata.
func (c *Cache) generateRequestHash(req Request, paramsMetadata map[string]interface{}) (string, error) {
	excludeSystem := c.config.ExcludeSystemPrompt != nil && *c.config.ExcludeSystemPrompt
	hashInput := map[string]interface{}{
		"input":  getNormalizedInput(req, excludeSystem),
		"params": paramsMetadata,
	}
	data, err := marshalSorted(hashInput)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request for hashing: %w", err)
	}
	return fmt.Sprintf("%x", xxhash.Sum64(data)), nil
}

// generateDirectCacheID returns a deterministic UUIDv5 for this exact request.
func (c *Cache) generateDirectCacheID(provider, model, cacheKey, requestHash, paramsHash string) (string, error) {
	cacheByProvider := c.config.CacheByProvider == nil || *c.config.CacheByProvider
	cacheByModel := c.config.CacheByModel == nil || *c.config.CacheByModel

	idInput := map[string]interface{}{
		"cache_key":    cacheKey,
		"request_hash": requestHash,
		"params_hash":  paramsHash,
	}
	if cacheByProvider && provider != "" {
		idInput["provider"] = provider
	}
	if cacheByModel && model != "" {
		idInput["model"] = model
	}
	data, err := marshalSorted(idInput)
	if err != nil {
		return "", err
	}
	return uuid.NewSHA1(directCacheNamespace, data).String(), nil
}

// buildParamsMetadata extracts the request params into a canonical map for
// hashing and metadata attachment. The map is also used to derive params_hash.
func (c *Cache) buildParamsMetadata(req Request) map[string]interface{} {
	m := paramsToMap(req.Params)
	// Include stream flag so streaming and non-streaming requests for the same
	// prompt don't collide in the same cache bucket.
	m["stream"] = req.Type.IsStream()
	return m
}
