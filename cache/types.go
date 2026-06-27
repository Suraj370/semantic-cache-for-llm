package cache

import (
	"encoding/json"
	"time"
)

// ── Request types ───────────────────────────────────────────────────────────

// RequestType identifies the kind of LLM call being cached.
type RequestType string

const (
	RequestTypeChat          RequestType = "chat"
	RequestTypeChatStream    RequestType = "chat_stream"
	RequestTypeText          RequestType = "text"
	RequestTypeTextStream    RequestType = "text_stream"
	RequestTypeEmbedding     RequestType = "embedding"
)

// IsStream returns true for streaming variants.
func (r RequestType) IsStream() bool {
	return r == RequestTypeChatStream || r == RequestTypeTextStream
}

// Message is a single conversation turn.
type Message struct {
	Role    string          `json:"role"`
	Content MessageContent  `json:"content"`
}

// MessageContent holds a message's text and/or multimodal blocks.
// When Blocks is non-empty it takes precedence over Text.
type MessageContent struct {
	Text   *string        `json:"text,omitempty"`
	Blocks []ContentBlock `json:"blocks,omitempty"`
}

// ContentBlock is a single typed content block within a message.
type ContentBlock struct {
	// Type is "text" or "image_url".
	Type     string  `json:"type"`
	Text     *string `json:"text,omitempty"`
	ImageURL *string `json:"image_url,omitempty"`
}

// Request is the cacheable representation of an LLM call.
// Fill only the fields relevant to the chosen RequestType.
type Request struct {
	// Type identifies the request shape. Required.
	Type RequestType `json:"type"`

	// Provider is the LLM provider name (e.g. "openai", "anthropic").
	// Included in the cache key when Config.CacheByProvider is true.
	Provider string `json:"provider,omitempty"`

	// Model is the model identifier. Included in the cache key when
	// Config.CacheByModel is true.
	Model string `json:"model,omitempty"`

	// Messages holds the conversation history for chat requests.
	Messages []Message `json:"messages,omitempty"`

	// Prompt holds the text input for text-completion requests.
	Prompt *string `json:"prompt,omitempty"`

	// EmbeddingInput is the text for embedding requests (not re-embedded by
	// the cache — the full text is hashed for direct lookup only).
	EmbeddingInput *string `json:"embedding_input,omitempty"`

	// Params are provider-specific parameters serialised as raw JSON. All
	// fields in Params participate in the params_hash, so changing any
	// parameter produces a different cache bucket.
	Params json.RawMessage `json:"params,omitempty"`
}

// ── Options ─────────────────────────────────────────────────────────────────

// LookupOptions controls cache behaviour for a single Lookup call.
type LookupOptions struct {
	// CacheKey is a tenant/feature scope key that partitions cache entries.
	// Required unless Config.DefaultCacheKey is set.
	CacheKey string

	// TTL overrides Config.TTL for this entry. Zero uses the config default.
	TTL time.Duration

	// Threshold overrides Config.Threshold for semantic similarity. Zero uses
	// the config default.
	Threshold float64

	// CacheType limits the lookup to a single path. An empty value (default)
	// runs both direct and semantic lookups.
	CacheType CacheType

	// NoStore skips writing the upstream response to cache after a miss.
	// The lookup still consults the cache normally.
	NoStore bool
}

// ── Results ──────────────────────────────────────────────────────────────────

// LookupResult is returned by a cache hit.
type LookupResult struct {
	// HitType is "direct" or "semantic".
	HitType CacheType

	// CacheID is the entry's storage ID. Pass to Cache.InvalidateByID to
	// remove this entry.
	CacheID string

	// Response holds the cached non-streaming response (nil for streaming hits).
	Response json.RawMessage

	// Stream delivers chunks for a cached streaming response (nil for
	// non-streaming hits). The channel is closed after the last chunk.
	Stream <-chan json.RawMessage

	// Latency is the time from Lookup entry to the first cached byte (ms).
	Latency int64

	// Similarity is the cosine similarity score for semantic hits (nil for
	// direct hits or when similarity is not available from the store).
	Similarity *float64

	// EmbeddingTokens is the number of tokens consumed by the embedding call
	// that produced the semantic hit (0 for direct hits).
	EmbeddingTokens int
}

// ── Miss handle ───────────────────────────────────────────────────────────────

// MissHandle is returned alongside (nil, MissHandle, nil) from Lookup on a
// cache miss. Call Store or StoreStream to write the upstream response so
// subsequent identical / similar requests are served from cache.
//
// Both methods are safe to call exactly once from any goroutine. Calling them
// more than once is a no-op after the first call.
type MissHandle struct {
	cache     *Cache
	requestID string
	storageID string
	opts      missOpts
	noStore   bool // true when LookupOptions.NoStore was set
}

// missOpts bundles the per-request values pre-computed by Lookup so Store
// doesn't have to repeat work.
type missOpts struct {
	provider   string
	model      string
	paramsHash string
	cacheKey   string
	embedding  []float32
	ttl        time.Duration
}

// CacheID returns the storage ID where the response will be written. Expose
// this to callers so they can reference the entry before it's stored (e.g. for
// logging).
func (m *MissHandle) CacheID() string {
	return m.storageID
}

// Store writes a non-streaming response to the cache.
// It is a no-op when LookupOptions.NoStore was set.
func (m *MissHandle) Store(response json.RawMessage) error {
	if m.noStore {
		return nil
	}
	return m.cache.storeResponse(m, response, 0, 0)
}

// StoreWithTokens writes a non-streaming response along with the token counts
// from the LLM call. inputTokens and outputTokens are used to calculate cost
// savings reported via MetricsRecorder.RecordCostSaved on future cache hits.
func (m *MissHandle) StoreWithTokens(response json.RawMessage, inputTokens, outputTokens int) error {
	if m.noStore {
		return nil
	}
	return m.cache.storeResponse(m, response, inputTokens, outputTokens)
}

// StoreStream writes a completed streaming response to the cache.
// chunks must be in delivery order; each element is the JSON encoding of one
// streaming chunk from the provider.
// It is a no-op when LookupOptions.NoStore was set.
func (m *MissHandle) StoreStream(chunks []json.RawMessage) error {
	if m.noStore {
		return nil
	}
	return m.cache.storeStreamResponse(m, chunks, 0, 0)
}

// StoreStreamWithTokens writes a completed streaming response along with token
// counts for cost-saving metrics. See StoreWithTokens for details.
func (m *MissHandle) StoreStreamWithTokens(chunks []json.RawMessage, inputTokens, outputTokens int) error {
	if m.noStore {
		return nil
	}
	return m.cache.storeStreamResponse(m, chunks, inputTokens, outputTokens)
}

// ── Internal stream chunk used during write ────────────────────────────────

// streamChunk is the internal representation of one streaming chunk retained
// during write-path accumulation. For external use see StoreStream.
type streamChunk struct {
	Timestamp time.Time
	Index     int
	Data      json.RawMessage
}
