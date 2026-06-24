// Package cache implements semantic caching for LLM requests using dual-path
// lookup: an O(1) direct hash match and an ANN-based semantic similarity search.
package cache

import (
	"encoding/json"
	"fmt"
	"time"
)

// CacheType narrows which lookup path is used for a request.
type CacheType string

const (
	// CacheTypeDirect runs only the deterministic hash lookup (O(1)).
	CacheTypeDirect CacheType = "direct"
	// CacheTypeSemantic runs only the embedding-based similarity search.
	CacheTypeSemantic CacheType = "semantic"
)

// Config is the configuration for a Cache instance.
//
// Modes:
//   - Semantic mode (default): set EmbeddingDimension > 1. Both direct hash
//     and embedding-based similarity search are enabled.
//   - Direct-only mode: set EmbeddingDimension = 1 and leave Embedder nil.
//     Only the deterministic hash path is used.
type Config struct {
	// Namespace is the vector store collection/index name. Defaults to
	// "SemanticCache" when empty.
	Namespace string `json:"namespace,omitempty"`

	// TTL is the default time-to-live for cache entries. Defaults to 5 minutes.
	TTL time.Duration `json:"ttl,omitempty"`

	// Threshold is the cosine similarity threshold for semantic hits.
	// Values closer to 1.0 require near-identical semantics; values around
	// 0.8 tolerate slight paraphrasing. Defaults to 0.8.
	Threshold float64 `json:"threshold,omitempty"`

	// EmbeddingDimension is the vector dimension for the backing store.
	// Must be > 0. Set to 1 for direct-only mode (no embeddings).
	EmbeddingDimension int `json:"embedding_dimension"`

	// DefaultCacheKey is used when the caller does not supply a per-request
	// key in LookupOptions.CacheKey. Caching is disabled when both are empty.
	DefaultCacheKey string `json:"default_cache_key,omitempty"`

	// ConversationHistoryThreshold skips caching when the request has more
	// messages than this. Defaults to 3. Set to 0 to disable the check.
	ConversationHistoryThreshold int `json:"conversation_history_threshold,omitempty"`

	// CacheByModel includes the model name in the cache key when true (default true).
	CacheByModel *bool `json:"cache_by_model,omitempty"`

	// CacheByProvider includes the provider name in the cache key when true (default true).
	CacheByProvider *bool `json:"cache_by_provider,omitempty"`

	// ExcludeSystemPrompt omits system messages from the cache key when true (default false).
	ExcludeSystemPrompt *bool `json:"exclude_system_prompt,omitempty"`
}

// UnmarshalJSON supports TTL as either a Go duration string ("5m") or seconds.
func (c *Config) UnmarshalJSON(data []byte) error {
	type alias Config
	aux := &struct {
		TTL json.RawMessage `json:"ttl,omitempty"`
		*alias
	}{alias: (*alias)(c)}
	if err := json.Unmarshal(data, aux); err != nil {
		return fmt.Errorf("failed to unmarshal cache config: %w", err)
	}
	if len(aux.TTL) == 0 || string(aux.TTL) == "null" {
		return nil
	}
	var s string
	if err := json.Unmarshal(aux.TTL, &s); err == nil {
		d, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("failed to parse TTL duration string %q: %w", s, err)
		}
		c.TTL = d
		return nil
	}
	var seconds float64
	if err := json.Unmarshal(aux.TTL, &seconds); err != nil {
		return fmt.Errorf("unsupported TTL value: %s", string(aux.TTL))
	}
	c.TTL = time.Duration(seconds * float64(time.Second))
	if c.TTL < 0 {
		return fmt.Errorf("TTL must be non-negative, got %v", c.TTL)
	}
	return nil
}

// Plugin-wide constants.
const (
	DefaultNamespace                    = "SemanticCache"
	DefaultTTL               time.Duration = 5 * time.Minute
	DefaultThreshold         float64       = 0.8
	DefaultConversationLimit int           = 3
	CacheConnectionTimeout   time.Duration = 5 * time.Second
	CreateNamespaceTimeout   time.Duration = 30 * time.Second
	CacheSetTimeout          time.Duration = 30 * time.Second
)

// VectorStoreProperties are the metadata properties registered when the
// namespace is created. The cache writes all of these on every entry.
// params_hash and from_semantic_cache are filter-only — they are not
// projected back on reads (see selectFields in cache.go).
var VectorStoreProperties = map[string]propertyDef{
	"response": {
		DataType:    propertyTypeString,
		Description: "JSON-encoded LLM response",
	},
	"stream_chunks": {
		DataType:    propertyTypeStringArray,
		Description: "JSON-encoded streaming response chunks (one per array element)",
	},
	"expires_at": {
		DataType:    propertyTypeInteger,
		Description: "Unix timestamp when the entry expires",
	},
	"cache_key": {
		DataType:    propertyTypeString,
		Description: "Caller-supplied tenant / feature scope key",
	},
	"provider": {
		DataType:    propertyTypeString,
		Description: "LLM provider name",
	},
	"model": {
		DataType:    propertyTypeString,
		Description: "LLM model name",
	},
	"params_hash": {
		DataType:    propertyTypeString,
		Description: "Hash of request parameters (filter-only)",
	},
	"from_semantic_cache": {
		DataType:    propertyTypeBoolean,
		Description: "Sentinel flag identifying entries written by this cache (filter-only)",
	},
}

// selectFields is the default projection for reads (excludes filter-only fields).
var selectFields = []string{"response", "stream_chunks", "expires_at", "cache_key", "provider", "model"}

// Property type aliases (matching vectorstore.VectorStorePropertyType values).
type propertyDef struct {
	DataType    string
	Description string
}

const (
	propertyTypeString      = "string"
	propertyTypeInteger     = "integer"
	propertyTypeBoolean     = "boolean"
	propertyTypeStringArray = "string[]"
)
