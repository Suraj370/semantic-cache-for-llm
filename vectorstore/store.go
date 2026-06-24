// Package vectorstore provides a generic interface and adapters for vector databases
// used as the backing store for the semantic cache.
package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/suraj370/semantic-cache/logger"
)

// VectorStoreType identifies the backing store implementation.
type VectorStoreType string

const (
	VectorStoreTypeWeaviate VectorStoreType = "weaviate"
	VectorStoreTypeRedis    VectorStoreType = "redis"
	VectorStoreTypeQdrant   VectorStoreType = "qdrant"
	VectorStoreTypePinecone VectorStoreType = "pinecone"
)

// Query represents a scalar-filter predicate applied alongside a vector search.
type Query struct {
	Field    string
	Operator QueryOperator
	Value    interface{}
}

// QueryOperator defines the comparison operation for a Query.
type QueryOperator string

const (
	QueryOperatorEqual              QueryOperator = "Equal"
	QueryOperatorNotEqual           QueryOperator = "NotEqual"
	QueryOperatorGreaterThan        QueryOperator = "GreaterThan"
	QueryOperatorLessThan           QueryOperator = "LessThan"
	QueryOperatorGreaterThanOrEqual QueryOperator = "GreaterThanOrEqual"
	QueryOperatorLessThanOrEqual    QueryOperator = "LessThanOrEqual"
	QueryOperatorLike               QueryOperator = "Like"
	QueryOperatorContainsAny        QueryOperator = "ContainsAny"
	QueryOperatorContainsAll        QueryOperator = "ContainsAll"
	QueryOperatorIsNull             QueryOperator = "IsNull"
	QueryOperatorIsNotNull          QueryOperator = "IsNotNull"
)

// SearchResult is a single entry returned by a vector search or point fetch.
type SearchResult struct {
	ID         string
	Score      *float64               // cosine similarity; nil for non-ANN lookups
	Properties map[string]interface{} // projected metadata fields
}

// DeleteResult reports the outcome of a single delete within a bulk operation.
type DeleteResult struct {
	ID     string
	Status DeleteStatus
	Error  string
}

// DeleteStatus indicates whether a delete succeeded or failed.
type DeleteStatus string

const (
	DeleteStatusSuccess DeleteStatus = "success"
	DeleteStatusError   DeleteStatus = "error"
)

// VectorStoreProperties describes a single metadata property for schema creation.
type VectorStoreProperties struct {
	DataType    VectorStorePropertyType `json:"data_type"`
	Description string                  `json:"description"`
}

// VectorStorePropertyType is the data type of a metadata property.
type VectorStorePropertyType string

const (
	VectorStorePropertyTypeString      VectorStorePropertyType = "string"
	VectorStorePropertyTypeInteger     VectorStorePropertyType = "integer"
	VectorStorePropertyTypeBoolean     VectorStorePropertyType = "boolean"
	VectorStorePropertyTypeStringArray VectorStorePropertyType = "string[]"
)

// disableScanFallbackKey is the context key for disabling full-scan fallback.
type disableScanFallbackKey struct{}

// VectorStore is the generic interface over vector database backends.
type VectorStore interface {
	// Ping verifies the store is reachable.
	Ping(ctx context.Context) error
	// CreateNamespace creates a namespace (collection/index) with the given schema.
	CreateNamespace(ctx context.Context, namespace string, dimension int, properties map[string]VectorStoreProperties) error
	// DeleteNamespace removes a namespace and all its entries.
	DeleteNamespace(ctx context.Context, namespace string) error
	// GetChunk retrieves a single entry by ID.
	GetChunk(ctx context.Context, namespace string, id string) (SearchResult, error)
	// GetChunks retrieves multiple entries by ID.
	GetChunks(ctx context.Context, namespace string, ids []string) ([]SearchResult, error)
	// GetAll returns entries matching the scalar filters, with pagination.
	GetAll(ctx context.Context, namespace string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error)
	// GetNearest returns the nearest entries to the given vector, filtered by queries.
	GetNearest(ctx context.Context, namespace string, vector []float32, queries []Query, selectFields []string, threshold float64, limit int64) ([]SearchResult, error)
	// RequiresVectors reports whether every stored entry must carry a vector.
	// Dedicated vector databases (Qdrant, Pinecone) return true; hybrid stores
	// (Weaviate, Redis) that can hold metadata-only entries return false.
	RequiresVectors() bool
	// Add stores a new entry with the given ID, optional vector, and metadata.
	Add(ctx context.Context, namespace string, id string, embedding []float32, metadata map[string]interface{}) error
	// Delete removes a single entry by ID.
	Delete(ctx context.Context, namespace string, id string) error
	// DeleteAll removes all entries matching the given scalar filters.
	DeleteAll(ctx context.Context, namespace string, queries []Query) ([]DeleteResult, error)
	// Close releases any resources held by the store for the namespace.
	Close(ctx context.Context, namespace string) error
}

// WithDisableScanFallback returns a context that tells the store not to fall
// back to full scans when an indexed search fails.
func WithDisableScanFallback(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, disableScanFallbackKey{}, true)
}

// IsScanFallbackDisabled reports whether the context disables scan fallback.
func IsScanFallbackDisabled(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	v, _ := ctx.Value(disableScanFallbackKey{}).(bool)
	return v
}

// Config is the top-level configuration for choosing and configuring a vector
// store backend. Pass this to NewVectorStore to get a ready VectorStore.
type Config struct {
	Enabled bool            `json:"enabled"`
	Type    VectorStoreType `json:"type"`
	Config  any             `json:"config"`
}

// UnmarshalJSON provides type-aware deserialization of the nested Config field.
func (c *Config) UnmarshalJSON(data []byte) error {
	type tempConfig struct {
		Enabled bool            `json:"enabled"`
		Type    string          `json:"type"`
		Config  json.RawMessage `json:"config"`
	}
	var tmp tempConfig
	if err := json.Unmarshal(data, &tmp); err != nil {
		return fmt.Errorf("failed to unmarshal vectorstore config: %w", err)
	}
	c.Enabled = tmp.Enabled
	c.Type = VectorStoreType(tmp.Type)

	switch c.Type {
	case VectorStoreTypeWeaviate:
		var cfg WeaviateConfig
		if err := json.Unmarshal(tmp.Config, &cfg); err != nil {
			return fmt.Errorf("failed to unmarshal weaviate config: %w", err)
		}
		c.Config = cfg
	case VectorStoreTypeRedis:
		var cfg RedisConfig
		if err := json.Unmarshal(tmp.Config, &cfg); err != nil {
			return fmt.Errorf("failed to unmarshal redis config: %w", err)
		}
		c.Config = cfg
	case VectorStoreTypeQdrant:
		var cfg QdrantConfig
		if err := json.Unmarshal(tmp.Config, &cfg); err != nil {
			return fmt.Errorf("failed to unmarshal qdrant config: %w", err)
		}
		c.Config = cfg
	case VectorStoreTypePinecone:
		var cfg PineconeConfig
		if err := json.Unmarshal(tmp.Config, &cfg); err != nil {
			return fmt.Errorf("failed to unmarshal pinecone config: %w", err)
		}
		c.Config = cfg
	default:
		return fmt.Errorf("unknown vector store type: %s", tmp.Type)
	}
	return nil
}

// NewVectorStore constructs and connects the vector store described by config.
func NewVectorStore(ctx context.Context, config *Config, log logger.Logger) (VectorStore, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	if !config.Enabled {
		return nil, fmt.Errorf("vector store is disabled")
	}
	switch config.Type {
	case VectorStoreTypeWeaviate:
		cfg, ok := config.Config.(WeaviateConfig)
		if !ok {
			return nil, fmt.Errorf("invalid weaviate config type")
		}
		return newWeaviateStore(ctx, &cfg, log)
	case VectorStoreTypeRedis:
		cfg, ok := config.Config.(RedisConfig)
		if !ok {
			return nil, fmt.Errorf("invalid redis config type")
		}
		return newRedisStore(ctx, cfg, log)
	case VectorStoreTypeQdrant:
		cfg, ok := config.Config.(QdrantConfig)
		if !ok {
			return nil, fmt.Errorf("invalid qdrant config type")
		}
		return newQdrantStore(ctx, &cfg, log)
	case VectorStoreTypePinecone:
		cfg, ok := config.Config.(PineconeConfig)
		if !ok {
			return nil, fmt.Errorf("invalid pinecone config type")
		}
		return newPineconeStore(ctx, &cfg, log)
	}
	return nil, fmt.Errorf("unknown vector store type: %s", config.Type)
}
