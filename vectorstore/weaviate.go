package vectorstore

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/suraj370/semantic-cache/logger"
	"github.com/suraj370/semantic-cache/types"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/auth"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/filters"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/graphql"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/grpc"
	"github.com/weaviate/weaviate/entities/models"
)

const DefaultClassName = "SemanticCacheStore"

// WeaviateConfig represents the configuration for the Weaviate vector store.
type WeaviateConfig struct {
	Scheme     string              `json:"scheme"`
	Host       *types.SecretVar    `json:"host"`
	GrpcConfig *WeaviateGrpcConfig `json:"grpc_config,omitempty"`
	APIKey     *types.SecretVar    `json:"api_key,omitempty"`
	Headers    map[string]string   `json:"headers,omitempty"`
	Timeout    types.Duration      `json:"timeout,omitempty"`
}

type WeaviateGrpcConfig struct {
	Host    *types.SecretVar `json:"host"`
	Secured bool             `json:"secured"`
}

// WeaviateStore represents the Weaviate vector store.
type WeaviateStore struct {
	client *weaviate.Client
	config *WeaviateConfig
	logger logger.Logger
}

// Ping checks if the Weaviate server is reachable.
func (s *WeaviateStore) Ping(ctx context.Context) error {
	_, err := s.client.Misc().MetaGetter().Do(ctx)
	return err
}

// Add stores a new object (with or without embedding).
func (s *WeaviateStore) Add(ctx context.Context, className string, id string, embedding []float32, metadata map[string]interface{}) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}
	obj := &models.Object{
		Class:      className,
		Properties: metadata,
	}
	var err error
	if len(embedding) > 0 {
		_, err = s.client.Data().Creator().
			WithClassName(className).
			WithID(id).
			WithProperties(obj.Properties).
			WithVector(embedding).
			Do(ctx)
	} else {
		_, err = s.client.Data().Creator().
			WithClassName(className).
			WithID(id).
			WithProperties(obj.Properties).
			Do(ctx)
	}
	return err
}

// GetChunk returns the metadata for a single entry by ID.
func (s *WeaviateStore) GetChunk(ctx context.Context, className string, id string) (SearchResult, error) {
	obj, err := s.client.Data().ObjectsGetter().
		WithClassName(className).
		WithID(id).
		Do(ctx)
	if err != nil {
		return SearchResult{}, err
	}
	if len(obj) == 0 {
		return SearchResult{}, fmt.Errorf("not found: %s", id)
	}
	props, ok := obj[0].Properties.(map[string]interface{})
	if !ok {
		return SearchResult{}, fmt.Errorf("invalid properties")
	}
	return SearchResult{ID: id, Properties: props}, nil
}

// GetChunks returns multiple objects by ID.
func (s *WeaviateStore) GetChunks(ctx context.Context, className string, ids []string) ([]SearchResult, error) {
	out := make([]SearchResult, 0, len(ids))
	for _, id := range ids {
		obj, err := s.client.Data().ObjectsGetter().
			WithClassName(className).
			WithID(id).
			Do(ctx)
		if err != nil {
			return nil, err
		}
		if len(obj) > 0 {
			props, ok := obj[0].Properties.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("invalid properties")
			}
			out = append(out, SearchResult{ID: id, Properties: props})
		}
	}
	return out, nil
}

// GetAll returns entries with optional filtering and cursor-based pagination.
func (s *WeaviateStore) GetAll(ctx context.Context, className string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error) {
	where := buildWeaviateFilter(queries)
	fields := []graphql.Field{
		{Name: "_additional", Fields: []graphql.Field{{Name: "id"}}},
	}
	for _, f := range selectFields {
		fields = append(fields, graphql.Field{Name: f})
	}
	search := s.client.GraphQL().Get().
		WithClassName(className).
		WithLimit(int(limit)).
		WithFields(fields...)
	if where != nil {
		search = search.WithWhere(where)
	}
	if cursor != nil {
		search = search.WithAfter(*cursor)
	}
	resp, err := search.Do(ctx)
	if err != nil {
		return nil, nil, err
	}
	if len(resp.Errors) > 0 {
		var msgs []string
		for _, e := range resp.Errors {
			msgs = append(msgs, e.Message)
		}
		return nil, nil, fmt.Errorf("graphql errors: %v", msgs)
	}
	data, ok := resp.Data["Get"].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("invalid graphql response: missing 'Get' key")
	}
	objsRaw, exists := data[className]
	if !exists {
		s.logger.Debug("no results for class %q", className)
		return nil, nil, nil
	}
	objs, ok := objsRaw.([]interface{})
	if !ok {
		s.logger.Debug("class %q data is not an array", className)
		return nil, nil, nil
	}
	results := make([]SearchResult, 0, len(objs))
	var nextCursor *string
	for _, o := range objs {
		obj, ok := o.(map[string]interface{})
		if !ok {
			continue
		}
		sr := SearchResult{Properties: obj}
		if add, ok := obj["_additional"].(map[string]interface{}); ok {
			if id, ok := add["id"].(string); ok {
				sr.ID = id
				nextCursor = &id
			}
		}
		results = append(results, sr)
	}
	return results, nextCursor, nil
}

// GetNearest returns the nearest entries to the given vector.
func (s *WeaviateStore) GetNearest(ctx context.Context, className string, vector []float32, queries []Query, selectFields []string, threshold float64, limit int64) ([]SearchResult, error) {
	where := buildWeaviateFilter(queries)
	fields := []graphql.Field{
		{Name: "_additional", Fields: []graphql.Field{{Name: "id"}, {Name: "certainty"}}},
	}
	for _, f := range selectFields {
		fields = append(fields, graphql.Field{Name: f})
	}
	nearVector := s.client.GraphQL().NearVectorArgBuilder().
		WithVector(vector).
		WithCertainty(float32(threshold))
	search := s.client.GraphQL().Get().
		WithClassName(className).
		WithNearVector(nearVector).
		WithLimit(int(limit)).
		WithFields(fields...)
	if where != nil {
		search = search.WithWhere(where)
	}
	resp, err := search.Do(ctx)
	if err != nil {
		return nil, err
	}
	if len(resp.Errors) > 0 {
		var msgs []string
		for _, e := range resp.Errors {
			msgs = append(msgs, e.Message)
		}
		return nil, fmt.Errorf("graphql errors: %v", msgs)
	}
	data, ok := resp.Data["Get"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid graphql response")
	}
	objsRaw, exists := data[className]
	if !exists {
		s.logger.Debug("no results for class %q", className)
		return nil, nil
	}
	objs, ok := objsRaw.([]interface{})
	if !ok {
		s.logger.Debug("class %q data is not an array", className)
		return nil, nil
	}
	results := make([]SearchResult, 0, len(objs))
	for _, o := range objs {
		obj, ok := o.(map[string]interface{})
		if !ok {
			continue
		}
		add, ok := obj["_additional"].(map[string]interface{})
		if !ok {
			continue
		}
		idRaw, ok := add["id"]
		if !ok || idRaw == nil {
			continue
		}
		id, ok := idRaw.(string)
		if !ok {
			continue
		}
		var score float64
		if certaintyRaw, exists := add["certainty"]; exists && certaintyRaw != nil {
			switch v := certaintyRaw.(type) {
			case float64:
				score = v
			case float32:
				score = float64(v)
			case int:
				score = float64(v)
			case int64:
				score = float64(v)
			}
		}
		results = append(results, SearchResult{ID: id, Score: &score, Properties: obj})
	}
	return results, nil
}

// Delete removes a single entry by ID.
func (s *WeaviateStore) Delete(ctx context.Context, className string, id string) error {
	return s.client.Data().Deleter().
		WithClassName(className).
		WithID(id).
		Do(ctx)
}

// DeleteAll removes all entries matching the queries.
func (s *WeaviateStore) DeleteAll(ctx context.Context, className string, queries []Query) ([]DeleteResult, error) {
	exists, err := s.client.Schema().ClassExistenceChecker().
		WithClassName(className).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to check class existence: %w", err)
	}
	if !exists {
		return []DeleteResult{}, nil
	}
	where := buildWeaviateFilter(queries)
	res, err := s.client.Batch().ObjectsBatchDeleter().
		WithClassName(className).
		WithWhere(where).
		Do(ctx)
	if err != nil {
		return nil, err
	}
	results := make([]DeleteResult, 0, len(res.Results.Objects))
	for _, obj := range res.Results.Objects {
		r := DeleteResult{ID: obj.ID.String()}
		if obj.Status != nil {
			switch *obj.Status {
			case "SUCCESS":
				r.Status = DeleteStatusSuccess
			case "FAILED":
				r.Status = DeleteStatusError
				if obj.Errors != nil {
					var msgs []string
					for _, e := range obj.Errors.Error {
						msgs = append(msgs, e.Message)
					}
					r.Error = strings.Join(msgs, ", ")
				}
			}
		}
		results = append(results, r)
	}
	return results, nil
}

// Close is a no-op for Weaviate (the HTTP client has no persistent connection).
func (s *WeaviateStore) Close(ctx context.Context, className string) error { return nil }

// RequiresVectors reports whether the store requires a vector on every entry.
func (s *WeaviateStore) RequiresVectors() bool { return true }

// newWeaviateStore creates and connects a Weaviate vector store.
func newWeaviateStore(ctx context.Context, config *WeaviateConfig, log logger.Logger) (*WeaviateStore, error) {
	if config.Scheme == "" || config.Host == nil || config.Host.GetValue() == "" {
		return nil, fmt.Errorf("weaviate scheme and host are required")
	}
	cfg := weaviate.Config{
		Scheme: config.Scheme,
		Host:   config.Host.GetValue(),
	}
	if config.APIKey != nil && config.APIKey.GetValue() != "" {
		cfg.AuthConfig = auth.ApiKey{Value: config.APIKey.GetValue()}
	}
	if config.GrpcConfig != nil {
		if config.GrpcConfig.Host == nil || config.GrpcConfig.Host.GetValue() == "" {
			return nil, fmt.Errorf("weaviate grpc host is required")
		}
		cfg.GrpcConfig = &grpc.Config{
			Host:    config.GrpcConfig.Host.GetValue(),
			Secured: config.GrpcConfig.Secured,
		}
	}
	if len(config.Headers) > 0 {
		cfg.Headers = config.Headers
	}
	client, err := weaviate.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create weaviate client: %w", err)
	}
	testCtx := ctx
	if config.Timeout > 0 {
		var cancel context.CancelFunc
		testCtx, cancel = context.WithTimeout(ctx, time.Duration(config.Timeout))
		defer cancel()
	}
	if _, err = client.Misc().MetaGetter().Do(testCtx); err != nil {
		return nil, fmt.Errorf("failed to connect to weaviate: %w", err)
	}
	return &WeaviateStore{client: client, config: config, logger: log}, nil
}

// CreateNamespace creates a Weaviate class with the given schema.
func (s *WeaviateStore) CreateNamespace(ctx context.Context, className string, dimension int, properties map[string]VectorStoreProperties) error {
	if err := validateClassName(className); err != nil {
		return err
	}
	exists, err := s.client.Schema().ClassExistenceChecker().WithClassName(className).Do(ctx)
	if err != nil {
		return fmt.Errorf("failed to check class existence: %w", err)
	}
	if exists {
		return nil
	}
	var weaviateProps []*models.Property
	for name, prop := range properties {
		var dataType []string
		switch prop.DataType {
		case VectorStorePropertyTypeString:
			dataType = []string{"string"}
		case VectorStorePropertyTypeInteger:
			dataType = []string{"int"}
		case VectorStorePropertyTypeBoolean:
			dataType = []string{"boolean"}
		case VectorStorePropertyTypeStringArray:
			dataType = []string{"string[]"}
		}
		weaviateProps = append(weaviateProps, &models.Property{
			Name:        name,
			DataType:    dataType,
			Description: prop.Description,
		})
	}
	classSchema := &models.Class{
		Class:           className,
		Properties:      weaviateProps,
		VectorIndexType: "hnsw",
		Vectorizer:      "none",
	}
	if dimension > 0 {
		classSchema.VectorIndexConfig = map[string]interface{}{"vectorDimensions": dimension}
	}
	if err := s.client.Schema().ClassCreator().WithClass(classSchema).Do(ctx); err != nil {
		return fmt.Errorf("failed to create class schema: %w", err)
	}
	return nil
}

// DeleteNamespace drops the Weaviate class.
func (s *WeaviateStore) DeleteNamespace(ctx context.Context, className string) error {
	exists, err := s.client.Schema().ClassExistenceChecker().WithClassName(className).Do(ctx)
	if err != nil {
		return fmt.Errorf("failed to check class existence: %w", err)
	}
	if !exists {
		return nil
	}
	return s.client.Schema().ClassDeleter().WithClassName(className).Do(ctx)
}

// buildWeaviateFilter converts []Query into a Weaviate WhereFilter.
func buildWeaviateFilter(queries []Query) *filters.WhereBuilder {
	if len(queries) == 0 {
		return nil
	}
	var operands []*filters.WhereBuilder
	for _, q := range queries {
		op := convertOperator(q.Operator)
		path := strings.Split(q.Field, ".")
		w := filters.Where().WithPath(path).WithOperator(op)
		switch q.Operator {
		case QueryOperatorIsNull:
			w = w.WithValueBoolean(true)
		case QueryOperatorIsNotNull:
			w = w.WithValueBoolean(false)
		default:
			switch v := q.Value.(type) {
			case string:
				w = w.WithValueString(v)
			case int:
				w = w.WithValueInt(int64(v))
			case int64:
				w = w.WithValueInt(v)
			case float32:
				w = w.WithValueNumber(float64(v))
			case float64:
				w = w.WithValueNumber(v)
			case bool:
				w = w.WithValueBoolean(v)
			default:
				w = w.WithValueString(fmt.Sprintf("%v", v))
			}
		}
		operands = append(operands, w)
	}
	if len(operands) == 1 {
		return operands[0]
	}
	return filters.Where().WithOperator(filters.And).WithOperands(operands)
}

func convertOperator(op QueryOperator) filters.WhereOperator {
	switch op {
	case QueryOperatorEqual:
		return filters.Equal
	case QueryOperatorNotEqual:
		return filters.NotEqual
	case QueryOperatorLessThan:
		return filters.LessThan
	case QueryOperatorLessThanOrEqual:
		return filters.LessThanEqual
	case QueryOperatorGreaterThan:
		return filters.GreaterThan
	case QueryOperatorGreaterThanOrEqual:
		return filters.GreaterThanEqual
	case QueryOperatorLike:
		return filters.Like
	case QueryOperatorContainsAny:
		return filters.ContainsAny
	case QueryOperatorContainsAll:
		return filters.ContainsAll
	case QueryOperatorIsNull:
		return filters.IsNull
	case QueryOperatorIsNotNull:
		return filters.IsNull // Weaviate uses IsNull=false for IsNotNull
	default:
		return filters.Equal
	}
}

func validateClassName(name string) error {
	if name == "" {
		return nil
	}
	if name[0] < 'A' || name[0] > 'Z' {
		return fmt.Errorf("weaviate class names must start with an uppercase letter; got %q, try %q", name, strings.ToUpper(name[:1])+name[1:])
	}
	return nil
}
