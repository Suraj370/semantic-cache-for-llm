package vectorstore

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/suraj370/semantic-cache/logger"
	"github.com/suraj370/semantic-cache/types"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
)

const qdrantMaxRecvMsgSize = 64 * 1024 * 1024 // 64 MB

// QdrantConfig holds connection settings for the Qdrant vector store.
type QdrantConfig struct {
	Host             types.SecretVar `json:"host"`
	Port             types.SecretVar `json:"port"`
	APIKey           types.SecretVar `json:"api_key,omitempty"`
	UseTLS           types.SecretVar `json:"use_tls,omitempty"`
	MaxRecvMsgSizeMB types.SecretVar `json:"max_recv_msg_size_mb,omitempty"`
}

// QdrantStore is the Qdrant VectorStore implementation.
type QdrantStore struct {
	client *qdrant.Client
	logger logger.Logger
}

// Ping checks if Qdrant is reachable.
func (s *QdrantStore) Ping(ctx context.Context) error {
	_, err := s.client.HealthCheck(ctx)
	return err
}

// CreateNamespace creates a Qdrant collection with the given schema.
func (s *QdrantStore) CreateNamespace(ctx context.Context, namespace string, dimension int, properties map[string]VectorStoreProperties) error {
	exists, err := s.client.CollectionExists(ctx, namespace)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}
	if exists {
		info, infoErr := s.client.GetCollectionInfo(ctx, namespace)
		if infoErr != nil {
			s.logger.Warn("could not inspect existing collection %q for dimension validation: %v", namespace, infoErr)
		} else if params := info.GetConfig().GetParams().GetVectorsConfig().GetParams(); params == nil {
			s.logger.Debug("collection %q uses named vectors — dimension check skipped", namespace)
		} else {
			existingDim := int(params.GetSize())
			if existingDim != dimension {
				return fmt.Errorf("namespace %q already exists with dimension %d but config requires %d", namespace, existingDim, dimension)
			}
		}
	}

	if !exists {
		if err := s.client.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: namespace,
			VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
				Size:     uint64(dimension),
				Distance: qdrant.Distance_Cosine,
			}),
		}); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
	}

	for fieldName, prop := range properties {
		var fieldType qdrant.FieldType
		switch prop.DataType {
		case VectorStorePropertyTypeInteger:
			fieldType = qdrant.FieldType_FieldTypeInteger
		case VectorStorePropertyTypeBoolean:
			fieldType = qdrant.FieldType_FieldTypeBool
		default:
			fieldType = qdrant.FieldType_FieldTypeKeyword
		}
		if _, err := s.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: namespace,
			FieldName:      fieldName,
			FieldType:      &fieldType,
		}); err != nil {
			s.logger.Debug("failed to create index for field %s: %v", fieldName, err)
		}
	}
	return nil
}

// DeleteNamespace drops a Qdrant collection.
func (s *QdrantStore) DeleteNamespace(ctx context.Context, namespace string) error {
	exists, err := s.client.CollectionExists(ctx, namespace)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}
	if !exists {
		return nil
	}
	return s.client.DeleteCollection(ctx, namespace)
}

// GetChunk retrieves a single point by ID.
func (s *QdrantStore) GetChunk(ctx context.Context, namespace string, id string) (SearchResult, error) {
	if strings.TrimSpace(id) == "" {
		return SearchResult{}, fmt.Errorf("id is required")
	}
	pointID, err := parsePointID(id)
	if err != nil {
		return SearchResult{}, fmt.Errorf("invalid id format: %w", err)
	}
	points, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: namespace,
		Ids:            []*qdrant.PointId{pointID},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return SearchResult{}, fmt.Errorf("failed to get point: %w", err)
	}
	if len(points) == 0 {
		return SearchResult{}, fmt.Errorf("not found: %s", id)
	}
	return SearchResult{ID: id, Properties: payloadToMap(points[0].Payload)}, nil
}

// GetChunks retrieves multiple points by ID.
func (s *QdrantStore) GetChunks(ctx context.Context, namespace string, ids []string) ([]SearchResult, error) {
	if len(ids) == 0 {
		return []SearchResult{}, nil
	}
	pointIDs := make([]*qdrant.PointId, 0, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) == "" {
			continue
		}
		pid, err := parsePointID(id)
		if err != nil {
			s.logger.Debug("skipping invalid id %s: %v", id, err)
			continue
		}
		pointIDs = append(pointIDs, pid)
	}
	if len(pointIDs) == 0 {
		return []SearchResult{}, nil
	}
	points, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: namespace,
		Ids:            pointIDs,
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get points: %w", err)
	}
	results := make([]SearchResult, 0, len(points))
	for _, p := range points {
		results = append(results, SearchResult{
			ID:         pointIDToString(p.Id),
			Properties: payloadToMap(p.Payload),
		})
	}
	return results, nil
}

// GetAll retrieves points with optional filtering and cursor pagination.
func (s *QdrantStore) GetAll(ctx context.Context, namespace string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error) {
	filter := buildQdrantFilter(queries)
	var offset *qdrant.PointId
	if cursor != nil && *cursor != "" {
		var err error
		if offset, err = parsePointID(*cursor); err != nil {
			s.logger.Debug("invalid cursor format: %v", err)
		}
	}
	scrollLimit := uint32(limit)
	if limit <= 0 {
		scrollLimit = 100
	}
	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: namespace,
		Filter:         filter,
		Limit:          &scrollLimit,
		Offset:         offset,
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to scroll points: %w", err)
	}
	results := make([]SearchResult, 0, len(scrollResult))
	var lastID string
	for _, p := range scrollResult {
		lastID = pointIDToString(p.Id)
		results = append(results, SearchResult{
			ID:         lastID,
			Properties: filterProperties(payloadToMap(p.Payload), selectFields),
		})
	}
	if len(scrollResult) >= int(scrollLimit) {
		return results, &lastID, nil
	}
	return results, nil, nil
}

// GetNearest returns points nearest to the given vector.
func (s *QdrantStore) GetNearest(ctx context.Context, namespace string, vector []float32, queries []Query, selectFields []string, threshold float64, limit int64) ([]SearchResult, error) {
	filter := buildQdrantFilter(queries)
	searchLimit := uint64(limit)
	if limit <= 0 {
		searchLimit = 10
	}
	searchResult, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: namespace,
		Query:          qdrant.NewQuery(vector...),
		Filter:         filter,
		Limit:          &searchLimit,
		WithPayload:    qdrant.NewWithPayload(true),
		ScoreThreshold: qdrant.PtrOf(float32(threshold)),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search points: %w", err)
	}
	results := make([]SearchResult, 0, len(searchResult))
	for _, p := range searchResult {
		score := float64(p.Score)
		results = append(results, SearchResult{
			ID:         pointIDToString(p.Id),
			Score:      &score,
			Properties: filterProperties(payloadToMap(p.Payload), selectFields),
		})
	}
	return results, nil
}

// Add upserts a point into Qdrant.
func (s *QdrantStore) Add(ctx context.Context, namespace string, id string, embedding []float32, metadata map[string]interface{}) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}
	pointID, err := parsePointID(id)
	if err != nil {
		return fmt.Errorf("invalid id format (must be UUID): %w", err)
	}
	point := &qdrant.PointStruct{
		Id:      pointID,
		Payload: mapToPayload(metadata),
	}
	if len(embedding) > 0 {
		point.Vectors = qdrant.NewVectors(embedding...)
	}
	if _, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: namespace,
		Points:         []*qdrant.PointStruct{point},
		Wait:           qdrant.PtrOf(true),
	}); err != nil {
		return fmt.Errorf("failed to upsert point: %w", err)
	}
	return nil
}

// Delete removes a single point by ID.
func (s *QdrantStore) Delete(ctx context.Context, namespace string, id string) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}
	pointID, err := parsePointID(id)
	if err != nil {
		return fmt.Errorf("invalid id format: %w", err)
	}
	_, err = s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: namespace,
		Points:         qdrant.NewPointsSelector(pointID),
	})
	return err
}

// DeleteAll removes all points matching the filter.
func (s *QdrantStore) DeleteAll(ctx context.Context, namespace string, queries []Query) ([]DeleteResult, error) {
	filter := buildQdrantFilter(queries)
	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: namespace,
		Filter:         filter,
		WithPayload:    qdrant.NewWithPayload(false),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to scroll points: %w", err)
	}
	if len(scrollResult) == 0 {
		return []DeleteResult{}, nil
	}
	results := make([]DeleteResult, len(scrollResult))
	for i, p := range scrollResult {
		results[i] = DeleteResult{ID: pointIDToString(p.Id), Status: DeleteStatusSuccess}
	}
	var deleteErr error
	if filter != nil {
		_, deleteErr = s.client.Delete(ctx, &qdrant.DeletePoints{
			CollectionName: namespace,
			Points:         qdrant.NewPointsSelectorFilter(filter),
		})
	} else {
		pointIDs := make([]*qdrant.PointId, len(scrollResult))
		for i, p := range scrollResult {
			pointIDs[i] = p.Id
		}
		_, deleteErr = s.client.Delete(ctx, &qdrant.DeletePoints{
			CollectionName: namespace,
			Points:         qdrant.NewPointsSelectorIDs(pointIDs),
		})
	}
	if deleteErr != nil {
		for i := range results {
			results[i].Status = DeleteStatusError
			results[i].Error = deleteErr.Error()
		}
	}
	return results, nil
}

// Close closes the Qdrant gRPC connection.
func (s *QdrantStore) Close(ctx context.Context, namespace string) error {
	return s.client.Close()
}

// RequiresVectors reports that Qdrant requires a vector on every point.
func (s *QdrantStore) RequiresVectors() bool { return true }

// newQdrantStore creates and connects a Qdrant vector store.
func newQdrantStore(ctx context.Context, config *QdrantConfig, log logger.Logger) (*QdrantStore, error) {
	if strings.TrimSpace(config.Host.GetValue()) == "" {
		return nil, fmt.Errorf("qdrant host is required")
	}
	maxRecvMsgSize := qdrantMaxRecvMsgSize
	if mb := config.MaxRecvMsgSizeMB.CoerceInt(0); mb > 0 {
		maxRecvMsgSize = mb * 1024 * 1024
	}
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   config.Host.GetValue(),
		Port:                   config.Port.CoerceInt(6334),
		APIKey:                 config.APIKey.GetValue(),
		UseTLS:                 config.UseTLS.CoerceBool(false),
		SkipCompatibilityCheck: true,
		GrpcOptions: []grpc.DialOption{
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxRecvMsgSize)),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create qdrant client: %w", err)
	}
	if _, err := client.HealthCheck(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to qdrant: %w", err)
	}
	return &QdrantStore{client: client, logger: log}, nil
}

func parsePointID(id string) (*qdrant.PointId, error) {
	if _, err := uuid.Parse(id); err != nil {
		return nil, err
	}
	return qdrant.NewID(id), nil
}

func pointIDToString(id *qdrant.PointId) string {
	if id == nil {
		return ""
	}
	switch v := id.PointIdOptions.(type) {
	case *qdrant.PointId_Uuid:
		return v.Uuid
	case *qdrant.PointId_Num:
		return fmt.Sprintf("%d", v.Num)
	}
	return ""
}

func payloadToMap(payload map[string]*qdrant.Value) map[string]interface{} {
	if payload == nil {
		return make(map[string]interface{})
	}
	result := make(map[string]interface{}, len(payload))
	for k, v := range payload {
		result[k] = valueToInterface(v)
	}
	return result
}

func valueToInterface(v *qdrant.Value) interface{} {
	if v == nil {
		return nil
	}
	switch val := v.Kind.(type) {
	case *qdrant.Value_StringValue:
		return val.StringValue
	case *qdrant.Value_IntegerValue:
		return val.IntegerValue
	case *qdrant.Value_DoubleValue:
		return val.DoubleValue
	case *qdrant.Value_BoolValue:
		return val.BoolValue
	case *qdrant.Value_ListValue:
		list := make([]interface{}, len(val.ListValue.Values))
		for i, item := range val.ListValue.Values {
			list[i] = valueToInterface(item)
		}
		return list
	case *qdrant.Value_StructValue:
		return payloadToMap(val.StructValue.Fields)
	}
	return nil
}

func mapToPayload(m map[string]interface{}) map[string]*qdrant.Value {
	if m == nil {
		return make(map[string]*qdrant.Value)
	}
	converted := make(map[string]interface{}, len(m))
	for k, v := range m {
		if ss, ok := v.([]string); ok {
			iface := make([]interface{}, len(ss))
			for i, s := range ss {
				iface[i] = s
			}
			converted[k] = iface
		} else {
			converted[k] = v
		}
	}
	return qdrant.NewValueMap(converted)
}

func filterProperties(props map[string]interface{}, selectFields []string) map[string]interface{} {
	if len(selectFields) == 0 {
		return props
	}
	out := make(map[string]interface{}, len(selectFields))
	for _, f := range selectFields {
		if v, ok := props[f]; ok {
			out[f] = v
		}
	}
	return out
}

func buildQdrantFilter(queries []Query) *qdrant.Filter {
	if len(queries) == 0 {
		return nil
	}
	var conditions []*qdrant.Condition
	for _, q := range queries {
		if c := buildQdrantCondition(q); c != nil {
			conditions = append(conditions, c)
		}
	}
	if len(conditions) == 0 {
		return nil
	}
	return &qdrant.Filter{Must: conditions}
}

func buildQdrantCondition(q Query) *qdrant.Condition {
	field := q.Field
	switch q.Operator {
	case QueryOperatorEqual:
		return buildMatchCondition(field, q.Value)
	case QueryOperatorNotEqual:
		if c := buildMatchCondition(field, q.Value); c != nil {
			return qdrant.NewFilterAsCondition(&qdrant.Filter{MustNot: []*qdrant.Condition{c}})
		}
		return nil
	case QueryOperatorGreaterThan:
		return buildRangeCondition(field, q.Value, "gt")
	case QueryOperatorGreaterThanOrEqual:
		return buildRangeCondition(field, q.Value, "gte")
	case QueryOperatorLessThan:
		return buildRangeCondition(field, q.Value, "lt")
	case QueryOperatorLessThanOrEqual:
		return buildRangeCondition(field, q.Value, "lte")
	case QueryOperatorIsNull:
		return qdrant.NewIsNull(field)
	case QueryOperatorIsNotNull:
		return qdrant.NewFilterAsCondition(&qdrant.Filter{MustNot: []*qdrant.Condition{qdrant.NewIsNull(field)}})
	case QueryOperatorContainsAny:
		switch v := q.Value.(type) {
		case []string:
			return qdrant.NewMatchKeywords(field, v...)
		case []int:
			int64s := make([]int64, len(v))
			for i, n := range v {
				int64s[i] = int64(n)
			}
			return qdrant.NewMatchInts(field, int64s...)
		case []int64:
			return qdrant.NewMatchInts(field, v...)
		}
		return buildMatchCondition(field, q.Value)
	case QueryOperatorContainsAll:
		if values, ok := q.Value.([]interface{}); ok {
			var must []*qdrant.Condition
			for _, v := range values {
				if c := buildMatchCondition(field, v); c != nil {
					must = append(must, c)
				}
			}
			if len(must) > 0 {
				return qdrant.NewFilterAsCondition(&qdrant.Filter{Must: must})
			}
		}
		return buildMatchCondition(field, q.Value)
	case QueryOperatorLike:
		if s, ok := q.Value.(string); ok {
			return qdrant.NewMatchText(field, s)
		}
		return nil
	default:
		return buildMatchCondition(field, q.Value)
	}
}

func buildMatchCondition(field string, value interface{}) *qdrant.Condition {
	switch v := value.(type) {
	case string:
		return qdrant.NewMatchKeyword(field, v)
	case int:
		return qdrant.NewMatchInt(field, int64(v))
	case int32:
		return qdrant.NewMatchInt(field, int64(v))
	case int64:
		return qdrant.NewMatchInt(field, v)
	case bool:
		return qdrant.NewMatchBool(field, v)
	default:
		return qdrant.NewMatchKeyword(field, fmt.Sprintf("%v", v))
	}
}

func buildRangeCondition(field string, value interface{}, op string) *qdrant.Condition {
	var f float64
	switch v := value.(type) {
	case int:
		f = float64(v)
	case int32:
		f = float64(v)
	case int64:
		f = float64(v)
	case float32:
		f = float64(v)
	case float64:
		f = v
	default:
		return nil
	}
	r := &qdrant.Range{}
	switch op {
	case "gt":
		r.Gt = qdrant.PtrOf(f)
	case "gte":
		r.Gte = qdrant.PtrOf(f)
	case "lt":
		r.Lt = qdrant.PtrOf(f)
	case "lte":
		r.Lte = qdrant.PtrOf(f)
	}
	return qdrant.NewRange(field, r)
}
