package vectorstore

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/suraj370/semantic-cache/logger"
	"github.com/suraj370/semantic-cache/types"
	"github.com/pinecone-io/go-pinecone/v5/pinecone"
	"google.golang.org/protobuf/types/known/structpb"
)

// PineconeConfig represents the configuration for the Pinecone vector store.
type PineconeConfig struct {
	APIKey    types.SecretVar `json:"api_key"`    // Pinecone API key - REQUIRED
	IndexHost types.SecretVar `json:"index_host"` // Index host URL from Pinecone console - REQUIRED
}

// PineconeStore represents the Pinecone vector store.
type PineconeStore struct {
	client     *pinecone.Client
	indexConn  *pinecone.IndexConnection
	config     *PineconeConfig
	logger     logger.Logger
	mu         sync.RWMutex
	namespaces map[string]*pinecone.IndexConnection
	dimension  int
}

// Ping checks if the Pinecone server is reachable.
func (s *PineconeStore) Ping(ctx context.Context) error {
	_, err := s.indexConn.DescribeIndexStats(ctx)
	return err
}

// CreateNamespace creates a new namespace in the Pinecone vector store.
func (s *PineconeStore) CreateNamespace(ctx context.Context, namespace string, dimension int, properties map[string]VectorStoreProperties) error {
	s.mu.Lock()
	s.dimension = dimension
	s.mu.Unlock()

	_, err := s.indexConn.DescribeIndexStats(ctx)
	if err != nil {
		return fmt.Errorf("failed to verify index connection: %w", err)
	}
	return nil
}

// DeleteNamespace deletes a namespace from the Pinecone vector store.
func (s *PineconeStore) DeleteNamespace(ctx context.Context, namespace string) error {
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return err
	}
	return idxConn.DeleteAllVectorsInNamespace(ctx)
}

// GetChunk retrieves a single vector from the Pinecone vector store.
func (s *PineconeStore) GetChunk(ctx context.Context, namespace string, id string) (SearchResult, error) {
	if strings.TrimSpace(id) == "" {
		return SearchResult{}, fmt.Errorf("id is required")
	}
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return SearchResult{}, err
	}
	res, err := idxConn.FetchVectors(ctx, []string{id})
	if err != nil {
		return SearchResult{}, fmt.Errorf("failed to fetch vector: %w", err)
	}
	if len(res.Vectors) == 0 {
		return SearchResult{}, fmt.Errorf("not found: %s", id)
	}
	vec, exists := res.Vectors[id]
	if !exists || vec == nil {
		return SearchResult{}, fmt.Errorf("not found: %s", id)
	}
	return SearchResult{
		ID:         id,
		Properties: metadataToMap(vec.Metadata),
	}, nil
}

// GetChunks retrieves multiple vectors from the Pinecone vector store.
func (s *PineconeStore) GetChunks(ctx context.Context, namespace string, ids []string) ([]SearchResult, error) {
	if len(ids) == 0 {
		return []SearchResult{}, nil
	}
	validIDs := make([]string, 0, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) != "" {
			validIDs = append(validIDs, id)
		}
	}
	if len(validIDs) == 0 {
		return []SearchResult{}, nil
	}
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return nil, err
	}
	res, err := idxConn.FetchVectors(ctx, validIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch vectors: %w", err)
	}
	results := make([]SearchResult, 0, len(res.Vectors))
	for id, vec := range res.Vectors {
		if vec != nil {
			results = append(results, SearchResult{
				ID:         id,
				Properties: metadataToMap(vec.Metadata),
			})
		}
	}
	return results, nil
}

// GetAll retrieves all vectors with optional filtering and pagination.
func (s *PineconeStore) GetAll(ctx context.Context, namespace string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error) {
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return nil, nil, err
	}
	topK := uint32(limit)
	if limit <= 0 {
		topK = 100
	}
	s.mu.RLock()
	dim := s.dimension
	s.mu.RUnlock()
	if dim <= 0 {
		return nil, nil, fmt.Errorf("dimension not set: CreateNamespace must be called before GetAll")
	}
	zeroVector := make([]float32, dim)
	queryReq := &pinecone.QueryByVectorValuesRequest{
		Vector:          zeroVector,
		TopK:            topK,
		IncludeValues:   false,
		IncludeMetadata: true,
	}
	if len(queries) > 0 {
		filter, filterErr := buildPineconeFilter(queries)
		if filterErr != nil {
			s.logger.Warn("failed to build pinecone filter: %v", filterErr)
		}
		if filter != nil {
			queryReq.MetadataFilter = filter
		}
	}
	res, err := idxConn.QueryByVectorValues(ctx, queryReq)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query vectors: %w", err)
	}
	results := make([]SearchResult, 0, len(res.Matches))
	for _, match := range res.Matches {
		if match.Vector == nil {
			continue
		}
		props := metadataToMap(match.Vector.Metadata)
		results = append(results, SearchResult{
			ID:         match.Vector.Id,
			Properties: filterPropertiesPinecone(props, selectFields),
		})
	}
	return results, nil, nil
}

// GetNearest retrieves the nearest vectors to a given vector.
func (s *PineconeStore) GetNearest(ctx context.Context, namespace string, vector []float32, queries []Query, selectFields []string, threshold float64, limit int64) ([]SearchResult, error) {
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return nil, err
	}
	topK := uint32(limit)
	if limit <= 0 {
		topK = 10
	}
	queryReq := &pinecone.QueryByVectorValuesRequest{
		Vector:          vector,
		TopK:            topK,
		IncludeValues:   false,
		IncludeMetadata: true,
	}
	if len(queries) > 0 {
		filter, filterErr := buildPineconeFilter(queries)
		if filterErr != nil {
			s.logger.Debug("failed to build pinecone filter: %v", filterErr)
		} else if filter != nil {
			queryReq.MetadataFilter = filter
		}
	}
	res, err := idxConn.QueryByVectorValues(ctx, queryReq)
	if err != nil {
		return nil, fmt.Errorf("failed to query vectors: %w", err)
	}
	results := make([]SearchResult, 0, len(res.Matches))
	for _, match := range res.Matches {
		if match.Vector == nil {
			continue
		}
		score := float64(match.Score)
		if score < threshold {
			continue
		}
		props := metadataToMap(match.Vector.Metadata)
		results = append(results, SearchResult{
			ID:         match.Vector.Id,
			Score:      &score,
			Properties: filterPropertiesPinecone(props, selectFields),
		})
	}
	return results, nil
}

// convertMetadataForStructpb converts []string fields to []interface{} for structpb compatibility.
func convertMetadataForStructpb(metadata map[string]interface{}) map[string]interface{} {
	if metadata == nil {
		return nil
	}
	converted := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
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
	return converted
}

// Add stores a new vector in the Pinecone vector store.
func (s *PineconeStore) Add(ctx context.Context, namespace string, id string, embedding []float32, metadata map[string]interface{}) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return err
	}
	var pbMetadata *structpb.Struct
	if len(metadata) > 0 {
		pbMetadata, err = structpb.NewStruct(convertMetadataForStructpb(metadata))
		if err != nil {
			return fmt.Errorf("failed to convert metadata: %w", err)
		}
	}
	vec := &pinecone.Vector{
		Id:       id,
		Metadata: pbMetadata,
	}
	if len(embedding) > 0 {
		vec.Values = &embedding
	}
	_, err = idxConn.UpsertVectors(ctx, []*pinecone.Vector{vec})
	if err != nil {
		return fmt.Errorf("failed to upsert vector: %w", err)
	}
	return nil
}

// Delete removes a vector from the Pinecone vector store.
func (s *PineconeStore) Delete(ctx context.Context, namespace string, id string) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return err
	}
	return idxConn.DeleteVectorsById(ctx, []string{id})
}

// DeleteAll removes multiple vectors matching the filter.
func (s *PineconeStore) DeleteAll(ctx context.Context, namespace string, queries []Query) ([]DeleteResult, error) {
	idxConn, err := s.getNamespaceConnection(namespace)
	if err != nil {
		return nil, err
	}
	if len(queries) > 0 {
		filter, filterErr := buildPineconeFilter(queries)
		if filterErr != nil {
			return nil, fmt.Errorf("failed to build filter: %w", filterErr)
		}
		if filter != nil {
			if err := idxConn.DeleteVectorsByFilter(ctx, filter); err != nil {
				return nil, fmt.Errorf("failed to delete vectors by filter: %w", err)
			}
			return []DeleteResult{}, nil
		}
	}
	listRes, err := idxConn.ListVectors(ctx, &pinecone.ListVectorsRequest{})
	if err != nil {
		return nil, fmt.Errorf("failed to list vectors: %w", err)
	}
	if len(listRes.VectorIds) == 0 {
		return []DeleteResult{}, nil
	}
	deleteIDs := make([]string, 0, len(listRes.VectorIds))
	for _, id := range listRes.VectorIds {
		if id != nil {
			deleteIDs = append(deleteIDs, *id)
		}
	}
	results := make([]DeleteResult, len(deleteIDs))
	for i, id := range deleteIDs {
		results[i] = DeleteResult{ID: id, Status: DeleteStatusSuccess}
	}
	if err := idxConn.DeleteVectorsById(ctx, deleteIDs); err != nil {
		for i := range results {
			results[i].Status = DeleteStatusError
			results[i].Error = err.Error()
		}
	}
	return results, nil
}

// Close closes the Pinecone client connection.
func (s *PineconeStore) Close(ctx context.Context, namespace string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if namespace != "" {
		if conn, exists := s.namespaces[namespace]; exists && conn != nil {
			conn.Close()
			delete(s.namespaces, namespace)
		}
		return nil
	}
	if s.indexConn != nil {
		s.indexConn.Close()
		s.indexConn = nil
	}
	for ns, conn := range s.namespaces {
		if conn != nil {
			conn.Close()
		}
		delete(s.namespaces, ns)
	}
	return nil
}

// RequiresVectors returns true because Pinecone requires vectors for all entries.
func (s *PineconeStore) RequiresVectors() bool { return true }

// newPineconeStore creates a new Pinecone vector store.
func newPineconeStore(ctx context.Context, config *PineconeConfig, log logger.Logger) (*PineconeStore, error) {
	if strings.TrimSpace(config.APIKey.GetValue()) == "" {
		return nil, fmt.Errorf("pinecone api_key is required")
	}
	if strings.TrimSpace(config.IndexHost.GetValue()) == "" {
		return nil, fmt.Errorf("pinecone index_host is required")
	}
	client, err := pinecone.NewClient(pinecone.NewClientParams{
		ApiKey: config.APIKey.GetValue(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create pinecone client: %w", err)
	}
	host := config.IndexHost.GetValue()
	if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
		if strings.HasPrefix(host, "localhost") || strings.HasPrefix(host, "127.0.0.1") {
			host = "http://" + host
		}
	}
	idxConn, err := client.Index(pinecone.NewIndexConnParams{Host: host})
	if err != nil {
		return nil, fmt.Errorf("failed to create index connection: %w", err)
	}
	if _, err = idxConn.DescribeIndexStats(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to pinecone index: %w", err)
	}
	return &PineconeStore{
		client:     client,
		indexConn:  idxConn,
		config:     config,
		logger:     log,
		namespaces: make(map[string]*pinecone.IndexConnection),
	}, nil
}

func (s *PineconeStore) getHostWithScheme() string {
	host := s.config.IndexHost.GetValue()
	if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
		if strings.HasPrefix(host, "localhost") || strings.HasPrefix(host, "127.0.0.1") {
			return "http://" + host
		}
	}
	return host
}

func (s *PineconeStore) getNamespaceConnection(namespace string) (*pinecone.IndexConnection, error) {
	if namespace == "" {
		return s.indexConn, nil
	}
	s.mu.RLock()
	if conn, exists := s.namespaces[namespace]; exists {
		s.mu.RUnlock()
		return conn, nil
	}
	s.mu.RUnlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if conn, exists := s.namespaces[namespace]; exists {
		return conn, nil
	}
	conn, err := s.client.Index(pinecone.NewIndexConnParams{
		Host:      s.getHostWithScheme(),
		Namespace: namespace,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create namespace connection: %w", err)
	}
	s.namespaces[namespace] = conn
	return conn, nil
}

func metadataToMap(metadata *structpb.Struct) map[string]interface{} {
	if metadata == nil {
		return make(map[string]interface{})
	}
	return metadata.AsMap()
}

func filterPropertiesPinecone(props map[string]interface{}, selectFields []string) map[string]interface{} {
	if len(selectFields) == 0 {
		return props
	}
	filtered := make(map[string]interface{}, len(selectFields))
	for _, field := range selectFields {
		if val, ok := props[field]; ok {
			filtered[field] = val
		}
	}
	return filtered
}

func matchesQueries(props map[string]interface{}, queries []Query) bool {
	for _, q := range queries {
		val, exists := props[q.Field]
		if !matchesQuery(val, exists, q) {
			return false
		}
	}
	return true
}

func matchesQuery(val interface{}, exists bool, q Query) bool {
	switch q.Operator {
	case QueryOperatorIsNull:
		return !exists || val == nil
	case QueryOperatorIsNotNull:
		return exists && val != nil
	case QueryOperatorEqual:
		return exists && fmt.Sprintf("%v", val) == fmt.Sprintf("%v", q.Value)
	case QueryOperatorNotEqual:
		return !exists || fmt.Sprintf("%v", val) != fmt.Sprintf("%v", q.Value)
	default:
		return true
	}
}

func buildPineconeFilter(queries []Query) (*structpb.Struct, error) {
	if len(queries) == 0 {
		return nil, nil
	}
	filterMap := make(map[string]interface{})
	for _, q := range queries {
		if cond := buildPineconeCondition(q); cond != nil {
			filterMap[q.Field] = cond
		}
	}
	if len(filterMap) == 0 {
		return nil, nil
	}
	return structpb.NewStruct(filterMap)
}

func buildPineconeCondition(q Query) interface{} {
	switch q.Operator {
	case QueryOperatorEqual:
		return map[string]interface{}{"$eq": q.Value}
	case QueryOperatorNotEqual:
		return map[string]interface{}{"$ne": q.Value}
	case QueryOperatorGreaterThan:
		return map[string]interface{}{"$gt": q.Value}
	case QueryOperatorGreaterThanOrEqual:
		return map[string]interface{}{"$gte": q.Value}
	case QueryOperatorLessThan:
		return map[string]interface{}{"$lt": q.Value}
	case QueryOperatorLessThanOrEqual:
		return map[string]interface{}{"$lte": q.Value}
	case QueryOperatorIsNull:
		return map[string]interface{}{"$eq": nil}
	case QueryOperatorIsNotNull:
		return map[string]interface{}{"$ne": nil}
	case QueryOperatorContainsAny:
		return map[string]interface{}{"$in": q.Value}
	case QueryOperatorContainsAll:
		values, ok := q.Value.([]interface{})
		if !ok {
			if ss, ok2 := q.Value.([]string); ok2 {
				values = make([]interface{}, len(ss))
				for i, s := range ss {
					values[i] = s
				}
			} else {
				return map[string]interface{}{"$eq": q.Value}
			}
		}
		andConds := make([]interface{}, len(values))
		for i, v := range values {
			andConds[i] = map[string]interface{}{"$eq": v}
		}
		return map[string]interface{}{"$and": andConds}
	default:
		return map[string]interface{}{"$eq": q.Value}
	}
}
