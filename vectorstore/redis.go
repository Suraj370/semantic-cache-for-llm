package vectorstore

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/suraj370/semantic-cache/logger"
	"github.com/suraj370/semantic-cache/types"
	"github.com/redis/go-redis/v9"
)

const (
	BatchLimit            = 100
	RedisMaxSearchResults = 10000
)

// RedisConfig holds connection settings for the Redis vector store.
type RedisConfig struct {
	Addr     *types.SecretVar `json:"addr"`
	Username *types.SecretVar `json:"username,omitempty"`
	Password *types.SecretVar `json:"password,omitempty"`
	DB       *types.SecretVar `json:"db,omitempty"`

	UseTLS             *types.SecretVar `json:"use_tls,omitempty"`
	InsecureSkipVerify *types.SecretVar `json:"insecure_skip_verify,omitempty"`
	CACertPEM          *types.SecretVar `json:"ca_cert_pem,omitempty"`

	ClusterMode *types.SecretVar `json:"cluster_mode,omitempty"`

	PoolSize        int            `json:"pool_size,omitempty"`
	MaxActiveConns  int            `json:"max_active_conns,omitempty"`
	MinIdleConns    int            `json:"min_idle_conns,omitempty"`
	MaxIdleConns    int            `json:"max_idle_conns,omitempty"`
	ConnMaxLifetime types.Duration `json:"conn_max_lifetime,omitempty"`
	ConnMaxIdleTime types.Duration `json:"conn_max_idle_time,omitempty"`
	DialTimeout     types.Duration `json:"dial_timeout,omitempty"`
	ReadTimeout     types.Duration `json:"read_timeout,omitempty"`
	WriteTimeout    types.Duration `json:"write_timeout,omitempty"`
	ContextTimeout  types.Duration `json:"context_timeout,omitempty"`
}

// RedisStore is the Redis VectorStore implementation backed by RediSearch.
type RedisStore struct {
	client redis.UniversalClient
	config RedisConfig
	logger logger.Logger

	namespaceFieldTypesMu sync.RWMutex
	namespaceFieldTypes   map[string]map[string]VectorStorePropertyType
}

// Ping checks if the Redis server is reachable.
func (s *RedisStore) Ping(ctx context.Context) error {
	return s.client.Ping(ctx).Err()
}

// CreateNamespace creates a RediSearch index with HNSW vector field.
func (s *RedisStore) CreateNamespace(ctx context.Context, namespace string, dimension int, properties map[string]VectorStoreProperties) error {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	infoResult := s.client.Do(ctx, "FT.INFO", namespace)
	if infoResult.Err() == nil {
		ftInfo, ftInfoErr := s.client.FTInfo(ctx, namespace).Result()
		if ftInfoErr != nil {
			s.logger.Warn("could not inspect existing index %q for dimension validation (check skipped): %v", namespace, ftInfoErr)
		} else {
			for _, attr := range ftInfo.Attributes {
				if strings.EqualFold(attr.Type, "VECTOR") && attr.Dim > 0 && attr.Dim != dimension {
					return fmt.Errorf("namespace %q already exists with dimension %d but config requires %d — update vector_store_namespace to a new name or drop the existing index manually", namespace, attr.Dim, dimension)
				}
			}
		}
		s.cacheNamespaceFieldTypes(namespace, properties)
		return nil
	}
	if err := infoResult.Err(); err != nil && strings.Contains(strings.ToLower(err.Error()), "unknown command") {
		return fmt.Errorf("search module not available: please use Redis Stack or a Valkey bundle with search support (FT.* commands required). original error: %w", err)
	}

	var metadataFields []string
	for fieldName := range properties {
		metadataFields = append(metadataFields, fieldName)
	}

	keyPrefix := fmt.Sprintf("%s:", namespace)
	if dimension <= 0 {
		return fmt.Errorf("redis vector index %q: dimension must be > 0 (got %d)", namespace, dimension)
	}

	args := []interface{}{
		"FT.CREATE", namespace,
		"ON", "HASH",
		"PREFIX", "1", keyPrefix,
		"SCHEMA",
		"embedding", "VECTOR", "HNSW", "6",
		"TYPE", "FLOAT32",
		"DIM", dimension,
		"DISTANCE_METRIC", "COSINE",
	}
	for _, field := range metadataFields {
		prop := properties[field]
		switch prop.DataType {
		case VectorStorePropertyTypeInteger:
			args = append(args, field, "NUMERIC")
		default:
			args = append(args, field, "TAG")
		}
	}

	if err := s.client.Do(ctx, args...).Err(); err != nil {
		return fmt.Errorf("failed to create semantic vector index %s: %w", namespace, err)
	}

	s.cacheNamespaceFieldTypes(namespace, properties)
	return nil
}

// GetChunk retrieves a single hash entry from Redis.
func (s *RedisStore) GetChunk(ctx context.Context, namespace string, id string) (SearchResult, error) {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	if strings.TrimSpace(id) == "" {
		return SearchResult{}, fmt.Errorf("id is required")
	}

	key := buildKey(namespace, id)
	result := s.client.HGetAll(ctx, key)
	if result.Err() != nil {
		return SearchResult{}, fmt.Errorf("failed to get chunk: %w", result.Err())
	}

	fields := result.Val()
	if len(fields) == 0 {
		return SearchResult{}, fmt.Errorf("chunk not found: %s", id)
	}

	searchResult := SearchResult{
		ID:         id,
		Properties: make(map[string]interface{}),
	}
	for k, v := range fields {
		searchResult.Properties[k] = v
	}
	return searchResult, nil
}

// GetChunks retrieves multiple hash entries via a pipeline.
func (s *RedisStore) GetChunks(ctx context.Context, namespace string, ids []string) ([]SearchResult, error) {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	if len(ids) == 0 {
		return []SearchResult{}, nil
	}

	keys := make([]string, len(ids))
	for i, id := range ids {
		if strings.TrimSpace(id) == "" {
			return nil, fmt.Errorf("id cannot be empty at index %d", i)
		}
		keys[i] = buildKey(namespace, id)
	}

	pipe := s.client.Pipeline()
	cmds := make([]*redis.MapStringStringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.HGetAll(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("failed to execute pipeline: %w", err)
	}

	var results []SearchResult
	for i, cmd := range cmds {
		if cmd.Err() != nil {
			s.logger.Debug("failed to get chunk %s: %v", ids[i], cmd.Err())
			continue
		}
		fields := cmd.Val()
		if len(fields) == 0 {
			continue
		}
		sr := SearchResult{
			ID:         ids[i],
			Properties: make(map[string]interface{}),
		}
		for k, v := range fields {
			sr.Properties[k] = v
		}
		results = append(results, sr)
	}
	return results, nil
}

// GetAll retrieves all entries matching the optional filter queries.
func (s *RedisStore) GetAll(ctx context.Context, namespace string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error) {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	if limit < 0 {
		limit = BatchLimit
	}

	redisQuery := buildRedisQuery(queries, s.getNamespaceFieldTypes(namespace))

	if limit == 0 {
		return s.getAllWithPagination(ctx, namespace, redisQuery, queries, selectFields)
	}

	searchLimit := limit
	if searchLimit > RedisMaxSearchResults {
		searchLimit = RedisMaxSearchResults
	}

	offset, err := parseOffsetCursor(cursor)
	if err != nil {
		return nil, nil, err
	}

	results, err := s.executeSearch(ctx, namespace, redisQuery, queries, selectFields, offset, int(searchLimit))
	if err != nil {
		return nil, nil, err
	}

	var nextCursor *string
	if cursor != nil && *cursor != "" {
		if len(results) == int(limit) && limit > 0 {
			off, parseErr := strconv.ParseInt(*cursor, 10, 64)
			if parseErr == nil {
				next := strconv.FormatInt(off+limit, 10)
				nextCursor = &next
			}
		}
	} else if len(results) == int(limit) && limit > 0 {
		next := strconv.FormatInt(limit, 10)
		nextCursor = &next
	}

	return results, nextCursor, nil
}

func (s *RedisStore) getAllWithPagination(ctx context.Context, namespace string, redisQuery string, queries []Query, selectFields []string) ([]SearchResult, *string, error) {
	var allResults []SearchResult
	offset := 0
	for {
		pageResults, err := s.executeSearch(ctx, namespace, redisQuery, queries, selectFields, offset, RedisMaxSearchResults)
		if err != nil {
			return nil, nil, err
		}
		if len(pageResults) == 0 {
			break
		}
		allResults = append(allResults, pageResults...)
		if len(pageResults) < RedisMaxSearchResults {
			break
		}
		offset += len(pageResults)
	}
	return allResults, nil, nil
}

func (s *RedisStore) executeSearch(ctx context.Context, namespace string, redisQuery string, queries []Query, selectFields []string, offset int, searchLimit int) ([]SearchResult, error) {
	args := []interface{}{"FT.SEARCH", namespace, redisQuery}

	if len(selectFields) > 0 {
		args = append(args, "RETURN", len(selectFields))
		for _, field := range selectFields {
			args = append(args, field)
		}
	}
	args = append(args, "LIMIT", offset, searchLimit, "DIALECT", "2")

	result := s.client.Do(ctx, args...)
	if result.Err() != nil {
		errMsg := strings.ToLower(result.Err().Error())
		if isQuerySyntaxError(errMsg) {
			s.logger.Debug("FT.SEARCH DIALECT fallback triggered for namespace %s: %s", namespace, result.Err())
			compatArgs := make([]interface{}, 0, len(args)-2)
			for i := 0; i < len(args); i++ {
				if i+1 < len(args) && args[i] == "DIALECT" {
					i++
					continue
				}
				compatArgs = append(compatArgs, args[i])
			}
			result = s.client.Do(ctx, compatArgs...)
		}
		if result.Err() != nil {
			errMsg = strings.ToLower(result.Err().Error())
			if isQuerySyntaxError(errMsg) {
				if IsScanFallbackDisabled(ctx) {
					return nil, fmt.Errorf("%w: %w", ErrQuerySyntax, result.Err())
				}
				s.logger.Debug("FT.SEARCH scan fallback triggered for namespace %s: %s", namespace, result.Err())
				scanResults, _, scanErr := s.getAllByScan(ctx, namespace, queries, selectFields, nil, int64(searchLimit))
				if scanErr != nil {
					return nil, scanErr
				}
				return scanResults, nil
			}
			return nil, fmt.Errorf("failed to search: %w", result.Err())
		}
	}

	results, err := s.parseSearchResults(result.Val(), namespace, selectFields)
	if err != nil {
		return nil, fmt.Errorf("failed to parse search results: %w", err)
	}
	return results, nil
}

func (s *RedisStore) getAllByScan(ctx context.Context, namespace string, queries []Query, selectFields []string, cursor *string, limit int64) ([]SearchResult, *string, error) {
	offset, err := parseOffsetCursor(cursor)
	if err != nil {
		return nil, nil, err
	}

	all, err := s.scanAllMatchingResults(ctx, namespace, queries, selectFields)
	if err != nil {
		return nil, nil, err
	}

	sort.Slice(all, func(i, j int) bool { return all[i].ID < all[j].ID })

	if offset > len(all) {
		offset = len(all)
	}
	if limit == 0 {
		return all[offset:], nil, nil
	}
	if limit < 0 {
		limit = BatchLimit
	}

	end := offset + int(limit)
	if end > len(all) {
		end = len(all)
	}

	results := all[offset:end]
	var next *string
	if end < len(all) {
		s := strconv.Itoa(end)
		next = &s
	}
	return results, next, nil
}

func (s *RedisStore) scanAllMatchingResults(ctx context.Context, namespace string, queries []Query, selectFields []string) ([]SearchResult, error) {
	if clusterClient, ok := s.client.(*redis.ClusterClient); ok {
		return s.scanAllMatchingResultsCluster(ctx, clusterClient, namespace, queries, selectFields)
	}
	return s.scanAllMatchingResultsSingle(ctx, s.client, namespace, queries, selectFields)
}

func (s *RedisStore) scanAllMatchingResultsSingle(ctx context.Context, client redis.Cmdable, namespace string, queries []Query, selectFields []string) ([]SearchResult, error) {
	pattern := buildKey(namespace, "*")
	var (
		scanCursor uint64
		all        []SearchResult
	)
	for {
		keys, nextCursor, err := client.Scan(ctx, scanCursor, pattern, BatchLimit).Result()
		if err != nil {
			return nil, fmt.Errorf("failed to scan keys: %w", err)
		}
		matches, err := s.fetchMatchingSearchResults(ctx, client, namespace, keys, queries, selectFields)
		if err != nil {
			return nil, err
		}
		all = append(all, matches...)
		scanCursor = nextCursor
		if scanCursor == 0 {
			break
		}
	}
	return all, nil
}

func (s *RedisStore) scanAllMatchingResultsCluster(ctx context.Context, client *redis.ClusterClient, namespace string, queries []Query, selectFields []string) ([]SearchResult, error) {
	var (
		all       []SearchResult
		allMu     sync.Mutex
		seenIDs   = make(map[string]struct{})
		seenIDsMu sync.Mutex
	)
	err := client.ForEachMaster(ctx, func(ctx context.Context, nodeClient *redis.Client) error {
		matches, err := s.scanAllMatchingResultsSingle(ctx, nodeClient, namespace, queries, selectFields)
		if err != nil {
			return err
		}
		unique := make([]SearchResult, 0, len(matches))
		seenIDsMu.Lock()
		for _, match := range matches {
			if _, ok := seenIDs[match.ID]; ok {
				continue
			}
			seenIDs[match.ID] = struct{}{}
			unique = append(unique, match)
		}
		seenIDsMu.Unlock()
		if len(unique) == 0 {
			return nil
		}
		allMu.Lock()
		all = append(all, unique...)
		allMu.Unlock()
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to scan cluster nodes: %w", err)
	}
	return all, nil
}

func (s *RedisStore) fetchMatchingSearchResults(ctx context.Context, client redis.Cmdable, namespace string, keys []string, queries []Query, selectFields []string) ([]SearchResult, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	pipe := client.Pipeline()
	cmds := make([]*redis.MapStringStringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.HGetAll(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("failed to fetch scanned keys: %w", err)
	}

	results := make([]SearchResult, 0, len(keys))
	for i, cmd := range cmds {
		if cmd.Err() != nil {
			continue
		}
		fields := cmd.Val()
		if len(fields) == 0 {
			continue
		}
		key := keys[i]
		id := strings.TrimPrefix(key, namespace+":")
		if id == key {
			continue
		}
		properties := make(map[string]interface{}, len(fields))
		for k, v := range fields {
			properties[k] = v
		}
		if !matchesQueriesForScan(properties, queries) {
			continue
		}
		sr := SearchResult{
			ID:         id,
			Properties: make(map[string]interface{}),
		}
		if len(selectFields) == 0 {
			sr.Properties = properties
		} else {
			for _, field := range selectFields {
				if val, ok := properties[field]; ok {
					sr.Properties[field] = val
				}
			}
		}
		results = append(results, sr)
	}
	return results, nil
}

func matchesQueriesForScan(properties map[string]interface{}, queries []Query) bool {
	for _, q := range queries {
		raw, exists := properties[q.Field]
		rawStr := fmt.Sprintf("%v", raw)
		queryStr := fmt.Sprintf("%v", q.Value)

		switch q.Operator {
		case QueryOperatorEqual:
			if !exists || rawStr != queryStr {
				return false
			}
		case QueryOperatorNotEqual:
			if exists && rawStr == queryStr {
				return false
			}
		case QueryOperatorIsNull:
			if exists {
				return false
			}
		case QueryOperatorIsNotNull:
			if !exists {
				return false
			}
		case QueryOperatorLike:
			if !exists || !strings.Contains(strings.ToLower(rawStr), strings.ToLower(queryStr)) {
				return false
			}
		case QueryOperatorGreaterThan:
			if !exists {
				return false
			}
			rawF, errR := strconv.ParseFloat(rawStr, 64)
			queryF, errQ := strconv.ParseFloat(queryStr, 64)
			if errR != nil || errQ != nil || rawF <= queryF {
				return false
			}
		case QueryOperatorGreaterThanOrEqual:
			if !exists {
				return false
			}
			rawF, errR := strconv.ParseFloat(rawStr, 64)
			queryF, errQ := strconv.ParseFloat(queryStr, 64)
			if errR != nil || errQ != nil || rawF < queryF {
				return false
			}
		case QueryOperatorLessThan:
			if !exists {
				return false
			}
			rawF, errR := strconv.ParseFloat(rawStr, 64)
			queryF, errQ := strconv.ParseFloat(queryStr, 64)
			if errR != nil || errQ != nil || rawF >= queryF {
				return false
			}
		case QueryOperatorLessThanOrEqual:
			if !exists {
				return false
			}
			rawF, errR := strconv.ParseFloat(rawStr, 64)
			queryF, errQ := strconv.ParseFloat(queryStr, 64)
			if errR != nil || errQ != nil || rawF > queryF {
				return false
			}
		case QueryOperatorContainsAny:
			if !exists {
				return false
			}
			propertyValues, ok := parseStringValuesForContains(raw)
			if !ok {
				return false
			}
			queryValues, ok := parseQueryContainsValues(q.Value)
			if !ok {
				return false
			}
			if !containsAnyString(propertyValues, queryValues) {
				return false
			}
		case QueryOperatorContainsAll:
			if !exists {
				return false
			}
			propertyValues, ok := parseStringValuesForContains(raw)
			if !ok {
				return false
			}
			queryValues, ok := parseQueryContainsValues(q.Value)
			if !ok {
				return false
			}
			if !containsAllStrings(propertyValues, queryValues) {
				return false
			}
		default:
			if !exists || rawStr != queryStr {
				return false
			}
		}
	}
	return true
}

func (s *RedisStore) parseSearchResults(result interface{}, namespace string, selectFields []string) ([]SearchResult, error) {
	results := []SearchResult{}
	switch typed := result.(type) {
	case map[interface{}]interface{}:
		rawResults, ok := typed["results"]
		if !ok {
			return results, nil
		}
		resultItems, ok := rawResults.([]interface{})
		if !ok {
			return results, nil
		}
		for _, item := range resultItems {
			if parsed, ok := parseSearchResultDocument(item, namespace, selectFields); ok {
				results = append(results, parsed)
			}
		}
		return results, nil
	case map[string]interface{}:
		rawResults, ok := typed["results"]
		if !ok {
			return results, nil
		}
		resultItems, ok := rawResults.([]interface{})
		if !ok {
			return results, nil
		}
		for _, item := range resultItems {
			if parsed, ok := parseSearchResultDocument(item, namespace, selectFields); ok {
				results = append(results, parsed)
			}
		}
		return results, nil
	case []interface{}:
		if len(typed) < 3 {
			return results, nil
		}
		for i := 1; i+1 < len(typed); i += 2 {
			doc := map[string]interface{}{
				"id":               typed[i],
				"extra_attributes": typed[i+1],
			}
			if parsed, ok := parseSearchResultDocument(doc, namespace, selectFields); ok {
				results = append(results, parsed)
			}
		}
		return results, nil
	default:
		return results, nil
	}
}

func parseSearchResultIDs(result interface{}, namespace string) []string {
	ids := make([]string, 0)
	appendID := func(value interface{}) {
		id, ok := toString(value)
		if !ok {
			return
		}
		id = strings.TrimSpace(id)
		if id == "" {
			return
		}
		if namespace != "" {
			prefix := namespace + ":"
			if strings.HasPrefix(id, prefix) {
				id = strings.TrimPrefix(id, prefix)
			}
		}
		if id == "" {
			return
		}
		ids = append(ids, id)
	}

	extractRESP3IDs := func(rawResults interface{}) {
		resultItems, ok := rawResults.([]interface{})
		if !ok {
			return
		}
		for _, item := range resultItems {
			switch doc := item.(type) {
			case map[string]interface{}:
				appendID(doc["id"])
			case map[interface{}]interface{}:
				appendID(doc["id"])
			default:
				appendID(item)
			}
		}
	}

	switch typed := result.(type) {
	case map[interface{}]interface{}:
		extractRESP3IDs(typed["results"])
	case map[string]interface{}:
		extractRESP3IDs(typed["results"])
	case []interface{}:
		if len(typed) < 2 {
			return ids
		}
		for i := 1; i < len(typed); i++ {
			appendID(typed[i])
			if i+1 < len(typed) {
				switch typed[i+1].(type) {
				case []interface{}, map[string]interface{}, map[interface{}]interface{}:
					i++
				}
			}
		}
	}
	return ids
}

func parseSearchResultDocument(resultItem interface{}, namespace string, selectFields []string) (SearchResult, bool) {
	var docMap map[string]interface{}
	switch item := resultItem.(type) {
	case map[string]interface{}:
		docMap = item
	case map[interface{}]interface{}:
		docMap = make(map[string]interface{}, len(item))
		for k, v := range item {
			docMap[fmt.Sprintf("%v", k)] = v
		}
	default:
		return SearchResult{}, false
	}

	idRaw, ok := docMap["id"]
	if !ok {
		return SearchResult{}, false
	}
	id, ok := toString(idRaw)
	if !ok {
		return SearchResult{}, false
	}
	docID := id
	if namespace != "" {
		prefix := namespace + ":"
		if strings.HasPrefix(id, prefix) {
			docID = strings.TrimPrefix(id, prefix)
		}
	}

	attrsRaw, ok := docMap["extra_attributes"]
	if !ok {
		return SearchResult{}, false
	}
	attrs := attributesToMap(attrsRaw)
	if attrs == nil {
		return SearchResult{}, false
	}

	sr := SearchResult{
		ID:         docID,
		Properties: make(map[string]interface{}, len(attrs)),
	}
	for fieldName, fieldValue := range attrs {
		if fieldName == "score" {
			sr.Properties[fieldName] = fieldValue
			if scoreFloat, ok := toFloat64(fieldValue); ok {
				sr.Score = &scoreFloat
			}
			continue
		}
		if len(selectFields) > 0 && !containsField(selectFields, fieldName) {
			continue
		}
		sr.Properties[fieldName] = fieldValue
	}
	return sr, true
}

func attributesToMap(value interface{}) map[string]interface{} {
	switch attrs := value.(type) {
	case map[string]interface{}:
		return attrs
	case map[interface{}]interface{}:
		out := make(map[string]interface{}, len(attrs))
		for k, v := range attrs {
			out[fmt.Sprintf("%v", k)] = v
		}
		return out
	case []interface{}:
		if len(attrs)%2 != 0 {
			return nil
		}
		out := make(map[string]interface{}, len(attrs)/2)
		for i := 0; i+1 < len(attrs); i += 2 {
			key, ok := toString(attrs[i])
			if !ok {
				continue
			}
			out[key] = attrs[i+1]
		}
		return out
	default:
		return nil
	}
}

func toString(value interface{}) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, true
	case []byte:
		return string(v), true
	default:
		return "", false
	}
}

func toFloat64(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case string:
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return 0, false
		}
		return parsed, true
	case []byte:
		parsed, err := strconv.ParseFloat(string(v), 64)
		if err != nil {
			return 0, false
		}
		return parsed, true
	default:
		return 0, false
	}
}

func containsField(fields []string, candidate string) bool {
	for _, field := range fields {
		if field == candidate {
			return true
		}
	}
	return false
}

func (s *RedisStore) cacheNamespaceFieldTypes(namespace string, properties map[string]VectorStoreProperties) {
	if strings.TrimSpace(namespace) == "" || len(properties) == 0 {
		return
	}
	fieldTypes := make(map[string]VectorStorePropertyType, len(properties))
	for field, prop := range properties {
		fieldTypes[field] = prop.DataType
	}
	s.namespaceFieldTypesMu.Lock()
	defer s.namespaceFieldTypesMu.Unlock()
	if s.namespaceFieldTypes == nil {
		s.namespaceFieldTypes = make(map[string]map[string]VectorStorePropertyType)
	}
	s.namespaceFieldTypes[namespace] = fieldTypes
}

func (s *RedisStore) deleteNamespaceFieldTypes(namespace string) {
	if strings.TrimSpace(namespace) == "" {
		return
	}
	s.namespaceFieldTypesMu.Lock()
	defer s.namespaceFieldTypesMu.Unlock()
	delete(s.namespaceFieldTypes, namespace)
}

func (s *RedisStore) getNamespaceFieldTypes(namespace string) map[string]VectorStorePropertyType {
	if strings.TrimSpace(namespace) == "" {
		return nil
	}
	s.namespaceFieldTypesMu.RLock()
	defer s.namespaceFieldTypesMu.RUnlock()
	fieldTypes, ok := s.namespaceFieldTypes[namespace]
	if !ok {
		return nil
	}
	copied := make(map[string]VectorStorePropertyType, len(fieldTypes))
	for field, dataType := range fieldTypes {
		copied[field] = dataType
	}
	return copied
}

func buildRedisQuery(queries []Query, fieldTypes map[string]VectorStorePropertyType) string {
	if len(queries) == 0 {
		return "*"
	}
	var conditions []string
	for _, query := range queries {
		if condition := buildRedisQueryCondition(query, fieldTypes); condition != "" {
			conditions = append(conditions, condition)
		}
	}
	if len(conditions) == 0 {
		return "*"
	}
	return strings.Join(conditions, " ")
}

func shouldUseNumericEquality(field string, value interface{}, fieldTypes map[string]VectorStorePropertyType) (string, bool) {
	if fieldTypes != nil {
		if dataType, ok := fieldTypes[field]; ok {
			if dataType == VectorStorePropertyTypeInteger {
				return normalizeNumericQueryValue(value)
			}
			return "", false
		}
	}
	return normalizeNumericQueryValue(value)
}

func normalizeNumericQueryValue(value interface{}) (string, bool) {
	switch v := value.(type) {
	case int:
		return strconv.FormatInt(int64(v), 10), true
	case int8:
		return strconv.FormatInt(int64(v), 10), true
	case int16:
		return strconv.FormatInt(int64(v), 10), true
	case int32:
		return strconv.FormatInt(int64(v), 10), true
	case int64:
		return strconv.FormatInt(v, 10), true
	case uint:
		return strconv.FormatUint(uint64(v), 10), true
	case uint8:
		return strconv.FormatUint(uint64(v), 10), true
	case uint16:
		return strconv.FormatUint(uint64(v), 10), true
	case uint32:
		return strconv.FormatUint(uint64(v), 10), true
	case uint64:
		return strconv.FormatUint(v, 10), true
	case float32:
		return strconv.FormatFloat(float64(v), 'f', -1, 32), true
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), true
	case string:
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			return "", false
		}
		if _, err := strconv.ParseFloat(trimmed, 64); err != nil {
			return "", false
		}
		return trimmed, true
	default:
		return "", false
	}
}

func buildRedisQueryCondition(query Query, fieldTypes map[string]VectorStorePropertyType) string {
	field := query.Field
	operator := query.Operator
	value := query.Value

	var stringValue string
	switch val := value.(type) {
	case string:
		stringValue = val
	case int, int64, float64, bool:
		stringValue = fmt.Sprintf("%v", val)
	default:
		jsonData, _ := json.Marshal(val)
		stringValue = string(jsonData)
	}

	escapedValue := escapeSearchValue(stringValue)

	switch operator {
	case QueryOperatorEqual:
		if numericValue, useNumeric := shouldUseNumericEquality(field, value, fieldTypes); useNumeric {
			return fmt.Sprintf("@%s:[%s %s]", field, numericValue, numericValue)
		}
		return fmt.Sprintf("@%s:{%s}", field, escapedValue)
	case QueryOperatorNotEqual:
		if numericValue, useNumeric := shouldUseNumericEquality(field, value, fieldTypes); useNumeric {
			return fmt.Sprintf("-@%s:[%s %s]", field, numericValue, numericValue)
		}
		return fmt.Sprintf("-@%s:{%s}", field, escapedValue)
	case QueryOperatorLike:
		return fmt.Sprintf("@%s:{%s}", field, escapedValue)
	case QueryOperatorGreaterThan:
		return fmt.Sprintf("@%s:[(%s +inf]", field, escapedValue)
	case QueryOperatorGreaterThanOrEqual:
		return fmt.Sprintf("@%s:[%s +inf]", field, escapedValue)
	case QueryOperatorLessThan:
		return fmt.Sprintf("@%s:[-inf (%s]", field, escapedValue)
	case QueryOperatorLessThanOrEqual:
		return fmt.Sprintf("@%s:[-inf %s]", field, escapedValue)
	case QueryOperatorIsNull:
		return fmt.Sprintf("-@%s:*", field)
	case QueryOperatorIsNotNull:
		return fmt.Sprintf("@%s:*", field)
	case QueryOperatorContainsAny:
		if values, ok := value.([]interface{}); ok {
			var orConditions []string
			for _, v := range values {
				orConditions = append(orConditions, fmt.Sprintf("@%s:{%s}", field, escapeSearchValue(fmt.Sprintf("%v", v))))
			}
			return fmt.Sprintf("(%s)", strings.Join(orConditions, " | "))
		}
		return fmt.Sprintf("@%s:{%s}", field, escapedValue)
	case QueryOperatorContainsAll:
		if values, ok := value.([]interface{}); ok {
			var andConditions []string
			for _, v := range values {
				andConditions = append(andConditions, fmt.Sprintf("@%s:{%s}", field, escapeSearchValue(fmt.Sprintf("%v", v))))
			}
			return strings.Join(andConditions, " ")
		}
		return fmt.Sprintf("@%s:{%s}", field, escapedValue)
	default:
		return fmt.Sprintf("@%s:{%s}", field, escapedValue)
	}
}

// GetNearest performs a KNN vector search with optional metadata filter.
func (s *RedisStore) GetNearest(ctx context.Context, namespace string, vector []float32, queries []Query, selectFields []string, threshold float64, limit int64) ([]SearchResult, error) {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	redisQuery := buildRedisQuery(queries, s.getNamespaceFieldTypes(namespace))
	queryBytes := float32SliceToBytes(vector)

	var hybridQuery string
	if len(queries) > 0 {
		hybridQuery = fmt.Sprintf("(%s)", redisQuery)
	} else {
		hybridQuery = "*"
	}

	knnLimit := limit
	if limit == 0 {
		knnLimit = math.MaxInt32
	}

	args := []interface{}{
		"FT.SEARCH", namespace,
		fmt.Sprintf("%s=>[KNN %d @embedding $vec AS score]", hybridQuery, knnLimit),
		"PARAMS", "2", "vec", queryBytes,
		"SORTBY", "score",
	}

	returnFields := []string{"score"}
	if len(selectFields) > 0 {
		returnFields = append(returnFields, selectFields...)
	}
	args = append(args, "RETURN", len(returnFields))
	for _, field := range returnFields {
		args = append(args, field)
	}

	searchLimit := limit
	if limit == 0 {
		searchLimit = math.MaxInt32
	}
	args = append(args, "LIMIT", 0, int(searchLimit), "DIALECT", "2")

	result := s.client.Do(ctx, args...)
	if result.Err() != nil {
		errMsg := strings.ToLower(result.Err().Error())
		if strings.Contains(errMsg, "unexpected argument `sortby`") || strings.Contains(errMsg, "unexpected argument sortby") {
			compatArgs := make([]interface{}, 0, len(args)-2)
			for i := 0; i < len(args); i++ {
				if i+1 < len(args) && args[i] == "SORTBY" {
					i++
					continue
				}
				compatArgs = append(compatArgs, args[i])
			}
			result = s.client.Do(ctx, compatArgs...)
		}
		if result.Err() != nil {
			return nil, fmt.Errorf("native vector search failed: %w", result.Err())
		}
	}

	results, err := s.parseSearchResults(result.Val(), namespace, selectFields)
	if err != nil {
		return nil, err
	}

	var filtered []SearchResult
	for _, res := range results {
		if scoreValue, exists := res.Properties["score"]; exists {
			score, ok := toFloat64(scoreValue)
			if !ok {
				continue
			}
			similarity := 1.0 - score
			res.Score = &similarity
			if similarity >= threshold {
				filtered = append(filtered, res)
			}
		} else {
			filtered = append(filtered, res)
		}
	}
	return filtered, nil
}

// Add stores a vector and metadata as a Redis hash.
func (s *RedisStore) Add(ctx context.Context, namespace string, id string, embedding []float32, metadata map[string]interface{}) error {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}

	key := buildKey(namespace, id)
	fields := make(map[string]interface{})

	if len(embedding) > 0 {
		fields["embedding"] = float32SliceToBytes(embedding)
	}

	for k, v := range metadata {
		switch val := v.(type) {
		case string:
			fields[k] = val
		case int, int64, float64, bool:
			fields[k] = fmt.Sprintf("%v", val)
		case []interface{}:
			b, err := json.Marshal(val)
			if err != nil {
				return fmt.Errorf("failed to marshal array metadata %s: %w", k, err)
			}
			fields[k] = string(b)
		default:
			jsonData, err := json.Marshal(val)
			if err != nil {
				return fmt.Errorf("failed to marshal metadata field %s: %w", k, err)
			}
			fields[k] = string(jsonData)
		}
	}

	if err := s.client.HSet(ctx, key, fields).Err(); err != nil {
		return fmt.Errorf("failed to store semantic cache entry: %w", err)
	}
	return nil
}

// Delete removes a single hash from Redis.
func (s *RedisStore) Delete(ctx context.Context, namespace string, id string) error {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("id is required")
	}

	key := buildKey(namespace, id)
	result := s.client.Del(ctx, key)
	if result.Err() != nil {
		return fmt.Errorf("failed to delete chunk %s: %w", id, result.Err())
	}
	if result.Val() == 0 {
		return fmt.Errorf("chunk not found: %s", id)
	}
	return nil
}

// DeleteAll deletes all entries matching the filter.
func (s *RedisStore) DeleteAll(ctx context.Context, namespace string, queries []Query) ([]DeleteResult, error) {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	return s.deleteAllBySnapshot(ctx, namespace, queries)
}

func (s *RedisStore) deleteAllBySnapshot(ctx context.Context, namespace string, queries []Query) ([]DeleteResult, error) {
	ids, err := s.getAllMatchingIDs(ctx, namespace, queries)
	if err != nil {
		return nil, fmt.Errorf("failed to find documents to delete: %w", err)
	}
	if len(ids) == 0 {
		return []DeleteResult{}, nil
	}

	var deleteResults []DeleteResult
	batchSize := BatchLimit
	for i := 0; i < len(ids); i += batchSize {
		end := i + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		batch := ids[i:end]

		pipe := s.client.Pipeline()
		cmds := make([]*redis.IntCmd, len(batch))
		for j, id := range batch {
			cmds[j] = pipe.Del(ctx, buildKey(namespace, id))
		}

		if _, err := pipe.Exec(ctx); err != nil {
			for _, id := range batch {
				deleteResults = append(deleteResults, DeleteResult{
					ID:     id,
					Status: DeleteStatusError,
					Error:  fmt.Sprintf("pipeline execution failed: %v", err),
				})
			}
			continue
		}

		for j, cmd := range cmds {
			id := batch[j]
			if cmd.Err() != nil {
				deleteResults = append(deleteResults, DeleteResult{ID: id, Status: DeleteStatusError, Error: cmd.Err().Error()})
			} else if cmd.Val() > 0 {
				deleteResults = append(deleteResults, DeleteResult{ID: id, Status: DeleteStatusSuccess})
			} else {
				deleteResults = append(deleteResults, DeleteResult{ID: id, Status: DeleteStatusError, Error: "document not found"})
			}
		}
	}
	return deleteResults, nil
}

func (s *RedisStore) getAllMatchingIDs(ctx context.Context, namespace string, queries []Query) ([]string, error) {
	redisQuery := buildRedisQuery(queries, s.getNamespaceFieldTypes(namespace))
	offset := 0
	ids := make([]string, 0)

	for {
		args := []interface{}{
			"FT.SEARCH", namespace,
			redisQuery,
			"RETURN", 0,
			"LIMIT", offset, BatchLimit,
			"DIALECT", "2",
		}
		result := s.client.Do(ctx, args...)
		if result.Err() != nil {
			errMsg := strings.ToLower(result.Err().Error())
			if isQuerySyntaxError(errMsg) {
				s.logger.Debug("FT.SEARCH DIALECT fallback triggered for namespace %s while collecting ids: %s", namespace, result.Err())
				compatArgs := make([]interface{}, 0, len(args)-2)
				for i := 0; i < len(args); i++ {
					if i+1 < len(args) && args[i] == "DIALECT" {
						i++
						continue
					}
					compatArgs = append(compatArgs, args[i])
				}
				result = s.client.Do(ctx, compatArgs...)
			}
			if result.Err() != nil {
				errMsg = strings.ToLower(result.Err().Error())
				if isQuerySyntaxError(errMsg) {
					if IsScanFallbackDisabled(ctx) {
						return nil, fmt.Errorf("failed to collect matching ids without scan fallback: %w", result.Err())
					}
					s.logger.Debug("FT.SEARCH scan fallback triggered for namespace %s while collecting ids: %s", namespace, result.Err())
					scanResults, _, scanErr := s.getAllByScan(ctx, namespace, queries, nil, nil, 0)
					if scanErr != nil {
						return nil, fmt.Errorf("failed to collect matching ids via scan fallback: %w", scanErr)
					}
					scanIDs := make([]string, 0, len(scanResults))
					for _, sr := range scanResults {
						scanIDs = append(scanIDs, sr.ID)
					}
					return scanIDs, nil
				}
				return nil, fmt.Errorf("failed to search for matching ids: %w", result.Err())
			}
		}

		pageIDs := parseSearchResultIDs(result.Val(), namespace)
		if len(pageIDs) == 0 {
			break
		}
		ids = append(ids, pageIDs...)
		if len(pageIDs) < BatchLimit {
			break
		}
		offset += len(pageIDs)
	}
	return ids, nil
}

// DeleteNamespace drops the RediSearch index (with document deletion via DD).
func (s *RedisStore) DeleteNamespace(ctx context.Context, namespace string) error {
	ctx, cancel := withTimeout(ctx, time.Duration(s.config.ContextTimeout))
	defer cancel()

	if err := s.client.Do(ctx, "FT.DROPINDEX", namespace, "DD").Err(); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "unknown index name") {
			s.deleteNamespaceFieldTypes(namespace)
			return s.deleteNamespaceKeys(ctx, namespace)
		}
		return fmt.Errorf("failed to drop semantic index %s: %w", namespace, err)
	}
	s.deleteNamespaceFieldTypes(namespace)
	return nil
}

func (s *RedisStore) deleteNamespaceKeys(ctx context.Context, namespace string) error {
	if clusterClient, ok := s.client.(*redis.ClusterClient); ok {
		return clusterClient.ForEachMaster(ctx, func(ctx context.Context, nodeClient *redis.Client) error {
			return s.deleteNamespaceKeysSingle(ctx, nodeClient, namespace)
		})
	}
	return s.deleteNamespaceKeysSingle(ctx, s.client, namespace)
}

func (s *RedisStore) deleteNamespaceKeysSingle(ctx context.Context, client redis.Cmdable, namespace string) error {
	pattern := buildKey(namespace, "*")
	var scanCursor uint64
	for {
		keys, nextCursor, err := client.Scan(ctx, scanCursor, pattern, BatchLimit).Result()
		if err != nil {
			return fmt.Errorf("failed to scan keys for namespace %s: %w", namespace, err)
		}
		if len(keys) > 0 {
			if err := client.Del(ctx, keys...).Err(); err != nil {
				return fmt.Errorf("failed to delete keys for namespace %s: %w", namespace, err)
			}
		}
		scanCursor = nextCursor
		if scanCursor == 0 {
			break
		}
	}
	return nil
}

// Close closes the Redis client connection.
func (s *RedisStore) Close(ctx context.Context, namespace string) error {
	return s.client.Close()
}

// RequiresVectors returns false because Redis can store entries without vectors.
func (s *RedisStore) RequiresVectors() bool { return false }

func escapeSearchValue(value string) string {
	replacer := strings.NewReplacer(
		"(", "\\(",
		")", "\\)",
		"[", "\\[",
		"]", "\\]",
		"{", "\\{",
		"}", "\\}",
		"*", "\\*",
		"?", "\\?",
		"|", "\\|",
		"&", "\\&",
		"!", "\\!",
		"@", "\\@",
		"#", "\\#",
		"$", "\\$",
		"%", "\\%",
		"^", "\\^",
		"~", "\\~",
		"`", "\\`",
		"\"", "\\\"",
		"'", "\\'",
		" ", "\\ ",
		"-", "\\-",
		".", "\\.",
		",", "\\,",
	)
	return replacer.Replace(value)
}

func float32SliceToBytes(floats []float32) []byte {
	b := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func isQuerySyntaxError(errMsg string) bool {
	return strings.Contains(errMsg, "missing `=>`") ||
		strings.Contains(errMsg, "invalid filter") ||
		strings.Contains(errMsg, "invalid query") ||
		strings.Contains(errMsg, "vector query clause is missing")
}

func parseOffsetCursor(cursor *string) (int, error) {
	if cursor == nil || *cursor == "" {
		return 0, nil
	}
	parsedOffset, err := strconv.ParseInt(*cursor, 10, 64)
	if err != nil {
		return 0, nil
	}
	if parsedOffset > math.MaxInt32 {
		return 0, fmt.Errorf("offset value %d exceeds maximum allowed value", parsedOffset)
	}
	if parsedOffset < 0 {
		return 0, fmt.Errorf("offset value %d cannot be negative", parsedOffset)
	}
	return int(parsedOffset), nil
}

func parseStringValuesForContains(value interface{}) ([]string, bool) {
	switch v := value.(type) {
	case []string:
		return v, true
	case []interface{}:
		out := make([]string, 0, len(v))
		for _, item := range v {
			out = append(out, fmt.Sprintf("%v", item))
		}
		return out, true
	case string:
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			return []string{}, true
		}
		if strings.HasPrefix(trimmed, "[") {
			var arr []interface{}
			if err := json.Unmarshal([]byte(trimmed), &arr); err == nil {
				out := make([]string, 0, len(arr))
				for _, item := range arr {
					out = append(out, fmt.Sprintf("%v", item))
				}
				return out, true
			}
		}
		return []string{v}, true
	default:
		return []string{fmt.Sprintf("%v", v)}, true
	}
}

func parseQueryContainsValues(value interface{}) ([]string, bool) {
	switch v := value.(type) {
	case []interface{}:
		out := make([]string, 0, len(v))
		for _, item := range v {
			out = append(out, fmt.Sprintf("%v", item))
		}
		return out, true
	case []string:
		return v, true
	default:
		return nil, false
	}
}

func containsAnyString(haystack []string, needles []string) bool {
	if len(needles) == 0 {
		return false
	}
	index := make(map[string]struct{}, len(haystack))
	for _, item := range haystack {
		index[item] = struct{}{}
	}
	for _, needle := range needles {
		if _, ok := index[needle]; ok {
			return true
		}
	}
	return false
}

func containsAllStrings(haystack []string, needles []string) bool {
	if len(needles) == 0 {
		return false
	}
	index := make(map[string]struct{}, len(haystack))
	for _, item := range haystack {
		index[item] = struct{}{}
	}
	for _, needle := range needles {
		if _, ok := index[needle]; !ok {
			return false
		}
	}
	return true
}

func buildKey(namespace, id string) string {
	return fmt.Sprintf("%s:%s", namespace, id)
}

// newRedisStore creates a new Redis vector store and verifies connectivity.
func newRedisStore(_ context.Context, config RedisConfig, log logger.Logger) (*RedisStore, error) {
	if config.Addr == nil || config.Addr.GetValue() == "" {
		return nil, fmt.Errorf("redis addr is required")
	}
	if config.Username == nil {
		sv := types.NewSecretVar("")
		config.Username = &sv
	}
	if config.Password == nil {
		sv := types.NewSecretVar("")
		config.Password = &sv
	}
	db := 0
	if config.DB != nil {
		db = config.DB.CoerceInt(0)
	}

	var tlsConfig *tls.Config
	if config.UseTLS.CoerceBool(false) {
		tlsConfig = &tls.Config{
			MinVersion:         tls.VersionTLS12,
			InsecureSkipVerify: config.InsecureSkipVerify.CoerceBool(false),
		}
		if config.CACertPEM != nil && config.CACertPEM.GetValue() != "" {
			rootCAs, err := systemCertPoolWithCA(config.CACertPEM.GetValue())
			if err != nil {
				return nil, fmt.Errorf("failed to configure Redis TLS CA certificate: %w", err)
			}
			tlsConfig.RootCAs = rootCAs
		}
	}

	clusterMode := config.ClusterMode.CoerceBool(false)

	var client redis.UniversalClient
	if clusterMode {
		if db != 0 {
			return nil, fmt.Errorf("redis cluster mode does not support database selection (DB must be 0)")
		}
		client = redis.NewClusterClient(&redis.ClusterOptions{
			Addrs:           []string{config.Addr.GetValue()},
			Username:        config.Username.GetValue(),
			Password:        config.Password.GetValue(),
			Protocol:        3,
			TLSConfig:       tlsConfig,
			PoolSize:        config.PoolSize,
			MaxActiveConns:  config.MaxActiveConns,
			MinIdleConns:    config.MinIdleConns,
			MaxIdleConns:    config.MaxIdleConns,
			ConnMaxLifetime: time.Duration(config.ConnMaxLifetime),
			ConnMaxIdleTime: time.Duration(config.ConnMaxIdleTime),
			DialTimeout:     time.Duration(config.DialTimeout),
			ReadTimeout:     time.Duration(config.ReadTimeout),
			WriteTimeout:    time.Duration(config.WriteTimeout),
		})
	} else {
		client = redis.NewClient(&redis.Options{
			Addr:            config.Addr.GetValue(),
			Username:        config.Username.GetValue(),
			Password:        config.Password.GetValue(),
			DB:              db,
			Protocol:        3,
			TLSConfig:       tlsConfig,
			PoolSize:        config.PoolSize,
			MaxActiveConns:  config.MaxActiveConns,
			MinIdleConns:    config.MinIdleConns,
			MaxIdleConns:    config.MaxIdleConns,
			ConnMaxLifetime: time.Duration(config.ConnMaxLifetime),
			ConnMaxIdleTime: time.Duration(config.ConnMaxIdleTime),
			DialTimeout:     time.Duration(config.DialTimeout),
			ReadTimeout:     time.Duration(config.ReadTimeout),
			WriteTimeout:    time.Duration(config.WriteTimeout),
		})
	}

	store := &RedisStore{
		client:              client,
		config:              config,
		logger:              log,
		namespaceFieldTypes: make(map[string]map[string]VectorStorePropertyType),
	}
	if err := store.Ping(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to connect to redis: %w", err)
	}
	return store, nil
}

func systemCertPoolWithCA(caCertPEM string) (*x509.CertPool, error) {
	rootCAs, err := x509.SystemCertPool()
	if err != nil {
		rootCAs = x509.NewCertPool()
	}
	if !rootCAs.AppendCertsFromPEM([]byte(caCertPEM)) {
		return nil, fmt.Errorf("failed to parse CA certificate PEM")
	}
	return rootCAs, nil
}
