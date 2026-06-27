package cache

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cespare/xxhash/v2"
)

// normalizeText lower-cases, trims whitespace, and strips trailing punctuation
// so that "Where is my order?" and "where is my order" hash identically.
func normalizeText(text string) string {
	t := strings.ToLower(strings.TrimSpace(text))
	return strings.TrimRight(t, "?!.,;:")
}

// marshalSorted encodes v as JSON. Go's encoding/json already sorts map keys at
// all levels, so the result is deterministic for any map[string]interface{} tree.
func marshalSorted(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

// hashMap returns a deterministic hex xxhash digest of m.
func hashMap(m map[string]interface{}) (string, error) {
	data, err := marshalSorted(m)
	if err != nil {
		return "", fmt.Errorf("failed to marshal map for hashing: %w", err)
	}
	return fmt.Sprintf("%x", xxhash.Sum64(data)), nil
}

// hashSortedSlice returns a deterministic hex xxhash digest of a sorted string
// slice (used for set-typed fields like stop sequences or modalities).
func hashSortedSlice(values []string) string {
	if len(values) == 0 {
		return ""
	}
	sorted := make([]string, len(values))
	copy(sorted, values)
	sort.Strings(sorted)
	data, _ := json.Marshal(sorted)
	return fmt.Sprintf("%x", xxhash.Sum64(data))
}

// extractTextForEmbedding converts a Request into a single normalized string
// that is sent to the embedding provider. Returns an error for request types
// that cannot be meaningfully embedded (embedding requests, raw binary inputs).
func extractTextForEmbedding(req Request, excludeSystem bool) (string, error) {
	switch req.Type {
	case RequestTypeChat, RequestTypeChatStream:
		var parts []string
		for _, msg := range req.Messages {
			if excludeSystem && strings.EqualFold(msg.Role, "system") {
				continue
			}
			content := extractMessageText(msg)
			if content == "" {
				continue
			}
			parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, normalizeText(content)))
		}
		if len(parts) == 0 {
			return "", fmt.Errorf("no embeddable text in chat messages")
		}
		return strings.Join(parts, "\n"), nil

	case RequestTypeText, RequestTypeTextStream:
		if req.Prompt == nil || *req.Prompt == "" {
			return "", fmt.Errorf("prompt is empty")
		}
		return normalizeText(*req.Prompt), nil

	case RequestTypeEmbedding:
		return "", fmt.Errorf("embedding requests are not re-embedded; use direct search only")

	default:
		return "", fmt.Errorf("unsupported request type for embedding: %s", req.Type)
	}
}

// extractMessageText flattens a Message's content into a plain string.
func extractMessageText(msg Message) string {
	if msg.Content.Text != nil {
		return *msg.Content.Text
	}
	if len(msg.Content.Blocks) > 0 {
		var parts []string
		for _, b := range msg.Content.Blocks {
			if b.Text != nil {
				parts = append(parts, *b.Text)
			}
		}
		return strings.Join(parts, " ")
	}
	return ""
}

// getNormalizedInput returns the request input with text fields normalized for
// hashing. For non-text types the raw input is returned unchanged.
func getNormalizedInput(req Request, excludeSystem bool) interface{} {
	switch req.Type {
	case RequestTypeChat, RequestTypeChatStream:
		msgs := make([]map[string]interface{}, 0, len(req.Messages))
		for _, msg := range req.Messages {
			if excludeSystem && strings.EqualFold(msg.Role, "system") {
				continue
			}
			content := extractMessageText(msg)
			msgs = append(msgs, map[string]interface{}{
				"role":    msg.Role,
				"content": normalizeText(content),
			})
		}
		return msgs

	case RequestTypeText, RequestTypeTextStream:
		if req.Prompt != nil {
			norm := normalizeText(*req.Prompt)
			return norm
		}
		return ""

	case RequestTypeEmbedding:
		if req.EmbeddingInput != nil {
			norm := normalizeText(*req.EmbeddingInput)
			return norm
		}
		return ""

	default:
		return nil
	}
}

// conversationMessageCount returns the number of non-system messages in a chat
// request, or 0 for non-chat request types.
func conversationMessageCount(req Request, excludeSystem bool) int {
	if req.Type != RequestTypeChat && req.Type != RequestTypeChatStream {
		return 0
	}
	if excludeSystem {
		count := 0
		for _, m := range req.Messages {
			if !strings.EqualFold(m.Role, "system") {
				count++
			}
		}
		return count
	}
	return len(req.Messages)
}

// isExpiredEntry returns (expired bool, parseFailed bool). A missing expires_at
// means the entry never expires.
func isExpiredEntry(properties map[string]interface{}) (bool, bool) {
	raw, exists := properties["expires_at"]
	if !exists || raw == nil {
		return false, false
	}
	var expiresAt int64
	switch v := raw.(type) {
	case string:
		parsed, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return false, true
		}
		expiresAt = parsed
	case float64:
		expiresAt = int64(v)
	case int64:
		expiresAt = v
	case int:
		expiresAt = int64(v)
	default:
		return false, true
	}
	return expiresAt < time.Now().Unix(), false
}

// parseStreamChunks converts the stream_chunks property (which different vector
// store backends return in different shapes) into a flat []string.
func parseStreamChunks(streamData interface{}) ([]string, error) {
	if streamData == nil {
		return nil, fmt.Errorf("stream_chunks is nil")
	}
	switch v := streamData.(type) {
	case []string:
		return v, nil
	case []interface{}:
		out := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				continue
			}
			out = append(out, s)
		}
		return out, nil
	case string:
		var arr []string
		if err := json.Unmarshal([]byte(v), &arr); err != nil {
			return nil, fmt.Errorf("failed to parse JSON stream_chunks string: %w", err)
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unsupported stream_chunks type: %T", streamData)
	}
}

// buildUnifiedMetadata constructs the property map written alongside the cached
// response. The caller adds "response" or "stream_chunks" before calling Add.
func buildUnifiedMetadata(provider, model, paramsHash, cacheKey string, ttl time.Duration) map[string]interface{} {
	m := map[string]interface{}{
		"provider":            provider,
		"model":               model,
		"cache_key":           cacheKey,
		"from_semantic_cache": true,
		"expires_at":          time.Now().Add(ttl).Unix(),
	}
	if paramsHash != "" {
		m["params_hash"] = paramsHash
	}
	return m
}

// paramsToMap unmarshals req.Params (raw JSON) into a map for hashing.
// Returns an empty map when Params is empty or nil.
func paramsToMap(params json.RawMessage) map[string]interface{} {
	if len(params) == 0 {
		return make(map[string]interface{})
	}
	var m map[string]interface{}
	if err := json.Unmarshal(params, &m); err != nil {
		return make(map[string]interface{})
	}
	return m
}

// readIntProperty reads an integer payload field from vector store properties.
// Qdrant returns JSON numbers as float64; this handles float64, int64, and int.
func readIntProperty(props map[string]interface{}, key string) int {
	v, ok := props[key]
	if !ok || v == nil {
		return 0
	}
	switch n := v.(type) {
	case float64:
		return int(n)
	case int64:
		return int(n)
	case int:
		return n
	}
	return 0
}
