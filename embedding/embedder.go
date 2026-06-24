// Package embedding provides the Embedder interface and provider implementations
// used by the semantic cache to convert text into vector representations.
package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Embedder converts text into a fixed-dimension float32 vector.
// Implement this interface to use any embedding provider with the cache.
type Embedder interface {
	// Embed returns the embedding vector for text and the number of input tokens
	// consumed (0 if the provider does not report usage).
	Embed(ctx context.Context, text string) ([]float32, int, error)

	// Dimension returns the vector dimension produced by this embedder. Must
	// match the dimension used when creating the vector store namespace.
	Dimension() int
}

// OpenAIConfig configures the OpenAI-compatible embeddings endpoint.
// Most providers (OpenAI, Azure OpenAI, Ollama, etc.) expose this API shape.
type OpenAIConfig struct {
	// APIKey is the bearer token sent in the Authorization header.
	APIKey string

	// Model is the embedding model name (e.g. "text-embedding-3-small").
	Model string

	// BaseURL is the base API URL. Defaults to "https://api.openai.com/v1" when empty.
	BaseURL string

	// Dimension is the output vector size. Must be > 0.
	// For OpenAI text-embedding-3-* models this can be set via the API; for
	// other models it is a property of the model itself.
	Dimension int

	// RequestedDimension, when > 0, is passed as the "dimensions" field in the
	// API request. Some models truncate their native dimension to this value.
	// Leave 0 to omit the field and get the model's native dimension.
	RequestedDimension int

	// Timeout for individual embedding requests. Defaults to 30s when zero.
	Timeout time.Duration
}

// OpenAIEmbedder calls an OpenAI-compatible embeddings endpoint.
type OpenAIEmbedder struct {
	cfg    OpenAIConfig
	client *http.Client
}

// NewOpenAIEmbedder creates an OpenAIEmbedder from the given config. Returns
// an error when required fields (APIKey, Model, Dimension) are missing.
func NewOpenAIEmbedder(cfg OpenAIConfig) (*OpenAIEmbedder, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("embedding: APIKey is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("embedding: Model is required")
	}
	if cfg.Dimension <= 0 {
		return nil, fmt.Errorf("embedding: Dimension must be > 0")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &OpenAIEmbedder{
		cfg:    cfg,
		client: &http.Client{Timeout: timeout},
	}, nil
}

// Dimension returns the configured vector dimension.
func (e *OpenAIEmbedder) Dimension() int {
	return e.cfg.Dimension
}

// embeddingRequest is the OpenAI embeddings request body.
type embeddingRequest struct {
	Input      string `json:"input"`
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions,omitempty"`
}

// embeddingResponse is the OpenAI embeddings response body.
type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// Embed sends text to the embeddings endpoint and returns the float32 vector.
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, int, error) {
	payload := embeddingRequest{
		Input: text,
		Model: e.cfg.Model,
	}
	if e.cfg.RequestedDimension > 0 {
		payload.Dimensions = e.cfg.RequestedDimension
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, fmt.Errorf("embedding: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, 0, fmt.Errorf("embedding: failed to build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("embedding: HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, 0, fmt.Errorf("embedding: failed to decode response (status %d): %w", resp.StatusCode, err)
	}
	if result.Error != nil {
		return nil, 0, fmt.Errorf("embedding: API error (%s): %s", result.Error.Type, result.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("embedding: unexpected status %d", resp.StatusCode)
	}
	if len(result.Data) == 0 {
		return nil, 0, fmt.Errorf("embedding: no embeddings returned")
	}

	raw := result.Data[0].Embedding
	vec := make([]float32, len(raw))
	for i, v := range raw {
		vec[i] = float32(v)
	}

	tokens := 0
	if result.Usage != nil {
		tokens = result.Usage.TotalTokens
	}
	return vec, tokens, nil
}
