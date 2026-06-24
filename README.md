# semantic-cache

A standalone Go library for caching LLM responses with dual-path lookup: an O(1) deterministic hash match and an ANN-based semantic similarity search. 

```
go get github.com/suraj370/semantic-cache-for-llm
```

---

## How it works

Every `Lookup` runs two paths in sequence:

```
                         Caller
                           │
                           ▼
             Cache.Lookup(ctx, req, opts)
                           │
             ┌─────────────┴──────────────┐
             │  no cache_key?             ├──── YES ──► (nil, nil, nil)
             │  conversation too long?    │             skipped
             └─────────────┬──────────────┘
                           │ NO
                           ▼
             params_hash = xxhash(temperature, model, …)
                           │
                           ▼
         ┌─────────────────────────────────────────┐
         │              DIRECT PATH                │
         │                                         │
         │  id = UUIDv5(                           │
         │    cacheKey + requestHash               │
         │    + paramsHash + provider + model      │
         │  )                                      │
         │                                         │
         │  VectorStore.GetChunk(id)    [O(1)]     │
         └──────────────────┬──────────────────────┘
                            │
               HIT ─────────┴───────── MISS
                │                         │
                ▼                         ▼
        ┌─────────────┐    ┌──────────────────────────────────────┐
        │             │    │           SEMANTIC PATH              │
        │  DIRECT HIT │    │                                      │
        │             │    │  text = flatten(messages)            │
        └──────┬──────┘    │  vec  = Embedder.Embed(text)        │
               │           │                                      │
               │           │  VectorStore.GetNearest(            │
               │           │    vec,                             │
               │           │    { cacheKey, paramsHash,          │
               │           │      provider, model },             │
               │           │    threshold, k=1                   │
               │           │  )        [HNSW / KNN]              │
               │           └──────────────┬───────────────────────┘
               │                          │
               │           HIT ───────────┴───────── MISS
               │            │                           │
               │            ▼                           ▼
               │   ┌──────────────────┐         ┌─────────────┐
               │   │                  │         │             │
               │   │  SEMANTIC HIT    │         │    MISS     │
               │   │                  │         │             │
               │   └────────┬─────────┘         └──────┬──────┘
               │            │                          │
               └─────┬──────┘                          ▼
                     │                          MissHandle
                     ▼                          .Store(response)
               LookupResult                    .StoreStream(chunks)
               .HitType    "direct"|"semantic" [async write-back]
               .Response   json.RawMessage
               .Stream     <-chan json.RawMessage
               .Similarity *float64
               .Latency    ms
```

### Cache key composition

```
Direct ID = UUIDv5(
    sha1_namespace,
    json({
        "cache_key":    <tenant scope>,
        "request_hash": xxhash(normalized messages + params),
        "params_hash":  xxhash(temperature, top_p, model, …),
        "provider":     <optional>,
        "model":        <optional>
    })
)
```

Changing any parameter, model, or prompt word produces a **different bucket** — no false direct hits. The semantic path then handles paraphrased variants above the similarity threshold.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     github.com/suraj370/semantic-cache               │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────────────┐ │
│  │  logger  │   │    types     │   │         embedding            │ │
│  │          │   │              │   │                              │ │
│  │ Logger   │   │ SecretVar    │   │  Embedder (interface)        │ │
│  │ interface│   │ (env or lit) │   │  OpenAIEmbedder              │ │
│  │ NoopLog  │   │ Duration     │   │  (Azure / Ollama compatible) │ │
│  └──────────┘   │ (human str)  │   └──────────────────────────────┘ │
│                 └──────────────┘                  │                  │
│                                                   │ []float32        │
│  ┌────────────────────────────────────────────────▼───────────────┐ │
│  │                        vectorstore                             │ │
│  │                                                                │ │
│  │  VectorStore (interface)                                       │ │
│  │  ├── Add(id, embedding, metadata)                              │ │
│  │  ├── GetChunk(id)          ← direct point fetch                │ │
│  │  ├── GetNearest(vec, filters, threshold, k)  ← ANN search     │ │
│  │  ├── GetAll / Delete / DeleteAll                               │ │
│  │  └── CreateNamespace / Ping / RequiresVectors                  │ │
│  │                                                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌────────┐  │ │
│  │  │   Weaviate  │  │    Redis    │  │  Qdrant  │  │Pinecone│  │ │
│  │  │  GraphQL +  │  │  RediSearch │  │   gRPC   │  │  REST  │  │ │
│  │  │  HNSW index │  │  FT.SEARCH  │  │  HNSW    │  │  KNN   │  │ │
│  │  │             │  │  KNN vector │  │          │  │        │  │ │
│  │  └─────────────┘  └─────────────┘  └──────────┘  └────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                           cache                                │ │
│  │                                                                │ │
│  │  Config                  LookupOptions                         │ │
│  │  ├── Namespace           ├── CacheKey (tenant scope)           │ │
│  │  ├── TTL                 ├── TTL override                      │ │
│  │  ├── Threshold           ├── Threshold override                │ │
│  │  ├── EmbeddingDimension  ├── CacheType (direct|semantic|both)  │ │
│  │  ├── CacheByModel        └── NoStore                           │ │
│  │  ├── CacheByProvider                                           │ │
│  │  └── ExcludeSystemPrompt                                       │ │
│  │                                                                │ │
│  │  Cache.Lookup() → LookupResult | MissHandle                    │ │
│  │  Cache.Invalidate(cacheKey)                                    │ │
│  │  Cache.InvalidateByID(id)                                      │ │
│  │  Cache.WaitForPendingOps()                                     │ │
│  │  Cache.Close()                                                 │ │
│  │                                                                │ │
│  │  Internal goroutines:                                          │ │
│  │  ├── async write workers  (writersWg)                          │ │
│  │  ├── cacheState reaper    (60 min TTL per request span)        │ │
│  │  └── streamAccumulator reaper (5 min idle TTL)                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "github.com/suraj370/semantic-cache/cache"
    "github.com/suraj370/semantic-cache/embedding"
    "github.com/suraj370/semantic-cache/vectorstore"
)

func main() {
    ctx := context.Background()

    // 1. Connect to a vector store.
    store, err := vectorstore.NewVectorStore(ctx, vectorstore.Config{
        Type: "redis",
        Config: &vectorstore.RedisConfig{
            // Addr, password etc — or set via env.REDIS_ADDR
        },
    }, nil)
    if err != nil {
        log.Fatal(err)
    }

    // 2. Configure an embedder (OpenAI-compatible endpoint).
    embedder, err := embedding.NewOpenAIEmbedder(embedding.OpenAIConfig{
        APIKey:    "sk-...",
        Model:     "text-embedding-3-small",
        Dimension: 1536,
    })
    if err != nil {
        log.Fatal(err)
    }

    // 3. Build the cache.
    c, err := cache.New(ctx, &cache.Config{
        EmbeddingDimension: embedder.Dimension(),
        DefaultCacheKey:    "myapp",
        Threshold:          0.85,
    }, store, embedder, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer c.Close()

    // 4. Use the cache around any LLM call.
    req := cache.Request{
        Type:     cache.RequestTypeChat,
        Provider: "openai",
        Model:    "gpt-4o",
        Messages: []cache.Message{
            {Role: "user", Content: cache.MessageContent{Text: strPtr("What is the capital of France?")}},
        },
    }

    result, miss, err := c.Lookup(ctx, "req-001", req, cache.LookupOptions{})
    if err != nil {
        log.Fatal(err)
    }

    if result != nil {
        fmt.Println("cache hit:", string(result.Response))
        return
    }

    // Cache miss — call the real LLM.
    llmResponse := json.RawMessage(`{"content":"Paris"}`)

    // Write back asynchronously.
    if err := miss.Store(llmResponse); err != nil {
        log.Println("cache store error:", err)
    }
    fmt.Println("llm response:", string(llmResponse))
}

func strPtr(s string) *string { return &s }
```

### Streaming responses

```go
result, miss, _ := c.Lookup(ctx, requestID, req, cache.LookupOptions{})

if result != nil && result.Stream != nil {
    // Replay cached chunks.
    for chunk := range result.Stream {
        writeToClient(chunk)
    }
    return
}

// Miss — stream from LLM and collect chunks.
var chunks []json.RawMessage
for chunk := range streamFromLLM() {
    writeToClient(chunk)
    chunks = append(chunks, chunk)
}
miss.StoreStream(chunks)
```

---

## Configuration reference

### `cache.Config`

| Field | Default | Description |
|---|---|---|
| `Namespace` | `"SemanticCache"` | Vector store collection / index name |
| `TTL` | `5m` | Entry lifetime (string `"5m"` or seconds in JSON) |
| `Threshold` | `0.8` | Cosine similarity cutoff for semantic hits |
| `EmbeddingDimension` | — | **Required.** Vector dimension. Set to `1` for direct-only mode |
| `DefaultCacheKey` | `""` | Fallback when `LookupOptions.CacheKey` is empty. Caching is skipped if both are empty |
| `ConversationHistoryThreshold` | `3` | Skip caching when message count exceeds this (long chats are unlikely to hit) |
| `CacheByModel` | `true` | Include model name in the cache key |
| `CacheByProvider` | `true` | Include provider name in the cache key |
| `ExcludeSystemPrompt` | `false` | Omit system messages from embedding and hash |

### `cache.LookupOptions`

| Field | Description |
|---|---|
| `CacheKey` | Tenant / feature scope (partitions entries) |
| `TTL` | Per-request TTL override |
| `Threshold` | Per-request similarity threshold override |
| `CacheType` | `"direct"`, `"semantic"`, or `""` (both) |
| `NoStore` | Consult cache normally but skip write-back on miss |

---

## Vector store backends

### Redis (RediSearch)

Requires the [RediSearch module](https://redis.io/docs/stack/search/). Uses `FT.CREATE` with HNSW index and `FT.SEARCH` with `KNN` for ANN queries. Cluster mode supported.

```go
vectorstore.Config{
    Type: "redis",
    Config: &vectorstore.RedisConfig{
        Addr:     types.NewSecretVar("localhost:6379"),  // or env.REDIS_ADDR
        Password: types.NewSecretVar("env.REDIS_PASS"),
    },
}
```

### Weaviate

Uses the GraphQL API for vector search and object CRUD.

```go
vectorstore.Config{
    Type: "weaviate",
    Config: &vectorstore.WeaviateConfig{
        Scheme: "http",
        Host:   &hostVar,   // types.SecretVar
        APIKey: &apiKeyVar,
    },
}
```

### Qdrant

Uses the gRPC client. Supports TLS, custom message size limits.

```go
vectorstore.Config{
    Type: "qdrant",
    Config: &vectorstore.QdrantConfig{
        Host:   types.NewSecretVar("localhost"),
        Port:   types.NewSecretVar("6334"),
        UseTLS: types.NewSecretVar("false"),
    },
}
```

### Pinecone

Uses namespace-based connection caching for efficient multi-tenant access.

```go
vectorstore.Config{
    Type: "pinecone",
    Config: &vectorstore.PineconeConfig{
        APIKey:    types.NewSecretVar("env.PINECONE_API_KEY"),
        IndexHost: types.NewSecretVar("env.PINECONE_HOST"),
    },
}
```

---

## SecretVar — environment variable references

Any `types.SecretVar` field accepts either a literal value or an `env.VAR_NAME` reference resolved at runtime:

```go
types.NewSecretVar("literal-value")
types.NewSecretVar("env.REDIS_PASSWORD")   // reads os.Getenv("REDIS_PASSWORD")
```

In JSON config files:

```json
{ "password": "env.REDIS_PASSWORD" }
```

---

## Modes

| Mode | `EmbeddingDimension` | `Embedder` | Behaviour |
|---|---|---|---|
| Full (default) | > 1 | provided | Direct hash first, then semantic similarity |
| Direct-only | `1` | `nil` | Only deterministic hash lookup, no embeddings |
| Semantic-only | > 1 | provided | Pass `CacheType: "semantic"` in `LookupOptions` |

---

## Custom embedder

Implement the `embedding.Embedder` interface to use any embedding provider:

```go
type Embedder interface {
    Embed(ctx context.Context, text string) ([]float32, int, error)
    Dimension() int
}
```

The built-in `OpenAIEmbedder` works with any OpenAI-compatible endpoint (Azure OpenAI, Ollama, Together AI, etc.) by setting `BaseURL`.

---

## Custom logger

Implement `logger.Logger` to forward logs to your preferred sink:

```go
type Logger interface {
    Debug(format string, args ...any)
    Info(format string, args ...any)
    Warn(format string, args ...any)
    Error(format string, args ...any)
}
```

Pass `nil` to `cache.New` to use the built-in no-op logger.

---

## Package layout

```
semantic-cache/
├── logger/          Logger interface + NoopLogger
├── types/           SecretVar, Duration
├── embedding/       Embedder interface, OpenAIEmbedder
├── vectorstore/     VectorStore interface + Weaviate, Redis, Qdrant, Pinecone
├── observability/
│   ├── prometheus.go   PrometheusRecorder (implements cache.MetricsRecorder)
│   └── dashboard.json  Grafana dashboard (import via UI or provisioning)
└── cache/
    ├── config.go    Config, constants, property schema
    ├── metrics.go   MetricsRecorder interface, LookupOutcome, WithMetrics option
    ├── types.go     Request, LookupOptions, LookupResult, MissHandle
    ├── cache.go     Cache struct, New, Lookup, Invalidate, Close
    ├── search.go    performDirectSearch, performSemanticSearch
    ├── state.go     per-request cacheState lifecycle
    ├── stream.go    StreamAccumulator, background reaper
    └── utils.go     hashing, normalization, TTL helpers
```

---

## Observability

### Prometheus metrics

The `observability` package provides a `MetricsRecorder` implementation backed by
[prometheus/client_golang](https://github.com/prometheus/client_golang).

```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/suraj370/semantic-cache/cache"
    "github.com/suraj370/semantic-cache/observability"
)

// Register metrics into the default Prometheus registry.
// Pass a custom prometheus.Registerer to isolate metrics (e.g. in tests).
rec, err := observability.NewPrometheusRecorder(nil)
if err != nil {
    log.Fatal(err)
}

c, err := cache.New(ctx, cfg, store, embedder, log, cache.WithMetrics(rec))
```

#### Metrics reference

| Metric | Type | Labels | Description |
|---|---|---|---|
| `semantic_cache_lookups_total` | Counter | `outcome` | Lookup calls by outcome |
| `semantic_cache_lookup_duration_seconds` | Histogram | `outcome` | End-to-end Lookup wall time |
| `semantic_cache_embedding_duration_seconds` | Histogram | — | Embedding provider call latency |
| `semantic_cache_embedding_tokens_total` | Counter | — | Tokens consumed by embedding calls |
| `semantic_cache_embedding_errors_total` | Counter | — | Failed embedding provider calls |
| `semantic_cache_store_duration_seconds` | Histogram | — | Async write-back latency |
| `semantic_cache_store_errors_total` | Counter | — | Failed cache write-back operations |
| `semantic_cache_evictions_total` | Counter | — | Lazily-expired entries detected |
| `semantic_cache_errors_total` | Counter | `operation` | Internal errors caught during lookup |

**`outcome` label values:** `direct_hit`, `semantic_hit`, `miss`, `skipped`

**`operation` label values:** `direct_search`, `semantic_search`

#### Custom MetricsRecorder

The `cache.MetricsRecorder` interface is backend-agnostic — wire any sink
(StatsD, OpenTelemetry, DataDog, etc.) by implementing it:

```go
type MetricsRecorder interface {
    RecordLookup(outcome LookupOutcome, latencyMs int64, embeddingTokens int)
    RecordEmbedding(latencyMs int64, tokens int, err error)
    RecordStore(latencyMs int64, err error)
    RecordEviction()
    RecordError(operation string)
}
```

### Grafana dashboard

A ready-to-import Grafana dashboard is provided at
[observability/dashboard.json](observability/dashboard.json).

Import it via **Dashboards → Import → Upload JSON file** in Grafana.
Set the Prometheus data source when prompted.

**Dashboard panels:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Hit Rate (%)   │  Direct Hit %  │  Semantic Hit %  │  Error Rate       │
├─────────────────────────────────┬────────────────────────────────────────┤
│  Lookup Outcomes (req/s)        │  Lookup Latency p50 / p95 / p99       │
├─────────────────────────────────┼────────────────────────────────────────┤
│  Embedding Latency p50/p95/p99  │  Embedding Token Usage                │
├─────────────────────────────────┼────────────────────────────────────────┤
│  Store Write Latency p50/p95/99 │  Evictions & Errors                   │
└─────────────────────────────────┴────────────────────────────────────────┘
```

### Quick-start with Docker Compose

Run Prometheus + Grafana locally with a single command (requires your app to
expose `/metrics` on port 2112):

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: semantic-cache
    static_configs:
      - targets: ["host.docker.internal:2112"]
```

Then expose your app's metrics endpoint:

```go
import "github.com/prometheus/client_golang/prometheus/promhttp"

http.Handle("/metrics", promhttp.Handler())
log.Fatal(http.ListenAndServe(":2112", nil))
```

---

## Requirements

- Go 1.25+
- One of the supported vector stores running and reachable
- An OpenAI-compatible embeddings endpoint (for semantic mode)
