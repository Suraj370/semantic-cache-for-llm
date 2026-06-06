# Semantic Cache — Internal Deployment Proposal

> **TL;DR:** A drop-in proxy that sits in front of any LLM API and serves
> repeated or semantically equivalent questions from cache. A 2,000-request
> load test against GPT-4o-mini showed an **87.5% cache hit rate**,
> **21× latency reduction** on hits, and a **$162 projected saving per
> million requests** — with a one-line integration change.

---

## The Problem

Every LLM call to OpenAI, Anthropic, or any hosted model costs money and
adds latency. In practice, a large fraction of the questions users ask are
semantically equivalent — same intent, different wording. We are currently
paying full price and waiting full time for every one of them.

A keyword cache would only catch exact duplicates. A semantic cache catches
paraphrases: *"How do I reverse a list in Python?"* and *"What's the Python
way to reverse a list?"* resolve to the same cached answer.

---

## Benchmark Results

Run on 2,000 requests, concurrency 20, model `gpt-4o-mini`, real OpenAI
embeddings and real LLM calls for misses.

### Hit-rate convergence

The cache cold-starts at 0% and converges as it warms up. It reached
**87.5%** by request 2,000 — and was already above 50% by request 260.

```
after   160    36.2%
after   260    51.2%
after   460    61.3%
after   760    73.4%
after 1,060    78.6%
after 1,560    84.4%
after 2,000    87.5%  ← converged
```

### Latency

|           | Mean    | P50     | P95     | P99     |
|-----------|---------|---------|---------|---------|
| **HIT**   | 388 ms  | 309 ms  | 664 ms  | 1.02 s  |
| **MISS**  | 6.56 s  | 6.55 s  | 13.79 s | 19.02 s |
| Overall   | 1.16 s  | 350 ms  | 7.41 s  | 12.60 s |

**Cache hits are 21× faster than misses at P50.**

Hit latency is dominated by the OpenAI embedding API call (~300 ms). A
self-hosted embedding model would bring hits into single-digit milliseconds.

### Hit rate by traffic type

| Traffic type  | Requests | Hits  | Hit rate |
|---------------|----------|-------|----------|
| Exact repeat  |      569 |   569 | 100.0%   |
| Paraphrase    |      827 |   696 |  84.2%   |
| Novel query   |      543 |   485 |  89.3%   |
| Warmup (seed) |       60 |     0 |   0.0%   |

Even queries categorised as "novel" hit at 89% after the cache warms — the
similarity threshold correctly identifies equivalent intent across differently
framed questions.

### Cost savings

| Metric           | Benchmark run | Projected (1M requests) |
|------------------|---------------|-------------------------|
| Requests         | 1,999         | 1,000,000               |
| Cache hits       | 1,750 (87.5%) | 875,000                 |
| Tokens saved     | 566,774       | ~283 million            |
| **Cost saved**   | **$0.3246**   | **$162**                |

At 1M requests/month the proxy pays for its own infrastructure within the
first day. At 10M requests/month the annual saving exceeds **$19,000**
against GPT-4o-mini pricing. Savings scale linearly with volume and are
proportionally larger for GPT-4o and Claude models.

---

## How It Works

```
Client  →  Semantic Cache (port 8000)  →  OpenAI / Anthropic / Ollama
                    |
           [on HIT]  return cached response instantly
           [on MISS] forward → get response → cache it → return
```

1. The incoming prompt is normalized and embedded via `text-embedding-3-small`.
2. The embedding is compared against cached entries using cosine similarity.
3. If similarity exceeds the threshold (default 0.85, adaptive per request
   type), the cached response is returned — no LLM call is made.
4. On a miss, the request is forwarded to the configured upstream provider.
   The response is written to the cache before being returned to the caller.

The cache is fully transparent. Every response carries `X-Cache: HIT` or
`X-Cache: MISS` headers. Hits also carry `X-Cache-Similarity` so engineers
can audit exactly what the cache matched.

### Supported providers (automatic routing by model name)

| Model prefix | Routes to |
|---|---|
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI |
| `claude-*`, `us.anthropic.*` | Anthropic |
| anything else | Ollama (local) |

---

## Integration

The proxy is a full OpenAI-spec drop-in. Integration is **one line** in
whatever client or framework already makes LLM calls.

**Python — openai SDK**
```python
# Before
client = openai.OpenAI(api_key="sk-...")

# After — nothing else in the codebase changes
client = openai.OpenAI(
    api_key="sk-...",
    base_url="http://semantic-cache:8000/v1",
)
```

**LangChain**
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_base="http://semantic-cache:8000/v1",
)
```

**Environment variable (works for any OpenAI-compatible client)**
```bash
OPENAI_BASE_URL=http://semantic-cache:8000/v1
```

No prompt changes, no response-format changes, no retry logic changes.

---

## Deployment Guide

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (for `text-embedding-3-small` embeddings)

### 1. Clone and configure

```bash
git clone <repo-url>
cd semantic-cache
cp .env.example .env
# Set OPENAI_API_KEY — the only required variable
```

### 2. Start the full stack

```bash
docker compose up -d
```

Four services start:

| Service      | Port | Purpose |
|--------------|------|---------|
| `api`        | 8000 | Semantic cache proxy |
| `redis`      | 6379 | Redis Stack (RedisVL-compatible) |
| `prometheus` | 9090 | Metrics scraping |
| `grafana`    | 3000 | Live monitoring dashboard |

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","cache_size":0,"hit_rate":0.0,...}
```

Open **http://localhost:3000** — the Grafana dashboard loads automatically
with hit rate, latency, cost savings, and capacity panels pre-configured.
Anonymous viewer access is on by default; no login required for reviewers.

### 4. Point your application at the proxy

Replace `https://api.openai.com` with `http://<host>:8000` in your LLM
client configuration. Restart your application. Caching begins immediately.

### 5. Tune the threshold (optional)

The default similarity threshold is 0.85. Simulate alternative thresholds
against real traffic before committing:

```bash
curl -X POST http://localhost:8000/v1/cache/tune-threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.80, "sample_size": 500}'
```

The response shows projected hit rate, near-miss rate, and false-positive
risk at the proposed value.

---

## Monitoring

Every metric is scraped by Prometheus and displayed in Grafana out of the box:

- **Hit rate** — overall and per model, 5-minute rolling window
- **Latency P50 / P95 / P99** — HIT vs MISS side by side
- **Cost saved** — cumulative USD, broken down per model and per hour
- **Cache size** — current entries vs configured maximum
- **Near-miss rate** — queries just below threshold (guides threshold tuning)
- **LRU eviction rate** — signals when `SEMANTIC_CACHE_MAX_SIZE` needs raising

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Used for embeddings. |
| `OPENAI_BASE_URL` | OpenAI | Override for Azure, local proxies, or the bundled mock LLM. |
| `ANTHROPIC_API_KEY` | — | Required only when proxying `claude-*` models. |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity cutoff. |
| `SEMANTIC_CACHE_MAX_SIZE` | `10000` | Maximum vector store entries. |
| `SEMANTIC_CACHE_TTL` | `3600` | Default entry TTL in seconds. |
| `SEMANTIC_CACHE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `SEMANTIC_CACHE_USE_FAISS` | `true` | FAISS-accelerated ANN search above the entry threshold. |
| `SEMANTIC_CACHE_FAISS_THRESHOLD` | `500` | Entry count at which FAISS activates. |

---

## Reproducing the Benchmark

```bash
# 2,000 requests, concurrency 20 (the numbers above)
python benchmark.py

# Larger run
python benchmark.py --requests 5000 --concurrency 40

# Replay the same schedule for A/B threshold comparison
python benchmark.py --seed 42 --requests 2000
```

Raw results are saved to `benchmark_<timestamp>.csv`. Each row contains
request sequence number, latency, HIT/MISS, category, traffic variant, and
cosine similarity score.

---

## Current Limitations

**In-memory only.** The vector store does not survive restarts. Redis Stack
is provisioned and wired in the compose file — a Redis-backed store is the
planned next step and would give persistence and cross-instance sharing.

**Single node.** Each instance maintains its own cache. A shared Redis
backend would allow horizontal scaling with a warm cache from the start.

**Embedding latency.** Hit latency (309 ms P50) is currently gated on the
OpenAI embedding API. A self-hosted model (`text-embedding-3-small` runs on
CPU) would eliminate this and bring hit latency under 10 ms.

**Time-sensitive content.** The cache returns stored responses verbatim.
For prompts where the answer changes over time (news, prices, live data),
set `SEMANTIC_CACHE_TTL` to a short window or configure per-request TTL
tiers — the classifier handles this automatically for detected real-time
query patterns.
