package eval_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/suraj370/semantic-cache/cache"
	"github.com/suraj370/semantic-cache/embedding"
	"github.com/suraj370/semantic-cache/observability"
	"github.com/suraj370/semantic-cache/types"
	"github.com/suraj370/semantic-cache/vectorstore"
)

// ── Eval-specific Prometheus metrics ──────────────────────────────────────────

var (
	evalF1 = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "eval_f1_score",
		Help: "F1 score of the current eval run (harmonic mean of precision and recall over cache hits).",
	})
	evalPrecision = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "eval_precision",
		Help: "Fraction of returned cache hits that were correct (TP / TP+FP).",
	})
	evalRecall = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "eval_recall",
		Help: "Fraction of expected cache hits that were found (TP / TP+FN).",
	})
	evalCategoryPass = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "eval_cases_pass_total",
		Help: "Number of passing eval cases per category.",
	}, []string{"category"})
	evalCategoryFail = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "eval_cases_fail_total",
		Help: "Number of failing eval cases per category.",
	}, []string{"category"})
)

func init() {
	prometheus.MustRegister(evalF1, evalPrecision, evalRecall, evalCategoryPass, evalCategoryFail)
}

// estimateTokens approximates token count from raw bytes (1 token ≈ 4 bytes).
func estimateTokens(data []byte) int {
	if n := len(data) / 4; n > 0 {
		return n
	}
	return 1
}

// startMetricsServer starts an HTTP server on :2112 exposing /metrics.
// The server stays alive for 30 s after the returned stop() is called so
// Prometheus has time to scrape the final values.
func startMetricsServer(t *testing.T) (stop func()) {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	srv := &http.Server{Addr: ":2112", Handler: mux}
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			t.Logf("metrics server: %v", err)
		}
	}()
	t.Logf("metrics server: http://localhost:2112/metrics")
	return func() {
		time.Sleep(30 * time.Second) // keep alive for final Prometheus scrape
		srv.Shutdown(context.Background())
	}
}

// ── Dataset types ─────────────────────────────────────────────────────────────

type TestCase struct {
	ID          string     `json:"id"`
	Category    string     `json:"category"`
	Description string     `json:"description"`
	Stored      StoredCase `json:"stored"`
	Query       QueryCase  `json:"query"`
	Expected    Expected   `json:"expected"`
}

type StoredCase struct {
	CacheKey string          `json:"cache_key"`
	Request  cache.Request   `json:"request"`
	Response json.RawMessage `json:"response"`
}

type QueryCase struct {
	CacheKey  string        `json:"cache_key"`
	Request   cache.Request `json:"request"`
	Threshold float64       `json:"threshold,omitempty"`
	CacheType string        `json:"cache_type,omitempty"`
}

type Expected struct {
	Outcome string `json:"outcome"` // "direct_hit" | "semantic_hit" | "miss" | "skipped"
	Note    string `json:"note"`
}

type EvalResult struct {
	ID         string
	Category   string
	Expected   string
	Got        string
	Passed     bool
	Latency    int64
	Similarity *float64
}

// ── Main test ─────────────────────────────────────────────────────────────────

// TestSemanticCacheEval runs the full evaluation dataset against a live Qdrant
// instance and exposes metrics on :2112 for Prometheus / Grafana.
//
// Required env vars:
//
//	OPENAI_API_KEY   embeddings API key (no chat completions are called)
//	QDRANT_HOST      Qdrant host       (default: localhost)
//	QDRANT_PORT      Qdrant gRPC port  (default: 6334)
func TestSemanticCacheEval(t *testing.T) {
	// load .env from project root if present (ignores error if file is missing)
	godotenv.Load("../.env")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set — skipping integration eval")
	}

	qdrantHost := os.Getenv("QDRANT_HOST")
	if qdrantHost == "" {
		qdrantHost = "localhost"
	}
	qdrantPort := os.Getenv("QDRANT_PORT")
	if qdrantPort == "" {
		qdrantPort = "6334"
	}

	// Start /metrics endpoint so Grafana can watch the run live.
	stopMetrics := startMetricsServer(t)
	defer stopMetrics()

	ctx := context.Background()

	// ── Load dataset ──────────────────────────────────────────────────────────
	data, err := os.ReadFile("testcases.json")
	if err != nil {
		t.Fatalf("failed to read testcases.json: %v", err)
	}
	var cases []TestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to parse testcases.json: %v", err)
	}
	t.Logf("loaded %d test cases", len(cases))

	// ── Connect to Qdrant ─────────────────────────────────────────────────────
	store, err := vectorstore.NewVectorStore(ctx, &vectorstore.Config{
		Enabled: true,
		Type:    vectorstore.VectorStoreTypeQdrant,
		Config: vectorstore.QdrantConfig{
			Host: types.NewSecretVar(qdrantHost),
			Port: types.NewSecretVar(qdrantPort),
		},
	}, nil)
	if err != nil {
		t.Fatalf("failed to connect to Qdrant at %s:%s: %v", qdrantHost, qdrantPort, err)
	}

	// ── Create embedder ───────────────────────────────────────────────────────
	embedder, err := embedding.NewOpenAIEmbedder(embedding.OpenAIConfig{
		APIKey:    apiKey,
		Model:     "text-embedding-3-small",
		Dimension: 1536,
	})
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	// ── Wire up PrometheusRecorder ────────────────────────────────────────────
	rec, err := observability.NewPrometheusRecorder(nil)
	if err != nil {
		t.Fatalf("failed to create prometheus recorder: %v", err)
	}

	// ── Unique namespace per run ──────────────────────────────────────────────
	namespace := "eval-" + uuid.New().String()[:8]
	t.Logf("namespace: %s", namespace)

	t.Cleanup(func() {
		cleanCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := store.DeleteNamespace(cleanCtx, namespace); err != nil {
			t.Logf("cleanup: could not delete namespace %s: %v", namespace, err)
		}
		store.Close(cleanCtx, namespace)
	})

	// ── Build cache ───────────────────────────────────────────────────────────
	cacheByModel := true
	cacheByProvider := true
	c, err := cache.New(ctx, &cache.Config{
		Namespace:                    namespace,
		TTL:                          2 * time.Hour,
		Threshold:                    0.90,
		EmbeddingDimension:           1536,
		DefaultCacheKey:              "eval",
		ConversationHistoryThreshold: 3,
		CacheByModel:                 &cacheByModel,
		CacheByProvider:              &cacheByProvider,
	}, store, embedder, nil, cache.WithMetrics(rec))
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer c.Close()

	// ── Phase 1: store all seed entries ───────────────────────────────────────
	t.Log("phase 1: storing seed entries …")
	for _, tc := range cases {
		opts := cache.LookupOptions{CacheKey: tc.Stored.CacheKey}
		_, miss, err := c.Lookup(ctx, "store-"+tc.ID, tc.Stored.Request, opts)
		if err != nil {
			t.Logf("[%s] store-lookup error: %v", tc.ID, err)
			continue
		}
		if miss == nil {
			continue // already cached (duplicate stored request)
		}
		reqBytes, _ := json.Marshal(tc.Stored.Request)
		in, out := estimateTokens(reqBytes), estimateTokens(tc.Stored.Response)
		if err := miss.StoreWithTokens(tc.Stored.Response, in, out); err != nil {
			t.Logf("[%s] store error: %v", tc.ID, err)
		}
	}
	c.WaitForPendingOps()
	t.Log("phase 1: done")

	// ── Phase 2: run queries ──────────────────────────────────────────────────
	t.Log("phase 2: running queries …")
	results := make([]EvalResult, 0, len(cases))

	for _, tc := range cases {
		opts := cache.LookupOptions{
			CacheKey:  tc.Query.CacheKey,
			Threshold: tc.Query.Threshold,
			CacheType: cache.CacheType(tc.Query.CacheType),
		}

		result, miss, err := c.Lookup(ctx, "query-"+tc.ID, tc.Query.Request, opts)
		if err != nil {
			t.Errorf("[%s] query lookup error: %v", tc.ID, err)
			continue
		}

		var got string
		var similarity *float64
		var latency int64

		switch {
		case result != nil:
			// HitType is CacheType ("direct"/"semantic"); map to outcome names
			switch result.HitType {
			case cache.CacheTypeDirect:
				got = "direct_hit"
			case cache.CacheTypeSemantic:
				got = "semantic_hit"
			default:
				got = string(result.HitType)
			}
			similarity = result.Similarity
			latency = result.Latency
			if result.Stream != nil {
				for range result.Stream {
				}
			}
		case miss != nil:
			got = "miss"
		default:
			got = "skipped"
		}

		passed := got == tc.Expected.Outcome
		results = append(results, EvalResult{
			ID: tc.ID, Category: tc.Category,
			Expected: tc.Expected.Outcome, Got: got,
			Passed: passed, Latency: latency, Similarity: similarity,
		})

		if !passed {
			sim := "n/a"
			if similarity != nil {
				sim = fmt.Sprintf("%.4f", *similarity)
			}
			t.Errorf("FAIL [%s] %-22s expected=%-13s got=%-13s sim=%s\n       %s",
				tc.ID, tc.Category, tc.Expected.Outcome, got, sim, tc.Expected.Note)
		} else {
			sim := ""
			if similarity != nil {
				sim = fmt.Sprintf(" sim=%.4f", *similarity)
			}
			t.Logf("PASS [%s] %-22s got=%-13s latency=%dms%s",
				tc.ID, tc.Category, got, latency, sim)
		}
	}

	// ── Phase 3: compute and publish metrics ──────────────────────────────────
	reportMetrics(t, results)
}

// ── Metrics reporting ─────────────────────────────────────────────────────────

func reportMetrics(t *testing.T, results []EvalResult) {
	t.Helper()

	byCategory := make(map[string][]EvalResult)
	for _, r := range results {
		byCategory[r.Category] = append(byCategory[r.Category], r)
	}

	cats := make([]string, 0, len(byCategory))
	for k := range byCategory {
		cats = append(cats, k)
	}
	sort.Strings(cats)

	t.Logf("\n%-26s  %5s  %5s  %5s", "Category", "Total", "Pass", "Fail")
	t.Logf("%s", strings.Repeat("─", 46))

	total, totalPass := 0, 0
	for _, cat := range cats {
		rs := byCategory[cat]
		pass := 0
		for _, r := range rs {
			if r.Passed {
				pass++
			}
		}
		fail := len(rs) - pass
		t.Logf("%-26s  %5d  %5d  %5d", cat, len(rs), pass, fail)
		total += len(rs)
		totalPass += pass

		// publish per-category gauges
		evalCategoryPass.WithLabelValues(cat).Set(float64(pass))
		evalCategoryFail.WithLabelValues(cat).Set(float64(fail))
	}
	t.Logf("%s", strings.Repeat("─", 46))
	t.Logf("%-26s  %5d  %5d  %5d", "TOTAL", total, totalPass, total-totalPass)

	// precision / recall over expected hits
	var tp, fp, fn, tn int
	for _, r := range results {
		wantHit := strings.HasSuffix(r.Expected, "_hit")
		gotHit := strings.HasSuffix(r.Got, "_hit")
		switch {
		case wantHit && gotHit:
			tp++
		case !wantHit && gotHit:
			fp++
		case wantHit && !gotHit:
			fn++
		default:
			tn++
		}
	}

	precision, recall, f1 := 0.0, 0.0, 0.0
	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn)
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}

	// publish summary gauges (Grafana reads these)
	evalPrecision.Set(precision)
	evalRecall.Set(recall)
	evalF1.Set(f1)

	t.Logf("\nPrecision : %.3f  (of all hits returned, how many were correct)", precision)
	t.Logf("Recall    : %.3f  (of all expected hits, how many were found)", recall)
	t.Logf("F1        : %.3f", f1)
	t.Logf("TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)
	t.Logf("\nFP > 0 → wrong answer served — raise Threshold in eval_test.go")
	t.Logf("FN > 0 → valid paraphrase missed — lower Threshold in eval_test.go")
}
