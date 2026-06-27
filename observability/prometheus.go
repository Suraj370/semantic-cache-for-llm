// Package observability provides a Prometheus MetricsRecorder for the semantic cache.
//
// Usage:
//
//	rec, err := observability.NewPrometheusRecorder(nil) // nil → prometheus.DefaultRegisterer
//	if err != nil {
//	    log.Fatal(err)
//	}
//	c, err := cache.New(ctx, cfg, store, embedder, log, cache.WithMetrics(rec))
package observability

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/suraj370/semantic-cache/cache"
)

// modelPricing maps model name → [inputPricePerMToken, outputPricePerMToken] in USD.
var modelPricing = map[string][2]float64{
	"gpt-4o":              {2.50, 10.00},
	"gpt-4o-mini":         {0.15, 0.60},
	"gpt-4-turbo":         {10.00, 30.00},
	"gpt-3.5-turbo":       {0.50, 1.50},
	"claude-opus-4-8":     {15.00, 75.00},
	"claude-sonnet-4-6":   {3.00, 15.00},
	"claude-haiku-4-5":    {0.80, 4.00},
	"gemini-1.5-pro":      {3.50, 10.50},
	"gemini-1.5-flash":    {0.075, 0.30},
	"gemini-2.0-flash":    {0.10, 0.40},
}

// costSaved returns estimated USD saved for a cache hit given token counts and model.
// Falls back to gpt-4o pricing when the model is not in the table.
func costSaved(inputTokens, outputTokens int, model string) float64 {
	pricing, ok := modelPricing[model]
	if !ok {
		pricing = modelPricing["gpt-4o"] // reasonable default
	}
	return float64(inputTokens)/1_000_000*pricing[0] +
		float64(outputTokens)/1_000_000*pricing[1]
}

// PrometheusRecorder implements cache.MetricsRecorder using the Prometheus
// client library. All metrics are registered under the "semantic_cache" namespace.
type PrometheusRecorder struct {
	lookups   *prometheus.CounterVec
	lookupDur *prometheus.HistogramVec
	embedDur  prometheus.Histogram
	embedToks prometheus.Counter
	embedErrs prometheus.Counter
	storeDur  prometheus.Histogram
	storeErrs prometheus.Counter
	evictions prometheus.Counter
	errors    *prometheus.CounterVec
	costSaved *prometheus.CounterVec
}

// NewPrometheusRecorder creates and registers all Prometheus metrics.
// Pass nil for reg to use prometheus.DefaultRegisterer.
// Returns an error if any metric fails to register (e.g. name conflict).
func NewPrometheusRecorder(reg prometheus.Registerer) (*PrometheusRecorder, error) {
	if reg == nil {
		reg = prometheus.DefaultRegisterer
	}

	r := &PrometheusRecorder{}
	var err error

	r.lookups = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "lookups_total",
		Help:      "Total number of cache lookups by outcome.",
	}, []string{"outcome"})

	r.lookupDur = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "semantic_cache",
		Name:      "lookup_duration_seconds",
		Help:      "Wall time of Cache.Lookup calls, in seconds, by outcome.",
		Buckets:   []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
	}, []string{"outcome"})

	r.embedDur = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "semantic_cache",
		Name:      "embedding_duration_seconds",
		Help:      "Wall time of embedding provider calls, in seconds.",
		Buckets:   []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10},
	})

	r.embedToks = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "embedding_tokens_total",
		Help:      "Total tokens consumed by embedding calls.",
	})

	r.embedErrs = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "embedding_errors_total",
		Help:      "Total number of failed embedding provider calls.",
	})

	r.storeDur = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "semantic_cache",
		Name:      "store_duration_seconds",
		Help:      "Wall time of async cache write-back calls, in seconds.",
		Buckets:   []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
	})

	r.storeErrs = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "store_errors_total",
		Help:      "Total number of failed cache write-back operations.",
	})

	r.evictions = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "evictions_total",
		Help:      "Total number of lazily-expired entries detected and scheduled for deletion.",
	})

	r.errors = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "errors_total",
		Help:      "Total number of internal errors caught during lookup, by operation.",
	}, []string{"operation"})

	r.costSaved = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "semantic_cache",
		Name:      "cost_saved_dollars_total",
		Help:      "Estimated USD saved by serving responses from cache instead of calling the LLM, by model.",
	}, []string{"model"})

	collectors := []prometheus.Collector{
		r.lookups, r.lookupDur,
		r.embedDur, r.embedToks, r.embedErrs,
		r.storeDur, r.storeErrs,
		r.evictions, r.errors,
		r.costSaved,
	}
	for _, col := range collectors {
		if err = reg.Register(col); err != nil {
			return nil, err
		}
	}

	return r, nil
}

// RecordLookup implements cache.MetricsRecorder.
func (r *PrometheusRecorder) RecordLookup(outcome cache.LookupOutcome, latencyMs int64, embeddingTokens int) {
	o := string(outcome)
	r.lookups.WithLabelValues(o).Inc()
	r.lookupDur.WithLabelValues(o).Observe(msToSeconds(latencyMs))
}

// RecordEmbedding implements cache.MetricsRecorder.
func (r *PrometheusRecorder) RecordEmbedding(latencyMs int64, tokens int, err error) {
	r.embedDur.Observe(msToSeconds(latencyMs))
	if tokens > 0 {
		r.embedToks.Add(float64(tokens))
	}
	if err != nil {
		r.embedErrs.Inc()
	}
}

// RecordStore implements cache.MetricsRecorder.
func (r *PrometheusRecorder) RecordStore(latencyMs int64, err error) {
	r.storeDur.Observe(msToSeconds(latencyMs))
	if err != nil {
		r.storeErrs.Inc()
	}
}

// RecordEviction implements cache.MetricsRecorder.
func (r *PrometheusRecorder) RecordEviction() {
	r.evictions.Inc()
}

// RecordError implements cache.MetricsRecorder.
func (r *PrometheusRecorder) RecordError(operation string) {
	r.errors.WithLabelValues(operation).Inc()
}

// RecordCostSaved implements cache.MetricsRecorder.
func (r *PrometheusRecorder) RecordCostSaved(inputTokens, outputTokens int, model string) {
	if inputTokens == 0 && outputTokens == 0 {
		return
	}
	r.costSaved.WithLabelValues(model).Add(costSaved(inputTokens, outputTokens, model))
}

func msToSeconds(ms int64) float64 {
	return float64(ms) / 1000.0
}
