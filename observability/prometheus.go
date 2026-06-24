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

	collectors := []prometheus.Collector{
		r.lookups, r.lookupDur,
		r.embedDur, r.embedToks, r.embedErrs,
		r.storeDur, r.storeErrs,
		r.evictions, r.errors,
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

func msToSeconds(ms int64) float64 {
	return float64(ms) / 1000.0
}
