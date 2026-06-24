package cache

// MetricsRecorder is the observability hook injected into Cache via WithMetrics.
// Implement it to send cache telemetry to any backend (Prometheus, StatsD,
// OpenTelemetry, etc.). All methods must be safe for concurrent use.
type MetricsRecorder interface {
	// RecordLookup records the outcome of a single Lookup call.
	// latencyMs is the total wall time of the Lookup in milliseconds.
	// embeddingTokens is the number of tokens consumed by the embedding call
	// that occurred during this lookup (0 for direct hits and skips).
	RecordLookup(outcome LookupOutcome, latencyMs int64, embeddingTokens int)

	// RecordEmbedding records a single call to the Embedder.
	// err is non-nil when the embedding provider returned an error.
	RecordEmbedding(latencyMs int64, tokens int, err error)

	// RecordStore records the async cache write-back that follows a miss.
	// err is non-nil when the vector store Add call failed.
	RecordStore(latencyMs int64, err error)

	// RecordEviction is called each time a lazily-expired entry is detected
	// during a lookup and scheduled for deletion.
	RecordEviction()

	// RecordError records an internal error that was caught and logged but
	// did not surface to the caller (e.g. a failing ANN search that fell
	// through to a miss). operation names: "direct_search", "semantic_search".
	RecordError(operation string)
}

// LookupOutcome classifies the result of a single Lookup call.
type LookupOutcome string

const (
	// LookupOutcomeDirectHit is returned when the deterministic hash matched an entry.
	LookupOutcomeDirectHit LookupOutcome = "direct_hit"
	// LookupOutcomeSemanticHit is returned when an ANN search found a similar entry.
	LookupOutcomeSemanticHit LookupOutcome = "semantic_hit"
	// LookupOutcomeMiss is returned when neither path found a cached response.
	LookupOutcomeMiss LookupOutcome = "miss"
	// LookupOutcomeSkipped is returned when the Lookup was bypassed entirely
	// (no cache key configured, conversation too long, etc.).
	LookupOutcomeSkipped LookupOutcome = "skipped"
)

// noopMetricsRecorder silently discards all metrics. Used when no recorder is
// attached so hot paths never need a nil check.
type noopMetricsRecorder struct{}

func (noopMetricsRecorder) RecordLookup(LookupOutcome, int64, int) {}
func (noopMetricsRecorder) RecordEmbedding(int64, int, error)      {}
func (noopMetricsRecorder) RecordStore(int64, error)               {}
func (noopMetricsRecorder) RecordEviction()                        {}
func (noopMetricsRecorder) RecordError(string)                     {}

// Option applies an optional setting to a Cache at construction time.
type Option func(*Cache)

// WithMetrics attaches a MetricsRecorder to the cache. Without this option the
// cache uses a no-op recorder and emits no metrics.
func WithMetrics(r MetricsRecorder) Option {
	return func(c *Cache) {
		if r != nil {
			c.metrics = r
		}
	}
}
