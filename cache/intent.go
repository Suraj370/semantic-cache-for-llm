package cache

import (
	"strings"
	"sync"
)

// IntentType identifies the customer-support intent category of a request.
type IntentType string

const (
	IntentPolicyFAQ     IntentType = "policy_faq"
	IntentOrderTracking IntentType = "order_tracking"
	IntentAccountMgmt   IntentType = "account_mgmt"
	IntentProductInfo   IntentType = "product_info"
	IntentComplaint     IntentType = "complaint"
	IntentUnknown       IntentType = "unknown"
)

// SkipSemantic returns true when semantic caching should be bypassed entirely.
// Complaint queries are emotional/situational — a cached answer from a different
// conversation is almost certainly wrong.
func (i IntentType) SkipSemantic() bool {
	return i == IntentComplaint
}

// defaultIntentThresholds are the starting cosine similarity thresholds per intent.
// -1 means skip semantic search entirely.
var defaultIntentThresholds = map[IntentType]float64{
	IntentPolicyFAQ:     0.70, // policy questions tolerate broad paraphrasing
	IntentOrderTracking: 0.72, // order queries are common; paraphrases are close
	IntentAccountMgmt:   0.74, // procedural answers are stable; moderate tolerance
	IntentProductInfo:   0.78, // product-specific — wrong answer is worse than a miss
	IntentComplaint:     -1,   // never serve a cached response for emotional queries
	IntentUnknown:       0.72, // most customer-support queries tolerate paraphrasing
}

// intentKeywords maps each intent to its keyword signals.
// detectIntent checks these in the order below (complaint first, then specificity descending).
var intentKeywords = map[IntentType][]string{
	IntentComplaint: {
		"unhappy", "frustrated", "complaint", "speak to a manager", "unacceptable",
		"terrible", "awful", "worst", "disgusted", "outraged", "file a complaint",
		"escalate", "demand refund", "very disappointed", "this is ridiculous",
	},
	IntentOrderTracking: {
		"my order", "order status", "track", "package", "delivery", "shipped",
		"where is", "estimated arrival", "out for delivery", "order number",
		"hasn't arrived", "not received", "my purchase", "my shipment",
		"when will", "how long until", "still waiting", "dispatch",
		"arrive", "in transit", "courier", "parcel",
	},
	IntentPolicyFAQ: {
		"return policy", "refund policy", "shipping policy", "how long does",
		"do you offer", "what is your", "how do i return", "return window",
		"free shipping", "shipping cost", "payment method", "accept",
		"can i return", "exchange", "how does", "eligible for",
	},
	IntentAccountMgmt: {
		"my account", "password", "reset password", "email address", "cancel",
		"subscription", "billing", "payment info", "account settings",
		"delete account", "log in", "sign in", "two factor", "two-factor",
		"update my", "change my", "profile",
	},
	IntentProductInfo: {
		"product", "item", "color", "colour", "size", "dimensions", "compatible",
		"in stock", "specification", "material", "warranty", "model number",
		"does it", "available in", "this item", "this product",
	},
}

// detectionOrder determines the priority when multiple intents match.
var detectionOrder = []IntentType{
	IntentComplaint,
	IntentOrderTracking,
	IntentPolicyFAQ,
	IntentAccountMgmt,
	IntentProductInfo,
}

// detectIntent returns the customer-support intent for req by keyword-matching
// the last user message. O(1) — no LLM call, no embedding.
func detectIntent(req Request) IntentType {
	msg := lastUserText(req)
	if msg == "" {
		return IntentUnknown
	}
	lower := strings.ToLower(msg)
	for _, intent := range detectionOrder {
		for _, kw := range intentKeywords[intent] {
			if strings.Contains(lower, kw) {
				return intent
			}
		}
	}
	return IntentUnknown
}

// lastUserText returns the raw text of the last user-role message, or the
// prompt for non-chat request types.
func lastUserText(req Request) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.EqualFold(req.Messages[i].Role, "user") {
			return extractMessageText(req.Messages[i])
		}
	}
	if req.Prompt != nil {
		return *req.Prompt
	}
	return ""
}

// ── AdaptiveThresholdManager ──────────────────────────────────────────────────

const (
	adaptiveRingSize   = 100
	adaptiveMinSamples = 20   // minimum outcomes before adjusting
	adaptiveTargetRate = 0.95 // target acceptance rate
	adaptiveStep       = 0.01 // threshold change per adjustment
	adaptiveMaxDelta   = 0.05 // max deviation from the intent default
)

// intentBucket tracks outcomes and the live threshold for one intent.
type intentBucket struct {
	buf     [adaptiveRingSize]bool
	head    int
	count   int     // real samples recorded (capped at ring size)
	current float64 // live adaptive threshold
	base    float64 // default (used as bounds anchor)
}

func (b *intentBucket) record(accepted bool) {
	b.buf[b.head] = accepted
	b.head = (b.head + 1) % adaptiveRingSize
	if b.count < adaptiveRingSize {
		b.count++
	}
}

func (b *intentBucket) acceptanceRate() (rate float64, ready bool) {
	if b.count < adaptiveMinSamples {
		return 0, false
	}
	n := b.count
	acc := 0
	for i := 0; i < n; i++ {
		if b.buf[i] {
			acc++
		}
	}
	return float64(acc) / float64(n), true
}

// AdaptiveThresholdManager keeps per-intent thresholds that drift toward the
// operating point where the target acceptance rate is met.
type AdaptiveThresholdManager struct {
	mu      sync.RWMutex
	buckets map[IntentType]*intentBucket
}

func newAdaptiveThresholdManager() *AdaptiveThresholdManager {
	m := &AdaptiveThresholdManager{
		buckets: make(map[IntentType]*intentBucket, len(defaultIntentThresholds)),
	}
	for intent, base := range defaultIntentThresholds {
		m.buckets[intent] = &intentBucket{current: base, base: base}
	}
	return m
}

// Threshold returns the current adaptive threshold for intent.
// Returns -1 for intents that should skip semantic search entirely.
func (m *AdaptiveThresholdManager) Threshold(intent IntentType) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if b, ok := m.buckets[intent]; ok {
		return b.current
	}
	return m.buckets[IntentUnknown].current
}

// RecordOutcome records whether a cached response was accepted or rejected by
// the end user, and nudges the intent's threshold if enough samples exist.
//
//   - acceptance rate < target → raise threshold (tighten, reduce false positives)
//   - acceptance rate > 99%    → lower threshold (loosen, improve recall)
func (m *AdaptiveThresholdManager) RecordOutcome(intent IntentType, accepted bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	b, ok := m.buckets[intent]
	if !ok {
		b = m.buckets[IntentUnknown]
	}
	b.record(accepted)

	rate, ready := b.acceptanceRate()
	if !ready {
		return
	}

	lo := b.base - adaptiveMaxDelta
	hi := b.base + adaptiveMaxDelta

	switch {
	case rate < adaptiveTargetRate:
		if next := b.current + adaptiveStep; next <= hi {
			b.current = next
		}
	case rate > 0.99:
		if next := b.current - adaptiveStep; next >= lo {
			b.current = next
		}
	}
}
