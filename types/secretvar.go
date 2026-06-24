// Package types provides shared value types used across the semantic cache library.
package types

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
)

// SecretVar holds a configuration value that can be supplied as a plain text
// literal or resolved at runtime from an environment variable.
//
// JSON unmarshaling accepts two forms:
//
//	{"value": "literal-string"}              — plain-text literal
//	{"value": "env.MY_VAR"}                  — resolves os.Getenv("MY_VAR") at unmarshal time
//	"literal-string"                         — shorthand plain-text string
//	"env.MY_VAR"                             — shorthand env-var reference
type SecretVar struct {
	val    string
	envKey string // non-empty when sourced from an env var
}

// NewSecretVar constructs a SecretVar from a string. Strings prefixed with
// "env." are treated as environment variable references; all others are literals.
func NewSecretVar(value string) SecretVar {
	if envKey, ok := strings.CutPrefix(value, "env."); ok {
		return SecretVar{val: os.Getenv(envKey), envKey: envKey}
	}
	return SecretVar{val: value}
}

// GetValue returns the resolved string value. For env-var references the
// value is resolved once at construction/unmarshal time.
func (s *SecretVar) GetValue() string {
	if s == nil {
		return ""
	}
	return s.val
}

// String implements fmt.Stringer.
func (s *SecretVar) String() string {
	return s.GetValue()
}

// CoerceInt converts the resolved value to int, returning defaultValue on
// empty or parse failure.
func (s *SecretVar) CoerceInt(defaultValue int) int {
	if s == nil || s.val == "" {
		return defaultValue
	}
	v, err := strconv.Atoi(s.val)
	if err != nil {
		return defaultValue
	}
	return v
}

// CoerceBool converts the resolved value to bool, returning defaultValue on
// empty or parse failure.
func (s *SecretVar) CoerceBool(defaultValue bool) bool {
	if s == nil || s.val == "" {
		return defaultValue
	}
	v, err := strconv.ParseBool(s.val)
	if err != nil {
		return defaultValue
	}
	return v
}

// IsSet returns true when the SecretVar has a non-empty resolved value.
func (s *SecretVar) IsSet() bool {
	return s != nil && s.val != ""
}

// MarshalJSON emits the resolved value as a JSON string.
func (s SecretVar) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.val)
}

// UnmarshalJSON accepts:
//
//	"literal"             — plain-text string
//	"env.MY_VAR"          — env-var reference
//	{"value":"..."}       — object with value field (same string rules apply)
func (s *SecretVar) UnmarshalJSON(data []byte) error {
	// Try object form first: {"value": "..."}
	var obj struct {
		Value string `json:"value"`
	}
	if json.Unmarshal(data, &obj) == nil && obj.Value != "" {
		*s = NewSecretVar(obj.Value)
		return nil
	}

	// Fall back to bare string form.
	var raw string
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*s = NewSecretVar(raw)
	return nil
}
