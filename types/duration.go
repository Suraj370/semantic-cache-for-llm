package types

import (
	"encoding/json"
	"fmt"
	"time"
)

// Duration is a time.Duration that accepts both human-readable duration strings
// and plain integer nanosecond values on JSON unmarshal.
//
// Accepted formats on unmarshal:
//   - String: "5s", "500ms", "1m30s", "2h"
//   - Integer: nanosecond count (same as default Go time.Duration JSON encoding)
//
// MarshalJSON is not overridden — the type marshals as its underlying int64
// nanosecond value, identical to time.Duration.
type Duration time.Duration

// D returns the underlying time.Duration.
func (d Duration) D() time.Duration {
	return time.Duration(d)
}

// String returns a human-readable representation.
func (d Duration) String() string {
	return time.Duration(d).String()
}

// UnmarshalJSON implements json.Unmarshaler.
func (d *Duration) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	if data[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		dur, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("invalid duration %q: use a Go duration string like \"5s\", \"500ms\", \"1m30s\"", s)
		}
		*d = Duration(dur)
		return nil
	}
	var n int64
	if err := json.Unmarshal(data, &n); err != nil {
		return fmt.Errorf("invalid duration: expected a duration string (e.g. \"5s\") or integer nanoseconds: %w", err)
	}
	*d = Duration(n)
	return nil
}
