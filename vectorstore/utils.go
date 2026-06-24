package vectorstore

import (
	"context"
	"time"
)

// withTimeout derives a child context with a timeout when timeout > 0.
// Returns a no-op cancel when no timeout is set, so call sites can always
// defer cancel() unconditionally.
func withTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout > 0 {
		return context.WithTimeout(ctx, timeout)
	}
	return ctx, func() {}
}
