// Package logger defines the Logger interface used across the semantic cache library.
package logger

// Logger is the logging interface used by cache internals and vector store adapters.
// Implement this to plug in your own logging backend (zerolog, zap, slog, etc.).
type Logger interface {
	Debug(format string, args ...any)
	Info(format string, args ...any)
	Warn(format string, args ...any)
	Error(format string, args ...any)
}

// NoopLogger discards all log output. Useful for testing or when logging is not needed.
type NoopLogger struct{}

func (NoopLogger) Debug(string, ...any) {}
func (NoopLogger) Info(string, ...any)  {}
func (NoopLogger) Warn(string, ...any)  {}
func (NoopLogger) Error(string, ...any) {}
