package vectorstore

import "errors"

var (
	// ErrNotFound is returned when a requested vector entry does not exist.
	ErrNotFound = errors.New("vectorstore: not found")
	// ErrNotSupported is returned when the operation is not supported by the store.
	ErrNotSupported = errors.New("vectorstore: operation not supported on this store")
	// ErrQuerySyntax is returned on malformed query expressions.
	ErrQuerySyntax = errors.New("vectorstore: query syntax error")
)
