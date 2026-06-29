FROM golang:1.25-alpine

WORKDIR /app

# Cache dependency layer — only re-downloaded when go.mod/go.sum change.
COPY go.mod go.sum ./
RUN go mod download

# Copy all source after deps are cached.
COPY . .

# Pre-compile to surface build errors before the 30-minute test run.
RUN go build ./...

# Output directory for the test log (mount ./output:/app/output from host).
RUN mkdir -p /app/output

CMD ["sh", "-c", "go test ./eval/ -v -run TestSemanticCacheEval -timeout 30m 2>&1 | tee /app/output/result.txt"]
