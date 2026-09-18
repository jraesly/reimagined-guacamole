GO ?= go
BIN := bin/probe

.PHONY: build test vet fmt check run clean

build:
	$(GO) build -o $(BIN) ./cmd/probe

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

fmt:
	@test -z "$$(gofmt -l .)" || { gofmt -l .; echo "run gofmt -w ."; exit 1; }

check: fmt vet test

run: build
	./$(BIN) fit --suggest

clean:
	rm -rf bin
