GO ?= go
BIN := bin/probe
PREFIX ?= $(shell [ -d /opt/homebrew/bin ] && echo /opt/homebrew || echo /usr/local)

.PHONY: build install test vet fmt check run clean

build:
	$(GO) build -o $(BIN) ./cmd/probe

install: build
	install -m 755 $(BIN) $(PREFIX)/bin/probe
	@echo "installed $(PREFIX)/bin/probe"

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
