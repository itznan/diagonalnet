# ==============================================================================
# DiagonNet - Pure Go Zero-Dependency Deep Learning Engine
# ==============================================================================

BINARY_NAME=diagonnet
BIN_DIR=bin
TARGET=$(BIN_DIR)/$(BINARY_NAME)

.PHONY: all build test clean audit benchmark deps-check run

all: test build

build:
	go build -o $(TARGET) .

test:
	go test -v ./...

audit: build
	$(TARGET) -audit -data data

benchmark: build
	$(TARGET) -benchmark

deps-check:
	@echo "Checking module graph..."
	go list -m all
	@echo "Checking stdlib imports..."
	go list -f "{{range .Imports}}{{println .}}{{end}}" ./...

clean:
	go clean
	rm -rf $(BIN_DIR)
