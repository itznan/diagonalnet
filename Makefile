# ==============================================================================
# DiagonalNet - Pure Go Zero-Dependency Deep Learning Engine
# ==============================================================================

BINARY_NAME=diagonalnet
BIN_DIR=bin
TARGET=$(BIN_DIR)/$(BINARY_NAME)

.PHONY: all build test clean audit deps-check run serve train

all: test build

build:
	go build -o $(TARGET) .

test:
	go test -v ./...

audit: build
	$(TARGET) -audit -data data

train: build
	$(TARGET) -train -data data -model weights/diagonalnet_model.bin -epochs 10 -lr 0.002 -batch 32

serve: build
	$(TARGET) -serve -port 8081 -model weights/diagonalnet_model.bin

deps-check:
	@echo "Checking module graph..."
	go list -m all
	@echo "Checking stdlib imports..."
	go list -f "{{range .Imports}}{{println .}}{{end}}" ./...

clean:
	go clean
	rm -rf $(BIN_DIR)

