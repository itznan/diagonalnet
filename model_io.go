package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

const ModelMagicHeader = "DIAGON01"

// SaveModelWeights writes parameter weights and class metadata to a binary file according to the DIAGON01 protocol:
// [Bytes 0..7]   : Magic Header String "DIAGON01"
// [Bytes 8..11]  : JSON Class Metadata Length (uint32 LittleEndian)
// [Bytes 12..N]  : JSON-encoded class name string slice
// [Bytes N+1..]  : Contiguous sequence of float32 parameter weights in binary LittleEndian
func SaveModelWeights(path string, params []*Parameter, classes []string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create weights directory: %w", err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// 1. Magic Header String: "DIAGON01"
	if _, err := writer.Write([]byte(ModelMagicHeader)); err != nil {
		return fmt.Errorf("failed to write magic header: %w", err)
	}

	// 2. JSON Class Metadata Length
	metaBytes, err := json.Marshal(classes)
	if err != nil {
		return fmt.Errorf("failed to encode class metadata: %w", err)
	}
	metaLen := uint32(len(metaBytes))
	if err := binary.Write(writer, binary.LittleEndian, metaLen); err != nil {
		return fmt.Errorf("failed to write metadata length: %w", err)
	}

	// 3. JSON-encoded class name string slice
	if _, err := writer.Write(metaBytes); err != nil {
		return fmt.Errorf("failed to write class metadata: %w", err)
	}

	// 4. Contiguous sequence of float32 parameter weights in binary LittleEndian
	buf := make([]byte, 4)
	for _, p := range params {
		if p == nil {
			continue
		}
		for _, val := range p.Data {
			bits := math.Float32bits(val)
			binary.LittleEndian.PutUint32(buf, bits)
			if _, err := writer.Write(buf); err != nil {
				return fmt.Errorf("failed to write weight data: %w", err)
			}
		}
	}

	return nil
}

// LoadModelWeights reads parameter weights and class metadata from a binary file.
func LoadModelWeights(path string, params []*Parameter) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	// 1. Validate Magic Header
	header := make([]byte, 8)
	if _, err := io.ReadFull(reader, header); err != nil {
		return nil, fmt.Errorf("failed to read magic header: %w", err)
	}
	if string(header) != ModelMagicHeader {
		return nil, fmt.Errorf("invalid model file format: expected header %q, got %q", ModelMagicHeader, string(header))
	}

	// 2. Read JSON Class Metadata Length
	var metaLen uint32
	if err := binary.Read(reader, binary.LittleEndian, &metaLen); err != nil {
		return nil, fmt.Errorf("failed to read metadata length: %w", err)
	}

	// 3. Read JSON Class Metadata
	metaBytes := make([]byte, metaLen)
	if _, err := io.ReadFull(reader, metaBytes); err != nil {
		return nil, fmt.Errorf("failed to read class metadata payload: %w", err)
	}

	var classes []string
	if err := json.Unmarshal(metaBytes, &classes); err != nil {
		return nil, fmt.Errorf("failed to decode class metadata JSON: %w", err)
	}

	// 4. Read contiguous float32 parameter weights
	buf := make([]byte, 4)
	for _, p := range params {
		if p == nil {
			continue
		}
		for i := 0; i < len(p.Data); i++ {
			if _, err := io.ReadFull(reader, buf); err != nil {
				if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
					return classes, fmt.Errorf("unexpected EOF while reading parameter weights")
				}
				return classes, fmt.Errorf("failed to read weight: %w", err)
			}
			bits := binary.LittleEndian.Uint32(buf)
			p.Data[i] = math.Float32frombits(bits)
		}
	}

	return classes, nil
}
