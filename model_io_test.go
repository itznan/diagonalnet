package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveAndLoadModelWeights(t *testing.T) {
	tempDir := filepath.Join(os.TempDir(), "diagonnet_test_io")
	defer os.RemoveAll(tempDir)

	modelPath := filepath.Join(tempDir, "subdir", "test_model.bin")

	classes := []string{"digit_0", "digit_1", "digit_2", "digit_3", "digit_4", "digit_5", "digit_6", "digit_7", "digit_8", "digit_9"}
	p1 := NewParameter(50)
	p2 := NewParameter(10)

	for i := range p1.Data {
		p1.Data[i] = float32(i) * 1.5
	}
	for i := range p2.Data {
		p2.Data[i] = float32(i) * -0.5
	}

	saveParams := []*Parameter{p1, p2}

	// 1. Save model weights
	if err := SaveModelWeights(modelPath, saveParams, classes); err != nil {
		t.Fatalf("SaveModelWeights failed: %v", err)
	}

	// 2. Prepare target parameters for loading
	loadP1 := NewParameter(50)
	loadP2 := NewParameter(10)
	loadParams := []*Parameter{loadP1, loadP2}

	// 3. Load model weights
	loadedClasses, err := LoadModelWeights(modelPath, loadParams)
	if err != nil {
		t.Fatalf("LoadModelWeights failed: %v", err)
	}

	// 4. Verify class metadata
	if len(loadedClasses) != len(classes) {
		t.Fatalf("classes length mismatch: expected %d, got %d", len(classes), len(loadedClasses))
	}
	for i, c := range loadedClasses {
		if c != classes[i] {
			t.Fatalf("class name mismatch at index %d: expected %s, got %s", i, classes[i], c)
		}
	}

	// 5. Verify weight values
	for i, val := range loadP1.Data {
		if val != p1.Data[i] {
			t.Fatalf("loadP1 mismatch at %d: expected %f, got %f", i, p1.Data[i], val)
		}
	}
	for i, val := range loadP2.Data {
		if val != p2.Data[i] {
			t.Fatalf("loadP2 mismatch at %d: expected %f, got %f", i, p2.Data[i], val)
		}
	}
}
