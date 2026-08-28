package main

import (
	"testing"
)

func TestTensorIndexAndStride(t *testing.T) {
	c, h, w := 3, 28, 28
	tensor := NewTensor(c, h, w)

	expectedLen := c * h * w
	if len(tensor.Data) != expectedLen {
		t.Fatalf("expected tensor length %d, got %d", expectedLen, len(tensor.Data))
	}

	targetC, targetY, targetX := 2, 4, 5
	expectedIndex := targetC*(h*w) + targetY*w + targetX
	val := float32(3.14)

	tensor.Set(targetC, targetY, targetX, val)

	// Verify direct slice access at expected index
	if tensor.Data[expectedIndex] != val {
		t.Fatalf("stride mismatch: expected tensor.Data[%d] == %f, got %f", expectedIndex, val, tensor.Data[expectedIndex])
	}

	// Verify Get accessor
	if got := tensor.Get(targetC, targetY, targetX); got != val {
		t.Fatalf("Get accessor mismatch: expected %f, got %f", val, got)
	}

	// Verify Index() method
	if idx := tensor.Index(targetC, targetY, targetX); idx != expectedIndex {
		t.Fatalf("Index method mismatch: expected %d, got %d", expectedIndex, idx)
	}
}

func TestTensorZeroAndClone(t *testing.T) {
	tensor := NewTensor(2, 4, 4)
	tensor.Set(1, 2, 3, 42.0)

	clone := tensor.Clone()
	if clone.Get(1, 2, 3) != 42.0 {
		t.Fatalf("clone value mismatch: expected 42.0, got %f", clone.Get(1, 2, 3))
	}

	tensor.Zero()
	if tensor.Get(1, 2, 3) != 0 {
		t.Fatalf("expected 0 after Zero(), got %f", tensor.Get(1, 2, 3))
	}

	// Ensure clone was independent deep copy
	if clone.Get(1, 2, 3) != 42.0 {
		t.Fatalf("clone modified after original Zero()")
	}

	// Verify shapes
	ch, ht, wd := clone.Shape()
	if ch != 2 || ht != 4 || wd != 4 {
		t.Fatalf("expected shape (2,4,4), got (%d,%d,%d)", ch, ht, wd)
	}
}
