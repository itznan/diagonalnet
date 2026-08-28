package main

import (
	"math/rand"
	"testing"
)

func TestReduceParameterGradients(t *testing.T) {
	L := 1024
	numWorkers := 8

	master := NewParameter(L)
	workers := make([]*Parameter, numWorkers)

	rng := rand.New(rand.NewSource(123))

	// Initialize worker gradients
	for w := 0; w < numWorkers; w++ {
		workers[w] = NewParameter(L)
		for i := 0; i < L; i++ {
			workers[w].Grad[i] = rng.Float32() * 10.0
		}
	}

	// Compute expected sum sequentially
	expected := make([]float32, L)
	for i := 0; i < L; i++ {
		for w := 0; w < numWorkers; w++ {
			expected[i] += workers[w].Grad[i]
		}
	}

	// Run parallel lock-free reduction
	ReduceParameterGradients(master, workers, numWorkers)

	// Verify correctness
	for i := 0; i < L; i++ {
		diff := master.Grad[i] - expected[i]
		if diff < -1e-4 || diff > 1e-4 {
			t.Fatalf("reduction mismatch at index %d: expected %f, got %f", i, expected[i], master.Grad[i])
		}
	}
}

func TestReduceGradientsMultiParam(t *testing.T) {
	numWorkers := 4
	masterParams := []*Parameter{
		NewParameter(100),
		NewParameter(250),
		NewParameter(10),
	}

	workerParams := make([][]*Parameter, numWorkers)
	for w := 0; w < numWorkers; w++ {
		workerParams[w] = []*Parameter{
			NewParameter(100),
			NewParameter(250),
			NewParameter(10),
		}
		for pIdx, p := range workerParams[w] {
			for i := range p.Grad {
				p.Grad[i] = float32(w + pIdx + 1)
			}
		}
	}

	ReduceGradients(masterParams, workerParams, numWorkers)

	// Check master 0: sum of w+0+1 for w=0..3 -> 1 + 2 + 3 + 4 = 10
	for _, v := range masterParams[0].Grad {
		if v != 10.0 {
			t.Fatalf("expected 10.0 for master[0], got %f", v)
		}
	}
	// Check master 1: sum of w+1+1 for w=0..3 -> 2 + 3 + 4 + 5 = 14
	for _, v := range masterParams[1].Grad {
		if v != 14.0 {
			t.Fatalf("expected 14.0 for master[1], got %f", v)
		}
	}
}
