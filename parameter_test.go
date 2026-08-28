package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestParameterAllocationAndBuffers(t *testing.T) {
	size := 128
	param := NewParameter(size)

	if param.Size() != size {
		t.Fatalf("expected size %d, got %d", size, param.Size())
	}
	if len(param.Data) != size || len(param.Grad) != size || len(param.M) != size || len(param.V) != size {
		t.Fatalf("buffer size mismatch")
	}

	param.Grad[0] = 5.5
	param.ZeroGrad()
	if param.Grad[0] != 0 {
		t.Fatalf("expected 0 after ZeroGrad(), got %f", param.Grad[0])
	}

	param.Data[10] = 3.14
	param.M[10] = 0.9
	param.V[10] = 0.999
	clone := param.Clone()

	if clone.Data[10] != 3.14 || clone.M[10] != 0.9 || clone.V[10] != 0.999 {
		t.Fatalf("clone data mismatch")
	}

	// Verify deep copy isolation
	clone.Data[10] = 99.0
	if param.Data[10] != 3.14 {
		t.Fatalf("clone mutation affected original parameter")
	}
}

func TestKaimingUniformInitialization(t *testing.T) {
	fanIn := 100
	size := 10000
	param := NewParameter(size)
	rng := rand.New(rand.NewSource(42))

	InitKaimingUniform(param, fanIn, rng)

	bound := float32(math.Sqrt(6.0 / float64(fanIn)))
	var sum float64
	for _, v := range param.Data {
		if v < -bound || v > bound {
			t.Fatalf("value %f out of bound [-%f, +%f]", v, bound, bound)
		}
		sum += float64(v)
	}

	mean := sum / float64(size)
	if math.Abs(mean) > 0.05 {
		t.Fatalf("expected mean near 0, got %f", mean)
	}
}

func TestKaimingNormalInitialization(t *testing.T) {
	fanIn := 100
	size := 20000
	param := NewParameter(size)
	rng := rand.New(rand.NewSource(42))

	InitKaimingNormal(param, fanIn, rng)

	expectedSigma := math.Sqrt(2.0 / float64(fanIn))
	var sum, sumSq float64
	for _, v := range param.Data {
		sum += float64(v)
		sumSq += float64(v * v)
	}

	mean := sum / float64(size)
	variance := (sumSq / float64(size)) - (mean * mean)
	stdDev := math.Sqrt(variance)

	if math.Abs(mean) > 0.02 {
		t.Fatalf("expected mean near 0, got %f", mean)
	}

	if math.Abs(stdDev-expectedSigma) > 0.02 {
		t.Fatalf("expected stdDev near %f, got %f", expectedSigma, stdDev)
	}
}
