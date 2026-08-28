package main

import (
	"math"
	"math/rand"
)

// Parameter encapsulates trainable weights, analytical gradients, and Adam momentum accumulators in contiguous memory.
type Parameter struct {
	Data []float32 // Trainable weights
	Grad []float32 // Analytical Jacobian gradients
	M    []float32 // Adam 1st moment vector
	V    []float32 // Adam 2nd raw moment vector
}

// NewParameter allocates a new Parameter struct with initialized contiguous buffers.
func NewParameter(size int) *Parameter {
	return &Parameter{
		Data: make([]float32, size),
		Grad: make([]float32, size),
		M:    make([]float32, size),
		V:    make([]float32, size),
	}
}

// Size returns the number of elements in the parameter buffer.
func (p *Parameter) Size() int {
	return len(p.Data)
}

// ZeroGrad resets the analytical gradient buffer to zero.
func (p *Parameter) ZeroGrad() {
	for i := range p.Grad {
		p.Grad[i] = 0
	}
}

// Clone creates an exact, independent deep copy of the Parameter.
func (p *Parameter) Clone() *Parameter {
	cp := NewParameter(len(p.Data))
	copy(cp.Data, p.Data)
	copy(cp.Grad, p.Grad)
	copy(cp.M, p.M)
	copy(cp.V, p.V)
	return cp
}

// InitKaimingUniform initializes weights using He uniform distribution:
// bound = sqrt(6 / fan_in), W ~ U(-bound, +bound)
func InitKaimingUniform(param *Parameter, fanIn int, rng *rand.Rand) {
	if fanIn <= 0 {
		fanIn = 1
	}
	bound := float32(math.Sqrt(6.0 / float64(fanIn)))
	for i := range param.Data {
		param.Data[i] = (rng.Float32()*2.0 - 1.0) * bound
	}
}

// InitKaimingNormal initializes weights using He normal distribution via Box-Muller transform:
// sigma = sqrt(2 / fan_in), z = sigma * sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
func InitKaimingNormal(param *Parameter, fanIn int, rng *rand.Rand) {
	if fanIn <= 0 {
		fanIn = 1
	}
	sigma := math.Sqrt(2.0 / float64(fanIn))
	for i := 0; i < len(param.Data); i += 2 {
		u1 := rng.Float64()
		if u1 < 1e-12 {
			u1 = 1e-12
		}
		u2 := rng.Float64()

		radius := math.Sqrt(-2.0 * math.Log(u1))
		theta := 2.0 * math.Pi * u2

		param.Data[i] = float32(sigma * radius * math.Cos(theta))
		if i+1 < len(param.Data) {
			param.Data[i+1] = float32(sigma * radius * math.Sin(theta))
		}
	}
}

// InitZeros initializes all parameter weights to 0.0.
func InitZeros(param *Parameter) {
	for i := range param.Data {
		param.Data[i] = 0
	}
}

// InitConstant initializes all parameter weights to a constant value.
func InitConstant(param *Parameter, val float32) {
	for i := range param.Data {
		param.Data[i] = val
	}
}
