package main

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// 1. Tensor Unit Tests
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

	if tensor.Data[expectedIndex] != val {
		t.Fatalf("stride mismatch: expected tensor.Data[%d] == %f, got %f", expectedIndex, val, tensor.Data[expectedIndex])
	}
	if got := tensor.Get(targetC, targetY, targetX); got != val {
		t.Fatalf("Get accessor mismatch: expected %f, got %f", val, got)
	}
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
	if clone.Get(1, 2, 3) != 42.0 {
		t.Fatalf("clone modified after original Zero()")
	}

	ch, ht, wd := clone.Shape()
	if ch != 2 || ht != 4 || wd != 4 {
		t.Fatalf("expected shape (2,4,4), got (%d,%d,%d)", ch, ht, wd)
	}
}

// 2. Parameter Unit Tests
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

// 3. Gradient Reduction Unit Tests
func TestReduceParameterGradients(t *testing.T) {
	L := 1024
	numWorkers := 8

	master := NewParameter(L)
	workers := make([]*Parameter, numWorkers)
	rng := rand.New(rand.NewSource(123))

	for w := 0; w < numWorkers; w++ {
		workers[w] = NewParameter(L)
		for i := 0; i < L; i++ {
			workers[w].Grad[i] = rng.Float32() * 10.0
		}
	}

	expected := make([]float32, L)
	for i := 0; i < L; i++ {
		for w := 0; w < numWorkers; w++ {
			expected[i] += workers[w].Grad[i]
		}
	}

	ReduceParameterGradients(master, workers, numWorkers)

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

	for _, v := range masterParams[0].Grad {
		if v != 10.0 {
			t.Fatalf("expected 10.0 for master[0], got %f", v)
		}
	}
	for _, v := range masterParams[1].Grad {
		if v != 14.0 {
			t.Fatalf("expected 14.0 for master[1], got %f", v)
		}
	}
}

// 4. Model IO Unit Tests
func TestSaveAndLoadModelWeights(t *testing.T) {
	tempDir := filepath.Join(os.TempDir(), "diagonnet_singlefile_test")
	defer os.RemoveAll(tempDir)

	modelPath := filepath.Join(tempDir, "weights", "model.bin")
	classes := []string{"zero", "one", "two"}
	p1 := NewParameter(20)
	p2 := NewParameter(5)

	for i := range p1.Data {
		p1.Data[i] = float32(i) * 2.0
	}
	for i := range p2.Data {
		p2.Data[i] = float32(i) * -1.0
	}

	if err := SaveModelWeights(modelPath, []*Parameter{p1, p2}, classes); err != nil {
		t.Fatalf("SaveModelWeights failed: %v", err)
	}

	loadP1 := NewParameter(20)
	loadP2 := NewParameter(5)
	loadedClasses, err := LoadModelWeights(modelPath, []*Parameter{loadP1, loadP2})
	if err != nil {
		t.Fatalf("LoadModelWeights failed: %v", err)
	}

	if len(loadedClasses) != len(classes) {
		t.Fatalf("classes length mismatch")
	}
	for i, c := range loadedClasses {
		if c != classes[i] {
			t.Fatalf("class mismatch at %d", i)
		}
	}
	for i, val := range loadP1.Data {
		if val != p1.Data[i] {
			t.Fatalf("p1 mismatch at %d", i)
		}
	}
	for i, val := range loadP2.Data {
		if val != p2.Data[i] {
			t.Fatalf("p2 mismatch at %d", i)
		}
	}
}

// 5. 13-Channel Manifold Unit Tests
func TestClamp(t *testing.T) {
	if got := clamp(-5, 0, 10); got != 0 {
		t.Fatalf("expected clamp(-5, 0, 10) == 0, got %d", got)
	}
	if got := clamp(15, 0, 10); got != 10 {
		t.Fatalf("expected clamp(15, 0, 10) == 10, got %d", got)
	}
	if got := clamp(5, 0, 10); got != 5 {
		t.Fatalf("expected clamp(5, 0, 10) == 5, got %d", got)
	}
}

func TestComputeManifoldSignatureAndParallel(t *testing.T) {
	h, w := 28, 28
	input := make([]float32, h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			input[y*w+x] = float32(y*w+x) / float32(h*w)
		}
	}

	manifold := ComputeManifold(input, h, w)

	if len(manifold) != 13*h*w {
		t.Fatalf("expected slice length %d, got %d", 13*h*w, len(manifold))
	}

	hw := h * w

	// Verify Channel 0 matches input exactly
	for i := 0; i < hw; i++ {
		if manifold[i] != input[i] {
			t.Fatalf("channel 0 mismatch at index %d", i)
		}
	}

	// Verify Channel 1 (Top-Left: dx=-1, dy=-1) at (10, 10)
	y, x := 10, 10
	expectedM1 := abs32(input[y*w+x] - input[(y-1)*w+(x-1)])
	if got := manifold[1*hw+y*w+x]; got != expectedM1 {
		t.Fatalf("M1 mismatch: expected %f, got %f", expectedM1, got)
	}

	// Verify All 8 Knight Channels (k in [0, 7] -> Channels 5..12) at (10, 10)
	for k := 0; k < 8; k++ {
		dx := KnightOffsets[k][0]
		dy := KnightOffsets[k][1]
		nx := clamp(x+dx, 0, w-1)
		ny := clamp(y+dy, 0, h-1)
		expectedVal := abs32(input[y*w+x] - input[ny*w+nx])
		if got := manifold[(k+5)*hw+y*w+x]; got != expectedVal {
			t.Fatalf("Knight channel %d mismatch: expected %f, got %f", 5+k, expectedVal, got)
		}
	}
}

// 6. Conv2DLayer Unit Tests
func TestConv2DLayerForward(t *testing.T) {
	inC, inH, inW := 2, 5, 5
	outC := 3
	K := 3
	S := 1
	P := 1

	conv := NewConv2DLayer(inC, outC, K, S, P, nil)

	InitConstant(conv.Weights, 1.0)
	InitConstant(conv.Bias, 0.5)

	input := NewTensor(inC, inH, inW)
	for i := range input.Data {
		input.Data[i] = 1.0
	}

	output := conv.Forward(input)

	c, h, w := output.Shape()
	if c != outC || h != inH || w != inW {
		t.Fatalf("unexpected output shape (%d, %d, %d), expected (%d, %d, %d)", c, h, w, outC, inH, inW)
	}

	centerVal := output.Get(0, 2, 2)
	if centerVal != 18.5 {
		t.Fatalf("expected center value 18.5, got %f", centerVal)
	}

	cornerVal := output.Get(0, 0, 0)
	if cornerVal != 8.5 {
		t.Fatalf("expected corner value 8.5, got %f", cornerVal)
	}
}

func TestConv2DLayerBackwardJacobian(t *testing.T) {
	inC, inH, inW := 2, 4, 4
	outC := 2
	K := 3
	S := 1
	P := 1

	rng := rand.New(rand.NewSource(99))
	conv := NewConv2DLayer(inC, outC, K, S, P, rng)

	input := NewTensor(inC, inH, inW)
	for i := range input.Data {
		input.Data[i] = rng.Float32()*2.0 - 1.0
	}

	// Forward pass
	output := conv.Forward(input)

	// Target tensor for Mean Squared Error loss: L = 0.5 * sum((Y - T)^2)
	target := NewTensor(outC, output.Height, output.Width)
	for i := range target.Data {
		target.Data[i] = rng.Float32()
	}

	gradOutput := NewTensor(outC, output.Height, output.Width)
	for i := range gradOutput.Data {
		gradOutput.Data[i] = output.Data[i] - target.Data[i]
	}

	conv.ZeroGrad()
	gradInput := conv.Backward(gradOutput)

	eps := float32(1e-3)

	// 1. Verify Weight Gradients vs Numerical Gradient
	for i := range conv.Weights.Data {
		orig := conv.Weights.Data[i]

		conv.Weights.Data[i] = orig + eps
		outPlus := conv.Forward(input)
		var lossPlus float32
		for j := range outPlus.Data {
			diff := outPlus.Data[j] - target.Data[j]
			lossPlus += 0.5 * diff * diff
		}

		conv.Weights.Data[i] = orig - eps
		outMinus := conv.Forward(input)
		var lossMinus float32
		for j := range outMinus.Data {
			diff := outMinus.Data[j] - target.Data[j]
			lossMinus += 0.5 * diff * diff
		}

		conv.Weights.Data[i] = orig

		numGrad := (lossPlus - lossMinus) / (2.0 * eps)
		anaGrad := conv.Weights.Grad[i]

		diff := float64(math.Abs(float64(anaGrad - numGrad)))
		if diff > 1e-2 {
			t.Fatalf("weight gradient mismatch at %d: analytical=%f, numerical=%f, diff=%f", i, anaGrad, numGrad, diff)
		}
	}

	// 2. Verify Bias Gradients vs Numerical Gradient
	for i := range conv.Bias.Data {
		orig := conv.Bias.Data[i]

		conv.Bias.Data[i] = orig + eps
		outPlus := conv.Forward(input)
		var lossPlus float32
		for j := range outPlus.Data {
			diff := outPlus.Data[j] - target.Data[j]
			lossPlus += 0.5 * diff * diff
		}

		conv.Bias.Data[i] = orig - eps
		outMinus := conv.Forward(input)
		var lossMinus float32
		for j := range outMinus.Data {
			diff := outMinus.Data[j] - target.Data[j]
			lossMinus += 0.5 * diff * diff
		}

		conv.Bias.Data[i] = orig

		numGrad := (lossPlus - lossMinus) / (2.0 * eps)
		anaGrad := conv.Bias.Grad[i]

		diff := float64(math.Abs(float64(anaGrad - numGrad)))
		if diff > 1e-2 {
			t.Fatalf("bias gradient mismatch at %d: analytical=%f, numerical=%f, diff=%f", i, anaGrad, numGrad, diff)
		}
	}

	// 3. Verify Input Gradients vs Numerical Gradient
	for i := range input.Data {
		orig := input.Data[i]

		input.Data[i] = orig + eps
		outPlus := conv.Forward(input)
		var lossPlus float32
		for j := range outPlus.Data {
			diff := outPlus.Data[j] - target.Data[j]
			lossPlus += 0.5 * diff * diff
		}

		input.Data[i] = orig - eps
		outMinus := conv.Forward(input)
		var lossMinus float32
		for j := range outMinus.Data {
			diff := outMinus.Data[j] - target.Data[j]
			lossMinus += 0.5 * diff * diff
		}

		input.Data[i] = orig

		numGrad := (lossPlus - lossMinus) / (2.0 * eps)
		anaGrad := gradInput.Data[i]

		diff := float64(math.Abs(float64(anaGrad - numGrad)))
		if diff > 1e-2 {
			t.Fatalf("input gradient mismatch at %d: analytical=%f, numerical=%f, diff=%f", i, anaGrad, numGrad, diff)
		}
	}
}
