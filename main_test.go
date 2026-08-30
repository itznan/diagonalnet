package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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
	tempDir := filepath.Join(os.TempDir(), "diagonalnet_singlefile_test")
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

	output := conv.Forward(input)

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

// 7. AdaptiveAvgPool2DLayer Unit Tests
func TestAdaptiveAvgPool2DLayer(t *testing.T) {
	pool := NewAdaptiveAvgPool2DLayer(2, 2)

	// Input 1 channel 4x4
	input := NewTensor(1, 4, 4)
	for i := range input.Data {
		input.Data[i] = float32(i)
	}

	output := pool.Forward(input)
	c, h, w := output.Shape()
	if c != 1 || h != 2 || w != 2 {
		t.Fatalf("unexpected pool shape (%d, %d, %d)", c, h, w)
	}

	// Quadrant 0: y in [0,1], x in [0,1] -> input indices (0, 1, 4, 5) -> values (0, 1, 4, 5) -> sum 10 / 4 = 2.5
	q0 := output.Get(0, 0, 0)
	if q0 != 2.5 {
		t.Fatalf("expected q0=2.5, got %f", q0)
	}

	gradOutput := NewTensor(1, 2, 2)
	gradOutput.Set(0, 0, 0, 4.0) // distributes 4.0 / 4 = 1.0 to each quadrant element
	gradInput := pool.Backward(gradOutput)

	if gradInput.Get(0, 0, 0) != 1.0 || gradInput.Get(0, 1, 1) != 1.0 {
		t.Fatalf("gradient distribution mismatch: expected 1.0, got %f", gradInput.Get(0, 0, 0))
	}
}

// 8. LinearLayer Unit Tests
func TestLinearLayerForwardAndBackward(t *testing.T) {
	inDim := 5
	outDim := 3
	rng := rand.New(rand.NewSource(42))
	dense := NewLinearLayer(inDim, outDim, rng)

	input := make([]float32, inDim)
	for i := range input {
		input[i] = rng.Float32()*2.0 - 1.0
	}

	output := dense.Forward(input)
	if len(output) != outDim {
		t.Fatalf("expected output len %d, got %d", outDim, len(output))
	}

	target := make([]float32, outDim)
	for i := range target {
		target[i] = rng.Float32()
	}

	gradOutput := make([]float32, outDim)
	for i := range gradOutput {
		gradOutput[i] = output[i] - target[i]
	}

	dense.ZeroGrad()
	gradInput := dense.Backward(gradOutput)

	eps := float32(1e-3)

	// Verify Weight Gradients
	for i := range dense.Weights.Data {
		orig := dense.Weights.Data[i]

		dense.Weights.Data[i] = orig + eps
		outPlus := dense.Forward(input)
		var lossPlus float32
		for j := range outPlus {
			diff := outPlus[j] - target[j]
			lossPlus += 0.5 * diff * diff
		}

		dense.Weights.Data[i] = orig - eps
		outMinus := dense.Forward(input)
		var lossMinus float32
		for j := range outMinus {
			diff := outMinus[j] - target[j]
			lossMinus += 0.5 * diff * diff
		}

		dense.Weights.Data[i] = orig
		numGrad := (lossPlus - lossMinus) / (2.0 * eps)
		anaGrad := dense.Weights.Grad[i]

		if math.Abs(float64(anaGrad-numGrad)) > 1e-2 {
			t.Fatalf("linear weight gradient mismatch at %d: ana=%f, num=%f", i, anaGrad, numGrad)
		}
	}

	// Verify Bias Gradients
	for i := range dense.Biases.Data {
		orig := dense.Biases.Data[i]

		dense.Biases.Data[i] = orig + eps
		outPlus := dense.Forward(input)
		var lossPlus float32
		for j := range outPlus {
			diff := outPlus[j] - target[j]
			lossPlus += 0.5 * diff * diff
		}

		dense.Biases.Data[i] = orig - eps
		outMinus := dense.Forward(input)
		var lossMinus float32
		for j := range outMinus {
			diff := outMinus[j] - target[j]
			lossMinus += 0.5 * diff * diff
		}

		dense.Biases.Data[i] = orig
		numGrad := (lossPlus - lossMinus) / (2.0 * eps)
		anaGrad := dense.Biases.Grad[i]

		if math.Abs(float64(anaGrad-numGrad)) > 1e-2 {
			t.Fatalf("linear bias gradient mismatch at %d: ana=%f, num=%f", i, anaGrad, numGrad)
		}
	}

	// Verify Input Gradients
	for j := range input {
		orig := input[j]

		input[j] = orig + eps
		outPlus := dense.Forward(input)
		var lossPlus float32
		for k := range outPlus {
			diff := outPlus[k] - target[k]
			lossPlus += 0.5 * diff * diff
		}

		input[j] = orig - eps
		outMinus := dense.Forward(input)
		var lossMinus float32
		for k := range outMinus {
			diff := outMinus[k] - target[k]
			lossMinus += 0.5 * diff * diff
		}

		input[j] = orig
		numGrad := (lossPlus - lossMinus) / (2.0 * eps)
		anaGrad := gradInput[j]

		if math.Abs(float64(anaGrad-numGrad)) > 1e-2 {
			t.Fatalf("linear input gradient mismatch at %d: ana=%f, num=%f", j, anaGrad, numGrad)
		}
	}
}

// 9. DropoutLayer Unit Tests
func TestDropoutLayer(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	dropout := NewDropoutLayer(0.2, rng)

	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}

	// 1. Training mode
	dropout.Training = true
	outTrain := dropout.Forward(input)

	for i, v := range outTrain {
		if dropout.Mask[i] == 1.0 {
			expected := input[i] * 1.25
			if v != expected {
				t.Fatalf("expected scaled value %f, got %f", expected, v)
			}
		} else {
			if v != 0.0 {
				t.Fatalf("expected dropped 0.0, got %f", v)
			}
		}
	}

	gradOut := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	gradIn := dropout.Backward(gradOut)
	for i, g := range gradIn {
		expectedG := gradOut[i] * dropout.Mask[i] * 1.25
		if g != expectedG {
			t.Fatalf("backward gradient mismatch: expected %f, got %f", expectedG, g)
		}
	}

	// 2. Inference mode (Identity)
	dropout.Training = false
	outEval := dropout.Forward(input)
	for i, v := range outEval {
		if v != input[i] {
			t.Fatalf("inference passthrough mismatch: expected %f, got %f", input[i], v)
		}
	}
}

// 10. ReLU Activation Unit Tests
func TestReLUScalar(t *testing.T) {
	if got := ReLU(5.5); got != 5.5 {
		t.Fatalf("expected ReLU(5.5) == 5.5, got %f", got)
	}
	if got := ReLU(-3.2); got != 0.0 {
		t.Fatalf("expected ReLU(-3.2) == 0.0, got %f", got)
	}
	if got := ReLU(0.0); got != 0.0 {
		t.Fatalf("expected ReLU(0.0) == 0.0, got %f", got)
	}

	if got := ReLUGrad(2.0, 3.5); got != 3.5 {
		t.Fatalf("expected ReLUGrad(2.0, 3.5) == 3.5, got %f", got)
	}
	if got := ReLUGrad(-2.0, 3.5); got != 0.0 {
		t.Fatalf("expected ReLUGrad(-2.0, 3.5) == 0.0, got %f", got)
	}
}

func TestReLULayerForwardAndBackward(t *testing.T) {
	relu := NewReLULayer()
	input := []float32{-2.5, -0.5, 0.0, 1.2, 3.8, -10.0, 7.5}
	expectedOutput := []float32{0.0, 0.0, 0.0, 1.2, 3.8, 0.0, 7.5}

	output := relu.Forward(input)
	for i, val := range output {
		if val != expectedOutput[i] {
			t.Fatalf("ReLU forward mismatch at index %d: expected %f, got %f", i, expectedOutput[i], val)
		}
	}

	gradOutput := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}
	expectedGradInput := []float32{0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0}

	gradInput := relu.Backward(gradOutput)
	for i, val := range gradInput {
		if val != expectedGradInput[i] {
			t.Fatalf("ReLU backward mismatch at index %d: expected %f, got %f", i, expectedGradInput[i], val)
		}
	}

	// Numerical Gradient Verification on non-zero points (away from x=0 kink)
	testInputs := []float32{-3.0, -1.5, 0.5, 2.0, 5.0}
	target := []float32{1.0, -0.5, 2.0, 1.5, 3.0}

	outTest := relu.Forward(testInputs)
	gradOutTest := make([]float32, len(testInputs))
	for i := range testInputs {
		gradOutTest[i] = outTest[i] - target[i]
	}
	anaGrad := relu.Backward(gradOutTest)

	eps := float32(1e-3)
	for i := range testInputs {
		orig := testInputs[i]

		testInputs[i] = orig + eps
		outP := relu.Forward(testInputs)
		var lossP float32
		for j := range outP {
			diff := outP[j] - target[j]
			lossP += 0.5 * diff * diff
		}

		testInputs[i] = orig - eps
		outM := relu.Forward(testInputs)
		var lossM float32
		for j := range outM {
			diff := outM[j] - target[j]
			lossM += 0.5 * diff * diff
		}

		testInputs[i] = orig
		numGrad := (lossP - lossM) / (2.0 * eps)

		if math.Abs(float64(anaGrad[i]-numGrad)) > 1e-2 {
			t.Fatalf("ReLU numerical gradient mismatch at %d: analytical=%f, numerical=%f", i, anaGrad[i], numGrad)
		}
	}
}

func TestReLULayerTensor(t *testing.T) {
	relu := NewReLULayer()
	tensor := NewTensor(2, 2, 2)
	tensor.Data = []float32{-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 0.0, 8.0}

	out := relu.ForwardTensor(tensor)
	expected := []float32{0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0}

	for i, v := range out.Data {
		if v != expected[i] {
			t.Fatalf("ReLU ForwardTensor mismatch at index %d: expected %f, got %f", i, expected[i], v)
		}
	}

	gradOut := NewTensor(2, 2, 2)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0
	}
	gradIn := relu.BackwardTensor(gradOut)
	expectedGrad := []float32{0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0}

	for i, v := range gradIn.Data {
		if v != expectedGrad[i] {
			t.Fatalf("ReLU BackwardTensor mismatch at index %d: expected %f, got %f", i, expectedGrad[i], v)
		}
	}
}

// 11. LeakyReLU Activation Unit Tests
func TestLeakyReLUScalar(t *testing.T) {
	alpha := float32(0.01)
	if got := LeakyReLU(4.0, alpha); got != 4.0 {
		t.Fatalf("expected LeakyReLU(4.0) == 4.0, got %f", got)
	}
	if got := LeakyReLU(-5.0, alpha); math.Abs(float64(got-(-0.05))) > 1e-5 {
		t.Fatalf("expected LeakyReLU(-5.0) == -0.05, got %f", got)
	}
	if got := LeakyReLU(0.0, alpha); got != 0.0 {
		t.Fatalf("expected LeakyReLU(0.0) == 0.0, got %f", got)
	}

	if got := LeakyReLUGrad(2.0, 5.0, alpha); got != 5.0 {
		t.Fatalf("expected LeakyReLUGrad(2.0, 5.0) == 5.0, got %f", got)
	}
	if got := LeakyReLUGrad(-2.0, 5.0, alpha); math.Abs(float64(got-0.05)) > 1e-5 {
		t.Fatalf("expected LeakyReLUGrad(-2.0, 5.0) == 0.05, got %f", got)
	}
}

func TestLeakyReLULayerForwardAndBackward(t *testing.T) {
	alpha := float32(0.02)
	leaky := NewLeakyReLULayer(alpha)

	input := []float32{-10.0, -2.0, 0.0, 3.0, 6.0}
	expectedOutput := []float32{-0.20, -0.04, 0.0, 3.0, 6.0}

	output := leaky.Forward(input)
	for i, val := range output {
		diff := math.Abs(float64(val - expectedOutput[i]))
		if diff > 1e-5 {
			t.Fatalf("LeakyReLU forward mismatch at %d: expected %f, got %f", i, expectedOutput[i], val)
		}
	}

	gradOutput := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	expectedGradInput := []float32{0.02, 0.04, 0.06, 4.0, 5.0}

	gradInput := leaky.Backward(gradOutput)
	for i, val := range gradInput {
		diff := math.Abs(float64(val - expectedGradInput[i]))
		if diff > 1e-5 {
			t.Fatalf("LeakyReLU backward mismatch at %d: expected %f, got %f", i, expectedGradInput[i], val)
		}
	}

	// Numerical Gradient Verification across both negative and positive domains
	testInputs := []float32{-4.0, -1.0, 1.5, 3.5}
	target := []float32{-0.1, 0.2, 1.0, 2.0}

	outTest := leaky.Forward(testInputs)
	gradOutTest := make([]float32, len(testInputs))
	for i := range testInputs {
		gradOutTest[i] = outTest[i] - target[i]
	}
	anaGrad := leaky.Backward(gradOutTest)

	eps := float32(1e-3)
	for i := range testInputs {
		orig := testInputs[i]

		testInputs[i] = orig + eps
		outP := leaky.Forward(testInputs)
		var lossP float32
		for j := range outP {
			diff := outP[j] - target[j]
			lossP += 0.5 * diff * diff
		}

		testInputs[i] = orig - eps
		outM := leaky.Forward(testInputs)
		var lossM float32
		for j := range outM {
			diff := outM[j] - target[j]
			lossM += 0.5 * diff * diff
		}

		testInputs[i] = orig
		numGrad := (lossP - lossM) / (2.0 * eps)

		if math.Abs(float64(anaGrad[i]-numGrad)) > 1e-2 {
			t.Fatalf("LeakyReLU numerical gradient mismatch at %d: analytical=%f, numerical=%f", i, anaGrad[i], numGrad)
		}
	}
}

func TestLeakyReLULayerTensor(t *testing.T) {
	leaky := NewLeakyReLULayer(0.1)
	tensor := NewTensor(1, 2, 2)
	tensor.Data = []float32{-5.0, 10.0, -20.0, 30.0}

	out := leaky.ForwardTensor(tensor)
	expected := []float32{-0.5, 10.0, -2.0, 30.0}

	for i, v := range out.Data {
		if math.Abs(float64(v-expected[i])) > 1e-5 {
			t.Fatalf("LeakyReLU ForwardTensor mismatch at index %d: expected %f, got %f", i, expected[i], v)
		}
	}

	gradOut := NewTensor(1, 2, 2)
	gradOut.Data = []float32{2.0, 2.0, 2.0, 2.0}
	gradIn := leaky.BackwardTensor(gradOut)
	expectedGrad := []float32{0.2, 2.0, 0.2, 2.0}

	for i, v := range gradIn.Data {
		if math.Abs(float64(v-expectedGrad[i])) > 1e-5 {
			t.Fatalf("LeakyReLU BackwardTensor mismatch at index %d: expected %f, got %f", i, expectedGrad[i], v)
		}
	}
}

// 12. Softmax Probability Distribution Unit Tests
func TestSoftmaxBasic(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := Softmax(logits)

	if len(probs) != len(logits) {
		t.Fatalf("expected len %d, got %d", len(logits), len(probs))
	}

	var sum float32
	for _, p := range probs {
		if p < 0 || p > 1 {
			t.Fatalf("probability %f out of range [0, 1]", p)
		}
		sum += p
	}

	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Fatalf("expected probability sum 1.0, got %f", sum)
	}

	// Verify monotonically increasing probabilities since logits are increasing
	if !(probs[0] < probs[1] && probs[1] < probs[2]) {
		t.Fatalf("expected strictly increasing probabilities, got %v", probs)
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	// Extreme logits that would cause standard exp() to overflow float32/float64 to +Inf
	extremeLogits := []float32{1000.0, 1002.0, 1005.0}
	probs := Softmax(extremeLogits)

	var sum float32
	for i, p := range probs {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
			t.Fatalf("Softmax produced NaN or Inf at index %d: %f", i, p)
		}
		if p < 0 || p > 1 {
			t.Fatalf("probability %f out of bounds", p)
		}
		sum += p
	}

	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Fatalf("expected sum 1.0 on extreme logits, got %f", sum)
	}

	// Difference between 1000 and 1005 is 5.0, exp(-5) / (exp(-5) + exp(-3) + exp(0))
	expectedP2 := float32(1.0 / (math.Exp(-5.0) + math.Exp(-3.0) + 1.0))
	if math.Abs(float64(probs[2]-expectedP2)) > 1e-4 {
		t.Fatalf("expected p[2] == %f, got %f", expectedP2, probs[2])
	}
}

func TestSoftmaxLayerForwardAndBackward(t *testing.T) {
	softmax := NewSoftmaxLayer()
	logits := []float32{1.5, 0.2, -0.8, 2.3}
	target := []float32{0.1, 0.2, 0.0, 0.7}

	probs := softmax.Forward(logits)
	gradOutput := make([]float32, len(logits))
	for i := range logits {
		gradOutput[i] = probs[i] - target[i]
	}

	anaGrad := softmax.Backward(gradOutput)

	eps := float32(1e-3)
	for i := range logits {
		orig := logits[i]

		logits[i] = orig + eps
		outP := Softmax(logits)
		var lossP float32
		for j := range outP {
			diff := outP[j] - target[j]
			lossP += 0.5 * diff * diff
		}

		logits[i] = orig - eps
		outM := Softmax(logits)
		var lossM float32
		for j := range outM {
			diff := outM[j] - target[j]
			lossM += 0.5 * diff * diff
		}

		logits[i] = orig
		numGrad := (lossP - lossM) / (2.0 * eps)

		if math.Abs(float64(anaGrad[i]-numGrad)) > 1e-2 {
			t.Fatalf("Softmax numerical gradient mismatch at %d: analytical=%f, numerical=%f", i, anaGrad[i], numGrad)
		}
	}
}

// 13. Categorical Cross-Entropy Loss & Analytical Softmax Derivatives Unit Tests
func TestCategoricalCrossEntropyValues(t *testing.T) {
	lossCriterion := NewCategoricalCrossEntropyLoss()

	// Perfect prediction p_target = 1.0 -> Loss ~ 0.0
	probsPerfect := []float32{0.0, 1.0, 0.0}
	loss0 := lossCriterion.Forward(probsPerfect, 1)
	if math.Abs(float64(loss0)) > 1e-4 {
		t.Fatalf("expected loss ~0.0 for p_target=1.0, got %f", loss0)
	}

	// p_target = 1/e ~ 0.367879 -> Loss ~ 1.0
	eInv := float32(math.Exp(-1.0))
	probsE := []float32{eInv, 1.0 - eInv}
	loss1 := lossCriterion.Forward(probsE, 0)
	if math.Abs(float64(loss1-1.0)) > 1e-4 {
		t.Fatalf("expected loss ~1.0 for p_target=1/e, got %f", loss1)
	}

	// Zero probability p_target = 0.0 -> Loss finite positive value due to eps=1e-15
	probsZero := []float32{1.0, 0.0}
	lossZero := lossCriterion.Forward(probsZero, 1)
	if math.IsNaN(float64(lossZero)) || math.IsInf(float64(lossZero), 0) || lossZero <= 0 {
		t.Fatalf("expected finite positive loss for p_target=0, got %f", lossZero)
	}
}

func TestCategoricalCrossEntropyOneHot(t *testing.T) {
	probs := []float32{0.1, 0.7, 0.2}
	oneHot := []float32{0.0, 1.0, 0.0}

	scalarLoss := CategoricalCrossEntropy(probs, 1)
	oneHotLoss := CategoricalCrossEntropyOneHot(probs, oneHot)

	if math.Abs(float64(scalarLoss-oneHotLoss)) > 1e-6 {
		t.Fatalf("mismatch between scalar loss (%f) and one-hot loss (%f)", scalarLoss, oneHotLoss)
	}
}

func TestSoftmaxCrossEntropyAnalyticalGradients(t *testing.T) {
	lossCriterion := NewCategoricalCrossEntropyLoss()

	logits := []float32{2.5, -1.0, 0.5, 3.2, -0.4}
	targetClass := 3

	loss, probs, anaGrad := lossCriterion.LossAndGrad(logits, targetClass)
	if loss <= 0 {
		t.Fatalf("expected positive loss, got %f", loss)
	}

	// Verify analytical gradient formula: dL/dz_i = p_i - 1(i == target)
	for i, p := range probs {
		var expectedGrad float32
		if i == targetClass {
			expectedGrad = p - 1.0
		} else {
			expectedGrad = p
		}
		if math.Abs(float64(anaGrad[i]-expectedGrad)) > 1e-6 {
			t.Fatalf("analytical gradient formula mismatch at %d: expected %f, got %f", i, expectedGrad, anaGrad[i])
		}
	}

	// Verify analytical gradient against finite-difference numerical gradients
	eps := float32(1e-3)
	for i := range logits {
		orig := logits[i]

		logits[i] = orig + eps
		probsP := Softmax(logits)
		lossP := lossCriterion.Forward(probsP, targetClass)

		logits[i] = orig - eps
		probsM := Softmax(logits)
		lossM := lossCriterion.Forward(probsM, targetClass)

		logits[i] = orig
		numGrad := (lossP - lossM) / (2.0 * eps)

		if math.Abs(float64(anaGrad[i]-numGrad)) > 1e-2 {
			t.Fatalf("Softmax-CrossEntropy numerical gradient mismatch at %d: analytical=%f, numerical=%f", i, anaGrad[i], numGrad)
		}
	}
}

// 14. Adam Optimizer Unit Tests
func TestAdamOptimizerSingleStep(t *testing.T) {
	param := NewParameter(2)
	param.Data[0] = 5.0
	param.Data[1] = -2.0
	param.Grad[0] = 0.5
	param.Grad[1] = -0.8

	cfg := AdamOptimizerConfig{
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  0.0,
	}

	opt := NewAdamOptimizer([]*Parameter{param}, cfg)
	opt.Step()

	// Verify step count
	if opt.StepCount != 1 {
		t.Fatalf("expected step count 1, got %d", opt.StepCount)
	}

	// Step 1 theoretical values:
	// For element 0 (g = 0.5):
	// m1 = (1 - 0.9) * 0.5 = 0.05
	// v1 = (1 - 0.999) * 0.25 = 0.00025
	// mHat = 0.05 / 0.1 = 0.5
	// vHat = 0.00025 / 0.001 = 0.25 -> sqrt(vHat) = 0.5
	// update = 0.01 * 0.5 / 0.5 = 0.01
	// new_data = 5.0 - 0.01 = 4.99
	if math.Abs(float64(param.M[0]-0.05)) > 1e-5 {
		t.Fatalf("expected m[0] == 0.05, got %f", param.M[0])
	}
	if math.Abs(float64(param.V[0]-0.00025)) > 1e-6 {
		t.Fatalf("expected v[0] == 0.00025, got %f", param.V[0])
	}
	if math.Abs(float64(param.Data[0]-4.99)) > 1e-4 {
		t.Fatalf("expected data[0] == 4.99, got %f", param.Data[0])
	}

	// For element 1 (g = -0.8, negative):
	// update = 0.01 * (-0.8) / 0.8 = -0.01
	// new_data = -2.0 - (-0.01) = -1.99
	if math.Abs(float64(param.Data[1]-(-1.99))) > 1e-4 {
		t.Fatalf("expected data[1] == -1.99, got %f", param.Data[1])
	}
}

func TestAdamOptimizerConvergence(t *testing.T) {
	// Minimize quadratic loss: L(w) = 0.5 * (w - 3.0)^2 -> dL/dw = w - 3.0
	param := NewParameter(1)
	param.Data[0] = 0.0 // Start at 0.0, optimal is 3.0

	cfg := AdamOptimizerConfig{
		LearningRate: 0.1,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
	}

	opt := NewAdamOptimizer([]*Parameter{param}, cfg)

	for step := 0; step < 150; step++ {
		w := param.Data[0]
		grad := w - 3.0
		param.Grad[0] = grad
		opt.Step()
	}

	finalVal := param.Data[0]
	if math.Abs(float64(finalVal-3.0)) > 1e-2 {
		t.Fatalf("Adam optimizer failed to converge to 3.0: final value = %f", finalVal)
	}
}

func TestAdamOptimizerMultiParamAndZeroGrad(t *testing.T) {
	p1 := NewParameter(10)
	p2 := NewParameter(5)

	for i := range p1.Grad {
		p1.Grad[i] = 1.0
	}
	for i := range p2.Grad {
		p2.Grad[i] = 2.0
	}

	opt := NewAdamOptimizer([]*Parameter{p1, p2}, DefaultAdamConfig())
	opt.ZeroGrad()

	for i, g := range p1.Grad {
		if g != 0 {
			t.Fatalf("p1 grad at %d not zeroed: %f", i, g)
		}
	}
	for i, g := range p2.Grad {
		if g != 0 {
			t.Fatalf("p2 grad at %d not zeroed: %f", i, g)
		}
	}
}

func TestAdamL2WeightDecayRegularization(t *testing.T) {
	// Parameter with weight theta = 10.0, raw grad = 0.0, lambda = 0.01
	// Effective regularized gradient: g_reg = 0.0 + 0.01 * 10.0 = 0.1
	param := NewParameter(1)
	param.Data[0] = 10.0
	param.Grad[0] = 0.0

	cfg := AdamOptimizerConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  0.01, // lambda = 0.01
	}

	opt := NewAdamOptimizer([]*Parameter{param}, cfg)
	opt.Step()

	// Effective gradient was 0.1. At t=1:
	// m1 = (1 - 0.9) * 0.1 = 0.01
	// v1 = (1 - 0.999) * 0.01 = 0.00001
	// mHat = 0.01 / 0.1 = 0.1
	// vHat = 0.00001 / 0.001 = 0.01 -> sqrt(vHat) = 0.1
	// update = 0.001 * 0.1 / 0.1 = 0.001
	// theta_new = 10.0 - 0.001 = 9.999
	if math.Abs(float64(param.Data[0]-9.999)) > 1e-4 {
		t.Fatalf("expected weight decay update to 9.999, got %f", param.Data[0])
	}
}

func TestAdamAnalyticalBiasCorrectionsMultiStep(t *testing.T) {
	// Step-by-step mathematical check of bias corrections across 5 steps with constant gradient
	param := NewParameter(1)
	param.Data[0] = 0.0
	gConst := float32(1.0)
	lr := float32(0.001)
	beta1 := float32(0.9)
	beta2 := float32(0.999)
	eps := float32(1e-8)

	var m, v float64
	for step := 1; step <= 5; step++ {
		param.Grad[0] = gConst
		StepParameter(param, step, lr, beta1, beta2, eps, 0.0)

		// Manual calculation
		m = 0.9*m + 0.1*1.0
		v = 0.999*v + 0.001*1.0

		biasCorr1 := 1.0 - math.Pow(0.9, float64(step))
		biasCorr2 := 1.0 - math.Pow(0.999, float64(step))

		expectedMHat := m / biasCorr1
		expectedVHat := v / biasCorr2

		actualM := float64(param.M[0])
		actualV := float64(param.V[0])

		if math.Abs(actualM-m) > 1e-5 {
			t.Fatalf("step %d: m mismatch: expected %f, got %f", step, m, actualM)
		}
		if math.Abs(actualV-v) > 1e-6 {
			t.Fatalf("step %d: v mismatch: expected %f, got %f", step, v, actualV)
		}

		actualMHat := actualM / biasCorr1
		actualVHat := actualV / biasCorr2
		if math.Abs(actualMHat-expectedMHat) > 1e-4 {
			t.Fatalf("step %d: mHat mismatch: expected %f, got %f", step, expectedMHat, actualMHat)
		}
		if math.Abs(actualVHat-expectedVHat) > 1e-4 {
			t.Fatalf("step %d: vHat mismatch: expected %f, got %f", step, expectedVHat, actualVHat)
		}
	}
}

// 15. Step Learning Rate Decay Scheduler Unit Tests
func TestStepLRSchedulerDefaultSchedule(t *testing.T) {
	param := NewParameter(1)
	opt := NewAdamOptimizer([]*Parameter{param}, DefaultAdamConfig())
	sched := NewStepLRScheduler(opt, DefaultStepLRSchedulerConfig())

	// Epochs 1 to 7: Initial LR = 0.002
	for epoch := 1; epoch <= 7; epoch++ {
		lr := sched.Step(epoch)
		if math.Abs(float64(lr-0.002)) > 1e-6 {
			t.Fatalf("epoch %d: expected LR 0.002, got %f", epoch, lr)
		}
		if math.Abs(float64(opt.Config.LearningRate-0.002)) > 1e-6 {
			t.Fatalf("epoch %d: optimizer LR mismatch: expected 0.002, got %f", epoch, opt.Config.LearningRate)
		}
	}

	// Epochs 8 to 16 (including 8-12): 50% decay -> LR = 0.001
	for epoch := 8; epoch <= 16; epoch++ {
		lr := sched.Step(epoch)
		if math.Abs(float64(lr-0.001)) > 1e-6 {
			t.Fatalf("epoch %d: expected LR 0.001 (50%% decay), got %f", epoch, lr)
		}
		if math.Abs(float64(opt.Config.LearningRate-0.001)) > 1e-6 {
			t.Fatalf("epoch %d: optimizer LR mismatch: expected 0.001, got %f", epoch, opt.Config.LearningRate)
		}
	}

	// Epochs 17+: 25% decay -> LR = 0.0005
	for epoch := 17; epoch <= 25; epoch++ {
		lr := sched.Step(epoch)
		if math.Abs(float64(lr-0.0005)) > 1e-6 {
			t.Fatalf("epoch %d: expected LR 0.0005 (25%% decay), got %f", epoch, lr)
		}
		if math.Abs(float64(opt.Config.LearningRate-0.0005)) > 1e-6 {
			t.Fatalf("epoch %d: optimizer LR mismatch: expected 0.0005, got %f", epoch, opt.Config.LearningRate)
		}
	}
}

func TestStepLRSchedulerJSONPersistence(t *testing.T) {
	tempDir := filepath.Join(os.TempDir(), "diagonalnet_scheduler_test")
	_ = os.RemoveAll(tempDir)
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "scheduler_settings.json")

	customCfg := StepLRSchedulerConfig{
		InitialLR: 0.005,
		Milestones: []LRMilestone{
			{Epoch: 5, Factor: 0.5},
			{Epoch: 10, Factor: 0.1},
		},
	}

	if err := SaveStepLRSchedulerConfig(configPath, &customCfg); err != nil {
		t.Fatalf("SaveStepLRSchedulerConfig failed: %v", err)
	}

	loadedCfg, err := LoadStepLRSchedulerConfig(configPath)
	if err != nil {
		t.Fatalf("LoadStepLRSchedulerConfig failed: %v", err)
	}

	if loadedCfg.InitialLR != 0.005 || len(loadedCfg.Milestones) != 2 {
		t.Fatalf("loaded config mismatch: %+v", loadedCfg)
	}
	if loadedCfg.Milestones[0].Epoch != 5 || loadedCfg.Milestones[0].Factor != 0.5 {
		t.Fatalf("milestone 0 mismatch")
	}

	param := NewParameter(1)
	opt := NewAdamOptimizer([]*Parameter{param}, DefaultAdamConfig())
	sched, err := NewStepLRSchedulerFromFile(opt, configPath)
	if err != nil {
		t.Fatalf("NewStepLRSchedulerFromFile failed: %v", err)
	}

	if sched.GetLR(1) != 0.005 {
		t.Fatalf("expected LR 0.005 at epoch 1, got %f", sched.GetLR(1))
	}
	if sched.GetLR(5) != 0.0025 {
		t.Fatalf("expected LR 0.0025 at epoch 5, got %f", sched.GetLR(5))
	}
	if math.Abs(float64(sched.GetLR(10)-0.0005)) > 1e-6 {
		t.Fatalf("expected LR 0.0005 at epoch 10, got %f", sched.GetLR(10))
	}
}

// 16. Dynamic Dataset Scanner & Bi-Directional Class Mapping Unit Tests
func TestDatasetMetadataTwoWayMapping(t *testing.T) {
	rawClasses := []string{"triangle", "circle", "square"}
	meta := NewDatasetMetadata(rawClasses)

	// Check alphabetical sort order
	if len(meta.Classes) != 3 {
		t.Fatalf("expected 3 classes, got %d", len(meta.Classes))
	}
	if meta.Classes[0] != "circle" || meta.Classes[1] != "square" || meta.Classes[2] != "triangle" {
		t.Fatalf("classes not sorted alphabetically: %v", meta.Classes)
	}
	if meta.NumClasses != 3 {
		t.Fatalf("expected NumClasses 3, got %d", meta.NumClasses)
	}

	// Verify ClassToIdx
	if meta.ClassToIdx["circle"] != 0 || meta.ClassToIdx["square"] != 1 || meta.ClassToIdx["triangle"] != 2 {
		t.Fatalf("invalid ClassToIdx mapping: %+v", meta.ClassToIdx)
	}

	// Verify IdxToClass
	if meta.IdxToClass[0] != "circle" || meta.IdxToClass[1] != "square" || meta.IdxToClass[2] != "triangle" {
		t.Fatalf("invalid IdxToClass mapping: %+v", meta.IdxToClass)
	}

	// Verify accessor methods
	idx, ok := meta.GetClassIndex("square")
	if !ok || idx != 1 {
		t.Fatalf("GetClassIndex failed for square: got %d, %v", idx, ok)
	}
	name, ok := meta.GetClassName(2)
	if !ok || name != "triangle" {
		t.Fatalf("GetClassName failed for index 2: got %s, %v", name, ok)
	}

	// Verify missing lookups
	if _, ok := meta.GetClassIndex("unknown"); ok {
		t.Fatalf("expected unknown class lookup to fail")
	}
	if _, ok := meta.GetClassName(99); ok {
		t.Fatalf("expected out-of-bounds index lookup to fail")
	}
}

func TestScanDatasetValidFilesystem(t *testing.T) {
	tempDir := filepath.Join(os.TempDir(), "diagonalnet_dataset_test")
	_ = os.RemoveAll(tempDir)
	defer os.RemoveAll(tempDir)

	// Create 3 subdirectories: dog, cat, bird
	classes := []string{"dog", "cat", "bird"}
	for _, c := range classes {
		cDir := filepath.Join(tempDir, c)
		if err := os.MkdirAll(cDir, 0755); err != nil {
			t.Fatalf("failed to create class dir: %v", err)
		}
		// Create mock image files
		_ = os.WriteFile(filepath.Join(cDir, "img1.png"), []byte{0x89, 0x50, 0x4E, 0x47}, 0644)
		_ = os.WriteFile(filepath.Join(cDir, "img2.JPG"), []byte{0xFF, 0xD8, 0xFF}, 0644)
		_ = os.WriteFile(filepath.Join(cDir, "img3.jpeg"), []byte{0xFF, 0xD8, 0xFF}, 0644)
		// Non-image file that should be ignored
		_ = os.WriteFile(filepath.Join(cDir, "notes.txt"), []byte("ignore me"), 0644)
	}

	ds, err := ScanDataset(tempDir)
	if err != nil {
		t.Fatalf("ScanDataset failed: %v", err)
	}

	// 3 classes: bird (0), cat (1), dog (2)
	if ds.Metadata.NumClasses != 3 {
		t.Fatalf("expected 3 classes, got %d", ds.Metadata.NumClasses)
	}
	if ds.Metadata.Classes[0] != "bird" || ds.Metadata.Classes[1] != "cat" || ds.Metadata.Classes[2] != "dog" {
		t.Fatalf("unexpected sorted classes: %v", ds.Metadata.Classes)
	}

	// Total images: 3 per class * 3 classes = 9 images
	if len(ds.Samples) != 9 {
		t.Fatalf("expected 9 samples, got %d", len(ds.Samples))
	}

	// Verify sample labels match class metadata
	for _, s := range ds.Samples {
		expectedIdx := ds.Metadata.ClassToIdx[s.Class]
		if s.ClassIndex != expectedIdx {
			t.Fatalf("sample classIndex mismatch for path %s: got %d, expected %d", s.Path, s.ClassIndex, expectedIdx)
		}
		if !IsValidImageExtension(s.Path) {
			t.Fatalf("sample has invalid image extension: %s", s.Path)
		}
	}
}

func TestScanDatasetErrorHandling(t *testing.T) {
	// 1. Non-existent directory
	_, err := ScanDataset("C:/non_existent_directory_diagonalnet_12345")
	if err == nil {
		t.Fatalf("expected error for non-existent directory")
	}

	// 2. Directory with only 1 class (< 2 classes)
	tempDir := filepath.Join(os.TempDir(), "diagonalnet_dataset_err1")
	_ = os.RemoveAll(tempDir)
	defer os.RemoveAll(tempDir)

	_ = os.MkdirAll(filepath.Join(tempDir, "only_one_class"), 0755)
	_ = os.WriteFile(filepath.Join(tempDir, "only_one_class", "a.png"), []byte("png"), 0644)

	_, err = ScanDataset(tempDir)
	if err == nil {
		t.Fatalf("expected error for directory with only 1 class")
	}

	// 3. Directory with subdirectories containing 0 valid images
	tempDir2 := filepath.Join(os.TempDir(), "diagonalnet_dataset_err2")
	_ = os.RemoveAll(tempDir2)
	defer os.RemoveAll(tempDir2)

	_ = os.MkdirAll(filepath.Join(tempDir2, "class_a"), 0755)
	_ = os.MkdirAll(filepath.Join(tempDir2, "class_b"), 0755)
	_ = os.WriteFile(filepath.Join(tempDir2, "class_a", "readme.txt"), []byte("no images"), 0644)
	_ = os.WriteFile(filepath.Join(tempDir2, "class_b", "readme.txt"), []byte("no images"), 0644)

	_, err = ScanDataset(tempDir2)
	if err == nil {
		t.Fatalf("expected error for classes containing zero images")
	}
}

// 17. Native Image Loading & Grayscale Conversion Unit Tests (Prompt 31)
func TestLoadImageFromFileAndTensor(t *testing.T) {
	tempDir := filepath.Join(os.TempDir(), "diagonalnet_img_test")
	_ = os.RemoveAll(tempDir)
	defer os.RemoveAll(tempDir)
	_ = os.MkdirAll(tempDir, 0755)

	imgFile := filepath.Join(tempDir, "test.png")
	rgba := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			rgba.Set(x, y, color.RGBA{R: uint8(x * 7), G: uint8(y * 7), B: 128, A: 255})
		}
	}

	f, err := os.Create(imgFile)
	if err != nil {
		t.Fatalf("failed to create image file: %v", err)
	}
	if err := png.Encode(f, rgba); err != nil {
		_ = f.Close()
		t.Fatalf("failed to encode png: %v", err)
	}
	_ = f.Close()

	gray, err := LoadImageFromFile(imgFile)
	if err != nil {
		t.Fatalf("LoadImageFromFile failed: %v", err)
	}

	if gray.Bounds().Dx() != 32 || gray.Bounds().Dy() != 32 {
		t.Fatalf("unexpected gray bounds: %v", gray.Bounds())
	}

	tensor := GrayImageToTensor(gray)
	if tensor.Channels != 1 || tensor.Height != 32 || tensor.Width != 32 {
		t.Fatalf("tensor shape mismatch: [%d, %d, %d]", tensor.Channels, tensor.Height, tensor.Width)
	}

	// Verify normalization within [0.0, 1.0]
	for i, v := range tensor.Data {
		if v < 0.0 || v > 1.0 {
			t.Fatalf("pixel at %d out of normalized range: %f", i, v)
		}
	}
}

// 18. Stratified Train/Test Dataset Splitting Unit Tests (Prompt 32)
func TestTrainTestSplitStratification(t *testing.T) {
	var items []ImageItem
	// Class 0: 40 items
	for i := 0; i < 40; i++ {
		items = append(items, ImageItem{Path: fmt.Sprintf("c0_%d.png", i), Class: "circle", ClassIndex: 0})
	}
	// Class 1: 30 items
	for i := 0; i < 30; i++ {
		items = append(items, ImageItem{Path: fmt.Sprintf("c1_%d.png", i), Class: "square", ClassIndex: 1})
	}
	// Class 2: 30 items
	for i := 0; i < 30; i++ {
		items = append(items, ImageItem{Path: fmt.Sprintf("c2_%d.png", i), Class: "triangle", ClassIndex: 2})
	}

	testRatio := 0.20
	seed := int64(12345)
	trainSet, valSet := TrainTestSplit(items, testRatio, seed)

	// Total validation count: floor(40*0.2) + floor(30*0.2) + floor(30*0.2) = 8 + 6 + 6 = 20
	// Total train count: (40-8) + (30-6) + (30-6) = 32 + 24 + 24 = 80
	if len(valSet) != 20 {
		t.Fatalf("expected 20 validation samples, got %d", len(valSet))
	}
	if len(trainSet) != 80 {
		t.Fatalf("expected 80 train samples, got %d", len(trainSet))
	}

	// Verify exact class counts in valSet
	valCounts := make(map[int]int)
	for _, item := range valSet {
		valCounts[item.ClassIndex]++
	}
	if valCounts[0] != 8 || valCounts[1] != 6 || valCounts[2] != 6 {
		t.Fatalf("stratified val class counts mismatch: %+v", valCounts)
	}

	// Verify exact class counts in trainSet
	trainCounts := make(map[int]int)
	for _, item := range trainSet {
		trainCounts[item.ClassIndex]++
	}
	if trainCounts[0] != 32 || trainCounts[1] != 24 || trainCounts[2] != 24 {
		t.Fatalf("stratified train class counts mismatch: %+v", trainCounts)
	}

	// Verify deterministic repeatability with same seed
	train2, val2 := TrainTestSplit(items, testRatio, seed)
	for i := range trainSet {
		if trainSet[i].Path != train2[i].Path {
			t.Fatalf("train split non-deterministic at %d", i)
		}
	}
	for i := range valSet {
		if valSet[i].Path != val2[i].Path {
			t.Fatalf("val split non-deterministic at %d", i)
		}
	}
}

// 19. Automated Dataset Health & Quality Auditor Unit Tests (Prompt 33)
func TestAuditDatasetQualityAndStats(t *testing.T) {
	tempDir := filepath.Join(os.TempDir(), "diagonalnet_audit_test")
	_ = os.RemoveAll(tempDir)
	defer os.RemoveAll(tempDir)

	c1 := filepath.Join(tempDir, "class_a")
	c2 := filepath.Join(tempDir, "class_b")
	_ = os.MkdirAll(c1, 0755)
	_ = os.MkdirAll(c2, 0755)

	savePNG := func(path string, w, h int, drawFunc func(img *image.RGBA)) {
		rgba := image.NewRGBA(image.Rect(0, 0, w, h))
		// Fill white background (255)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				rgba.Set(x, y, color.RGBA{255, 255, 255, 255})
			}
		}
		if drawFunc != nil {
			drawFunc(rgba)
		}
		f, _ := os.Create(path)
		_ = png.Encode(f, rgba)
		_ = f.Close()
	}

	// 1. Normal drawing in class_a (100 foreground pixels)
	savePNG(filepath.Join(c1, "normal1.png"), 32, 32, func(img *image.RGBA) {
		for y := 10; y < 20; y++ {
			for x := 10; x < 20; x++ {
				img.Set(x, y, color.RGBA{0, 0, 0, 255})
			}
		}
	})

	// 2. Tiny drawing in class_a (< 30 foreground pixels, e.g. 5 pixels)
	savePNG(filepath.Join(c1, "tiny1.png"), 32, 32, func(img *image.RGBA) {
		for x := 0; x < 5; x++ {
			img.Set(x, 0, color.RGBA{0, 0, 0, 255})
		}
	})

	// 3. Blank image in class_a (0 foreground pixels)
	savePNG(filepath.Join(c1, "blank1.png"), 32, 32, nil)

	// 4. Corrupt image in class_a
	_ = os.WriteFile(filepath.Join(c1, "corrupt1.png"), []byte("not a real png"), 0644)

	// 5. Normal drawings in class_b
	savePNG(filepath.Join(c2, "normal2.png"), 32, 32, func(img *image.RGBA) {
		for y := 5; y < 15; y++ {
			for x := 5; x < 15; x++ {
				img.Set(x, y, color.RGBA{0, 0, 0, 255})
			}
		}
	})
	savePNG(filepath.Join(c2, "normal3.png"), 32, 32, func(img *image.RGBA) {
		for y := 8; y < 18; y++ {
			for x := 8; x < 18; x++ {
				img.Set(x, y, color.RGBA{0, 0, 0, 255})
			}
		}
	})

	report, err := AuditDataset(tempDir)
	if err != nil {
		t.Fatalf("AuditDataset failed: %v", err)
	}

	if report.NumClasses != 2 {
		t.Fatalf("expected 2 classes, got %d", report.NumClasses)
	}
	if report.TotalSamples != 6 {
		t.Fatalf("expected 6 total samples, got %d", report.TotalSamples)
	}
	if report.CorruptCount != 1 {
		t.Fatalf("expected 1 corrupt file, got %d", report.CorruptCount)
	}
	if report.BlankCount != 1 {
		t.Fatalf("expected 1 blank image, got %d", report.BlankCount)
	}
	if report.TinyCount != 1 {
		t.Fatalf("expected 1 tiny outlier, got %d", report.TinyCount)
	}
	if report.ValidCount != 3 {
		t.Fatalf("expected 3 valid images, got %d", report.ValidCount)
	}

	// Verify bounding box calculation for normal1.png (10x10 bbox)
	resNormal := AuditImage(filepath.Join(c1, "normal1.png"), "class_a", 0)
	if resNormal.BBoxWidth != 10 || resNormal.BBoxHeight != 10 {
		t.Fatalf("normal bbox mismatch: %d x %d", resNormal.BBoxWidth, resNormal.BBoxHeight)
	}
	if math.Abs(resNormal.AspectRatio-1.0) > 1e-4 {
		t.Fatalf("normal aspect ratio mismatch: %f", resNormal.AspectRatio)
	}

	// Verify PrintAuditReport doesn't panic
	PrintAuditReport(report)
}

// 20. Tight Bounding Box Locator Unit Tests (Prompt 35)
func TestFindBoundingBox(t *testing.T) {
	// 100x100 grayscale image with foreground in [20, 35] x [40, 75]
	gray := image.NewGray(image.Rect(0, 0, 100, 100))
	for y := 40; y <= 75; y++ {
		for x := 20; x <= 35; x++ {
			gray.SetGray(x, y, color.Gray{Y: 255})
		}
	}

	bbox := FindBoundingBox(gray, 10)
	if bbox == nil {
		t.Fatalf("expected valid bounding box, got nil")
	}

	if bbox.MinX != 20 || bbox.MaxX != 35 || bbox.MinY != 40 || bbox.MaxY != 75 {
		t.Fatalf("bounding box coordinate mismatch: %+v", bbox)
	}
	if bbox.Width() != 16 || bbox.Height() != 36 {
		t.Fatalf("bounding box dimension mismatch: %d x %d", bbox.Width(), bbox.Height())
	}

	// Test blank image returns nil
	blank := image.NewGray(image.Rect(0, 0, 50, 50))
	if bboxBlank := FindBoundingBox(blank, 10); bboxBlank != nil {
		t.Fatalf("expected nil bounding box for blank image, got %+v", bboxBlank)
	}

	// Test Tensor version
	tensor := GrayImageToTensor(gray)
	bboxTensor := FindBoundingBoxTensor(tensor, 0.05)
	if bboxTensor == nil || bboxTensor.MinX != 20 || bboxTensor.MaxX != 35 || bboxTensor.MinY != 40 || bboxTensor.MaxY != 75 {
		t.Fatalf("tensor bounding box mismatch: %+v", bboxTensor)
	}
}

// 21. Scale-Invariant Proportional Padding & Centering Unit Tests (Prompt 36)
func TestPadAndCenterProportions(t *testing.T) {
	// Let W_bbox = 20, H_bbox = 30 -> D = 30
	// pad = max(2, floor(0.22 * 30)) = max(2, 6) = 6
	// S = 30 + 2*6 = 42
	// Target occupancy: D / S = 30 / 42 ~= 71.4%
	gray := image.NewGray(image.Rect(0, 0, 100, 100))
	for y := 10; y < 40; y++ {
		for x := 10; x < 30; x++ {
			gray.SetGray(x, y, color.Gray{Y: 200})
		}
	}

	bbox := FindBoundingBox(gray, 10)
	if bbox == nil {
		t.Fatalf("expected bounding box")
	}
	if bbox.Width() != 20 || bbox.Height() != 30 {
		t.Fatalf("expected bbox 20x30, got %dx%d", bbox.Width(), bbox.Height())
	}

	centered := PadAndCenter(gray, bbox)
	if centered.Bounds().Dx() != 42 || centered.Bounds().Dy() != 42 {
		t.Fatalf("expected centered canvas 42x42, got %dx%d", centered.Bounds().Dx(), centered.Bounds().Dy())
	}

	// Verify foreground pixels are centered:
	// offsetX = (42 - 20) / 2 = 11
	// offsetY = (42 - 30) / 2 = 6
	if centered.GrayAt(11, 6).Y != 200 {
		t.Fatalf("top-left of centered foreground not found at (11, 6)")
	}
	if centered.GrayAt(11+19, 6+29).Y != 200 {
		t.Fatalf("bottom-right of centered foreground not found at (30, 35)")
	}
	// Verify padding area is 0
	if centered.GrayAt(0, 0).Y != 0 {
		t.Fatalf("padding margin at (0, 0) should be 0")
	}

	// Test Tensor version
	tensor := GrayImageToTensor(gray)
	centeredTensor := PadAndCenterTensor(tensor, bbox)
	if centeredTensor.Height != 42 || centeredTensor.Width != 42 {
		t.Fatalf("expected centered tensor 42x42, got %dx%d", centeredTensor.Height, centeredTensor.Width)
	}
	if centeredTensor.Get(0, 6, 11) == 0 {
		t.Fatalf("top-left of centered tensor foreground not found at (11, 6)")
	}
}

// 22. Peak Stroke Luminosity & Dynamic Contrast Stretching Unit Tests (Prompt 37)
func TestContrastStretch(t *testing.T) {
	// Case 1: Faint image with L_max = 100 in (30, 240)
	gray := image.NewGray(image.Rect(0, 0, 10, 10))
	gray.SetGray(2, 2, color.Gray{Y: 100}) // L_max
	gray.SetGray(3, 3, color.Gray{Y: 50})
	gray.SetGray(4, 4, color.Gray{Y: 0})

	stretched := ContrastStretch(gray)
	// Scale = 255.0 / 100 = 2.55
	// Pixel 100 -> 255
	if stretched.GrayAt(2, 2).Y != 255 {
		t.Fatalf("expected stretched max 255, got %d", stretched.GrayAt(2, 2).Y)
	}
	// Pixel 50 -> round(50 * 2.55) = 128
	if stretched.GrayAt(3, 3).Y != 128 {
		t.Fatalf("expected scaled pixel 128, got %d", stretched.GrayAt(3, 3).Y)
	}
	if stretched.GrayAt(4, 4).Y != 0 {
		t.Fatalf("expected background 0, got %d", stretched.GrayAt(4, 4).Y)
	}

	// Case 2: High contrast image with L_max >= 240 (e.g. 250) -> should remain unchanged
	highContrast := image.NewGray(image.Rect(0, 0, 5, 5))
	highContrast.SetGray(1, 1, color.Gray{Y: 250})
	highContrast.SetGray(2, 2, color.Gray{Y: 100})
	stretchedHigh := ContrastStretch(highContrast)
	if stretchedHigh.GrayAt(1, 1).Y != 250 || stretchedHigh.GrayAt(2, 2).Y != 100 {
		t.Fatalf("expected high contrast to remain unchanged")
	}

	// Case 3: Very faint image with L_max <= 30 (e.g. 20) -> should remain unchanged
	veryFaint := image.NewGray(image.Rect(0, 0, 5, 5))
	veryFaint.SetGray(1, 1, color.Gray{Y: 20})
	stretchedFaint := ContrastStretch(veryFaint)
	if stretchedFaint.GrayAt(1, 1).Y != 20 {
		t.Fatalf("expected very faint image to remain unchanged")
	}

	// Tensor version test
	tensor := GrayImageToTensor(gray)
	stretchedTensor := ContrastStretchTensor(tensor)
	if math.Abs(float64(stretchedTensor.Get(0, 2, 2)-1.0)) > 1e-4 {
		t.Fatalf("expected tensor max ~1.0, got %f", stretchedTensor.Get(0, 2, 2))
	}
}

// 23. Sub-Pixel Bilinear Interpolation Resampling Unit Tests (Prompt 38)
func TestResizeBilinearInterpolation(t *testing.T) {
	// Create 2x2 image:
	// [10,  20]
	// [30,  40]
	src := image.NewGray(image.Rect(0, 0, 2, 2))
	src.SetGray(0, 0, color.Gray{Y: 10})
	src.SetGray(1, 0, color.Gray{Y: 20})
	src.SetGray(0, 1, color.Gray{Y: 30})
	src.SetGray(1, 1, color.Gray{Y: 40})

	// Resize to 4x4
	dst := ResizeBilinear(src, 4, 4)
	if dst.Bounds().Dx() != 4 || dst.Bounds().Dy() != 4 {
		t.Fatalf("expected 4x4 bounds, got %v", dst.Bounds())
	}

	// Center-most values should smoothly interpolate between 10 and 40
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			val := dst.GrayAt(x, y).Y
			if val < 10 || val > 40 {
				t.Fatalf("interpolated pixel at (%d, %d) out of bounds: %d", x, y, val)
			}
		}
	}

	// Top-left should be closest to 10, bottom-right should be closest to 40
	if dst.GrayAt(0, 0).Y > dst.GrayAt(3, 3).Y {
		t.Fatalf("expected monotonic gradient from top-left to bottom-right")
	}

	// Resize to standard 100x100 grid
	dst100 := ResizeBilinear(src, 100, 100)
	if dst100.Bounds().Dx() != 100 || dst100.Bounds().Dy() != 100 {
		t.Fatalf("expected 100x100 bounds, got %v", dst100.Bounds())
	}

	// Tensor version test
	tensor := GrayImageToTensor(src)
	dstTensor := ResizeBilinearTensor(tensor, 100, 100)
	if dstTensor.Channels != 1 || dstTensor.Height != 100 || dstTensor.Width != 100 {
		t.Fatalf("tensor resize shape mismatch: [%d, %d, %d]", dstTensor.Channels, dstTensor.Height, dstTensor.Width)
	}
}

// 24. Continuous Coordinate Rotation & 2D Translation Unit Tests (Prompt 39)
func TestRotateImageAndShift(t *testing.T) {
	// 1. Test RotateImage with 0 degrees (identity)
	src := image.NewGray(image.Rect(0, 0, 20, 20))
	src.SetGray(10, 10, color.Gray{Y: 255})
	rot0 := RotateImage(src, 0.0)
	if rot0.GrayAt(10, 10).Y != 255 {
		t.Fatalf("0-deg rotation should preserve center pixel")
	}

	// 2. Test RotateImage with small angle (e.g. 15 degrees)
	rot15 := RotateImage(src, 15.0)
	if rot15.Bounds().Dx() != 20 || rot15.Bounds().Dy() != 20 {
		t.Fatalf("rotation bounds mismatch: %v", rot15.Bounds())
	}
	// Center pixel (10, 10) is the pivot, should remain bright
	if rot15.GrayAt(10, 10).Y == 0 {
		t.Fatalf("pivot center pixel should remain non-zero")
	}

	// 3. Test ShiftImage
	shifted := ShiftImage(src, 3, -2)
	// (10, 10) shifted by (+3, -2) becomes (13, 8)
	if shifted.GrayAt(13, 8).Y != 255 {
		t.Fatalf("shifted pixel at (13, 8) not found")
	}
	if shifted.GrayAt(10, 10).Y != 0 {
		t.Fatalf("original location (10, 10) should be 0 after shift")
	}
	// Margin checks
	if shifted.GrayAt(0, 0).Y != 0 || shifted.GrayAt(19, 19).Y != 0 {
		t.Fatalf("margins should be 0")
	}
}

// 25. Slant Shear & Morphological Dilation / Erosion Unit Tests (Prompt 40)
func TestShearMorphologyAndAugmentImage(t *testing.T) {
	// 1. Test ShearImage
	src := image.NewGray(image.Rect(0, 0, 21, 21))
	// Vertical line at x=10
	for y := 0; y < 21; y++ {
		src.SetGray(10, y, color.Gray{Y: 255})
	}
	sheared := ShearImage(src, 0.3)
	// Center row (y=10, cy=10) should stay at x=10
	if sheared.GrayAt(10, 10).Y != 255 {
		t.Fatalf("center row pivot should remain at x=10")
	}
	// Top row (y=0, dy=-10) -> xSrc = x - (-10)*0.3 = x + 3 = 10 -> x = 7
	if sheared.GrayAt(7, 0).Y < 150 {
		t.Fatalf("top row of sheared slant expected near x=7, got %d", sheared.GrayAt(7, 0).Y)
	}
	// Bottom row (y=20, dy=+10) -> xSrc = x - (+10)*0.3 = x - 3 = 10 -> x = 13
	if sheared.GrayAt(13, 20).Y < 150 {
		t.Fatalf("bottom row of sheared slant expected near x=13, got %d", sheared.GrayAt(13, 20).Y)
	}

	// 2. Test MorphDilation (3x3 max filter)
	pointImg := image.NewGray(image.Rect(0, 0, 10, 10))
	pointImg.SetGray(5, 5, color.Gray{Y: 200})
	dilated := MorphDilation(pointImg)
	// 3x3 window around (5,5) should all be 200
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dilated.GrayAt(5+dx, 5+dy).Y != 200 {
				t.Fatalf("dilated pixel at (%d, %d) mismatch: %d", 5+dx, 5+dy, dilated.GrayAt(5+dx, 5+dy).Y)
			}
		}
	}
	if dilated.GrayAt(0, 0).Y != 0 {
		t.Fatalf("unaffected area should be 0")
	}

	// 3. Test MorphErosion (3x3 min filter)
	boxImg := image.NewGray(image.Rect(0, 0, 10, 10))
	for y := 4; y <= 6; y++ {
		for x := 4; x <= 6; x++ {
			boxImg.SetGray(x, y, color.Gray{Y: 200})
		}
	}
	eroded := MorphErosion(boxImg)
	// Only center (5, 5) had all 8 neighbors at 200
	if eroded.GrayAt(5, 5).Y != 200 {
		t.Fatalf("eroded center pixel should survive: %d", eroded.GrayAt(5, 5).Y)
	}
	// Outer border pixels should be eroded to 0
	if eroded.GrayAt(4, 4).Y != 0 || eroded.GrayAt(6, 6).Y != 0 {
		t.Fatalf("outer border should be eroded to 0")
	}

	// 4. Test AugmentImage returns 15 valid variants
	testImg := image.NewGray(image.Rect(0, 0, 32, 32))
	for i := 10; i < 22; i++ {
		testImg.SetGray(i, i, color.Gray{Y: 255})
	}
	variants := AugmentImage(testImg)
	if len(variants) != 15 {
		t.Fatalf("expected 15 augmented variants, got %d", len(variants))
	}
	for idx, v := range variants {
		if v == nil {
			t.Fatalf("variant %d is nil", idx)
		}
		if v.Bounds().Dx() != 32 || v.Bounds().Dy() != 32 {
			t.Fatalf("variant %d bounds mismatch: %v", idx, v.Bounds())
		}
	}
}

// 26. DiagonalNet Full Model Forward & Analytical Backward Unit Tests (Prompt 41)
func TestDiagonalNetModelForwardBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	model := NewDiagonalNetModel(3, rng)
	model.SetTraining(false)

	// Create sample 1-channel 100x100 tensor
	input := NewTensor(1, 100, 100)
	for y := 30; y < 70; y++ {
		for x := 30; x < 70; x++ {
			input.Set(0, y, x, 0.8)
		}
	}

	// Forward pass
	logits := model.Forward(input)
	if len(logits) != 3 {
		t.Fatalf("expected 3 logits for 3 classes, got %d", len(logits))
	}
	probs := Softmax(logits)
	var sumP float32
	for _, p := range probs {
		sumP += p
	}
	if math.Abs(float64(sumP-1.0)) > 1e-4 {
		t.Fatalf("expected Softmax probabilities to sum to 1.0, got %f", sumP)
	}

	// ForwardBackward pass
	model.SetTraining(true)
	model.ZeroGrad()
	loss, probsBW := model.ForwardBackward(input, 1)
	if loss <= 0 {
		t.Fatalf("expected positive cross entropy loss, got %f", loss)
	}
	if len(probsBW) != 3 {
		t.Fatalf("expected 3 probabilities, got %d", len(probsBW))
	}

	// Verify gradients were accumulated in parameters
	for idx, p := range model.Parameters() {
		hasGrad := false
		for _, g := range p.Grad {
			if g != 0 {
				hasGrad = true
				break
			}
		}
		if !hasGrad {
			t.Fatalf("expected non-zero analytical gradients in parameter %d", idx)
		}
	}
}

// 27. BatchTrainer Data-Parallel Worker Replicas & Master Reduction Unit Tests (Prompts 41 & 42)
func TestBatchTrainerDataParallelTraining(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	master := NewDiagonalNetModel(2, rng)

	optConfig := DefaultAdamConfig()
	optConfig.LearningRate = 0.01
	opt := NewAdamOptimizer(master.Parameters(), optConfig)

	numWorkers := 4
	trainer := NewBatchTrainer(master, opt, numWorkers)

	if len(trainer.Workers) != numWorkers {
		t.Fatalf("expected %d workers, got %d", numWorkers, len(trainer.Workers))
	}

	// Create a synthetic batch of 8 samples (4 of class 0, 4 of class 1)
	batch := make([]Sample, 8)
	for i := 0; i < 8; i++ {
		tensor := NewTensor(1, 100, 100)
		target := i % 2
		// Distinct spatial signatures for each class
		if target == 0 {
			for y := 20; y < 40; y++ {
				for x := 20; x < 80; x++ {
					tensor.Set(0, y, x, 0.9)
				}
			}
		} else {
			for y := 20; y < 80; y++ {
				for x := 40; x < 60; x++ {
					tensor.Set(0, y, x, 0.9)
				}
			}
		}
		batch[i] = Sample{Input: tensor, TargetClass: target}
	}

	// Train 5 batches and track loss
	initialLoss, _ := trainer.TrainBatch(batch)
	if initialLoss <= 0 {
		t.Fatalf("expected positive initial loss, got %f", initialLoss)
	}

	for epoch := 0; epoch < 4; epoch++ {
		trainer.TrainBatch(batch)
	}

	// Evaluate on dataset
	valLoss, valAcc := trainer.Evaluate(batch)
	// Cross-entropy is non-negative, but this 8-sample toy batch is separable enough that the
	// network drives it to ~0 within 5 steps, at which point -log(p) lands on float noise a
	// hair either side of zero. Only a genuinely negative loss indicates a broken reduction.
	if valLoss < -1e-6 {
		t.Fatalf("expected non-negative validation loss, got %f", valLoss)
	}
	if valAcc < 0 || valAcc > 1.0 {
		t.Fatalf("invalid accuracy: %f", valAcc)
	}
}

// 28. Best-Model Validation Accuracy Checkpointing & Weight Restoration Unit Tests (Prompt 43)
func TestModelCheckpointBestAccuracyAndRestoration(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	model := NewDiagonalNetModel(2, rng)
	cp := NewModelCheckpoint()

	if cp.BestValAcc != -1.0 || cp.BestEpoch != -1 {
		t.Fatalf("initial checkpoint state mismatch: acc=%f, epoch=%d", cp.BestValAcc, cp.BestEpoch)
	}

	// Epoch 1: valAcc = 0.60
	model.FC.Biases.Data[0] = 1.0
	updated1 := cp.Update(model, 1, 0.60)
	if !updated1 || cp.BestEpoch != 1 || math.Abs(cp.BestValAcc-0.60) > 1e-6 {
		t.Fatalf("epoch 1 checkpoint update failed")
	}

	// Epoch 2: valAcc = 0.85 (Better)
	model.FC.Biases.Data[0] = 2.5
	updated2 := cp.Update(model, 2, 0.85)
	if !updated2 || cp.BestEpoch != 2 || math.Abs(cp.BestValAcc-0.85) > 1e-6 {
		t.Fatalf("epoch 2 checkpoint update failed")
	}

	// Epoch 3: valAcc = 0.70 (Worse / Overfitting)
	model.FC.Biases.Data[0] = -5.0
	updated3 := cp.Update(model, 3, 0.70)
	if updated3 {
		t.Fatalf("epoch 3 should not have updated checkpoint with worse accuracy")
	}
	if cp.BestEpoch != 2 {
		t.Fatalf("checkpoint best epoch should remain 2, got %d", cp.BestEpoch)
	}

	// Restore best weights
	cp.RestoreBest(model)
	if math.Abs(float64(model.FC.Biases.Data[0]-2.5)) > 1e-6 {
		t.Fatalf("restored weights mismatch: expected 2.5, got %f", model.FC.Biases.Data[0])
	}
}

// 29. Comprehensive Multi-Class Classification Evaluation Metrics Unit Tests (Prompt 44)
func TestMultiClassEvaluationMetrics(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	model := NewDiagonalNetModel(3, rng)

	classes := []string{"circle", "square", "triangle"}
	samples := make([]Sample, 12)

	for i := 0; i < 12; i++ {
		tensor := NewTensor(1, 100, 100)
		target := i % 3
		// Fill synthetic feature mark
		tensor.Set(0, 50, 50, float32(target+1)*0.3)
		samples[i] = Sample{Input: tensor, TargetClass: target}
	}

	report := ComputeEvaluationMetrics(model, samples, classes)

	if report.NumClasses != 3 {
		t.Fatalf("expected 3 classes, got %d", report.NumClasses)
	}
	if report.TotalSamples != 12 {
		t.Fatalf("expected 12 total samples, got %d", report.TotalSamples)
	}
	if report.Accuracy < 0 || report.Accuracy > 1.0 {
		t.Fatalf("invalid accuracy: %f", report.Accuracy)
	}
	if report.MacroF1 < 0 || report.MacroF1 > 1.0 {
		t.Fatalf("invalid Macro-F1: %f", report.MacroF1)
	}

	// Verify per-class formulas
	for c := 0; c < 3; c++ {
		cm := report.ClassMetrics[c]
		if cm.ClassName != classes[c] {
			t.Fatalf("class name mismatch at %d: %s", c, cm.ClassName)
		}
		if cm.TP+cm.FN != cm.Support {
			t.Fatalf("support mismatch for class %s: TP=%d, FN=%d, Support=%d", cm.ClassName, cm.TP, cm.FN, cm.Support)
		}
		if cm.TP+cm.FP > 0 {
			expectedPrec := float64(cm.TP) / float64(cm.TP+cm.FP)
			if math.Abs(cm.Precision-expectedPrec) > 1e-6 {
				t.Fatalf("precision formula mismatch: got %f, expected %f", cm.Precision, expectedPrec)
			}
		}
		if cm.TP+cm.FN > 0 {
			expectedRec := float64(cm.TP) / float64(cm.TP+cm.FN)
			if math.Abs(cm.Recall-expectedRec) > 1e-6 {
				t.Fatalf("recall formula mismatch: got %f, expected %f", cm.Recall, expectedRec)
			}
		}
	}

	// Print test table to stdout
	PrintEvaluationReport(report)
}

// 33. Embedded HTML5 Web Application Content Verification (Prompt 46)
func TestEmbeddedWebAppHTML(t *testing.T) {
	if len(webAppHTML) < 500 {
		t.Fatalf("expected substantial embedded webAppHTML, got %d bytes", len(webAppHTML))
	}
	requiredSubstrings := []string{
		`<!DOCTYPE html>`,
		`<canvas id="paintCanvas" width="400" height="400">`,
		`btnClear`,
		`btnPredict`,
		`topClass`,
		`topConfidence`,
		`latencyBadge`,
		`classList`,
		`/api/predict`,
		`/api/info`,
	}
	for _, sub := range requiredSubstrings {
		if !strings.Contains(webAppHTML, sub) {
			t.Fatalf("webAppHTML missing essential substring: %q", sub)
		}
	}
}

// 34. Web Image Preprocessing Pipeline Verification (Prompt 47)
func TestPreprocessWebImagePipeline(t *testing.T) {
	// 1. Valid Drawing (200x200 square drawn in center of 400x400 canvas)
	img := image.NewRGBA(image.Rect(0, 0, 400, 400))
	for y := 100; y < 300; y++ {
		for x := 100; x < 300; x++ {
			img.Set(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}

	tensor, isBlank := PreprocessWebImage(img)
	if isBlank {
		t.Fatalf("expected valid non-blank image, got isBlank=true")
	}
	if tensor.Channels != 1 || tensor.Height != InputSize || tensor.Width != InputSize {
		t.Fatalf("expected preprocessed tensor [1, %d, %d], got [%d, %d, %d]",
			InputSize, InputSize, tensor.Channels, tensor.Height, tensor.Width)
	}

	// 2. Blank Image (all black)
	blankImg := image.NewRGBA(image.Rect(0, 0, 400, 400))
	blankTensor, isBlank := PreprocessWebImage(blankImg)
	if !isBlank {
		t.Fatalf("expected isBlank=true for all-black canvas")
	}
	if blankTensor.Channels != 1 || blankTensor.Height != InputSize || blankTensor.Width != InputSize {
		t.Fatalf("expected blank tensor [1, %d, %d], got [%d, %d, %d]",
			InputSize, InputSize, blankTensor.Channels, blankTensor.Height, blankTensor.Width)
	}
}

// 35. HTTP Inference Server Routes, /api/info & /api/predict (Prompt 47 & 48)
func TestInferenceServerHTTPRoutesAndPredict(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	classes := []string{"circle", "square", "triangle"}
	model := NewDiagonalNetModel(len(classes), rng)
	server := NewInferenceServer(model, classes, 8081)

	// 1. Test GET / (HTML Application)
	reqRoot := httptest.NewRequest(http.MethodGet, "/", nil)
	recRoot := httptest.NewRecorder()
	server.ServeHTTP(recRoot, reqRoot)

	if recRoot.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for /, got %d", recRoot.Code)
	}
	if !strings.Contains(recRoot.Header().Get("Content-Type"), "text/html") {
		t.Fatalf("expected text/html Content-Type, got %s", recRoot.Header().Get("Content-Type"))
	}

	// 2. Test GET /api/info (Metadata)
	reqInfo := httptest.NewRequest(http.MethodGet, "/api/info", nil)
	recInfo := httptest.NewRecorder()
	server.ServeHTTP(recInfo, reqInfo)

	if recInfo.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for /api/info, got %d", recInfo.Code)
	}
	var infoMap map[string]interface{}
	if err := json.NewDecoder(recInfo.Body).Decode(&infoMap); err != nil {
		t.Fatalf("failed to decode /api/info JSON: %v", err)
	}
	if int(infoMap["num_classes"].(float64)) != 3 {
		t.Fatalf("expected 3 num_classes in /api/info, got %v", infoMap["num_classes"])
	}

	// 3. Test POST /api/predict (Synthetic PNG Base64 Drawing)
	drawImg := image.NewRGBA(image.Rect(0, 0, 400, 400))
	for y := 150; y < 250; y++ {
		for x := 150; x < 250; x++ {
			drawImg.Set(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, drawImg); err != nil {
		t.Fatalf("failed to encode synthetic PNG: %v", err)
	}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBuf.Bytes())

	payload, _ := json.Marshal(PredictRequest{Image: dataURL})
	reqPredict := httptest.NewRequest(http.MethodPost, "/api/predict", bytes.NewReader(payload))
	recPredict := httptest.NewRecorder()
	server.ServeHTTP(recPredict, reqPredict)

	if recPredict.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for /api/predict, got %d: %s", recPredict.Code, recPredict.Body.String())
	}

	var predResp PredictResponse
	if err := json.NewDecoder(recPredict.Body).Decode(&predResp); err != nil {
		t.Fatalf("failed to decode /api/predict JSON: %v", err)
	}

	if predResp.PredictedClass == "" {
		t.Fatalf("expected non-empty PredictedClass")
	}
	if len(predResp.Confidences) != 3 {
		t.Fatalf("expected 3 class confidences, got %d", len(predResp.Confidences))
	}
	if predResp.LatencyMs < 0 {
		t.Fatalf("expected positive latency, got %f", predResp.LatencyMs)
	}
	if predResp.IsBlank {
		t.Fatalf("expected isBlank=false for drawn square")
	}
}

func TestMaxPool2DLayerForwardAndBackward(t *testing.T) {
	// Create a 1 x 4 x 4 input tensor
	input := NewTensor(1, 4, 4)
	input.Data = []float32{
		1, 3, 2, 4,
		5, 2, 8, 1,
		4, 7, 3, 6,
		2, 1, 9, 5,
	}

	pool := NewMaxPool2DLayer(2)
	out := pool.Forward(input)

	if out.Channels != 1 || out.Height != 2 || out.Width != 2 {
		t.Fatalf("expected shape [1 x 2 x 2], got [%d x %d x %d]", out.Channels, out.Height, out.Width)
	}

	// Expected max values in 2x2 windows:
	// Top-left: max(1, 3, 5, 2) = 5 (idx 4)
	// Top-right: max(2, 4, 8, 1) = 8 (idx 6)
	// Bottom-left: max(4, 7, 2, 1) = 7 (idx 9)
	// Bottom-right: max(3, 6, 9, 5) = 9 (idx 14)
	expectedOut := []float32{5, 8, 7, 9}
	expectedArgMax := []int{4, 6, 9, 14}

	for i := range expectedOut {
		if math.Abs(float64(out.Data[i]-expectedOut[i])) > 1e-5 {
			t.Errorf("out[%d]: expected %f, got %f", i, expectedOut[i], out.Data[i])
		}
		if pool.ArgMax[i] != expectedArgMax[i] {
			t.Errorf("argMax[%d]: expected %d, got %d", i, expectedArgMax[i], pool.ArgMax[i])
		}
	}

	// Test Backward gradient routing
	gradOut := NewTensor(1, 2, 2)
	gradOut.Data = []float32{1.5, 2.5, 3.5, 4.5}

	gradIn := pool.Backward(gradOut)
	if gradIn.Channels != 1 || gradIn.Height != 4 || gradIn.Width != 4 {
		t.Fatalf("expected gradIn shape [1 x 4 x 4], got [%d x %d x %d]", gradIn.Channels, gradIn.Height, gradIn.Width)
	}

	expectedGradIn := make([]float32, 16)
	expectedGradIn[4] = 1.5
	expectedGradIn[6] = 2.5
	expectedGradIn[9] = 3.5
	expectedGradIn[14] = 4.5

	for i := range expectedGradIn {
		if math.Abs(float64(gradIn.Data[i]-expectedGradIn[i])) > 1e-5 {
			t.Errorf("gradIn[%d]: expected %f, got %f", i, expectedGradIn[i], gradIn.Data[i])
		}
	}
}

// 36. HTTP Inference Server Deep Stats Integration Test
func TestInferenceServerDeepStats(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	classes := []string{"0", "1", "2", "3", "4"}
	model := NewDiagonalNetModel(len(classes), rng)
	server := NewInferenceServer(model, classes, 8081)

	// Draw a circle in the center of 400x400 canvas
	drawImg := image.NewRGBA(image.Rect(0, 0, 400, 400))
	for y := 0; y < 400; y++ {
		for x := 0; x < 400; x++ {
			dx := float64(x - 200)
			dy := float64(y - 200)
			dist := math.Sqrt(dx*dx + dy*dy)
			if dist >= 60 && dist <= 80 {
				drawImg.Set(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}

	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, drawImg); err != nil {
		t.Fatalf("failed to encode synthetic PNG: %v", err)
	}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBuf.Bytes())

	payload, _ := json.Marshal(PredictRequest{Image: dataURL})
	reqPredict := httptest.NewRequest(http.MethodPost, "/api/predict", bytes.NewReader(payload))
	recPredict := httptest.NewRecorder()
	server.ServeHTTP(recPredict, reqPredict)

	if recPredict.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 for /api/predict, got %d: %s", recPredict.Code, recPredict.Body.String())
	}

	var predResp PredictResponse
	if err := json.NewDecoder(recPredict.Body).Decode(&predResp); err != nil {
		t.Fatalf("failed to decode /api/predict JSON: %v", err)
	}

	if predResp.Stats == nil {
		t.Fatalf("expected non-nil Stats in PredictResponse")
	}

	stats := predResp.Stats

	// 1. Information Theory
	if stats.EntropyBits < 0 || stats.EntropyBits > stats.MaxEntropyBits+1e-3 {
		t.Errorf("entropy out of range: %f (max: %f)", stats.EntropyBits, stats.MaxEntropyBits)
	}
	if stats.Perplexity < 1.0 || stats.Perplexity > float64(len(classes))+1e-3 {
		t.Errorf("perplexity out of range: %f", stats.Perplexity)
	}
	if len(stats.RawLogits) != len(classes) {
		t.Errorf("expected %d raw logits, got %d", len(classes), len(stats.RawLogits))
	}

	// 2. Geometry
	if stats.Geometry.BBoxWidth <= 0 || stats.Geometry.BBoxHeight <= 0 {
		t.Errorf("invalid bbox dimensions: %dx%d", stats.Geometry.BBoxWidth, stats.Geometry.BBoxHeight)
	}
	if stats.Geometry.ForegroundPixels <= 0 {
		t.Errorf("expected positive foreground pixels, got %d", stats.Geometry.ForegroundPixels)
	}
	if len(stats.Geometry.Resampled28x28) != InputSize*InputSize {
		t.Errorf("expected %d resampled pixels, got %d", InputSize*InputSize, len(stats.Geometry.Resampled28x28))
	}

	// 3. 13-Channel Manifold
	if len(stats.Manifold.ChannelNames) != 13 {
		t.Errorf("expected 13 channel names, got %d", len(stats.Manifold.ChannelNames))
	}
	if len(stats.Manifold.ChannelGrids) != 13 {
		t.Errorf("expected 13 channel grids, got %d", len(stats.Manifold.ChannelGrids))
	}
	for i, grid := range stats.Manifold.ChannelGrids {
		if len(grid) != InputSize*InputSize {
			t.Errorf("channel %d grid length %d != %d", i, len(grid), InputSize*InputSize)
		}
	}

	// 4. Layer Activations
	if len(stats.Layers.FC1HiddenVector) != 128 {
		t.Errorf("expected 128 FC1 hidden values, got %d", len(stats.Layers.FC1HiddenVector))
	}

	// 5. Performance Timing
	if stats.Timing.TotalUs <= 0 {
		t.Errorf("expected positive total latency, got %f", stats.Timing.TotalUs)
	}
	if stats.Timing.ThroughputFps <= 0 {
		t.Errorf("expected positive throughput FPS, got %f", stats.Timing.ThroughputFps)
	}

	// 6. Runtime Health
	if stats.Runtime.CPUCores <= 0 {
		t.Errorf("expected positive CPU cores, got %d", stats.Runtime.CPUCores)
	}
}
















