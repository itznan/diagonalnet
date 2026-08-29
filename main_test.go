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



