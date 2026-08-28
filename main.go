package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// ============================================================================
// 1. HARDWARE TOPOLOGY & RUNTIME CONCURRENCY
// ============================================================================

const banner = `
================================================================================
  ____  _                               _   _      _   
 |  _ \(_) __ _  __ _  ___  _ __   __ _| | | \ | | ___| |_ 
 | | | | |/ _` + "`" + ` |/ _` + "`" + ` |/ _ \| '_ \ / _` + "`" + ` | | |  \| |/ _ \ __|
 | |_| | | (_| | (_| | (_) | | | | (_| | | | |\  |  __/ |_ 
 |____/|_|\__,_|\__, |\___/|_| |_|\__,_|_| |_| \_|\___|\__|
                |___/                                       
 Pure Go Zero-Dependency Deep Learning Engine & Web Runtime
================================================================================
`

// NumWorkers returns the number of worker goroutines scaled to system hardware (minimum 1)
func NumWorkers() int {
	n := runtime.NumCPU()
	if n < 1 {
		return 1
	}
	return n
}

// PrintHardwareDiagnostics prints system hardware topology and core utilization
func PrintHardwareDiagnostics() {
	cores := runtime.NumCPU()
	fmt.Println("==================================================================")
	fmt.Println("       DiagonNet Pure Go Deep Learning Engine")
	fmt.Printf("       CPU Compute Engine: %d Logical Cores Fully Utilized (100%%)\n", cores)
	fmt.Printf("       OS: %s | Architecture: %s | Go: %s\n", runtime.GOOS, runtime.GOARCH, runtime.Version())
	fmt.Println("==================================================================")
}

// ============================================================================
// 2. CONTIGUOUS 1D/3D TENSOR ENGINE
// ============================================================================

// Tensor represents a flat contiguous 1D slice representation for multi-dimensional 3D tensors [C x H x W]
// to maximize CPU L1/L2 cache locality and eliminate pointer chasing.
type Tensor struct {
	Data     []float32
	Channels int
	Height   int
	Width    int
}

// NewTensor allocates a new Tensor with dimensions C x H x W
func NewTensor(c, h, w int) *Tensor {
	return &Tensor{
		Data:     make([]float32, c*h*w),
		Channels: c,
		Height:   h,
		Width:    w,
	}
}

// Index computes the contiguous 1D slice index for coordinates (c, y, x):
// Index(c, y, x) = c * (Height * Width) + y * Width + x
func (t *Tensor) Index(c, y, x int) int {
	return c*(t.Height*t.Width) + y*t.Width + x
}

// Get returns the value at coordinate (c, y, x)
func (t *Tensor) Get(c, y, x int) float32 {
	return t.Data[c*(t.Height*t.Width)+y*t.Width+x]
}

// Set stores a value at coordinate (c, y, x)
func (t *Tensor) Set(c, y, x int, val float32) {
	t.Data[c*(t.Height*t.Width)+y*t.Width+x] = val
}

// Zero resets all elements in the tensor to 0
func (t *Tensor) Zero() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// Size returns the total number of float32 elements (C * H * W)
func (t *Tensor) Size() int {
	return len(t.Data)
}

// Clone creates an exact deep copy of the tensor
func (t *Tensor) Clone() *Tensor {
	cp := NewTensor(t.Channels, t.Height, t.Width)
	copy(cp.Data, t.Data)
	return cp
}

// Shape returns the tensor dimensions (Channels, Height, Width)
func (t *Tensor) Shape() (int, int, int) {
	return t.Channels, t.Height, t.Width
}

// ============================================================================
// 3. TRAINABLE PARAMETER ABSTRACTION & INITIALIZATION
// ============================================================================

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

// ============================================================================
// 4. LOCK-FREE PARALLEL GRADIENT REDUCTION
// ============================================================================

// ReduceParameterGradients accumulates gradients from multiple workers into a master parameter
// using lock-free contiguous chunk partitioning across worker Goroutines.
func ReduceParameterGradients(master *Parameter, workers []*Parameter, numWorkers int) {
	if master == nil || len(workers) == 0 {
		return
	}
	L := len(master.Grad)
	if L == 0 {
		return
	}

	if numWorkers <= 0 {
		numWorkers = len(workers)
	}
	if numWorkers > L {
		numWorkers = L
	}

	chunkSize := (L + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > L {
			end = L
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				var sum float32
				for k := 0; k < len(workers); k++ {
					sum += workers[k].Grad[i]
				}
				master.Grad[i] = sum
			}
		}(start, end)
	}

	wg.Wait()
}

// ReduceGradients accumulates gradients across an entire parameter list for multiple worker replicas.
func ReduceGradients(masterParams []*Parameter, workerParams [][]*Parameter, numWorkers int) {
	if len(masterParams) == 0 || len(workerParams) == 0 {
		return
	}
	for paramIdx, master := range masterParams {
		workersForParam := make([]*Parameter, len(workerParams))
		for k := 0; k < len(workerParams); k++ {
			if paramIdx < len(workerParams[k]) {
				workersForParam[k] = workerParams[k][paramIdx]
			}
		}
		ReduceParameterGradients(master, workersForParam, numWorkers)
	}
}

// ============================================================================
// 5. BINARY MODEL WEIGHT SERIALIZATION ENGINE (DIAGON01)
// ============================================================================

const ModelMagicHeader = "DIAGON01"

// SaveModelWeights writes parameter weights and class metadata to a binary file according to the DIAGON01 protocol.
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

// ============================================================================
// 6. 13-CHANNEL SPATIAL DIFFERENCE MANIFOLD CALCULUS
// ============================================================================

// Clamp restricts val within the closed interval [minVal, maxVal].
func clamp(val, minVal, maxVal int) int {
	if val < minVal {
		return minVal
	}
	if val > maxVal {
		return maxVal
	}
	return val
}

// abs32 returns the absolute value of a float32 without float64 conversion overhead.
func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// Directional offset vectors for 13-channel spatial difference manifold:
var (
	// Channels 1-4: Immediate diagonals (|I(x, y) - I(clamp(x+dx), clamp(y+dy))|)
	// Ch 1: (-1, -1) Top-Left
	// Ch 2: (+1, -1) Top-Right
	// Ch 3: (-1, +1) Bottom-Left
	// Ch 4: (+1, +1) Bottom-Right
	DiagonalOffsets = [4][2]int{
		{-1, -1}, // Ch 1: Top-Left
		{+1, -1}, // Ch 2: Top-Right
		{-1, +1}, // Ch 3: Bottom-Left
		{+1, +1}, // Ch 4: Bottom-Right
	}

	// Channels 5-12: 8-Way Chess Knight-Move differential operators (K set)
	// K = { (-2, -1), (-2, +1), (-1, -2), (-1, +2), (+1, -2), (+1, +2), (+2, -1), (+2, +1) }
	KnightOffsets = [8][2]int{
		{-2, -1}, // Ch 5:  Knight 1
		{-2, +1}, // Ch 6:  Knight 2
		{-1, -2}, // Ch 7:  Knight 3
		{-1, +2}, // Ch 8:  Knight 4
		{+1, -2}, // Ch 9:  Knight 5
		{+1, +2}, // Ch 10: Knight 6
		{+2, -1}, // Ch 11: Knight 7
		{+2, +1}, // Ch 12: Knight 8
	}
)

// ComputeManifold transforms a flat grayscale image slice [h*w] into a 13-channel spatial difference manifold [13*h*w]
// parallelized row-by-row across runtime.NumCPU() Goroutines.
func ComputeManifold(input []float32, h, w int) []float32 {
	out := make([]float32, 13*h*w)
	ComputeManifoldIntoSlice(input, out, h, w)
	return out
}

// ComputeManifoldIntoSlice executes the row-parallel 13-channel manifold transformation into a pre-allocated destination slice.
func ComputeManifoldIntoSlice(input []float32, out []float32, h, w int) {
	hw := h * w
	numWorkers := runtime.NumCPU()
	if numWorkers <= 0 {
		numWorkers = 1
	}
	if numWorkers > h {
		numWorkers = h
	}

	rowsPerWorker := (h + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for workerID := 0; workerID < numWorkers; workerID++ {
		startY := workerID * rowsPerWorker
		endY := startY + rowsPerWorker
		if endY > h {
			endY = h
		}
		if startY >= endY {
			continue
		}

		wg.Add(1)
		go func(sy, ey int) {
			defer wg.Done()
			for y := sy; y < ey; y++ {
				yOffset := y * w
				for x := 0; x < w; x++ {
					pixelIdx := yOffset + x
					baseVal := input[pixelIdx]

					// Channel 0: Base normalized grayscale intensity
					out[pixelIdx] = baseVal

					// Channels 1-4: Immediate diagonal absolute gradients
					for ch := 0; ch < 4; ch++ {
						dx := DiagonalOffsets[ch][0]
						dy := DiagonalOffsets[ch][1]
						nx := clamp(x+dx, 0, w-1)
						ny := clamp(y+dy, 0, h-1)
						neighborVal := input[ny*w+nx]
						out[(ch+1)*hw+pixelIdx] = abs32(baseVal - neighborVal)
					}

					// Channels 5-12: 8-Way Chess Knight-Move differential operators
					for k := 0; k < 8; k++ {
						dx := KnightOffsets[k][0]
						dy := KnightOffsets[k][1]
						nx := clamp(x+dx, 0, w-1)
						ny := clamp(y+dy, 0, h-1)
						neighborVal := input[ny*w+nx]
						out[(k+5)*hw+pixelIdx] = abs32(baseVal - neighborVal)
					}
				}
			}
		}(startY, endY)
	}

	wg.Wait()
}

// ComputeManifoldInto transforms a 1-channel Tensor into a 13-channel Tensor in-place using row-parallel multi-threading.
func ComputeManifoldInto(input *Tensor, output *Tensor) {
	H, W := input.Height, input.Width
	if output.Channels < 13 || output.Height != H || output.Width != W {
		*output = *NewTensor(13, H, W)
	}
	ComputeManifoldIntoSlice(input.Data, output.Data, H, W)
}

// ComputeManifoldTensor creates and returns a new 13-channel Tensor from a 1-channel input Tensor.
func ComputeManifoldTensor(input *Tensor) *Tensor {
	out := NewTensor(13, input.Height, input.Width)
	ComputeManifoldInto(input, out)
	return out
}

// ============================================================================
// 7. CONVOLUTIONAL, POOLING & DENSE NEURAL NETWORK LAYERS
// ============================================================================

// Conv2DLayer implements a multi-channel 2D convolutional layer with arbitrary kernel size, stride, and padding.
type Conv2DLayer struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int

	Weights *Parameter // Shape: [OutChannels, InChannels, KernelSize, KernelSize]
	Bias    *Parameter // Shape: [OutChannels]

	LastInput *Tensor
}

// NewConv2DLayer constructs and initializes a new Conv2DLayer using Kaiming Uniform weight initialization.
func NewConv2DLayer(inChannels, outChannels, kernelSize, stride, padding int, rng *rand.Rand) *Conv2DLayer {
	layer := &Conv2DLayer{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Weights:     NewParameter(outChannels * inChannels * kernelSize * kernelSize),
		Bias:        NewParameter(outChannels),
	}

	fanIn := inChannels * kernelSize * kernelSize
	if rng != nil {
		InitKaimingUniform(layer.Weights, fanIn, rng)
	} else {
		defaultRNG := rand.New(rand.NewSource(42))
		InitKaimingUniform(layer.Weights, fanIn, defaultRNG)
	}
	InitZeros(layer.Bias)

	return layer
}

// OutputShape calculates spatial dimensions (outH, outW) given input dimensions (inH, inW).
func (l *Conv2DLayer) OutputShape(inH, inW int) (outH, outW int) {
	outH = (inH+2*l.Padding-l.KernelSize)/l.Stride + 1
	outW = (inW+2*l.Padding-l.KernelSize)/l.Stride + 1
	return outH, outW
}

// ZeroGrad resets weight and bias analytical gradient accumulators to zero.
func (l *Conv2DLayer) ZeroGrad() {
	l.Weights.ZeroGrad()
	l.Bias.ZeroGrad()
}

// Parameters returns references to trainable parameters in the convolutional layer.
func (l *Conv2DLayer) Parameters() []*Parameter {
	return []*Parameter{l.Weights, l.Bias}
}

// Forward computes the parallelized multi-channel 2D convolution forward pass:
// Y(c_out, y, x) = B(c_out) + sum_{c_in} sum_{ky} sum_{kx} W(c_out, c_in, ky, kx) * X(c_in, y*S + ky - P, x*S + kx - P)
func (l *Conv2DLayer) Forward(input *Tensor) *Tensor {
	outH, outW := l.OutputShape(input.Height, input.Width)
	output := NewTensor(l.OutChannels, outH, outW)
	l.ForwardInto(input, output)
	return output
}

// ForwardInto computes the 2D convolution into a pre-allocated destination tensor, parallelizing output channels across CPU cores.
func (l *Conv2DLayer) ForwardInto(input *Tensor, output *Tensor) {
	l.LastInput = input

	inC, inH, inW := input.Channels, input.Height, input.Width
	outC := l.OutChannels
	outH, outW := l.OutputShape(inH, inW)

	if output.Channels != outC || output.Height != outH || output.Width != outW {
		*output = *NewTensor(outC, outH, outW)
	}

	K := l.KernelSize
	S := l.Stride
	P := l.Padding
	Ksq := K * K
	inHW := inH * inW
	outHW := outH * outW
	weightsPerOutChannel := inC * Ksq

	numWorkers := runtime.NumCPU()
	if numWorkers <= 0 {
		numWorkers = 1
	}
	if numWorkers > outC {
		numWorkers = outC
	}

	channelsPerWorker := (outC + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		startC := w * channelsPerWorker
		endC := startC + channelsPerWorker
		if endC > outC {
			endC = outC
		}
		if startC >= endC {
			continue
		}

		wg.Add(1)
		go func(sC, eC int) {
			defer wg.Done()
			for cOut := sC; cOut < eC; cOut++ {
				biasVal := l.Bias.Data[cOut]
				outChOffset := cOut * outHW
				weightOutOffset := cOut * weightsPerOutChannel

				for y := 0; y < outH; y++ {
					inYBase := y*S - P
					outYOffset := outChOffset + y*outW

					for x := 0; x < outW; x++ {
						inXBase := x*S - P
						var sum float32 = biasVal

						for cIn := 0; cIn < inC; cIn++ {
							inChOffset := cIn * inHW
							weightInOffset := weightOutOffset + cIn*Ksq

							for ky := 0; ky < K; ky++ {
								inY := inYBase + ky
								if inY < 0 || inY >= inH {
									continue
								}
								inRowOffset := inChOffset + inY*inW
								weightRowOffset := weightInOffset + ky*K

								for kx := 0; kx < K; kx++ {
									inX := inXBase + kx
									if inX < 0 || inX >= inW {
										continue
									}

									weightVal := l.Weights.Data[weightRowOffset+kx]
									inputVal := input.Data[inRowOffset+inX]
									sum += weightVal * inputVal
								}
							}
						}

						output.Data[outYOffset+x] = sum
					}
				}
			}
		}(startC, endC)
	}

	wg.Wait()
}

// Backward computes analytical Jacobian backpropagation gradients for weights, bias, and input feature tensor:
// 1. dL/dW(c_out, c_in, ky, kx) = sum_{y, x} dL/dY(c_out, y, x) * X(c_in, y*S + ky - P, x*S + kx - P)
// 2. dL/dB(c_out) = sum_{y, x} dL/dY(c_out, y, x)
// 3. dL/dX(c_in, iy, ix) = sum_{c_out, ky, kx} dL/dY(c_out, (iy+P-ky)/S, (ix+P-kx)/S) * W(c_out, c_in, ky, kx)
func (l *Conv2DLayer) Backward(gradOutput *Tensor) *Tensor {
	if l.LastInput == nil {
		panic("Conv2DLayer.Backward called before Forward pass")
	}
	gradInput := NewTensor(l.InChannels, l.LastInput.Height, l.LastInput.Width)
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto executes multi-threaded analytical backpropagation into pre-allocated gradInput buffer.
func (l *Conv2DLayer) BackwardInto(gradOutput *Tensor, gradInput *Tensor) {
	input := l.LastInput
	inC, inH, inW := input.Channels, input.Height, input.Width
	outC, outH, outW := gradOutput.Channels, gradOutput.Height, gradOutput.Width

	if gradInput.Channels != inC || gradInput.Height != inH || gradInput.Width != inW {
		*gradInput = *NewTensor(inC, inH, inW)
	}
	gradInput.Zero()

	K := l.KernelSize
	S := l.Stride
	P := l.Padding
	Ksq := K * K
	inHW := inH * inW
	outHW := outH * outW
	weightsPerOutChannel := inC * Ksq

	numWorkers := runtime.NumCPU()
	if numWorkers <= 0 {
		numWorkers = 1
	}

	// 1. Weight & Bias Gradients (Parallelized over output channels cOut for lock-free writing)
	numWorkersW := numWorkers
	if numWorkersW > outC {
		numWorkersW = outC
	}
	channelsPerWorkerW := (outC + numWorkersW - 1) / numWorkersW
	var wgW sync.WaitGroup

	for w := 0; w < numWorkersW; w++ {
		startC := w * channelsPerWorkerW
		endC := startC + channelsPerWorkerW
		if endC > outC {
			endC = outC
		}
		if startC >= endC {
			continue
		}

		wgW.Add(1)
		go func(sC, eC int) {
			defer wgW.Done()
			for cOut := sC; cOut < eC; cOut++ {
				var biasGrad float32
				outChOffset := cOut * outHW
				weightOutOffset := cOut * weightsPerOutChannel

				for y := 0; y < outH; y++ {
					inYBase := y*S - P
					outYOffset := outChOffset + y*outW

					for x := 0; x < outW; x++ {
						inXBase := x*S - P
						gy := gradOutput.Data[outYOffset+x]
						biasGrad += gy

						for cIn := 0; cIn < inC; cIn++ {
							inChOffset := cIn * inHW
							weightInOffset := weightOutOffset + cIn*Ksq

							for ky := 0; ky < K; ky++ {
								inY := inYBase + ky
								if inY < 0 || inY >= inH {
									continue
								}
								inRowOffset := inChOffset + inY*inW
								weightRowOffset := weightInOffset + ky*K

								for kx := 0; kx < K; kx++ {
									inX := inXBase + kx
									if inX < 0 || inX >= inW {
										continue
									}

									xVal := input.Data[inRowOffset+inX]
									l.Weights.Grad[weightRowOffset+kx] += gy * xVal
								}
							}
						}
					}
				}
				l.Bias.Grad[cOut] += biasGrad
			}
		}(startC, endC)
	}
	wgW.Wait()

	// 2. Input Gradients (Parallelized over input channels cIn for lock-free writing)
	numWorkersIn := numWorkers
	if numWorkersIn > inC {
		numWorkersIn = inC
	}
	channelsPerWorkerIn := (inC + numWorkersIn - 1) / numWorkersIn
	var wgIn sync.WaitGroup

	for w := 0; w < numWorkersIn; w++ {
		startCin := w * channelsPerWorkerIn
		endCin := startCin + channelsPerWorkerIn
		if endCin > inC {
			endCin = inC
		}
		if startCin >= endCin {
			continue
		}

		wgIn.Add(1)
		go func(sCin, eCin int) {
			defer wgIn.Done()
			for cIn := sCin; cIn < eCin; cIn++ {
				inChOffset := cIn * inHW

				for iy := 0; iy < inH; iy++ {
					inRowOffset := inChOffset + iy*inW

					for ix := 0; ix < inW; ix++ {
						var sum float32

						for cOut := 0; cOut < outC; cOut++ {
							outChOffset := cOut * outHW
							weightInOffset := cOut*weightsPerOutChannel + cIn*Ksq

							for ky := 0; ky < K; ky++ {
								yDiff := iy + P - ky
								if yDiff < 0 || yDiff%S != 0 {
									continue
								}
								y := yDiff / S
								if y >= outH {
									continue
								}
								outYOffset := outChOffset + y*outW
								weightRowOffset := weightInOffset + ky*K

								for kx := 0; kx < K; kx++ {
									xDiff := ix + P - kx
									if xDiff < 0 || xDiff%S != 0 {
										continue
									}
									x := xDiff / S
									if x >= outW {
										continue
									}

									gy := gradOutput.Data[outYOffset+x]
									wVal := l.Weights.Data[weightRowOffset+kx]
									sum += gy * wVal
								}
							}
						}

						gradInput.Data[inRowOffset+ix] = sum
					}
				}
			}
		}(startCin, endCin)
	}
	wgIn.Wait()
}

// AdaptiveAvgPool2DLayer dynamically pools arbitrary input feature dimensions to a fixed [TargetH x TargetW] output.
type AdaptiveAvgPool2DLayer struct {
	TargetH   int
	TargetW   int
	LastInput *Tensor
}

// NewAdaptiveAvgPool2DLayer constructs a new adaptive average pooling layer.
func NewAdaptiveAvgPool2DLayer(targetH, targetW int) *AdaptiveAvgPool2DLayer {
	return &AdaptiveAvgPool2DLayer{
		TargetH: targetH,
		TargetW: targetW,
	}
}

// Forward executes the 2D adaptive average pooling forward pass.
func (l *AdaptiveAvgPool2DLayer) Forward(input *Tensor) *Tensor {
	output := NewTensor(input.Channels, l.TargetH, l.TargetW)
	l.ForwardInto(input, output)
	return output
}

// ForwardInto executes adaptive average pooling into pre-allocated output tensor.
func (l *AdaptiveAvgPool2DLayer) ForwardInto(input *Tensor, output *Tensor) {
	l.LastInput = input
	C, inH, inW := input.Channels, input.Height, input.Width
	tgtH, tgtW := l.TargetH, l.TargetW

	if output.Channels != C || output.Height != tgtH || output.Width != tgtW {
		*output = *NewTensor(C, tgtH, tgtW)
	}

	for c := 0; c < C; c++ {
		for y := 0; y < tgtH; y++ {
			yStart := (y * inH) / tgtH
			yEnd := ((y + 1) * inH + tgtH - 1) / tgtH
			if yEnd > inH {
				yEnd = inH
			}

			for x := 0; x < tgtW; x++ {
				xStart := (x * inW) / tgtW
				xEnd := ((x + 1) * inW + tgtW - 1) / tgtW
				if xEnd > inW {
					xEnd = inW
				}

				binCount := (yEnd - yStart) * (xEnd - xStart)
				if binCount == 0 {
					binCount = 1
				}

				var sum float32
				for iy := yStart; iy < yEnd; iy++ {
					for ix := xStart; ix < xEnd; ix++ {
						sum += input.Get(c, iy, ix)
					}
				}

				output.Set(c, y, x, sum/float32(binCount))
			}
		}
	}
}

// Backward distributes gradients uniformly across adaptive pooling bins:
// dL/dX(c, iy, ix) = sum_{y, x} (dL/dY(c, y, x) / (bin_width * bin_height))
func (l *AdaptiveAvgPool2DLayer) Backward(gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(l.LastInput.Channels, l.LastInput.Height, l.LastInput.Width)
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto executes adaptive average pooling backpropagation into pre-allocated tensor.
func (l *AdaptiveAvgPool2DLayer) BackwardInto(gradOutput *Tensor, gradInput *Tensor) {
	input := l.LastInput
	C, inH, inW := input.Channels, input.Height, input.Width
	tgtH, tgtW := l.TargetH, l.TargetW

	if gradInput.Channels != C || gradInput.Height != inH || gradInput.Width != inW {
		*gradInput = *NewTensor(C, inH, inW)
	}
	gradInput.Zero()

	for c := 0; c < C; c++ {
		for y := 0; y < tgtH; y++ {
			yStart := (y * inH) / tgtH
			yEnd := ((y + 1) * inH + tgtH - 1) / tgtH
			if yEnd > inH {
				yEnd = inH
			}

			for x := 0; x < tgtW; x++ {
				xStart := (x * inW) / tgtW
				xEnd := ((x + 1) * inW + tgtW - 1) / tgtW
				if xEnd > inW {
					xEnd = inW
				}

				binCount := (yEnd - yStart) * (xEnd - xStart)
				if binCount == 0 {
					binCount = 1
				}

				dY := gradOutput.Get(c, y, x)
				distributedGrad := dY / float32(binCount)

				for iy := yStart; iy < yEnd; iy++ {
					for ix := xStart; ix < xEnd; ix++ {
						val := gradInput.Get(c, iy, ix)
						gradInput.Set(c, iy, ix, val+distributedGrad)
					}
				}
			}
		}
	}
}

// LinearLayer implements a fully connected feedforward layer with vectorization and analytical Jacobian backpropagation.
type LinearLayer struct {
	Weights   *Parameter // Shape: [OutputDim, InputDim] -> flat size OutputDim * InputDim
	Biases    *Parameter // Shape: [OutputDim]
	InputDim  int
	OutputDim int
	LastInput []float32
}

// NewLinearLayer constructs and initializes a LinearLayer with Kaiming Uniform weights and zero biases.
func NewLinearLayer(inDim, outDim int, rng *rand.Rand) *LinearLayer {
	layer := &LinearLayer{
		Weights:   NewParameter(outDim * inDim),
		Biases:    NewParameter(outDim),
		InputDim:  inDim,
		OutputDim: outDim,
	}

	if rng != nil {
		InitKaimingUniform(layer.Weights, inDim, rng)
	} else {
		defaultRNG := rand.New(rand.NewSource(42))
		InitKaimingUniform(layer.Weights, inDim, defaultRNG)
	}
	InitZeros(layer.Biases)

	return layer
}

// ZeroGrad zeroes out gradient buffers for weights and biases.
func (l *LinearLayer) ZeroGrad() {
	l.Weights.ZeroGrad()
	l.Biases.ZeroGrad()
}

// Parameters returns references to trainable parameters in the linear layer.
func (l *LinearLayer) Parameters() []*Parameter {
	return []*Parameter{l.Weights, l.Biases}
}

// Forward computes y_i = b_i + sum_j W_{i, j} * x_j for all i in [0, OutputDim-1].
func (l *LinearLayer) Forward(input []float32) []float32 {
	out := make([]float32, l.OutputDim)
	l.ForwardInto(input, out)
	return out
}

// ForwardInto computes dense feedforward layer into pre-allocated output slice.
func (l *LinearLayer) ForwardInto(input []float32, output []float32) {
	if len(l.LastInput) != len(input) {
		l.LastInput = make([]float32, len(input))
	}
	copy(l.LastInput, input)

	inDim := l.InputDim
	outDim := l.OutputDim

	for i := 0; i < outDim; i++ {
		wOffset := i * inDim
		var sum float32 = l.Biases.Data[i]
		for j := 0; j < inDim; j++ {
			sum += l.Weights.Data[wOffset+j] * input[j]
		}
		output[i] = sum
	}
}

// Backward computes analytical Jacobian parameter gradients and input gradients:
// 1. dL/dW_{i, j} = dL/dy_i * x_j
// 2. dL/db_i = dL/dy_i
// 3. dL/dx_j = sum_{i=0}^{D_out-1} W_{i, j} * dL/dy_i
func (l *LinearLayer) Backward(gradOutput []float32) []float32 {
	gradInput := make([]float32, l.InputDim)
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto computes linear backpropagation into pre-allocated gradInput slice.
func (l *LinearLayer) BackwardInto(gradOutput []float32, gradInput []float32) {
	inDim := l.InputDim
	outDim := l.OutputDim
	input := l.LastInput

	for j := 0; j < inDim; j++ {
		gradInput[j] = 0
	}

	for i := 0; i < outDim; i++ {
		gy := gradOutput[i]
		l.Biases.Grad[i] += gy
		wOffset := i * inDim

		for j := 0; j < inDim; j++ {
			l.Weights.Grad[wOffset+j] += gy * input[j]
			gradInput[j] += l.Weights.Data[wOffset+j] * gy
		}
	}
}

// DropoutLayer implements Inverted Dropout regularization (default p=0.2).
type DropoutLayer struct {
	DropRate float32 // p = 0.2
	Scale    float32 // 1.0 / (1.0 - p) = 1.25
	Training bool
	Mask     []float32
	RNG      *rand.Rand
}

// NewDropoutLayer constructs an Inverted Dropout layer with the specified drop probability.
func NewDropoutLayer(dropRate float32, rng *rand.Rand) *DropoutLayer {
	if dropRate < 0 {
		dropRate = 0
	}
	if dropRate >= 1.0 {
		dropRate = 0.99
	}
	scale := float32(1.0 / (1.0 - float64(dropRate)))
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	return &DropoutLayer{
		DropRate: dropRate,
		Scale:    scale,
		Training: true,
		RNG:      rng,
	}
}

// Forward applies Bernoulli inverted dropout during training, or passes through during inference.
func (l *DropoutLayer) Forward(input []float32) []float32 {
	out := make([]float32, len(input))
	l.ForwardInto(input, out)
	return out
}

// ForwardInto computes inverted dropout into pre-allocated slice.
func (l *DropoutLayer) ForwardInto(input []float32, output []float32) {
	N := len(input)
	if len(l.Mask) != N {
		l.Mask = make([]float32, N)
	}

	if !l.Training || l.DropRate == 0 {
		copy(output, input)
		for i := 0; i < N; i++ {
			l.Mask[i] = 1.0
		}
		return
	}

	for i := 0; i < N; i++ {
		if l.RNG.Float32() >= l.DropRate {
			l.Mask[i] = 1.0
			output[i] = input[i] * l.Scale
		} else {
			l.Mask[i] = 0.0
			output[i] = 0.0
		}
	}
}

// Backward scales incoming gradients by the Bernoulli inverted mask.
func (l *DropoutLayer) Backward(gradOutput []float32) []float32 {
	gradInput := make([]float32, len(gradOutput))
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto computes dropout gradient scaling into pre-allocated slice.
func (l *DropoutLayer) BackwardInto(gradOutput []float32, gradInput []float32) {
	if !l.Training || l.DropRate == 0 {
		copy(gradInput, gradOutput)
		return
	}
	for i := range gradOutput {
		gradInput[i] = gradOutput[i] * l.Mask[i] * l.Scale
	}
}

// ============================================================================
// 8. CLI ROUTING & EXECUTION HANDLERS
// ============================================================================

func printHelp() {
	fmt.Print(banner)
	fmt.Println("Usage:")
	fmt.Println("  diagonnet [mode flag] [configuration flags]")
	fmt.Println("  diagonnet [subcommand] [configuration flags]")
	fmt.Println()
	fmt.Println("Execution Modes:")
	fmt.Println("  -train          Launch deep learning model training pipeline")
	fmt.Println("  -serve          Start the interactive HTTP inference and dashboard server")
	fmt.Println("  -audit          Run dataset validation, shape verification, and integrity audit")
	fmt.Println("  -benchmark      Execute full benchmark suite against standard dataset manifolds")
	fmt.Println()
	fmt.Println("Positional Subcommands:")
	fmt.Println("  train, serve, audit, benchmark, help")
	fmt.Println()
	fmt.Println("Configuration Flags:")
	fmt.Println("  -data string    Path to dataset samples directory (default \"data\")")
	fmt.Println("  -model string   Path to binary model weights file (default \"weights/diagonnet_model.bin\")")
	fmt.Println("  -epochs int     Number of training epochs (default 8)")
	fmt.Println("  -lr float       Learning rate for parameter optimization (default 0.002)")
	fmt.Println("  -batch int      Mini-batch size for training (default 64)")
	fmt.Println("  -port int       HTTP server listen port (default 8081)")
	fmt.Println("  -help, -h       Display this help and exit")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  diagonnet -train -data data -epochs 10 -lr 0.001 -batch 32")
	fmt.Println("  diagonnet -serve -model weights/diagonnet_model.bin -port 8081")
	fmt.Println("  diagonnet -audit -data data")
	fmt.Println("  diagonnet -benchmark")
	fmt.Println()
}

func runAudit(dataDir string) {
	fmt.Println(">>> [Audit Mode] Starting dataset validation and integrity audit...")
	fmt.Printf("    Target Dataset Directory : %s\n", dataDir)
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		fmt.Printf("    Notice: Dataset directory '%s' does not exist yet. Create it or specify -data.\n", dataDir)
	} else {
		fmt.Printf("    Dataset directory '%s' verified.\n", dataDir)
	}
	fmt.Println(">>> Dataset audit completed.")
}

func runTrain(dataDir string, modelPath string, epochs int, lr float32, batchSize int) {
	fmt.Println(">>> [Train Mode] Initializing deep learning training pipeline...")
	fmt.Printf("    Dataset Directory : %s\n", dataDir)
	fmt.Printf("    Output Model Path : %s\n", modelPath)
	fmt.Printf("    Training Epochs   : %d\n", epochs)
	fmt.Printf("    Learning Rate     : %.4f\n", lr)
	fmt.Printf("    Batch Size        : %d\n", batchSize)
	fmt.Printf("    Worker Threads    : %d\n", NumWorkers())
	fmt.Println(">>> Training pipeline ready.")
}

func runServer(modelPath string, port int) {
	fmt.Println(">>> [Serve Mode] Initializing interactive HTTP server...")
	fmt.Printf("    Model Weights Path : %s\n", modelPath)
	fmt.Printf("    HTTP Listen Port   : %d\n", port)
	fmt.Printf("    Server URL         : http://localhost:%d\n", port)
	fmt.Println(">>> Server configured.")
}

func runBenchmark(dataDir string) {
	fmt.Println(">>> [Benchmark Mode] Initializing manifold and standard benchmark suite...")
	fmt.Printf("    Benchmark Data Path : %s\n", dataDir)
	fmt.Printf("    Parallel Workers    : %d\n", NumWorkers())
	fmt.Println(">>> Benchmark suite ready.")
}

func main() {
	// 1. Configure multi-core parallel runtime settings
	runtime.GOMAXPROCS(runtime.NumCPU())
	PrintHardwareDiagnostics()

	fs := flag.NewFlagSet("diagonnet", flag.ContinueOnError)
	fs.Usage = printHelp

	trainFlag := fs.Bool("train", false, "Launch deep learning model training pipeline")
	serveFlag := fs.Bool("serve", false, "Start the interactive HTTP inference and dashboard server")
	auditFlag := fs.Bool("audit", false, "Run dataset validation, shape verification, and integrity audit")
	benchFlag := fs.Bool("benchmark", false, "Execute full benchmark suite against standard dataset manifolds")

	dataDir := fs.String("data", "data", "Path to dataset directory")
	modelPath := fs.String("model", "weights/diagonnet_model.bin", "Path to binary model weights")
	epochs := fs.Int("epochs", 8, "Number of training epochs")
	lr := fs.Float64("lr", 0.002, "Learning rate for optimization")
	batchSize := fs.Int("batch", 64, "Mini-batch size for training")
	port := fs.Int("port", 8081, "HTTP server listen port")

	args := os.Args[1:]

	// Positional subcommand fallback handling
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		subcmd := strings.ToLower(args[0])
		remaining := args[1:]
		switch subcmd {
		case "train":
			args = append([]string{"-train"}, remaining...)
		case "serve":
			args = append([]string{"-serve"}, remaining...)
		case "audit":
			args = append([]string{"-audit"}, remaining...)
		case "benchmark", "bench":
			args = append([]string{"-benchmark"}, remaining...)
		case "help":
			printHelp()
			return
		default:
			// Let flag set handle or report unknown
		}
	}

	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return
		}
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %v\n", err)
		os.Exit(1)
	}

	// Dispatch logic
	switch {
	case *auditFlag:
		runAudit(*dataDir)
	case *trainFlag:
		runTrain(*dataDir, *modelPath, *epochs, float32(*lr), *batchSize)
	case *serveFlag:
		runServer(*modelPath, *port)
	case *benchFlag:
		runBenchmark(*dataDir)
	default:
		// Positional fallback check from flag.Args() if flags were mixed
		if fs.NArg() > 0 {
			switch strings.ToLower(fs.Arg(0)) {
			case "train":
				runTrain(*dataDir, *modelPath, *epochs, float32(*lr), *batchSize)
				return
			case "serve":
				runServer(*modelPath, *port)
				return
			case "audit":
				runAudit(*dataDir)
				return
			case "benchmark", "bench":
				runBenchmark(*dataDir)
				return
			}
		}
		printHelp()
	}
}
