package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// ============================================================================
// 1. HARDWARE TOPOLOGY & RUNTIME CONCURRENCY
// ============================================================================

const banner = `
================================================================================
  ____  _                         _ _   _      _   
 |  _ \(_) __ _  __ _  ___  _ __   __ _| | \ | | ___| |_ 
 | | | | |/ _` + "`" + ` |/ _` + "`" + ` |/ _ \| '_ \ / _` + "`" + ` | |  \| |/ _ \ __|
 | |_| | | (_| | (_| | (_) | | | | (_| | | |\  |  __/ |_ 
 |____/|_|\__,_|\__, |\___/|_| |_|\__,_|_|_| \_|\___|\__|
                |___/                                     
 Pure Go Zero-Dependency Deep Learning Engine & Web Runtime
================================================================================
`

// InputSize is the square spatial resolution (InputSize x InputSize) that every image is
// resampled to after bounding-box cropping, centering and contrast stretching.
//
// Training and live web inference all read this single constant. If the training
// pipeline and the inference pipeline disagree on this value the convolutional receptive fields
// seen at serve time no longer match the ones the weights were fitted on, which silently
// destroys accuracy. Keep it as the one knob.
const InputSize = 28

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
	fmt.Println("       DiagonalNet Pure Go Deep Learning Engine")
	fmt.Printf("       CPU Compute Engine: %d Logical Cores Fully Utilized (100%%)\n", cores)
	fmt.Printf("       OS: %s | Architecture: %s | Go: %s\n", runtime.GOOS, runtime.GOARCH, runtime.Version())
	fmt.Println("==================================================================")
}

// ============================================================================
// 2. CONTIGUOUS 1D/3D TENSOR ENGINE
// ============================================================================

// --- 2.1 TENSOR STRUCT & FLAT MEMORY ALLOCATION ---

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

// --- 2.2 FLAT-STRIDED INDEXING & O(1) ELEMENT ACCESS ---

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

// --- 2.3 MEMORY ZEROING, SIZING & DEEP COPY ISOLATION ---

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

// AdamOptimizerConfig defines hyperparameters for the Adam optimization algorithm.
type AdamOptimizerConfig struct {
	LearningRate float32 // Learning rate eta (default: 0.001)
	Beta1        float32 // 1st moment decay beta_1 (default: 0.9)
	Beta2        float32 // 2nd raw moment decay beta_2 (default: 0.999)
	Eps          float32 // Numerical stability epsilon (default: 1e-8)
	WeightDecay  float32 // Optional L2 weight decay (default: 0.0)
}

// DefaultAdamConfig returns the standard Adam hyperparameter configuration:
// lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 1e-4 (L2 regularization lambda)
func DefaultAdamConfig() AdamOptimizerConfig {
	return AdamOptimizerConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  1e-4, // lambda = 10^-4 L2 weight decay regularization
	}
}

// AdamOptimizer implements the Adam optimization algorithm with L2 weight decay regularization,
// 1st (mean) and 2nd (variance) moment tracking, and analytical bias corrections.
//
// Formulations for Step t:
// 1. L2 Regularized Gradient:   g_t <- g_t + lambda * theta_t (lambda = 10^-4)
// 2. 1st Moment EMA (Mean):     m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
// 3. 2nd Moment EMA (Variance): v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
// 4. Analytical Bias Correction:
//       m_hat_t = m_t / (1 - beta1^t)
//       v_hat_t = v_t / (1 - beta2^t)
// 5. Weight Parameter Update:   theta_{t+1} = theta_t - alpha * m_hat_t / (sqrt(v_hat_t) + eps)
type AdamOptimizer struct {
	Params    []*Parameter
	Config    AdamOptimizerConfig
	StepCount int
}

// NewAdamOptimizer constructs an Adam optimizer for the specified slice of trainable parameters.
func NewAdamOptimizer(params []*Parameter, config AdamOptimizerConfig) *AdamOptimizer {
	if config.Beta1 <= 0 || config.Beta1 >= 1 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 || config.Beta2 >= 1 {
		config.Beta2 = 0.999
	}
	if config.Eps <= 0 {
		config.Eps = 1e-8
	}
	if config.LearningRate <= 0 {
		config.LearningRate = 0.001
	}

	return &AdamOptimizer{
		Params:    params,
		Config:    config,
		StepCount: 0,
	}
}

// ZeroGrad resets gradient accumulators for all registered parameters to zero.
func (opt *AdamOptimizer) ZeroGrad() {
	for _, p := range opt.Params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}

// Step performs a single Adam optimization parameter update across all registered parameters.
func (opt *AdamOptimizer) Step() {
	opt.StepCount++
	t := opt.StepCount
	lr := opt.Config.LearningRate
	beta1 := opt.Config.Beta1
	beta2 := opt.Config.Beta2
	eps := opt.Config.Eps
	wd := opt.Config.WeightDecay

	for _, p := range opt.Params {
		if p == nil {
			continue
		}
		StepParameter(p, t, lr, beta1, beta2, eps, wd)
	}
}

// StepParameter executes the Adam parameter update for a single Parameter buffer at time step t.
func StepParameter(param *Parameter, t int, lr, beta1, beta2, eps, weightDecay float32) {
	if param == nil || t <= 0 {
		return
	}
	tF := float64(t)
	b1 := float64(beta1)
	b2 := float64(beta2)
	learningRate := float64(lr)
	epsilon := float64(eps)
	wd := float64(weightDecay)

	biasCorrection1 := 1.0 - math.Pow(b1, tF)
	biasCorrection2 := 1.0 - math.Pow(b2, tF)

	L := len(param.Data)
	for i := 0; i < L; i++ {
		g := float64(param.Grad[i])
		if wd > 0 {
			g += wd * float64(param.Data[i])
		}

		// 1st moment vector: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
		m := b1*float64(param.M[i]) + (1.0-b1)*g
		param.M[i] = float32(m)

		// 2nd raw moment vector: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
		v := b2*float64(param.V[i]) + (1.0-b2)*(g*g)
		param.V[i] = float32(v)

		// Bias-corrected moment estimates
		mHat := m / biasCorrection1
		vHat := v / biasCorrection2

		// Weight update step
		update := (learningRate * mHat) / (math.Sqrt(vHat) + epsilon)
		param.Data[i] -= float32(update)
	}
}

// LRMilestone defines an epoch threshold and the multiplier factor applied to the initial learning rate.
type LRMilestone struct {
	Epoch  int     `json:"epoch"`
	Factor float32 `json:"factor"`
}

// StepLRSchedulerConfig defines configurable parameters for the step learning rate decay scheduler.
type StepLRSchedulerConfig struct {
	InitialLR  float32       `json:"initial_lr"`
	Milestones []LRMilestone `json:"milestones"`
}

// DefaultStepLRSchedulerConfig returns the default milestone decay schedule:
// Initial LR: alpha_0 = 0.002
// Epochs 1 - 7:   factor = 1.0  (alpha = 0.002)
// Epochs 8 - 16:  factor = 0.5  (alpha = 0.001, 50% decay)
// Epochs 17+:     factor = 0.25 (alpha = 0.0005, 25% decay)
func DefaultStepLRSchedulerConfig() StepLRSchedulerConfig {
	return StepLRSchedulerConfig{
		InitialLR: 0.002,
		Milestones: []LRMilestone{
			{Epoch: 8, Factor: 0.5},
			{Epoch: 17, Factor: 0.25},
		},
	}
}

// SaveStepLRSchedulerConfig saves scheduler settings to a JSON configuration file.
func SaveStepLRSchedulerConfig(path string, cfg *StepLRSchedulerConfig) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory for scheduler config: %w", err)
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode scheduler config JSON: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// LoadStepLRSchedulerConfig loads scheduler settings from a JSON configuration file.
func LoadStepLRSchedulerConfig(path string) (*StepLRSchedulerConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read scheduler config file: %w", err)
	}
	var cfg StepLRSchedulerConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to decode scheduler config JSON: %w", err)
	}
	return &cfg, nil
}

// StepLRScheduler manages learning rate milestone decay scheduling during training.
type StepLRScheduler struct {
	Optimizer *AdamOptimizer
	Config    StepLRSchedulerConfig
	CurrentLR float32
	LastEpoch int
}

// NewStepLRScheduler constructs a new StepLRScheduler for an Adam optimizer with the provided config.
func NewStepLRScheduler(optimizer *AdamOptimizer, config StepLRSchedulerConfig) *StepLRScheduler {
	if config.InitialLR <= 0 {
		config.InitialLR = 0.002
	}
	sched := &StepLRScheduler{
		Optimizer: optimizer,
		Config:    config,
		CurrentLR: config.InitialLR,
		LastEpoch: 0,
	}
	if optimizer != nil {
		optimizer.Config.LearningRate = sched.CurrentLR
	}
	return sched
}

// NewStepLRSchedulerFromFile constructs a StepLRScheduler by loading configuration from a JSON file.
// If the file does not exist, it initializes with DefaultStepLRSchedulerConfig and saves the default config file.
func NewStepLRSchedulerFromFile(optimizer *AdamOptimizer, configPath string) (*StepLRScheduler, error) {
	var cfg *StepLRSchedulerConfig
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		defaultCfg := DefaultStepLRSchedulerConfig()
		_ = SaveStepLRSchedulerConfig(configPath, &defaultCfg)
		cfg = &defaultCfg
	} else {
		loadedCfg, err := LoadStepLRSchedulerConfig(configPath)
		if err != nil {
			return nil, err
		}
		cfg = loadedCfg
	}
	return NewStepLRScheduler(optimizer, *cfg), nil
}

// GetLR calculates the scheduled learning rate for any specified 1-indexed epoch.
func (s *StepLRScheduler) GetLR(epoch int) float32 {
	if epoch <= 0 {
		epoch = 1
	}
	factor := float32(1.0)
	// Milestones are evaluated in ascending order
	for _, m := range s.Config.Milestones {
		if epoch >= m.Epoch {
			factor = m.Factor
		}
	}
	return s.Config.InitialLR * factor
}

// Step advances the scheduler to the specified epoch, updates optimizer learning rate,
// and logs adjustments cleanly to stdout whenever the learning rate changes.
func (s *StepLRScheduler) Step(epoch int) float32 {
	s.LastEpoch = epoch
	newLR := s.GetLR(epoch)
	oldLR := s.CurrentLR

	if newLR != oldLR {
		factor := float32(1.0)
		if s.Config.InitialLR > 0 {
			factor = newLR / s.Config.InitialLR
		}
		fmt.Printf(">>> [LR Scheduler] Epoch %d: Learning rate adjusted from %.6f -> %.6f (scale %.2f)\n", epoch, oldLR, newLR, factor)
		s.CurrentLR = newLR
	}

	if s.Optimizer != nil {
		s.Optimizer.Config.LearningRate = newLR
	}

	return newLR
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

	if L < 4096 || numWorkers <= 1 {
		for i := 0; i < L; i++ {
			var sum float32
			for k := 0; k < len(workers); k++ {
				sum += workers[k].Grad[i]
			}
			master.Grad[i] = sum
		}
		return
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
// 6. DATASET SCANNER, IMAGE PREPROCESSING & 15X DATA AUGMENTATION
// ============================================================================

// --- 6.1 DYNAMIC DATASET SCANNER & CLASS MAPPING ---

// DatasetMetadata stores dynamic bi-directional mappings between class names and integer indices.
type DatasetMetadata struct {
	Classes    []string       `json:"classes"`      // e.g. ["circle", "square", "triangle"]
	ClassToIdx map[string]int `json:"class_to_idx"` // "circle" -> 0, "square" -> 1, "triangle" -> 2
	IdxToClass map[int]string `json:"idx_to_class"` // 0 -> "circle", 1 -> "square", 2 -> "triangle"
	NumClasses int            `json:"num_classes"`  // K = len(Classes)
}

// NewDatasetMetadata constructs bi-directional mappings from a slice of class name strings.
// Class names are sorted alphabetically to ensure 100% deterministic, reproducible index assignments.
func NewDatasetMetadata(classes []string) DatasetMetadata {
	sortedClasses := make([]string, len(classes))
	copy(sortedClasses, classes)
	sort.Strings(sortedClasses)

	classToIdx := make(map[string]int, len(sortedClasses))
	idxToClass := make(map[int]string, len(sortedClasses))

	for idx, name := range sortedClasses {
		classToIdx[name] = idx
		idxToClass[idx] = name
	}

	return DatasetMetadata{
		Classes:    sortedClasses,
		ClassToIdx: classToIdx,
		IdxToClass: idxToClass,
		NumClasses: len(sortedClasses),
	}
}

// GetClassIndex returns the integer index for a class name.
func (m *DatasetMetadata) GetClassIndex(className string) (int, bool) {
	idx, ok := m.ClassToIdx[className]
	return idx, ok
}

// GetClassName returns the class name string for an integer index.
func (m *DatasetMetadata) GetClassName(classIndex int) (string, bool) {
	name, ok := m.IdxToClass[classIndex]
	return name, ok
}

// ImageSample represents an individual image file path, its class name, and integer class label.
type ImageSample struct {
	Path       string `json:"path"`
	Class      string `json:"class"`
	ClassIndex int    `json:"class_index"`
}

// Dataset contains discovered image samples and dynamic class metadata.
type Dataset struct {
	Metadata DatasetMetadata `json:"metadata"`
	Samples  []ImageSample   `json:"samples"`
}

// IsValidImageExtension returns true if file extension matches .png, .jpg, or .jpeg (case-insensitive).
func IsValidImageExtension(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".png" || ext == ".jpg" || ext == ".jpeg"
}

// ScanDataset dynamically scans the filesystem directory without hardcoded labels:
// 1. Reads directory dataDir.
// 2. Discovers all immediate subdirectories as classes.
// 3. Scans for .png, .jpg, .jpeg image files inside each class directory.
// 4. Returns an error if directory contains fewer than 2 valid classes or zero images.
func ScanDataset(dataDir string) (*Dataset, error) {
	info, err := os.Stat(dataDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("dataset directory does not exist: %s", dataDir)
		}
		return nil, fmt.Errorf("failed to access dataset directory %s: %w", dataDir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("dataset path is not a directory: %s", dataDir)
	}

	entries, err := os.ReadDir(dataDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read dataset directory %s: %w", dataDir, err)
	}

	var classNames []string
	classFiles := make(map[string][]string)

	for _, entry := range entries {
		if entry.IsDir() {
			className := entry.Name()
			subDirPath := filepath.Join(dataDir, className)

			subEntries, err := os.ReadDir(subDirPath)
			if err != nil {
				continue
			}

			var imgPaths []string
			for _, subEntry := range subEntries {
				if !subEntry.IsDir() && IsValidImageExtension(subEntry.Name()) {
					imgPaths = append(imgPaths, filepath.Join(subDirPath, subEntry.Name()))
				}
			}

			if len(imgPaths) > 0 {
				classNames = append(classNames, className)
				classFiles[className] = imgPaths
			}
		}
	}

	if len(classNames) < 2 {
		return nil, fmt.Errorf("dataset requires at least 2 valid classes containing images, found %d", len(classNames))
	}

	metadata := NewDatasetMetadata(classNames)
	var samples []ImageSample

	for _, className := range metadata.Classes {
		idx := metadata.ClassToIdx[className]
		for _, imgPath := range classFiles[className] {
			samples = append(samples, ImageSample{
				Path:       imgPath,
				Class:      className,
				ClassIndex: idx,
			})
		}
	}

	if len(samples) == 0 {
		return nil, errors.New("dataset contains zero valid images")
	}

	return &Dataset{
		Metadata: metadata,
		Samples:  samples,
	}, nil
}

// ImageItem is an alias for ImageSample representing a dataset item.
type ImageItem = ImageSample

// --- 6.2 8-BIT GRAYSCALE LOADER & TENSOR NORMALIZATION ---

// LoadImageFromFile opens a PNG or JPEG file from disk, decodes it, and converts it into
// a standard 8-bit grayscale luminosity representation (*image.Gray) using pure Go standard library.
func LoadImageFromFile(path string) (*image.Gray, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image %s: %w", path, err)
	}
	defer file.Close()

	src, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("decode error on %s: %w", path, err)
	}

	bounds := src.Bounds()
	gray := image.NewGray(bounds)
	draw.Draw(gray, bounds, src, bounds.Min, draw.Src)
	return gray, nil
}

// GrayImageToTensor converts an *image.Gray into a 1xHxW Tensor normalized to [0.0, 1.0].
func GrayImageToTensor(gray *image.Gray) *Tensor {
	bounds := gray.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	t := NewTensor(1, h, w)

	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := gray.GrayAt(x, y).Y
			t.Data[idx] = float32(pixel) / 255.0
			idx++
		}
	}
	return t
}

// --- 6.3 BOUNDING BOX EXTRACTION & BLANK FILTERING ---

// BoundingBox defines the tight rectangular coordinate bounds of foreground pixels.
type BoundingBox struct {
	MinX int `json:"min_x"`
	MinY int `json:"min_y"`
	MaxX int `json:"max_x"`
	MaxY int `json:"max_y"`
}

// Width returns the horizontal pixel extent: MaxX - MinX + 1.
func (b *BoundingBox) Width() int {
	if b == nil {
		return 0
	}
	return b.MaxX - b.MinX + 1
}

// Height returns the vertical pixel extent: MaxY - MinY + 1.
func (b *BoundingBox) Height() int {
	if b == nil {
		return 0
	}
	return b.MaxY - b.MinY + 1
}

// FindBoundingBox locates the tight bounding box enclosing foreground drawing pixels in an *image.Gray.
//
// Specifications:
// 1. Minimum luminosity threshold = 10 (or configurable threshold).
// 2. Finds min(x), min(y), max(x), max(y) where pixel luminosity > threshold.
// 3. If no pixel exceeds threshold, returns nil indicating a blank image.
func FindBoundingBox(gray *image.Gray, threshold uint8) *BoundingBox {
	if gray == nil {
		return nil
	}
	bounds := gray.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	if w == 0 || h == 0 {
		return nil
	}

	minX, maxX := w, -1
	minY, maxY := h, -1

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := gray.GrayAt(x, y).Y
			if pixel > threshold {
				relX := x - bounds.Min.X
				relY := y - bounds.Min.Y
				if relX < minX {
					minX = relX
				}
				if relX > maxX {
					maxX = relX
				}
				if relY < minY {
					minY = relY
				}
				if relY > maxY {
					maxY = relY
				}
			}
		}
	}

	if maxX < 0 || maxY < 0 {
		return nil // Blank image
	}

	return &BoundingBox{
		MinX: minX,
		MinY: minY,
		MaxX: maxX,
		MaxY: maxY,
	}
}

// FindBoundingBoxTensor locates the tight bounding box on a 1xHxW Tensor.
func FindBoundingBoxTensor(t *Tensor, threshold float32) *BoundingBox {
	if t == nil || t.Height == 0 || t.Width == 0 {
		return nil
	}
	minX, maxX := t.Width, -1
	minY, maxY := t.Height, -1

	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			val := t.Get(0, y, x)
			if val > threshold {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}

	if maxX < 0 || maxY < 0 {
		return nil
	}

	return &BoundingBox{
		MinX: minX,
		MinY: minY,
		MaxX: maxX,
		MaxY: maxY,
	}
}

// --- 6.4 SCALE-INVARIANT PROPORTIONAL PADDING & CENTERING ---

// PadAndCenter centers the cropped bounding box foreground into a square canvas
// with scale-invariant proportional padding (22% margin per side, ~70% foreground occupancy).
//
// Algorithm:
// 1. D = max(W_bbox, H_bbox)
// 2. pad = max(2, floor(0.22 * D))
// 3. S = D + 2 * pad
// 4. Center bounding box into S x S canvas.
func PadAndCenter(src *image.Gray, bbox *BoundingBox) *image.Gray {
	if src == nil || bbox == nil {
		return src
	}
	wBbox := bbox.Width()
	hBbox := bbox.Height()
	if wBbox <= 0 || hBbox <= 0 {
		return src
	}

	d := wBbox
	if hBbox > d {
		d = hBbox
	}

	pad := int(math.Floor(0.22 * float64(d)))
	if pad < 2 {
		pad = 2
	}

	s := d + 2*pad
	dst := image.NewGray(image.Rect(0, 0, s, s))

	offsetX := (s - wBbox) / 2
	offsetY := (s - hBbox) / 2

	bounds := src.Bounds()
	for dy := 0; dy < hBbox; dy++ {
		srcY := bounds.Min.Y + bbox.MinY + dy
		dstY := offsetY + dy
		for dx := 0; dx < wBbox; dx++ {
			srcX := bounds.Min.X + bbox.MinX + dx
			dstX := offsetX + dx
			if srcX < bounds.Max.X && srcY < bounds.Max.Y && dstX < s && dstY < s {
				dst.SetGray(dstX, dstY, src.GrayAt(srcX, srcY))
			}
		}
	}

	return dst
}

// PadAndCenterTensor centers a bounding box on a 1xHxW Tensor into an SxS square Tensor.
func PadAndCenterTensor(src *Tensor, bbox *BoundingBox) *Tensor {
	if src == nil || bbox == nil {
		return src
	}
	wBbox := bbox.Width()
	hBbox := bbox.Height()
	if wBbox <= 0 || hBbox <= 0 {
		return src
	}

	d := wBbox
	if hBbox > d {
		d = hBbox
	}

	pad := int(math.Floor(0.22 * float64(d)))
	if pad < 2 {
		pad = 2
	}

	s := d + 2*pad
	dst := NewTensor(src.Channels, s, s)

	offsetX := (s - wBbox) / 2
	offsetY := (s - hBbox) / 2

	for c := 0; c < src.Channels; c++ {
		for dy := 0; dy < hBbox; dy++ {
			srcY := bbox.MinY + dy
			dstY := offsetY + dy
			for dx := 0; dx < wBbox; dx++ {
				srcX := bbox.MinX + dx
				dstX := offsetX + dx
				if srcX < src.Width && srcY < src.Height && dstX < s && dstY < s {
					val := src.Get(c, srcY, srcX)
					dst.Set(c, dstY, dstX, val)
				}
			}
		}
	}

	return dst
}

// --- 6.5 PEAK STROKE LUMINOSITY CONTRAST STRETCHING ---

// ContrastStretch applies adaptive peak luminosity stretching to normalize faint and heavy pen strokes.
//
// Formulation:
// 1. Find maximum pixel value L_max in the image.
// 2. If 30 < L_max < 240, apply scale factor s = 255.0 / L_max.
// 3. Adjusted pixel: y' = min(255, round(y * s)).
func ContrastStretch(src *image.Gray) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	if w == 0 || h == 0 {
		return src
	}

	// 1. Find L_max
	var lMax uint8 = 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			p := src.GrayAt(x, y).Y
			if p > lMax {
				lMax = p
			}
		}
	}

	// 2. Check threshold: if not in (30, 240), return copy
	dst := image.NewGray(bounds)
	if lMax <= 30 || lMax >= 240 {
		draw.Draw(dst, bounds, src, bounds.Min, draw.Src)
		return dst
	}

	// 3. Scale pixels
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			p := src.GrayAt(x, y).Y
			val := int(math.Round(float64(p) * 255.0 / float64(lMax)))
			if val > 255 {
				val = 255
			}
			dst.SetGray(x, y, color.Gray{Y: uint8(val)})
		}
	}

	return dst
}

// ContrastStretchTensor applies adaptive peak luminosity stretching to a 1xHxW Tensor.
func ContrastStretchTensor(src *Tensor) *Tensor {
	if src == nil {
		return nil
	}
	var maxVal float32 = 0
	for _, v := range src.Data {
		if v > maxVal {
			maxVal = v
		}
	}

	dst := src.Clone()
	if maxVal <= (30.0/255.0) || maxVal >= (240.0/255.0) {
		return dst
	}

	scale := float32(1.0) / maxVal
	for i, v := range dst.Data {
		val := v * scale
		if val > 1.0 {
			val = 1.0
		}
		dst.Data[i] = val
	}
	return dst
}

// --- 6.6 SUB-PIXEL BILINEAR RESAMPLING (CANONICAL INPUTSIZE) ---

// ResizeBilinear resamples an *image.Gray to target dimensions (targetW x targetH)
// using sub-pixel bilinear interpolation with continuous half-pixel centering.
//
// Formulation:
// For target coordinate (x, y):
// x_src = (x + 0.5) * (W_src / W_target) - 0.5
// y_src = (y + 0.5) * (H_src / H_target) - 0.5
//
// Interpolate using the 4 bounding integer neighbors (x0, y0), (x1, y0), (x0, y1), (x1, y1)
// with fractional weights fx = x_src - x0, fy = y_src - y0.
func ResizeBilinear(src *image.Gray, targetW, targetH int) *image.Gray {
	if src == nil {
		return nil
	}
	if targetW <= 0 || targetH <= 0 {
		return image.NewGray(image.Rect(0, 0, 0, 0))
	}

	bounds := src.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()
	if srcW == 0 || srcH == 0 {
		return image.NewGray(image.Rect(0, 0, targetW, targetH))
	}

	dst := image.NewGray(image.Rect(0, 0, targetW, targetH))

	scaleX := float64(srcW) / float64(targetW)
	scaleY := float64(srcH) / float64(targetH)

	for ty := 0; ty < targetH; ty++ {
		srcY := (float64(ty) + 0.5)*scaleY - 0.5
		if srcY < 0 {
			srcY = 0
		}
		if srcY > float64(srcH-1) {
			srcY = float64(srcH - 1)
		}

		y0 := int(math.Floor(srcY))
		y1 := y0 + 1
		if y1 >= srcH {
			y1 = srcH - 1
		}
		fy := srcY - float64(y0)

		for tx := 0; tx < targetW; tx++ {
			srcX := (float64(tx) + 0.5)*scaleX - 0.5
			if srcX < 0 {
				srcX = 0
			}
			if srcX > float64(srcW-1) {
				srcX = float64(srcW - 1)
			}

			x0 := int(math.Floor(srcX))
			x1 := x0 + 1
			if x1 >= srcW {
				x1 = srcW - 1
			}
			fx := srcX - float64(x0)

			// 4 neighbor pixel values
			p00 := float64(src.GrayAt(bounds.Min.X+x0, bounds.Min.Y+y0).Y)
			p10 := float64(src.GrayAt(bounds.Min.X+x1, bounds.Min.Y+y0).Y)
			p01 := float64(src.GrayAt(bounds.Min.X+x0, bounds.Min.Y+y1).Y)
			p11 := float64(src.GrayAt(bounds.Min.X+x1, bounds.Min.Y+y1).Y)

			// Bilinear interpolation
			val := (1.0-fx)*(1.0-fy)*p00 +
				fx*(1.0-fy)*p10 +
				(1.0-fx)*fy*p01 +
				fx*fy*p11

			vInt := int(math.Round(val))
			if vInt < 0 {
				vInt = 0
			}
			if vInt > 255 {
				vInt = 255
			}

			dst.SetGray(tx, ty, color.Gray{Y: uint8(vInt)})
		}
	}

	return dst
}

// ResizeBilinearTensor resamples a [Channels x Height x Width] Tensor to [Channels x targetH x targetW]
// using sub-pixel bilinear interpolation.
func ResizeBilinearTensor(src *Tensor, targetW, targetH int) *Tensor {
	if src == nil {
		return nil
	}
	if targetW <= 0 || targetH <= 0 {
		return NewTensor(src.Channels, 0, 0)
	}

	dst := NewTensor(src.Channels, targetH, targetW)
	if src.Width == 0 || src.Height == 0 {
		return dst
	}

	scaleX := float32(src.Width) / float32(targetW)
	scaleY := float32(src.Height) / float32(targetH)

	for ty := 0; ty < targetH; ty++ {
		srcY := (float32(ty) + 0.5)*scaleY - 0.5
		if srcY < 0 {
			srcY = 0
		}
		if srcY > float32(src.Height-1) {
			srcY = float32(src.Height - 1)
		}

		y0 := int(math.Floor(float64(srcY)))
		y1 := y0 + 1
		if y1 >= src.Height {
			y1 = src.Height - 1
		}
		fy := srcY - float32(y0)

		for tx := 0; tx < targetW; tx++ {
			srcX := (float32(tx) + 0.5)*scaleX - 0.5
			if srcX < 0 {
				srcX = 0
			}
			if srcX > float32(src.Width-1) {
				srcX = float32(src.Width - 1)
			}

			x0 := int(math.Floor(float64(srcX)))
			x1 := x0 + 1
			if x1 >= src.Width {
				x1 = src.Width - 1
			}
			fx := srcX - float32(x0)

			for c := 0; c < src.Channels; c++ {
				p00 := src.Get(c, y0, x0)
				p10 := src.Get(c, y0, x1)
				p01 := src.Get(c, y1, x0)
				p11 := src.Get(c, y1, x1)

				val := (1.0-fx)*(1.0-fy)*p00 +
					fx*(1.0-fy)*p10 +
					(1.0-fx)*fy*p01 +
					fx*fy*p11

				if val < 0.0 {
					val = 0.0
				}
				if val > 1.0 {
					val = 1.0
				}
				dst.Set(c, ty, tx, val)
			}
		}
	}

	return dst
}

// sampleBilinearGray interpolates pixel luminosity from continuous coordinates (xSrc, ySrc).
func sampleBilinearGray(src *image.Gray, xSrc, ySrc float64) uint8 {
	bounds := src.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()
	if srcW == 0 || srcH == 0 {
		return 0
	}
	if xSrc < 0 || xSrc > float64(srcW-1) || ySrc < 0 || ySrc > float64(srcH-1) {
		return 0
	}

	x0 := int(math.Floor(xSrc))
	x1 := x0 + 1
	if x1 >= srcW {
		x1 = srcW - 1
	}
	fx := xSrc - float64(x0)

	y0 := int(math.Floor(ySrc))
	y1 := y0 + 1
	if y1 >= srcH {
		y1 = srcH - 1
	}
	fy := ySrc - float64(y0)

	p00 := float64(src.GrayAt(bounds.Min.X+x0, bounds.Min.Y+y0).Y)
	p10 := float64(src.GrayAt(bounds.Min.X+x1, bounds.Min.Y+y0).Y)
	p01 := float64(src.GrayAt(bounds.Min.X+x0, bounds.Min.Y+y1).Y)
	p11 := float64(src.GrayAt(bounds.Min.X+x1, bounds.Min.Y+y1).Y)

	val := (1.0-fx)*(1.0-fy)*p00 +
		fx*(1.0-fy)*p10 +
		(1.0-fx)*fy*p01 +
		fx*fy*p11

	vInt := int(math.Round(val))
	if vInt < 0 {
		vInt = 0
	}
	if vInt > 255 {
		vInt = 255
	}
	return uint8(vInt)
}

// RotateImage rotates an *image.Gray around its center (cx, cy) by angleDeg degrees
// using backward continuous coordinate rotation with bilinear sampling.
func RotateImage(src *image.Gray, angleDeg float64) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	if w == 0 || h == 0 || angleDeg == 0 {
		dst := image.NewGray(bounds)
		draw.Draw(dst, bounds, src, bounds.Min, draw.Src)
		return dst
	}

	dst := image.NewGray(bounds)
	cx := float64(w-1) / 2.0
	cy := float64(h-1) / 2.0

	rad := angleDeg * math.Pi / 180.0
	cosA := math.Cos(rad)
	sinA := math.Sin(rad)

	for y := 0; y < h; y++ {
		dy := float64(y) - cy
		for x := 0; x < w; x++ {
			dx := float64(x) - cx

			// Backward rotation mapping: [x_src, y_src]^T = R(-theta) [dx, dy]^T + [cx, cy]^T
			xSrc := cx + dx*cosA + dy*sinA
			ySrc := cy - dx*sinA + dy*cosA

			pix := sampleBilinearGray(src, xSrc, ySrc)
			dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: pix})
		}
	}

	return dst
}

// ShiftImage translates an *image.Gray by (dx, dy) pixels, filling exposed margins with 0 (black).
func ShiftImage(src *image.Gray, dx, dy int) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	dst := image.NewGray(bounds)
	if w == 0 || h == 0 {
		return dst
	}

	for y := 0; y < h; y++ {
		srcY := y - dy
		for x := 0; x < w; x++ {
			srcX := x - dx
			if srcX >= 0 && srcX < w && srcY >= 0 && srcY < h {
				p := src.GrayAt(bounds.Min.X+srcX, bounds.Min.Y+srcY)
				dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, p)
			} else {
				dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: 0})
			}
		}
	}

	return dst
}

// ShearImage applies affine horizontal slant shear: x_src = x - (y - cy) * shearX.
func ShearImage(src *image.Gray, shearX float64) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	if w == 0 || h == 0 || shearX == 0 {
		dst := image.NewGray(bounds)
		draw.Draw(dst, bounds, src, bounds.Min, draw.Src)
		return dst
	}

	dst := image.NewGray(bounds)
	cy := float64(h-1) / 2.0

	for y := 0; y < h; y++ {
		dy := float64(y) - cy
		for x := 0; x < w; x++ {
			xSrc := float64(x) - dy*shearX
			ySrc := float64(y)

			pix := sampleBilinearGray(src, xSrc, ySrc)
			dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: pix})
		}
	}

	return dst
}

// MorphDilation applies a 3x3 maximum filter for pen stroke thickening.
func MorphDilation(src *image.Gray) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	dst := image.NewGray(bounds)
	if w == 0 || h == 0 {
		return dst
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var maxVal uint8 = 0
			for dy := -1; dy <= 1; dy++ {
				ny := y + dy
				if ny < 0 || ny >= h {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					nx := x + dx
					if nx < 0 || nx >= w {
						continue
					}
					p := src.GrayAt(bounds.Min.X+nx, bounds.Min.Y+ny).Y
					if p > maxVal {
						maxVal = p
					}
				}
			}
			dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: maxVal})
		}
	}

	return dst
}

// MorphErosion applies a 3x3 minimum filter for pen stroke thinning.
func MorphErosion(src *image.Gray) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	dst := image.NewGray(bounds)
	if w == 0 || h == 0 {
		return dst
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var minVal uint8 = 255
			// Clamp the 3x3 window to the canvas (replicate-edge) instead of treating
			// out-of-bounds neighbours as black. Zero-padding forced every border pixel to 0
			// regardless of its value, carving a 1px black frame out of each eroded variant.
			for dy := -1; dy <= 1; dy++ {
				ny := clamp(y+dy, 0, h-1)
				for dx := -1; dx <= 1; dx++ {
					nx := clamp(x+dx, 0, w-1)
					p := src.GrayAt(bounds.Min.X+nx, bounds.Min.Y+ny).Y
					if p < minVal {
						minVal = p
					}
				}
			}
			dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: minVal})
		}
	}

	return dst
}

// ScaleImage rescales an *image.Gray about its center by independent horizontal and vertical
// factors using backward mapping with bilinear sampling. Factors above 1.0 magnify the drawing.
//
// Only factors >= 1.0 are safe for augmentation here: the backward map reads from a sub-region of
// the source, so nothing is ever sampled out of bounds. A factor below 1.0 would need source
// pixels outside the canvas and would silently clip the drawing to black.
func ScaleImage(src *image.Gray, scaleX, scaleY float64) *image.Gray {
	if src == nil {
		return nil
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	if w == 0 || h == 0 || (scaleX == 1.0 && scaleY == 1.0) {
		dst := image.NewGray(bounds)
		draw.Draw(dst, bounds, src, bounds.Min, draw.Src)
		return dst
	}
	if scaleX <= 0 {
		scaleX = 1.0
	}
	if scaleY <= 0 {
		scaleY = 1.0
	}

	dst := image.NewGray(bounds)
	cx := float64(w-1) / 2.0
	cy := float64(h-1) / 2.0

	for y := 0; y < h; y++ {
		ySrc := cy + (float64(y)-cy)/scaleY
		for x := 0; x < w; x++ {
			xSrc := cx + (float64(x)-cx)/scaleX
			pix := sampleBilinearGray(src, xSrc, ySrc)
			dst.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: pix})
		}
	}

	return dst
}

// --- 6.7 15-VARIANT GEOMETRY & MORPHOLOGY DATA AUGMENTOR ---

// AugmentImage generates 15 geometric and morphological variants per training image:
//  1. Original image
//  2-5. Rotations: -15 deg, +15 deg, -10 deg, +10 deg
//  6-9. Aspect / scale jitter: wider, taller, larger, and a mild wide-and-tall stretch
//  10-11. Combined rotate + shear (slant with tilt)
//  12-13. Horizontal shears: -0.20, +0.20
//  14. Morphological dilation (stroke thickening)
//  15. Morphological erosion (stroke thinning)
//
// Pure translations were deliberately removed. Every sample is bounding-box cropped and
// re-centered downstream (FindBoundingBox -> PadAndCenter), which exactly undoes a translation,
// so the six shift variants were producing byte-identical copies of the original. That meant 40%
// of the training set was duplicated originals: it inflated the sample count, biased the model
// toward the un-augmented pose, and contributed nothing to invariance. The replacements below all
// change the bounding box shape or the stroke statistics, so they survive normalization and teach
// real aspect-ratio, slant and stroke-width invariance.
func AugmentImage(src *image.Gray) []*image.Gray {
	if src == nil {
		return nil
	}
	variants := make([]*image.Gray, 0, 15)

	// 1. Original
	orig := image.NewGray(src.Bounds())
	draw.Draw(orig, src.Bounds(), src, src.Bounds().Min, draw.Src)
	variants = append(variants, orig)

	// 2-5. Rotations
	variants = append(variants, RotateImage(src, -15.0))
	variants = append(variants, RotateImage(src, 15.0))
	variants = append(variants, RotateImage(src, -10.0))
	variants = append(variants, RotateImage(src, 10.0))

	// 6-9. Aspect ratio and scale jitter (all factors >= 1.0, see ScaleImage)
	variants = append(variants, ScaleImage(src, 1.25, 1.00)) // wider strokes/shape
	variants = append(variants, ScaleImage(src, 1.00, 1.25)) // taller strokes/shape
	variants = append(variants, ScaleImage(src, 1.15, 1.15)) // uniformly larger
	variants = append(variants, ScaleImage(src, 1.30, 1.10)) // mild wide stretch

	// 10-11. Combined tilt + slant, for poses neither rotation nor shear alone covers
	variants = append(variants, ShearImage(RotateImage(src, -8.0), 0.12))
	variants = append(variants, ShearImage(RotateImage(src, 8.0), -0.12))

	// 12-13. Horizontal Shears
	variants = append(variants, ShearImage(src, -0.20))
	variants = append(variants, ShearImage(src, 0.20))

	// 14. Morphological Dilation
	variants = append(variants, MorphDilation(src))

	// 15. Morphological Erosion
	variants = append(variants, MorphErosion(src))

	return variants
}

// --- 6.8 STRATIFIED TRAIN / VALIDATION DATASET SPLITTER ---

// TrainTestSplit partitions samples into stratified train and validation sets ensuring
// equal/proportional class representation in both splits according to testRatio.
//
// Requirements:
// 1. Group samples by class label.
// 2. For each class group of size N_c, assign floor(N_c * testRatio) items to validation set and remaining to train set.
// 3. Shuffle final splits with deterministic random seed.
func TrainTestSplit(items []ImageItem, testRatio float64, seed int64) ([]ImageItem, []ImageItem) {
	if len(items) == 0 {
		return nil, nil
	}
	if testRatio <= 0 {
		train := make([]ImageItem, len(items))
		copy(train, items)
		return train, nil
	}
	if testRatio >= 1.0 {
		val := make([]ImageItem, len(items))
		copy(val, items)
		return nil, val
	}

	// 1. Group samples by class index
	classBuckets := make(map[int][]ImageItem)
	for _, item := range items {
		classBuckets[item.ClassIndex] = append(classBuckets[item.ClassIndex], item)
	}

	rng := rand.New(rand.NewSource(seed))

	var trainSet []ImageItem
	var valSet []ImageItem

	// Sort class keys for 100% reproducible splitting
	var classKeys []int
	for k := range classBuckets {
		classKeys = append(classKeys, k)
	}
	sort.Ints(classKeys)

	for _, k := range classKeys {
		bucket := classBuckets[k]
		n := len(bucket)
		if n == 0 {
			continue
		}

		// Shuffle bucket elements before splitting
		shuffled := make([]ImageItem, n)
		copy(shuffled, bucket)
		rng.Shuffle(n, func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		nVal := int(math.Floor(float64(n) * testRatio))
		if nVal > n {
			nVal = n
		}

		valSet = append(valSet, shuffled[:nVal]...)
		trainSet = append(trainSet, shuffled[nVal:]...)
	}

	// 3. Shuffle final splits with deterministic random seed
	rng.Shuffle(len(trainSet), func(i, j int) {
		trainSet[i], trainSet[j] = trainSet[j], trainSet[i]
	})
	rng.Shuffle(len(valSet), func(i, j int) {
		valSet[i], valSet[j] = valSet[j], valSet[i]
	})

	return trainSet, valSet
}

// ImageAuditResult contains quality analysis for an individual image.
type ImageAuditResult struct {
	Path            string
	Class           string
	ClassIndex      int
	IsCorrupt       bool
	IsBlank         bool
	IsTiny          bool
	ForegroundCount int
	BBoxWidth       int
	BBoxHeight      int
	AspectRatio     float64
	StrokeDensity   float64
}

// ClassAuditStats holds aggregated metrics for a single class.
type ClassAuditStats struct {
	ClassName    string
	ClassIndex   int
	TotalSamples int
	ValidCount   int
	CorruptCount int
	BlankCount   int
	TinyCount    int
	AvgBBoxW     float64
	AvgBBoxH     float64
	AvgAspect    float64
	AvgDensity   float64
}

// DuplicateGroup tracks duplicate files sharing identical SHA-256 hash digests.
type DuplicateGroup struct {
	Hash  string
	Paths []string
}

// ComputeFileSHA256 computes the hexadecimal SHA-256 hash of a file.
func ComputeFileSHA256(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

// --- 6.9 AUTOMATED DATASET QUALITY, HEALTH & STROKE DENSITY AUDITOR ---

// DatasetAuditReport aggregates health metrics across all classes in the dataset.
type DatasetAuditReport struct {
	DataDir        string
	NumClasses     int
	Classes        []string
	TotalSamples   int
	ValidCount     int
	CorruptCount   int
	BlankCount     int
	TinyCount      int
	DuplicateFiles int
	Duplicates     []DuplicateGroup
	ClassStats     []ClassAuditStats
	Results        []ImageAuditResult
}

// AuditImage performs quality and bounding box analysis on a single image file:
// 1. Detects unreadable/corrupt image files.
// 2. Detects 100% blank images (0 foreground pixels).
// 3. Detects tiny outlier drawings (< 30 foreground pixels).
// 4. Computes bounding box dimensions, aspect ratio, and stroke density.
func AuditImage(path string, className string, classIndex int) ImageAuditResult {
	res := ImageAuditResult{
		Path:       path,
		Class:      className,
		ClassIndex: classIndex,
	}

	gray, err := LoadImageFromFile(path)
	if err != nil {
		res.IsCorrupt = true
		return res
	}

	bounds := gray.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	totalPixels := w * h
	if totalPixels == 0 {
		res.IsCorrupt = true
		return res
	}

	// Compute average intensity to determine background luminosity
	var sumLum uint64
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			sumLum += uint64(gray.GrayAt(x, y).Y)
		}
	}
	meanLum := float64(sumLum) / float64(totalPixels)
	isLightBackground := meanLum > 128.0

	minX, maxX := w, -1
	minY, maxY := h, -1
	fgCount := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pix := gray.GrayAt(x, y).Y
			var isForeground bool
			if isLightBackground {
				isForeground = pix < 240
			} else {
				isForeground = pix > 15
			}

			if isForeground {
				fgCount++
				relX := x - bounds.Min.X
				relY := y - bounds.Min.Y
				if relX < minX {
					minX = relX
				}
				if relX > maxX {
					maxX = relX
				}
				if relY < minY {
					minY = relY
				}
				if relY > maxY {
					maxY = relY
				}
			}
		}
	}

	res.ForegroundCount = fgCount
	if fgCount == 0 {
		res.IsBlank = true
		return res
	}
	if fgCount < 30 {
		res.IsTiny = true
	}

	bboxW := maxX - minX + 1
	bboxH := maxY - minY + 1
	res.BBoxWidth = bboxW
	res.BBoxHeight = bboxH

	if bboxH > 0 {
		res.AspectRatio = float64(bboxW) / float64(bboxH)
	} else {
		res.AspectRatio = 1.0
	}

	if bboxW*bboxH > 0 {
		res.StrokeDensity = float64(fgCount) / float64(bboxW*bboxH)
	}

	return res
}

// AuditDataset executes a complete health and quality scan over a dataset directory.
func AuditDataset(dataDir string) (*DatasetAuditReport, error) {
	ds, err := ScanDataset(dataDir)
	if err != nil {
		return nil, err
	}

	report := &DatasetAuditReport{
		DataDir:      dataDir,
		NumClasses:   ds.Metadata.NumClasses,
		Classes:      ds.Metadata.Classes,
		TotalSamples: len(ds.Samples),
	}

	fileHashMap := make(map[string][]string)
	classResults := make(map[int][]ImageAuditResult)
	for _, sample := range ds.Samples {
		res := AuditImage(sample.Path, sample.Class, sample.ClassIndex)
		report.Results = append(report.Results, res)
		classResults[sample.ClassIndex] = append(classResults[sample.ClassIndex], res)

		if hash, err := ComputeFileSHA256(sample.Path); err == nil {
			fileHashMap[hash] = append(fileHashMap[hash], sample.Path)
		}

		if res.IsCorrupt {
			report.CorruptCount++
		} else if res.IsBlank {
			report.BlankCount++
		} else if res.IsTiny {
			report.TinyCount++
		} else {
			report.ValidCount++
		}
	}

	// Aggregate duplicate SHA-256 groups
	for hash, paths := range fileHashMap {
		if len(paths) > 1 {
			report.Duplicates = append(report.Duplicates, DuplicateGroup{
				Hash:  hash,
				Paths: paths,
			})
			report.DuplicateFiles += len(paths) - 1
		}
	}
	sort.Slice(report.Duplicates, func(i, j int) bool {
		return report.Duplicates[i].Hash < report.Duplicates[j].Hash
	})

	for idx, clsName := range ds.Metadata.Classes {
		items := classResults[idx]
		stats := ClassAuditStats{
			ClassName:    clsName,
			ClassIndex:   idx,
			TotalSamples: len(items),
		}

		var sumW, sumH, sumAspect, sumDensity float64
		validForGeometry := 0

		for _, item := range items {
			if item.IsCorrupt {
				stats.CorruptCount++
			} else if item.IsBlank {
				stats.BlankCount++
			} else {
				if item.IsTiny {
					stats.TinyCount++
				} else {
					stats.ValidCount++
				}
				sumW += float64(item.BBoxWidth)
				sumH += float64(item.BBoxHeight)
				sumAspect += item.AspectRatio
				sumDensity += item.StrokeDensity
				validForGeometry++
			}
		}

		if validForGeometry > 0 {
			stats.AvgBBoxW = sumW / float64(validForGeometry)
			stats.AvgBBoxH = sumH / float64(validForGeometry)
			stats.AvgAspect = sumAspect / float64(validForGeometry)
			stats.AvgDensity = sumDensity / float64(validForGeometry)
		}

		report.ClassStats = append(report.ClassStats, stats)
	}

	return report, nil
}

// PrintAuditReport prints a clean tabular dataset audit and health report to stdout.
func PrintAuditReport(report *DatasetAuditReport) {
	fmt.Println("====================================================================================================")
	fmt.Println("                            DIAGONALNET DATASET AUDIT & HEALTH REPORT")
	fmt.Println("====================================================================================================")
	fmt.Printf(" Target Directory   : %s\n", report.DataDir)
	fmt.Printf(" Discovered Classes : %d %v (K=%d)\n", report.NumClasses, report.Classes, report.NumClasses)
	fmt.Printf(" Total Samples      : %d | Clean Valid: %d | Corrupt: %d | Blank: %d | Tiny Outliers: %d | Duplicates: %d\n",
		report.TotalSamples, report.ValidCount, report.CorruptCount, report.BlankCount, report.TinyCount, report.DuplicateFiles)
	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Printf(" %-15s | %7s | %5s | %7s | %5s | %4s | %-14s | %-10s | %s\n",
		"Class Name", "Samples", "Valid", "Corrupt", "Blank", "Tiny", "Avg BBox (WxH)", "Avg Aspect", "Stroke Dens")
	fmt.Println("----------------------------------------------------------------------------------------------------")

	var totalW, totalH, totalAspect, totalDensity float64
	validClasses := 0

	for _, s := range report.ClassStats {
		fmt.Printf(" %-15s | %7d | %5d | %7d | %5d | %4d | %5.1f x %-6.1f | %10.2f | %10.1f%%\n",
			s.ClassName, s.TotalSamples, s.ValidCount, s.CorruptCount, s.BlankCount, s.TinyCount,
			s.AvgBBoxW, s.AvgBBoxH, s.AvgAspect, s.AvgDensity*100.0)
		if s.ValidCount+s.TinyCount > 0 {
			totalW += s.AvgBBoxW
			totalH += s.AvgBBoxH
			totalAspect += s.AvgAspect
			totalDensity += s.AvgDensity
			validClasses++
		}
	}
	fmt.Println("----------------------------------------------------------------------------------------------------")
	avgW, avgH, avgAspect, avgDens := 0.0, 0.0, 0.0, 0.0
	if validClasses > 0 {
		avgW = totalW / float64(validClasses)
		avgH = totalH / float64(validClasses)
		avgAspect = totalAspect / float64(validClasses)
		avgDens = totalDensity / float64(validClasses)
	}
	fmt.Printf(" %-15s | %7d | %5d | %7d | %5d | %4d | %5.1f x %-6.1f | %10.2f | %10.1f%%\n",
		"SUMMARY", report.TotalSamples, report.ValidCount, report.CorruptCount, report.BlankCount, report.TinyCount,
		avgW, avgH, avgAspect, avgDens*100.0)
	fmt.Println("====================================================================================================")

	if len(report.Duplicates) > 0 {
		fmt.Printf("\n [DUPLICATES DETECTED] %d exact duplicate file instance(s) found across dataset:\n", report.DuplicateFiles)
		for _, dup := range report.Duplicates {
			fmt.Printf("  • SHA-256 [%s...]:\n", dup.Hash[:16])
			for _, p := range dup.Paths {
				fmt.Printf("      - %s\n", p)
			}
		}
		fmt.Println("====================================================================================================")
	}
}

// ============================================================================
// 7. 13-CHANNEL SPATIAL DIFFERENCE MANIFOLD CALCULUS
// ============================================================================

// --- 7.1 BOUNDARY CLAMPING & FAST ABSOLUTE VALUE PRIMITIVES ---

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

// --- 7.2 DIRECTIONAL OFFSETS: 4 IMMEDIATE DIAGONALS & 8 CHESS KNIGHT-MOVES ---

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

// --- 7.3 ROW-PARALLEL MULTI-CORE 13-CHANNEL MANIFOLD GENERATOR ---

// ComputeManifold transforms a flat grayscale image slice [h*w] into a 13-channel spatial difference manifold [13*h*w]
// parallelized row-by-row across runtime.NumCPU() Goroutines.
func ComputeManifold(input []float32, h, w int) []float32 {
	out := make([]float32, 13*h*w)
	ComputeManifoldIntoSlice(input, out, h, w)
	return out
}

// ComputeManifoldIntoSlice executes the 13-channel manifold transformation into a pre-allocated destination slice.
func ComputeManifoldIntoSlice(input []float32, out []float32, h, w int) {
	hw := h * w
	for y := 0; y < h; y++ {
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
}

// --- 7.4 TENSOR-LEVEL IN-PLACE & ALLOCATING MANIFOLD TRANSFORMERS ---

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
// 8. CONVOLUTIONAL, POOLING & DENSE NEURAL NETWORK LAYERS
// ============================================================================

// --- 8.1 2D CONVOLUTIONAL LAYER & ANALYTICAL JACOBIAN BACKPROPAGATION ---

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

	for cOut := 0; cOut < outC; cOut++ {
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

// BackwardInto executes analytical backpropagation into pre-allocated gradInput buffer.
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

	for cOut := 0; cOut < outC; cOut++ {
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

							wIdx := weightRowOffset + kx
							inIdx := inRowOffset + inX
							l.Weights.Grad[wIdx] += gy * input.Data[inIdx]
							gradInput.Data[inIdx] += gy * l.Weights.Data[wIdx]
						}
					}
				}
			}
		}
		l.Bias.Grad[cOut] += biasGrad
	}
}

// --- 8.2 2D MAX POOLING LAYER & SPARSE ARGMAX BACKWARD ROUTING ---

// MaxPool2DLayer performs 2D max pooling with ArgMax coordinate caching for exact sparse backpropagation.
type MaxPool2DLayer struct {
	KernelSize int
	ArgMax     []int   // Cached 1D input indices for the argmax of each output position
	LastInput  *Tensor // Cached input tensor
	LastOutput *Tensor // Cached output tensor
}

// NewMaxPool2DLayer constructs a MaxPool2DLayer.
func NewMaxPool2DLayer(kernelSize int) *MaxPool2DLayer {
	if kernelSize <= 0 {
		kernelSize = 2
	}
	return &MaxPool2DLayer{
		KernelSize: kernelSize,
	}
}

// Forward computes max pooling over non-overlapping KxK spatial windows and caches ArgMax indices.
func (l *MaxPool2DLayer) Forward(input *Tensor) *Tensor {
	l.LastInput = input
	k := l.KernelSize
	c := input.Channels
	hIn := input.Height
	wIn := input.Width

	hOut := hIn / k
	wOut := wIn / k
	if hOut <= 0 {
		hOut = 1
	}
	if wOut <= 0 {
		wOut = 1
	}

	output := NewTensor(c, hOut, wOut)
	l.LastOutput = output
	l.ArgMax = make([]int, len(output.Data))

	for ch := 0; ch < c; ch++ {
		inChOffset := ch * (hIn * wIn)
		outChOffset := ch * (hOut * wOut)

		for y := 0; y < hOut; y++ {
			inYStart := y * k
			outRowOffset := outChOffset + y*wOut

			for x := 0; x < wOut; x++ {
				inXStart := x * k
				outIdx := outRowOffset + x

				maxVal := float32(-math.MaxFloat32)
				maxIdx := -1

				for ky := 0; ky < k; ky++ {
					iy := inYStart + ky
					if iy >= hIn {
						continue
					}
					inRowOffset := inChOffset + iy*wIn

					for kx := 0; kx < k; kx++ {
						ix := inXStart + kx
						if ix >= wIn {
							continue
						}
						currIdx := inRowOffset + ix
						val := input.Data[currIdx]

						if val > maxVal || maxIdx == -1 {
							maxVal = val
							maxIdx = currIdx
						}
					}
				}

				output.Data[outIdx] = maxVal
				l.ArgMax[outIdx] = maxIdx
			}
		}
	}

	return output
}

// Backward routes output gradients directly to the cached ArgMax locations in input space.
func (l *MaxPool2DLayer) Backward(gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(l.LastInput.Channels, l.LastInput.Height, l.LastInput.Width)
	for i, goVal := range gradOutput.Data {
		if i < len(l.ArgMax) && l.ArgMax[i] >= 0 && l.ArgMax[i] < len(gradInput.Data) {
			gradInput.Data[l.ArgMax[i]] += goVal
		}
	}
	return gradInput
}

// --- 8.3 2D ADAPTIVE AVERAGE POOLING LAYER (SPATIAL BINNING) ---

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

// --- 8.4 FULLY CONNECTED (DENSE) LINEAR LAYER & JACOBIAN MATMUL ---

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

// --- 8.5 INVERTED BERNOULLI DROPOUT REGULARIZATION LAYER ---

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

// ReLU calculates the Rectified Linear Unit activation scalar: max(0, x).
func ReLU(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUGrad calculates the analytical gradient of ReLU: gy if x > 0 else 0.
func ReLUGrad(x, gy float32) float32 {
	if x > 0 {
		return gy
	}
	return 0
}

// --- 8.6 RECTIFIED LINEAR UNIT (RELU) ACTIVATION LAYER ---

// ReLULayer implements Rectified Linear Unit activation with analytical Jacobian backpropagation.
// Forward:  y_i = max(0, x_i)
// Backward: dL/dx_i = dL/dy_i if x_i > 0 else 0
type ReLULayer struct {
	LastInput []float32
}

// NewReLULayer constructs a new ReLU activation layer.
func NewReLULayer() *ReLULayer {
	return &ReLULayer{}
}

// Forward computes y_i = max(0, x_i) for flat slice inputs.
func (l *ReLULayer) Forward(input []float32) []float32 {
	out := make([]float32, len(input))
	l.ForwardInto(input, out)
	return out
}

// ForwardInto computes ReLU into a pre-allocated destination slice.
func (l *ReLULayer) ForwardInto(input []float32, output []float32) {
	if len(l.LastInput) != len(input) {
		l.LastInput = make([]float32, len(input))
	}
	copy(l.LastInput, input)

	for i, x := range input {
		if x > 0 {
			output[i] = x
		} else {
			output[i] = 0
		}
	}
}

// Backward computes analytical Jacobian gradient dL/dx_i = dL/dy_i if x_i > 0 else 0.
func (l *ReLULayer) Backward(gradOutput []float32) []float32 {
	gradInput := make([]float32, len(gradOutput))
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto computes ReLU analytical Jacobian gradient into a pre-allocated gradInput slice.
func (l *ReLULayer) BackwardInto(gradOutput []float32, gradInput []float32) {
	for i, gy := range gradOutput {
		if l.LastInput[i] > 0 {
			gradInput[i] = gy
		} else {
			gradInput[i] = 0
		}
	}
}

// ForwardTensor executes ReLU forward pass over a 3D Tensor.
func (l *ReLULayer) ForwardTensor(input *Tensor) *Tensor {
	out := NewTensor(input.Channels, input.Height, input.Width)
	l.ForwardTensorInto(input, out)
	return out
}

// ForwardTensorInto executes ReLU forward pass into a pre-allocated 3D Tensor.
func (l *ReLULayer) ForwardTensorInto(input *Tensor, output *Tensor) {
	if output.Channels != input.Channels || output.Height != input.Height || output.Width != input.Width {
		*output = *NewTensor(input.Channels, input.Height, input.Width)
	}
	l.ForwardInto(input.Data, output.Data)
}

// BackwardTensor executes ReLU analytical gradient backpropagation over a 3D Tensor.
func (l *ReLULayer) BackwardTensor(gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(gradOutput.Channels, gradOutput.Height, gradOutput.Width)
	l.BackwardTensorInto(gradOutput, gradInput)
	return gradInput
}

// BackwardTensorInto executes ReLU analytical gradient backpropagation into a pre-allocated 3D Tensor.
func (l *ReLULayer) BackwardTensorInto(gradOutput *Tensor, gradInput *Tensor) {
	if gradInput.Channels != gradOutput.Channels || gradInput.Height != gradOutput.Height || gradInput.Width != gradOutput.Width {
		*gradInput = *NewTensor(gradOutput.Channels, gradOutput.Height, gradOutput.Width)
	}
	l.BackwardInto(gradOutput.Data, gradInput.Data)
}

// LeakyReLU calculates the Leaky ReLU activation scalar: x if x > 0 else alpha * x.
func LeakyReLU(x, alpha float32) float32 {
	if x > 0 {
		return x
	}
	return alpha * x
}

// LeakyReLUGrad calculates the analytical gradient of Leaky ReLU: gy if x > 0 else alpha * gy.
func LeakyReLUGrad(x, gy, alpha float32) float32 {
	if x > 0 {
		return gy
	}
	return alpha * gy
}

// --- 8.7 LEAKY RECTIFIED LINEAR UNIT (LEAKY RELU) ACTIVATION LAYER ---

// LeakyReLULayer implements Leaky ReLU activation with analytical Jacobian backpropagation.
// Forward:  y_i = x_i if x_i > 0 else alpha * x_i
// Backward: dL/dx_i = dL/dy_i if x_i > 0 else alpha * dL/dy_i
type LeakyReLULayer struct {
	Alpha     float32
	LastInput []float32
}

// NewLeakyReLULayer constructs a new LeakyReLU activation layer with specified alpha slope (default 0.01).
func NewLeakyReLULayer(alpha float32) *LeakyReLULayer {
	if alpha <= 0 {
		alpha = 0.01
	}
	return &LeakyReLULayer{
		Alpha: alpha,
	}
}

// Forward computes y_i = x_i if x_i > 0 else alpha * x_i for flat slice inputs.
func (l *LeakyReLULayer) Forward(input []float32) []float32 {
	out := make([]float32, len(input))
	l.ForwardInto(input, out)
	return out
}

// ForwardInto computes LeakyReLU into a pre-allocated destination slice.
func (l *LeakyReLULayer) ForwardInto(input []float32, output []float32) {
	if len(l.LastInput) != len(input) {
		l.LastInput = make([]float32, len(input))
	}
	copy(l.LastInput, input)

	alpha := l.Alpha
	for i, x := range input {
		if x > 0 {
			output[i] = x
		} else {
			output[i] = alpha * x
		}
	}
}

// Backward computes analytical Jacobian gradient dL/dx_i = dL/dy_i if x_i > 0 else alpha * dL/dy_i.
func (l *LeakyReLULayer) Backward(gradOutput []float32) []float32 {
	gradInput := make([]float32, len(gradOutput))
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto computes LeakyReLU analytical Jacobian gradient into a pre-allocated gradInput slice.
func (l *LeakyReLULayer) BackwardInto(gradOutput []float32, gradInput []float32) {
	alpha := l.Alpha
	for i, gy := range gradOutput {
		if l.LastInput[i] > 0 {
			gradInput[i] = gy
		} else {
			gradInput[i] = alpha * gy
		}
	}
}

// ForwardTensor executes LeakyReLU forward pass over a 3D Tensor.
func (l *LeakyReLULayer) ForwardTensor(input *Tensor) *Tensor {
	out := NewTensor(input.Channels, input.Height, input.Width)
	l.ForwardTensorInto(input, out)
	return out
}

// ForwardTensorInto executes LeakyReLU forward pass into a pre-allocated 3D Tensor.
func (l *LeakyReLULayer) ForwardTensorInto(input *Tensor, output *Tensor) {
	if output.Channels != input.Channels || output.Height != input.Height || output.Width != input.Width {
		*output = *NewTensor(input.Channels, input.Height, input.Width)
	}
	l.ForwardInto(input.Data, output.Data)
}

// BackwardTensor executes LeakyReLU analytical gradient backpropagation over a 3D Tensor.
func (l *LeakyReLULayer) BackwardTensor(gradOutput *Tensor) *Tensor {
	gradInput := NewTensor(gradOutput.Channels, gradOutput.Height, gradOutput.Width)
	l.BackwardTensorInto(gradOutput, gradInput)
	return gradInput
}

// BackwardTensorInto executes LeakyReLU analytical gradient backpropagation into a pre-allocated 3D Tensor.
func (l *LeakyReLULayer) BackwardTensorInto(gradOutput *Tensor, gradInput *Tensor) {
	if gradInput.Channels != gradOutput.Channels || gradInput.Height != gradOutput.Height || gradInput.Width != gradOutput.Width {
		*gradInput = *NewTensor(gradOutput.Channels, gradOutput.Height, gradOutput.Width)
	}
	l.BackwardInto(gradOutput.Data, gradInput.Data)
}

// Softmax computes the numerically stable softmax probability distribution over an arbitrary 1D logits slice:
// m = max_j(z_j), e_i = exp(z_i - m), p_i = e_i / sum_{j=0}^{K-1} e_j
func Softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	probs := make([]float32, len(logits))
	SoftmaxInto(logits, probs)
	return probs
}

// SoftmaxInto computes numerically stable softmax into a pre-allocated destination slice.
func SoftmaxInto(logits []float32, probs []float32) {
	if len(logits) == 0 {
		return
	}
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sumExp float32
	for i, v := range logits {
		e := float32(math.Exp(float64(v - maxLogit)))
		probs[i] = e
		sumExp += e
	}
	if sumExp == 0 {
		sumExp = 1e-12
	}
	invSum := 1.0 / sumExp
	for i := range probs {
		probs[i] *= invSum
	}
}

// SoftmaxGrad computes analytical Jacobian vector product for Softmax:
// dL/dz_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
func SoftmaxGrad(probs []float32, gradOutput []float32) []float32 {
	gradInput := make([]float32, len(probs))
	SoftmaxGradInto(probs, gradOutput, gradInput)
	return gradInput
}

// SoftmaxGradInto computes Softmax analytical Jacobian gradient into a pre-allocated slice.
func SoftmaxGradInto(probs []float32, gradOutput []float32, gradInput []float32) {
	var dot float32
	for i := range probs {
		dot += gradOutput[i] * probs[i]
	}
	for i := range probs {
		gradInput[i] = probs[i] * (gradOutput[i] - dot)
	}
}

// --- 8.8 NUMERICALLY STABLE SOFTMAX PROBABILITY LAYER ---

// SoftmaxLayer implements a stateful Softmax probability distribution layer with analytical Jacobian autograd.
type SoftmaxLayer struct {
	LastProbs []float32
}

// NewSoftmaxLayer constructs a new Softmax layer.
func NewSoftmaxLayer() *SoftmaxLayer {
	return &SoftmaxLayer{}
}

// Forward executes the numerically stable Softmax forward pass.
func (l *SoftmaxLayer) Forward(logits []float32) []float32 {
	out := make([]float32, len(logits))
	l.ForwardInto(logits, out)
	return out
}

// ForwardInto executes Softmax forward pass into a pre-allocated destination slice.
func (l *SoftmaxLayer) ForwardInto(logits []float32, output []float32) {
	if len(l.LastProbs) != len(logits) {
		l.LastProbs = make([]float32, len(logits))
	}
	SoftmaxInto(logits, output)
	copy(l.LastProbs, output)
}

// Backward computes analytical Jacobian backpropagation gradient: dL/dz_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j)).
func (l *SoftmaxLayer) Backward(gradOutput []float32) []float32 {
	gradInput := make([]float32, len(gradOutput))
	l.BackwardInto(gradOutput, gradInput)
	return gradInput
}

// BackwardInto computes Softmax analytical Jacobian backpropagation into pre-allocated gradInput slice.
func (l *SoftmaxLayer) BackwardInto(gradOutput []float32, gradInput []float32) {
	SoftmaxGradInto(l.LastProbs, gradOutput, gradInput)
}

// --- 8.9 CATEGORICAL CROSS-ENTROPY LOSS & ANALYTICAL LOGIT GRADIENTS ---

// CategoricalCrossEntropy computes cross-entropy loss for target class index:
// L = -ln(p_target + eps), eps = 1e-15
func CategoricalCrossEntropy(probs []float32, targetClass int) float32 {
	const eps = 1e-15
	if targetClass < 0 || targetClass >= len(probs) {
		return 0
	}
	p := float64(probs[targetClass])
	if p < 0 {
		p = 0
	}
	return float32(-math.Log(p + eps))
}

// CategoricalCrossEntropyOneHot computes loss for one-hot target distribution:
// L = -sum_{k=0}^{K-1} y_k * ln(p_k + eps)
func CategoricalCrossEntropyOneHot(probs []float32, targetOneHot []float32) float32 {
	const eps = 1e-15
	var totalLoss float64
	for i, y := range targetOneHot {
		if y > 0 && i < len(probs) {
			p := float64(probs[i])
			if p < 0 {
				p = 0
			}
			totalLoss -= float64(y) * math.Log(p+eps)
		}
	}
	return float32(totalLoss)
}

// SoftmaxCrossEntropyGrad computes analytical pre-softmax logit gradients:
// dL/dz_i = p_i - 1(i == target)
func SoftmaxCrossEntropyGrad(probs []float32, targetClass int) []float32 {
	grad := make([]float32, len(probs))
	SoftmaxCrossEntropyGradInto(probs, targetClass, grad)
	return grad
}

// SoftmaxCrossEntropyGradInto computes analytical pre-softmax logit gradients into a pre-allocated slice:
// dL/dz_i = p_i - 1(i == target)
func SoftmaxCrossEntropyGradInto(probs []float32, targetClass int, gradLogits []float32) {
	for i, p := range probs {
		if i == targetClass {
			gradLogits[i] = p - 1.0
		} else {
			gradLogits[i] = p
		}
	}
}

// SoftmaxCrossEntropyGradOneHotInto computes analytical gradients for soft/one-hot distributions:
// dL/dz_i = p_i - y_i
func SoftmaxCrossEntropyGradOneHotInto(probs []float32, targetOneHot []float32, gradLogits []float32) {
	for i, p := range probs {
		if i < len(targetOneHot) {
			gradLogits[i] = p - targetOneHot[i]
		} else {
			gradLogits[i] = p
		}
	}
}

// CategoricalCrossEntropyLoss implements the Categorical Cross-Entropy loss criterion
// with analytical gradients with respect to pre-softmax logits.
// Loss:              L = -ln(p_target + eps)
// Logit Derivative:  dL/dz_i = p_i - 1(i == target)
type CategoricalCrossEntropyLoss struct {
	Eps float64
}

// NewCategoricalCrossEntropyLoss constructs a new Categorical Cross-Entropy loss criterion.
func NewCategoricalCrossEntropyLoss() *CategoricalCrossEntropyLoss {
	return &CategoricalCrossEntropyLoss{
		Eps: 1e-15,
	}
}

// Forward computes loss given predicted class probabilities and target class label: L = -ln(p_target + eps)
func (c *CategoricalCrossEntropyLoss) Forward(probs []float32, targetClass int) float32 {
	if targetClass < 0 || targetClass >= len(probs) {
		return 0
	}
	p := float64(probs[targetClass])
	if p < 0 {
		p = 0
	}
	return float32(-math.Log(p + c.Eps))
}

// Backward computes analytical gradient of loss w.r.t pre-softmax logits: dL/dz_i = p_i - 1(i == target)
func (c *CategoricalCrossEntropyLoss) Backward(probs []float32, targetClass int) []float32 {
	return SoftmaxCrossEntropyGrad(probs, targetClass)
}

// BackwardInto computes analytical gradient of loss w.r.t pre-softmax logits into a pre-allocated buffer.
func (c *CategoricalCrossEntropyLoss) BackwardInto(probs []float32, targetClass int, gradLogits []float32) {
	SoftmaxCrossEntropyGradInto(probs, targetClass, gradLogits)
}

// LossAndGrad computes both the loss scalar and the analytical pre-softmax logit gradients in a single pass.
func (c *CategoricalCrossEntropyLoss) LossAndGrad(logits []float32, targetClass int) (float32, []float32, []float32) {
	probs := Softmax(logits)
	loss := c.Forward(probs, targetClass)
	grad := c.Backward(probs, targetClass)
	return loss, probs, grad
}

// LossAndGradInto computes loss, probabilities, and logit gradients into pre-allocated slices.
func (c *CategoricalCrossEntropyLoss) LossAndGradInto(logits []float32, targetClass int, probs []float32, gradLogits []float32) float32 {
	SoftmaxInto(logits, probs)
	loss := c.Forward(probs, targetClass)
	c.BackwardInto(probs, targetClass, gradLogits)
	return loss
}

// ============================================================================
// 9. END-TO-END MODEL ARCHITECTURE & MULTI-CORE BATCH TRAINER
// ============================================================================

// Sample encapsulates an input feature tensor and integer class target.
type Sample struct {
	Input       *Tensor // [1 x H x W] or [13 x H x W]
	TargetClass int     // Integer label in [0, K-1]
}

// DiagonalNetModel represents the complete neural network architecture for DiagonalNet:
//
//	13-Channel Manifold [13 x S x S]
//	  -> Conv2D (13->16, K=3, S=1, P=1) -> ReLU -> MaxPool (2x2)
//	  -> Conv2D (16->32, K=3, S=1, P=1) -> ReLU -> MaxPool (2x2)
//	  -> AdaptiveAvgPool (4x4)                     [32*4*4 = 512 features]
//	  -> Linear (512->128) -> ReLU -> Dropout (p=0.2)
//	  -> Linear (128->K)
//
// Two stride-1 convolution stages separated by max pooling give the network a genuine
// feature hierarchy (edges -> stroke junctions -> shape parts), and the 128-unit hidden layer
// makes the classifier head non-linear. The previous single Conv(stride 2) -> AvgPool -> Linear
// stack collapsed each feature map into 16 coarse 4x4 averages and then fed them straight to a
// linear readout, which is close to a linear model over blurred features and badly underfits.
// Dropout now sits after the hidden ReLU (regularizing a learned representation) rather than
// directly on the pooled convolution features, where it was just injecting input noise.
type DiagonalNetModel struct {
	NumClasses int

	Conv1 *Conv2DLayer
	ReLU1 *ReLULayer
	Pool1 *MaxPool2DLayer

	Conv2 *Conv2DLayer
	ReLU2 *ReLULayer
	Pool2 *MaxPool2DLayer

	Pool *AdaptiveAvgPool2DLayer

	FC1     *LinearLayer
	ReLU3   *ReLULayer
	Dropout *DropoutLayer
	FC      *LinearLayer

	LossFn *CategoricalCrossEntropyLoss
}

// DiagonNetModel is an alias for DiagonalNetModel for backward compatibility.
type DiagonNetModel = DiagonalNetModel

// DiagonalNet feature-stack dimensions.
const (
	diagonalConv1Channels = 16
	diagonalConv2Channels = 32
	diagonalPoolTarget    = 4
	diagonalHiddenUnits   = 128

	diagonConv1Channels = diagonalConv1Channels
	diagonConv2Channels = diagonalConv2Channels
	diagonPoolTarget    = diagonalPoolTarget
	diagonHiddenUnits   = diagonalHiddenUnits
)

// NewDiagonalNetModel constructs a DiagonalNet classification model configured dynamically for K classes.
func NewDiagonalNetModel(numClasses int, rng *rand.Rand) *DiagonalNetModel {
	if numClasses < 2 {
		numClasses = 2
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	conv1RNG := rand.New(rand.NewSource(rng.Int63()))
	conv2RNG := rand.New(rand.NewSource(rng.Int63()))
	fc1RNG := rand.New(rand.NewSource(rng.Int63()))
	fc2RNG := rand.New(rand.NewSource(rng.Int63()))
	dropRNG := rand.New(rand.NewSource(rng.Int63()))

	flatDim := diagonalConv2Channels * diagonalPoolTarget * diagonalPoolTarget

	return &DiagonalNetModel{
		NumClasses: numClasses,

		Conv1: NewConv2DLayer(13, diagonalConv1Channels, 3, 1, 1, conv1RNG),
		ReLU1: NewReLULayer(),
		Pool1: NewMaxPool2DLayer(2),

		Conv2: NewConv2DLayer(diagonalConv1Channels, diagonalConv2Channels, 3, 1, 1, conv2RNG),
		ReLU2: NewReLULayer(),
		Pool2: NewMaxPool2DLayer(2),

		Pool: NewAdaptiveAvgPool2DLayer(diagonalPoolTarget, diagonalPoolTarget),

		FC1:     NewLinearLayer(flatDim, diagonalHiddenUnits, fc1RNG),
		ReLU3:   NewReLULayer(),
		Dropout: NewDropoutLayer(0.2, dropRNG),
		FC:      NewLinearLayer(diagonalHiddenUnits, numClasses, fc2RNG),

		LossFn: NewCategoricalCrossEntropyLoss(),
	}
}

// NewDiagonNetModel is an alias for NewDiagonalNetModel for backward compatibility.
func NewDiagonNetModel(numClasses int, rng *rand.Rand) *DiagonalNetModel {
	return NewDiagonalNetModel(numClasses, rng)
}

// Parameters returns all trainable parameter buffers in the model.
//
// The order is part of the on-disk checkpoint format: SaveModelWeights and LoadModelWeights walk
// this slice sequentially, so appending or reordering entries invalidates existing .bin files.
func (m *DiagonalNetModel) Parameters() []*Parameter {
	return []*Parameter{
		m.Conv1.Weights, m.Conv1.Bias,
		m.Conv2.Weights, m.Conv2.Bias,
		m.FC1.Weights, m.FC1.Biases,
		m.FC.Weights, m.FC.Biases,
	}
}

// ZeroGrad resets analytical Jacobian gradient buffers for all parameters to zero.
func (m *DiagonalNetModel) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

// SetTraining toggles training vs evaluation mode (affecting Dropout regularization).
func (m *DiagonalNetModel) SetTraining(training bool) {
	m.Dropout.Training = training
}

// CloneForWorker constructs an isolated model replica for a parallel batch worker,
// with independent gradient and layer state buffers.
func (m *DiagonalNetModel) CloneForWorker(workerID int) *DiagonalNetModel {
	// Each replica gets its own RNG stream so the workers draw independent dropout masks,
	// while staying reproducible across runs for a given worker id.
	rng := rand.New(rand.NewSource(int64(1000 + workerID*37)))
	replica := NewDiagonalNetModel(m.NumClasses, rng)
	replica.SyncWeightsFrom(m)
	return replica
}

// SyncWeightsFrom copies trainable weight vectors from the master model into the replica.
func (m *DiagonalNetModel) SyncWeightsFrom(master *DiagonalNetModel) {
	dst := m.Parameters()
	src := master.Parameters()
	for i := range dst {
		copy(dst[i].Data, src[i].Data)
	}
}

// SnapshotWeights creates deep copies of all trainable parameter weights in the model.
func (m *DiagonalNetModel) SnapshotWeights() [][]float32 {
	params := m.Parameters()
	snapshot := make([][]float32, len(params))
	for i, p := range params {
		if p != nil {
			snapshot[i] = make([]float32, len(p.Data))
			copy(snapshot[i], p.Data)
		}
	}
	return snapshot
}

// RestoreWeights restores trainable parameter weights from a saved snapshot.
func (m *DiagonalNetModel) RestoreWeights(snapshot [][]float32) {
	params := m.Parameters()
	for i, p := range params {
		if p != nil && i < len(snapshot) && snapshot[i] != nil {
			copy(p.Data, snapshot[i])
		}
	}
}

// forwardFeatures runs the shared convolutional trunk and dense head, returning the class logits.
// The layers cache their own activations, so ForwardBackward can immediately backpropagate.
func (m *DiagonalNetModel) forwardFeatures(input *Tensor) []float32 {
	var manifold *Tensor
	if input.Channels == 1 {
		manifold = ComputeManifoldTensor(input)
	} else {
		manifold = input
	}

	// Convolutional feature hierarchy
	c1 := m.Conv1.Forward(manifold) // [16 x S x S]
	a1 := m.ReLU1.ForwardTensor(c1)
	p1 := m.Pool1.Forward(a1)    // [16 x S/2 x S/2]
	c2 := m.Conv2.Forward(p1)    // [32 x S/2 x S/2]
	a2 := m.ReLU2.ForwardTensor(c2)
	p2 := m.Pool2.Forward(a2)    // [32 x S/4 x S/4]
	pooled := m.Pool.Forward(p2) // [32 x 4 x 4]

	// Non-linear dense classifier head
	h := m.FC1.Forward(pooled.Data) // [128]
	hAct := m.ReLU3.Forward(h)
	hDrop := m.Dropout.Forward(hAct)
	return m.FC.Forward(hDrop) // [K]
}

// Forward executes the full model inference forward pass, returning unnormalized class logits.
func (m *DiagonalNetModel) Forward(input *Tensor) []float32 {
	return m.forwardFeatures(input)
}

// ForwardBackward executes forward evaluation, cross-entropy loss computation, and full analytical Jacobian
// backpropagation through all layers, accumulating gradients into parameter buffers.
func (m *DiagonalNetModel) ForwardBackward(input *Tensor, targetClass int) (float32, []float32) {
	// 1. Forward Pass
	logits := m.forwardFeatures(input)

	// 2. Numerically stable Softmax & Categorical Cross-Entropy Loss
	probs := Softmax(logits)
	loss := m.LossFn.Forward(probs, targetClass)

	// 3. Analytical pre-softmax logit gradient: dL/dz_i = p_i - 1(i == target)
	gradLogits := SoftmaxCrossEntropyGrad(probs, targetClass)

	// 4. Dense head: Linear -> Dropout -> ReLU -> Linear
	gradDrop := m.FC.Backward(gradLogits)    // [128]
	gradAct := m.Dropout.Backward(gradDrop)  // [128]
	gradHidden := m.ReLU3.Backward(gradAct)  // [128]
	gradPooled := m.FC1.Backward(gradHidden) // [32*4*4]

	// 5. Adaptive Average Pooling Analytical Backpropagation
	pooledGrad := &Tensor{
		Data:     gradPooled,
		Channels: m.Conv2.OutChannels,
		Height:   m.Pool.TargetH,
		Width:    m.Pool.TargetW,
	}
	gradP2 := m.Pool.Backward(pooledGrad) // [32 x S/4 x S/4]

	// 6. Second convolutional stage: MaxPool -> ReLU -> Conv
	gradA2 := m.Pool2.Backward(gradP2)
	gradC2 := m.ReLU2.BackwardTensor(gradA2)
	gradP1 := m.Conv2.Backward(gradC2) // [16 x S/2 x S/2]

	// 7. First convolutional stage: MaxPool -> ReLU -> Conv
	gradA1 := m.Pool1.Backward(gradP1)
	gradC1 := m.ReLU1.BackwardTensor(gradA1)
	m.Conv1.Backward(gradC1)

	return loss, probs
}

// BatchTrainer coordinates multi-threaded data-parallel batch training across N worker model replicas.
type BatchTrainer struct {
	MasterModel *DiagonalNetModel
	Optimizer   *AdamOptimizer
	NumWorkers  int
	Workers     []*DiagonalNetModel
}

// NewBatchTrainer constructs a BatchTrainer coordinating N worker model replicas (scaled to runtime.NumCPU()).
func NewBatchTrainer(master *DiagonalNetModel, optimizer *AdamOptimizer, numWorkers int) *BatchTrainer {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
		if numWorkers <= 0 {
			numWorkers = 1
		}
	}

	workers := make([]*DiagonalNetModel, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workers[i] = master.CloneForWorker(i)
	}

	return &BatchTrainer{
		MasterModel: master,
		Optimizer:   optimizer,
		NumWorkers:  numWorkers,
		Workers:     workers,
	}
}

// TrainBatch executes concurrent data-parallel batch training across N worker replicas:
// 1. Synchronizes master weights to worker replicas.
// 2. Chunks batch into N slices of size ceil(B / N).
// 3. Concurrently computes forward, loss, and analytical backward passes on worker replicas.
// 4. Sums worker gradients into Master parameters in parallel using lock-free chunk reduction.
// 5. Scales gradients by 1 / len(batch).
// 6. Executes Adam optimizer step.
func (bt *BatchTrainer) TrainBatch(batch []Sample) (float32, float32) {
	B := len(batch)
	if B == 0 {
		return 0, 0
	}

	bt.MasterModel.SetTraining(true)

	// 1. Sync weights and zero worker gradients
	for _, w := range bt.Workers {
		w.SyncWeightsFrom(bt.MasterModel)
		w.ZeroGrad()
		w.SetTraining(true)
	}

	// 2. Split batch into N chunks
	N := bt.NumWorkers
	if N > B {
		N = B
	}
	chunkSize := (B + N - 1) / N

	var wg sync.WaitGroup
	losses := make([]float32, N)
	correctCounts := make([]int, N)

	for w := 0; w < N; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > B {
			end = B
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(workerIdx, s, e int) {
			defer wg.Done()
			worker := bt.Workers[workerIdx]
			var localLoss float32
			var localCorrect int

			for i := s; i < e; i++ {
				loss, probs := worker.ForwardBackward(batch[i].Input, batch[i].TargetClass)
				localLoss += loss

				// Determine predicted class
				predClass := 0
				maxP := probs[0]
				for k := 1; k < len(probs); k++ {
					if probs[k] > maxP {
						maxP = probs[k]
						predClass = k
					}
				}
				if predClass == batch[i].TargetClass {
					localCorrect++
				}
			}

			losses[workerIdx] = localLoss
			correctCounts[workerIdx] = localCorrect
		}(w, start, end)
	}

	// 1. Wait for all workers to finish with sync.WaitGroup
	wg.Wait()

	// Compute total loss and accuracy
	var totalLoss float32
	var totalCorrect int
	for w := 0; w < N; w++ {
		totalLoss += losses[w]
		totalCorrect += correctCounts[w]
	}
	avgLoss := totalLoss / float32(B)
	accuracy := float32(totalCorrect) / float32(B)

	// 2. Accumulate worker parameter gradients into Master parameters in parallel
	masterParams := bt.MasterModel.Parameters()
	workerParams := make([][]*Parameter, N)
	for w := 0; w < N; w++ {
		workerParams[w] = bt.Workers[w].Parameters()
	}

	bt.MasterModel.ZeroGrad()
	ReduceGradients(masterParams, workerParams, bt.NumWorkers)

	// 3. Scale gradients by 1 / len(batch)
	scale := float32(1.0) / float32(B)
	for _, p := range masterParams {
		for i := range p.Grad {
			p.Grad[i] *= scale
		}
	}

	// 4. Execute optimizer.Step() on Master model
	if bt.Optimizer.Params == nil || len(bt.Optimizer.Params) == 0 {
		bt.Optimizer.Params = masterParams
	}
	bt.Optimizer.Step()

	return avgLoss, accuracy
}

// Evaluate computes loss and classification accuracy on a validation dataset without weight updates.
func (bt *BatchTrainer) Evaluate(samples []Sample) (float32, float32) {
	if len(samples) == 0 {
		return 0, 0
	}
	bt.MasterModel.SetTraining(false)

	N := bt.NumWorkers
	if N <= 0 {
		N = 1
	}
	if N > len(samples) {
		N = len(samples)
	}

	chunkSize := (len(samples) + N - 1) / N
	var wg sync.WaitGroup
	losses := make([]float32, N)
	correctCounts := make([]int, N)

	for w := 0; w < N; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(samples) {
			end = len(samples)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(workerIdx, s, e int) {
			defer wg.Done()
			var worker *DiagonNetModel
			if workerIdx < len(bt.Workers) {
				worker = bt.Workers[workerIdx]
				worker.SyncWeightsFrom(bt.MasterModel)
				worker.SetTraining(false)
			} else {
				worker = bt.MasterModel
			}

			var localLoss float32
			var localCorrect int

			for i := s; i < e; i++ {
				logits := worker.Forward(samples[i].Input)
				probs := Softmax(logits)
				loss := worker.LossFn.Forward(probs, samples[i].TargetClass)
				localLoss += loss

				predClass := 0
				maxP := probs[0]
				for k := 1; k < len(probs); k++ {
					if probs[k] > maxP {
						maxP = probs[k]
						predClass = k
					}
				}
				if predClass == samples[i].TargetClass {
					localCorrect++
				}
			}

			losses[workerIdx] = localLoss
			correctCounts[workerIdx] = localCorrect
		}(w, start, end)
	}
	wg.Wait()

	var totalLoss float32
	var totalCorrect int
	for w := 0; w < N; w++ {
		totalLoss += losses[w]
		totalCorrect += correctCounts[w]
	}

	return totalLoss / float32(len(samples)), float32(totalCorrect) / float32(len(samples))
}

// ModelCheckpoint preserves parameter weights from the epoch achieving the highest validation accuracy.
type ModelCheckpoint struct {
	BestValAcc  float64
	BestEpoch   int
	BestWeights [][]float32
}

// NewModelCheckpoint constructs a new ModelCheckpoint tracker.
func NewModelCheckpoint() *ModelCheckpoint {
	return &ModelCheckpoint{
		BestValAcc:  -1.0,
		BestEpoch:   -1,
		BestWeights: nil,
	}
}

// Update records model weights if current validation accuracy strictly exceeds previous best accuracy.
// Returns true if a new best accuracy was achieved.
func (cp *ModelCheckpoint) Update(model *DiagonalNetModel, epoch int, valAcc float64) bool {
	if valAcc > cp.BestValAcc {
		cp.BestValAcc = valAcc
		cp.BestEpoch = epoch
		cp.BestWeights = model.SnapshotWeights()
		return true
	}
	return false
}

// RestoreBest restores the optimal historical weights into the model.
func (cp *ModelCheckpoint) RestoreBest(model *DiagonalNetModel) {
	if cp.BestWeights != nil && model != nil {
		model.RestoreWeights(cp.BestWeights)
	}
}

// ClassMetrics holds precision, recall, and F1-score for an individual class.
type ClassMetrics struct {
	ClassIndex int
	ClassName  string
	TP         int
	FP         int
	FN         int
	Support    int
	Precision  float64
	Recall     float64
	F1Score    float64
}

// EvaluationReport contains comprehensive multi-class classification evaluation metrics.
type EvaluationReport struct {
	NumClasses      int
	TotalSamples    int
	Accuracy        float64
	MacroPrecision  float64
	MacroRecall     float64
	MacroF1         float64
	ClassMetrics    []ClassMetrics
	ConfusionMatrix [][]int
}

// ComputeEvaluationMetrics computes confusion matrix, per-class Precision, Recall, F1, and Macro-F1.
func ComputeEvaluationMetrics(model *DiagonalNetModel, samples []Sample, classNames []string) EvaluationReport {
	K := model.NumClasses
	if len(classNames) < K {
		classNames = make([]string, K)
		for i := 0; i < K; i++ {
			classNames[i] = fmt.Sprintf("Class_%d", i)
		}
	}

	model.SetTraining(false)

	// Build K x K Confusion Matrix [actual][predicted]
	matrix := make([][]int, K)
	for i := 0; i < K; i++ {
		matrix[i] = make([]int, K)
	}

	totalSamples := len(samples)
	correct := 0

	for _, sample := range samples {
		logits := model.Forward(sample.Input)
		probs := Softmax(logits)

		predClass := 0
		maxP := probs[0]
		for k := 1; k < len(probs); k++ {
			if probs[k] > maxP {
				maxP = probs[k]
				predClass = k
			}
		}

		actual := sample.TargetClass
		if actual >= 0 && actual < K && predClass >= 0 && predClass < K {
			matrix[actual][predClass]++
			if actual == predClass {
				correct++
			}
		}
	}

	var accuracy float64
	if totalSamples > 0 {
		accuracy = float64(correct) / float64(totalSamples)
	}

	classMetrics := make([]ClassMetrics, K)
	var sumPrec, sumRec, sumF1 float64

	for c := 0; c < K; c++ {
		tp := matrix[c][c]

		// False Positives: predicted as c, but actual != c (column sum - TP)
		fp := 0
		for a := 0; a < K; a++ {
			if a != c {
				fp += matrix[a][c]
			}
		}

		// False Negatives: actual is c, but predicted != c (row sum - TP)
		fn := 0
		for p := 0; p < K; p++ {
			if p != c {
				fn += matrix[c][p]
			}
		}

		support := tp + fn

		var prec float64
		if tp+fp > 0 {
			prec = float64(tp) / float64(tp+fp)
		}

		var rec float64
		if tp+fn > 0 {
			rec = float64(tp) / float64(tp+fn)
		}

		var f1 float64
		if prec+rec > 0 {
			f1 = 2.0 * (prec * rec) / (prec + rec)
		}

		classMetrics[c] = ClassMetrics{
			ClassIndex: c,
			ClassName:  classNames[c],
			TP:         tp,
			FP:         fp,
			FN:         fn,
			Support:    support,
			Precision:  prec,
			Recall:     rec,
			F1Score:    f1,
		}

		sumPrec += prec
		sumRec += rec
		sumF1 += f1
	}

	macroPrec := sumPrec / float64(K)
	macroRec := sumRec / float64(K)
	macroF1 := sumF1 / float64(K)

	return EvaluationReport{
		NumClasses:      K,
		TotalSamples:    totalSamples,
		Accuracy:        accuracy,
		MacroPrecision:  macroPrec,
		MacroRecall:     macroRec,
		MacroF1:         macroF1,
		ClassMetrics:    classMetrics,
		ConfusionMatrix: matrix,
	}
}

// PrintEvaluationReport outputs a clean tabular evaluation report with per-class and macro metrics.
func PrintEvaluationReport(report EvaluationReport) {
	fmt.Println("=======================================================================================")
	fmt.Println("                      DIAGONALNET MODEL EVALUATION REPORT                              ")
	fmt.Println("=======================================================================================")
	fmt.Printf(" Total Samples Tested : %d\n", report.TotalSamples)
	fmt.Printf(" Overall Accuracy     : %6.2f%% (%d / %d)\n", report.Accuracy*100.0, int(report.Accuracy*float64(report.TotalSamples)+0.5), report.TotalSamples)
	fmt.Printf(" Macro-Precision      : %6.2f%%\n", report.MacroPrecision*100.0)
	fmt.Printf(" Macro-Recall         : %6.2f%%\n", report.MacroRecall*100.0)
	fmt.Printf(" Macro-F1 Score       : %6.2f%%\n", report.MacroF1*100.0)
	fmt.Println("---------------------------------------------------------------------------------------")
	fmt.Printf(" %-16s | %7s | %4s | %4s | %4s | %10s | %8s | %8s\n",
		"Class Name", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1-Score")
	fmt.Println("---------------------------------------------------------------------------------------")

	for _, cm := range report.ClassMetrics {
		fmt.Printf(" %-16s | %7d | %4d | %4d | %4d | %9.2f%% | %7.2f%% | %7.2f%%\n",
			cm.ClassName, cm.Support, cm.TP, cm.FP, cm.FN, cm.Precision*100.0, cm.Recall*100.0, cm.F1Score*100.0)
	}

	fmt.Println("---------------------------------------------------------------------------------------")
	fmt.Printf(" %-16s | %7d | %4s | %4s | %4s | %9.2f%% | %7.2f%% | %7.2f%%\n",
		"MACRO AVERAGE", report.TotalSamples, "-", "-", "-", report.MacroPrecision*100.0, report.MacroRecall*100.0, report.MacroF1*100.0)
	fmt.Println("=======================================================================================")
}

// CountModelParameters computes the total number of trainable floating-point weights and biases in parameter buffers.
func CountModelParameters(params []*Parameter) int {
	total := 0
	for _, p := range params {
		if p != nil {
			total += len(p.Data)
		}
	}
	return total
}

// ============================================================================
// 10. EMBEDDED HTML5 CANVAS & REAL-TIME WEB INFERENCE SERVER
// ============================================================================

// webAppHTML embeds the complete single-page interactive drawing canvas application with deep neural diagnostics.
const webAppHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DiagonalNet | Real-Time Neural Drawing Canvas & Deep Diagnostics</title>
<style>
  :root {
    --bg-main: #070b14;
    --bg-card: #0f172a;
    --bg-card-alt: #162036;
    --border-color: #1e293b;
    --border-bright: #334155;
    --accent-blue: #38bdf8;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-indigo: #6366f1;
    --accent-purple: #a855f7;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  body { background: var(--bg-main); color: var(--text-primary); min-height: 100vh; display: flex; flex-direction: column; }
  header { background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border-color); padding: 0.85rem 1.75rem; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 50; }
  .logo-title { display: flex; align-items: center; gap: 0.75rem; font-size: 1.25rem; font-weight: 700; background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header-badges { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
  .badge { background: #1e293b; border: 1px solid var(--border-color); color: var(--accent-blue); padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.35rem; }
  .badge.emerald { color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.08); }
  .badge.purple { color: var(--accent-purple); border-color: rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.08); }
  .main-container { flex: 1; max-width: 1440px; width: 100%; margin: 0 auto; padding: 1.5rem; display: grid; grid-template-columns: 430px 1fr; gap: 1.5rem; }
  @media (max-width: 1100px) { .main-container { grid-template-columns: 1fr; } }
  .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.85rem; padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); }
  .canvas-wrapper { position: relative; width: 400px; height: 400px; margin: 0 auto; border-radius: 0.75rem; overflow: hidden; border: 2px solid var(--border-bright); box-shadow: inset 0 2px 12px rgba(0,0,0,0.8), 0 0 20px rgba(56, 189, 248, 0.08); }
  canvas#paintCanvas { width: 400px; height: 400px; background: #000000; cursor: crosshair; touch-action: none; display: block; }
  .controls { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; }
  button { padding: 0.6rem 0.9rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; transition: all 0.15s ease-in-out; border: none; font-size: 0.85rem; display: inline-flex; align-items: center; justify-content: center; gap: 0.4rem; }
  .btn-clear { background: #1e293b; color: #f8fafc; border: 1px solid var(--border-bright); }
  .btn-clear:hover { background: #334155; }
  .btn-predict { background: linear-gradient(135deg, #0284c7, #06b6d4); color: white; }
  .btn-predict:hover { opacity: 0.92; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3); }
  .btn-action { background: var(--bg-card-alt); color: var(--text-secondary); border: 1px solid var(--border-bright); padding: 0.35rem 0.65rem; font-size: 0.75rem; border-radius: 0.4rem; }
  .btn-action:hover { color: var(--text-primary); border-color: var(--accent-blue); background: #1e293b; }
  .btn-preset { background: #1e293b; border: 1px solid var(--border-color); color: var(--text-secondary); padding: 0.3rem 0.55rem; font-size: 0.75rem; border-radius: 0.35rem; font-family: monospace; }
  .btn-preset:hover { background: #334155; color: var(--accent-blue); border-color: var(--accent-blue); }
  .presets-row { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; }
  .prediction-banner { background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(6, 182, 212, 0.04)); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 0.75rem; padding: 1rem 1.25rem; display: flex; align-items: center; justify-content: space-between; position: relative; overflow: hidden; }
  .prediction-banner::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent-blue), var(--accent-emerald), var(--accent-purple)); }
  .pred-label-group { display: flex; flex-direction: column; gap: 0.15rem; }
  .pred-sub { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
  .pred-name { font-size: 2.25rem; font-weight: 800; color: var(--accent-blue); line-height: 1.1; text-shadow: 0 0 20px rgba(56, 189, 248, 0.4); }
  .latency-tag { font-family: monospace; font-size: 0.85rem; color: var(--accent-emerald); background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); padding: 0.3rem 0.6rem; border-radius: 0.4rem; }
  .metric-pill { font-family: monospace; font-size: 0.75rem; color: var(--text-secondary); background: #1e293b; border: 1px solid var(--border-color); padding: 0.2rem 0.5rem; border-radius: 0.35rem; display: inline-flex; align-items: center; gap: 0.3rem; }
  
  /* Tabs */
  .tabs-nav { display: flex; gap: 0.35rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; overflow-x: auto; scrollbar-width: none; }
  .tab-btn { background: transparent; border: none; color: var(--text-secondary); padding: 0.5rem 0.85rem; font-size: 0.82rem; font-weight: 600; border-radius: 0.4rem; cursor: pointer; transition: all 0.15s; white-space: nowrap; }
  .tab-btn:hover { color: var(--text-primary); background: #1e293b; }
  .tab-btn.active { color: var(--accent-blue); background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); }
  .tab-pane { display: none; flex-direction: column; gap: 1rem; animation: fadeIn 0.15s ease-in; }
  .tab-pane.active { display: flex; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

  /* Probability Bars */
  .class-list { display: flex; flex-direction: column; gap: 0.45rem; max-height: 280px; overflow-y: auto; padding-right: 0.35rem; }
  .class-row { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.82rem; }
  .class-info { display: flex; justify-content: space-between; font-weight: 500; }
  .progress-bg { height: 7px; background: #1e293b; border-radius: 9999px; overflow: hidden; border: 1px solid #334155; }
  .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); border-radius: 9999px; transition: width 0.12s ease-out; }
  .class-row.top .progress-fill { background: linear-gradient(90deg, var(--accent-emerald), #34d399); }
  
  /* Stats Grids */
  .stat-grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; }
  .stat-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }
  .stat-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.6rem; }
  @media (max-width: 700px) { .stat-grid-4, .stat-grid-3 { grid-template-columns: repeat(2, 1fr); } }
  .stat-box { background: var(--bg-card-alt); border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 0.65rem 0.8rem; display: flex; flex-direction: column; gap: 0.2rem; }
  .stat-box-title { font-size: 0.7rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.05em; }
  .stat-box-val { font-size: 1.15rem; font-weight: 700; color: var(--text-primary); font-family: monospace; }
  .stat-box-val.accent { color: var(--accent-blue); }
  .stat-box-val.emerald { color: var(--accent-emerald); }
  .stat-box-val.amber { color: var(--accent-amber); }
  .stat-box-val.purple { color: var(--accent-purple); }
  .stat-box-sub { font-size: 0.7rem; color: var(--text-secondary); }

  /* 13-Manifold Heatmaps */
  .manifold-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(105px, 1fr)); gap: 0.6rem; max-height: 440px; overflow-y: auto; padding-right: 0.35rem; }
  .manifold-card { background: var(--bg-card-alt); border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 0.5rem; display: flex; flex-direction: column; align-items: center; gap: 0.35rem; cursor: pointer; transition: transform 0.15s, border-color 0.15s; }
  .manifold-card:hover { transform: translateY(-2px); border-color: var(--accent-blue); }
  .manifold-card canvas { width: 84px; height: 84px; image-rendering: pixelated; border-radius: 0.35rem; background: #000; border: 1px solid var(--border-bright); }
  .manifold-title { font-size: 0.68rem; font-weight: 600; color: var(--text-secondary); text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; width: 100%; }
  .manifold-sub { font-size: 0.65rem; color: var(--text-muted); font-family: monospace; }

  /* Timing Chart */
  .stage-bar-wrapper { display: flex; height: 16px; border-radius: 9999px; overflow: hidden; background: #1e293b; border: 1px solid var(--border-bright); margin: 0.4rem 0; }
  .stage-segment { height: 100%; transition: width 0.15s; }
  .stage-legend { display: flex; flex-wrap: wrap; gap: 0.6rem; font-size: 0.72rem; }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

  /* Dense Vector Visualizer */
  .vector-chart { display: flex; align-items: flex-end; gap: 2px; height: 64px; background: var(--bg-card-alt); padding: 6px 8px; border-radius: 0.5rem; border: 1px solid var(--border-color); overflow-x: auto; }
  .vector-bar { flex: 1; min-width: 2px; background: var(--accent-blue); border-radius: 1px; transition: height 0.1s; }
  
  /* Tables */
  .data-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; text-align: left; }
  .data-table th { color: var(--text-muted); font-weight: 600; padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border-color); text-transform: uppercase; font-size: 0.68rem; }
  .data-table td { padding: 0.45rem 0.6rem; border-bottom: 1px solid rgba(30, 41, 59, 0.6); color: var(--text-secondary); font-family: monospace; }
  .data-table tr:hover td { background: rgba(56, 189, 248, 0.04); color: var(--text-primary); }

  /* Modal */
  .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.75); backdrop-filter: blur(4px); z-index: 100; display: none; align-items: center; justify-content: center; padding: 1rem; }
  .modal-overlay.active { display: flex; }
  .modal-box { background: var(--bg-card); border: 1px solid var(--border-bright); border-radius: 0.85rem; max-width: 520px; width: 100%; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.8); }
</style>
</head>
<body>
  <header>
    <div class="logo-title">
      <span>⬡</span>
      <span>DiagonalNet 13-Manifold Neural Engine</span>
    </div>
    <div class="header-badges">
      <span class="badge" id="hdrCores">⚡ CPU Cores: &mdash;</span>
      <span class="badge emerald" id="hdrFps">FPS: &mdash;</span>
      <span class="badge purple" id="hdrMem">RAM: &mdash;</span>
      <button class="btn-action" id="btnExportJson">📥 Export Telemetry</button>
    </div>
  </header>

  <div class="main-container">
    <!-- Left Column: Canvas, Presets, Morphology -->
    <div style="display: flex; flex-direction: column; gap: 1.25rem;">
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-weight: 700; font-size: 0.95rem;">Interactive Sketch Canvas</span>
          <span style="font-size: 0.78rem; color: var(--text-secondary);" id="canvasDimLabel">400 &times; 400 px</span>
        </div>
        <div class="canvas-wrapper">
          <canvas id="paintCanvas" width="400" height="400"></canvas>
        </div>
        <div class="controls">
          <button class="btn-clear" id="btnClear">Clear (C)</button>
          <div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: var(--text-secondary);">
            <span>Brush:</span>
            <input id="brushSize" type="range" min="10" max="40" value="22" style="accent-color: var(--accent-blue); cursor: pointer; width: 70px;">
            <span id="brushVal" style="font-family: monospace; min-width: 20px;">22</span>
          </div>
          <button class="btn-predict" id="btnPredict">Predict (↵)</button>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: var(--text-secondary); padding-top: 0.25rem;">
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="chkAutoPredict" checked style="accent-color: var(--accent-emerald);">
            <span>Live Continuous Autograd</span>
          </label>
          <span style="font-size: 0.72rem; color: var(--text-muted);">Shortcuts: <b>C</b> / <b>Esc</b></span>
        </div>
      </div>

      <!-- Quick Draw Presets -->
      <div class="card" style="padding: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; font-weight: 600;">
          <span>⚡ Quick Sample Presets</span>
          <span style="font-size: 0.72rem; color: var(--text-muted);">Click to inject test pattern</span>
        </div>
        <div class="presets-row" id="presetsRow">
          <button class="btn-preset" data-preset="0">0</button>
          <button class="btn-preset" data-preset="1">1</button>
          <button class="btn-preset" data-preset="2">2</button>
          <button class="btn-preset" data-preset="3">3</button>
          <button class="btn-preset" data-preset="4">4</button>
          <button class="btn-preset" data-preset="5">5</button>
          <button class="btn-preset" data-preset="6">6</button>
          <button class="btn-preset" data-preset="7">7</button>
          <button class="btn-preset" data-preset="8">8</button>
          <button class="btn-preset" data-preset="9">9</button>
          <button class="btn-preset" data-preset="Circle">Circle</button>
          <button class="btn-preset" data-preset="Cross">Cross</button>
          <button class="btn-preset" data-preset="Box">Box</button>
          <button class="btn-preset" data-preset="Diagonal">Diagonal</button>
        </div>
      </div>

      <!-- Input Spatial Geometry Card -->
      <div class="card" style="padding: 1rem;">
        <div style="font-weight: 600; font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
          <span>📐 Input Spatial Morphology</span>
          <span class="badge" style="font-size: 0.7rem;">28&times;28 Resampled</span>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
          <div style="display: flex; flex-direction: column; align-items: center; gap: 0.25rem;">
            <canvas id="preview28Canvas" width="28" height="28" style="width: 84px; height: 84px; image-rendering: pixelated; border-radius: 0.4rem; background: #000; border: 1px solid var(--border-bright);"></canvas>
            <span style="font-size: 0.65rem; color: var(--text-muted);">Centered Tensor</span>
          </div>
          <div style="flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; font-size: 0.75rem;">
            <div class="stat-box" style="padding: 0.4rem 0.5rem;">
              <span class="stat-box-title">BBox Size</span>
              <span class="stat-box-val" id="geomBBoxDim" style="font-size: 0.9rem;">&mdash;</span>
            </div>
            <div class="stat-box" style="padding: 0.4rem 0.5rem;">
              <span class="stat-box-title">Aspect Ratio</span>
              <span class="stat-box-val" id="geomAspect" style="font-size: 0.9rem;">&mdash;</span>
            </div>
            <div class="stat-box" style="padding: 0.4rem 0.5rem;">
              <span class="stat-box-title">Stroke Mass</span>
              <span class="stat-box-val" id="geomFgCount" style="font-size: 0.9rem;">&mdash;</span>
            </div>
            <div class="stat-box" style="padding: 0.4rem 0.5rem;">
              <span class="stat-box-title">Fill Density</span>
              <span class="stat-box-val" id="geomDensity" style="font-size: 0.9rem;">&mdash;</span>
            </div>
          </div>
        </div>
        <div style="font-size: 0.72rem; color: var(--text-muted); display: flex; justify-content: space-between; font-family: monospace;">
          <span>Centroid: <span id="geomCentroid" style="color: var(--text-secondary);">&mdash;</span></span>
          <span>Coverage: <span id="geomCoverage" style="color: var(--text-secondary);">&mdash;</span></span>
        </div>
      </div>
    </div>

    <!-- Right Column: Deep Diagnostics & Tabs -->
    <div style="display: flex; flex-direction: column; gap: 1.25rem;">
      <!-- Hero Prediction Banner -->
      <div class="prediction-banner">
        <div class="pred-label-group">
          <span class="pred-sub">Top Predicted Class &bull; 13-Manifold CNN</span>
          <span class="pred-name" id="topClass">&mdash;</span>
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.2rem;">
            <span class="metric-pill" id="marginBadge">Margin: &mdash;</span>
            <span class="metric-pill" id="entropyBadge">Entropy: &mdash;</span>
          </div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem;">
          <span id="topConfidence" style="font-size: 2rem; font-weight: 800; color: var(--text-primary);">0.0%</span>
          <span class="latency-tag" id="latencyBadge">&mdash; ms</span>
          <span style="font-size: 0.72rem; color: var(--text-muted); font-family: monospace;" id="fpsBadge">&mdash; inf/sec</span>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <div class="tabs-nav">
        <button class="tab-btn active" data-tab="tab-probs">📊 Class Probabilities</button>
        <button class="tab-btn" data-tab="tab-perf">⚡ Performance &amp; Profiler</button>
        <button class="tab-btn" data-tab="tab-manifold">🧬 13-Channel Manifold</button>
        <button class="tab-btn" data-tab="tab-layers">🧠 Layer Activations</button>
        <button class="tab-btn" data-tab="tab-arch">⚙️ Model Architecture</button>
      </div>

      <!-- Tab 1: Probabilities & Decision Theory -->
      <div class="tab-pane active" id="tab-probs">
        <div class="card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; font-size: 0.9rem;">Softmax Probability Distribution</span>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: var(--text-secondary);">
              <span>Temperature T:</span>
              <input type="range" id="tempSlider" min="0.2" max="2.5" step="0.1" value="1.0" style="width: 75px; accent-color: var(--accent-cyan); cursor: pointer;">
              <span id="tempVal" style="font-family: monospace;">1.0</span>
            </div>
          </div>

          <div class="class-list" id="classList">
            <div style="color: var(--text-secondary); font-size: 0.85rem; text-align: center; padding: 2rem 0;">Draw on the canvas to evaluate live class probabilities.</div>
          </div>

          <!-- Information Theory Metrics -->
          <div style="font-weight: 700; font-size: 0.85rem; margin-top: 0.5rem;">Information-Theoretic Decision Metrics</div>
          <div class="stat-grid-4">
            <div class="stat-box">
              <span class="stat-box-title">Shannon Entropy</span>
              <span class="stat-box-val accent" id="infoEntropy">&mdash;</span>
              <span class="stat-box-sub" id="infoEntropyMax">Max: &mdash; bits</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Uncertainty</span>
              <span class="stat-box-val amber" id="infoUncertainty">&mdash;</span>
              <span class="stat-box-sub">Normalized Index</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Perplexity</span>
              <span class="stat-box-val emerald" id="infoPerplexity">&mdash;</span>
              <span class="stat-box-sub">2^H Candidates</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Gini Impurity</span>
              <span class="stat-box-val purple" id="infoGini">&mdash;</span>
              <span class="stat-box-sub">1 - ∑ p_i^2</span>
            </div>
          </div>

          <!-- Raw Logits vs Softmax Table -->
          <div style="font-weight: 700; font-size: 0.85rem; margin-top: 0.5rem;">Top Class Logits &amp; Softmax Scores</div>
          <div style="max-height: 160px; overflow-y: auto;">
            <table class="data-table" id="logitsTable">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Raw Logit (z_i)</th>
                  <th>Exp(z_i / T)</th>
                  <th>Softmax P(y=i)</th>
                </tr>
              </thead>
              <tbody id="logitsTableBody">
                <tr><td colspan="4" style="text-align:center;">No data available</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Tab 2: Performance Profiler -->
      <div class="tab-pane" id="tab-perf">
        <div class="card">
          <div style="font-weight: 700; font-size: 0.9rem;">Sub-Millisecond Inference Pipeline Breakdown</div>
          
          <!-- Stacked Timing Bar -->
          <div class="stage-bar-wrapper" id="stageTimingBar">
            <div class="stage-segment" style="width: 20%; background: var(--accent-blue);" title="Preprocess"></div>
            <div class="stage-segment" style="width: 25%; background: var(--accent-purple);" title="Manifold"></div>
            <div class="stage-segment" style="width: 35%; background: var(--accent-cyan);" title="Conv1 & 2"></div>
            <div class="stage-segment" style="width: 15%; background: var(--accent-emerald);" title="Dense"></div>
            <div class="stage-segment" style="width: 5%; background: var(--accent-amber);" title="Softmax"></div>
          </div>

          <div class="stage-legend">
            <span><span class="legend-dot" style="background: var(--accent-blue);"></span> Preprocess</span>
            <span><span class="legend-dot" style="background: var(--accent-purple);"></span> 13-Manifold</span>
            <span><span class="legend-dot" style="background: var(--accent-cyan);"></span> Conv Stages</span>
            <span><span class="legend-dot" style="background: var(--accent-emerald);"></span> Dense Head</span>
            <span><span class="legend-dot" style="background: var(--accent-amber);"></span> Softmax</span>
          </div>

          <table class="data-table" style="margin-top: 0.5rem;">
            <thead>
              <tr>
                <th>Pipeline Stage</th>
                <th>Latency (µs)</th>
                <th>Latency (ms)</th>
                <th>Share (%)</th>
              </tr>
            </thead>
            <tbody id="timingTableBody">
              <tr><td>Preprocessing &amp; BBox</td><td id="tPreUs">&mdash;</td><td id="tPreMs">&mdash;</td><td id="tPrePct">&mdash;</td></tr>
              <tr><td>13-Manifold Spatial Calculus</td><td id="tManUs">&mdash;</td><td id="tManMs">&mdash;</td><td id="tManPct">&mdash;</td></tr>
              <tr><td>Conv1 &amp; Conv2 Convolutions</td><td id="tConvUs">&mdash;</td><td id="tConvMs">&mdash;</td><td id="tConvPct">&mdash;</td></tr>
              <tr><td>Adaptive Pool &amp; Dense FC1/FC2</td><td id="tDenseUs">&mdash;</td><td id="tDenseMs">&mdash;</td><td id="tDensePct">&mdash;</td></tr>
              <tr><td>Softmax &amp; Decision Logic</td><td id="tSoftUs">&mdash;</td><td id="tSoftMs">&mdash;</td><td id="tSoftPct">&mdash;</td></tr>
              <tr style="font-weight: 700; color: var(--text-primary);"><td>Total End-to-End Latency</td><td id="tTotUs">&mdash;</td><td id="tTotMs">&mdash;</td><td>100.0%</td></tr>
            </tbody>
          </table>

          <div style="font-weight: 700; font-size: 0.85rem; margin-top: 0.6rem;">Host Engine &amp; Go Runtime Health</div>
          <div class="stat-grid-4">
            <div class="stat-box">
              <span class="stat-box-title">Engine Speed</span>
              <span class="stat-box-val emerald" id="perfThroughput">&mdash;</span>
              <span class="stat-box-sub">Inferences / sec</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Avg Latency (30f)</span>
              <span class="stat-box-val accent" id="perfAvgLatency">&mdash;</span>
              <span class="stat-box-sub">Rolling Window</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Go Heap Alloc</span>
              <span class="stat-box-val purple" id="perfHeapAlloc">&mdash;</span>
              <span class="stat-box-sub">Active Memory</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">GC Cycles</span>
              <span class="stat-box-val amber" id="perfNumGC">&mdash;</span>
              <span class="stat-box-sub">Total Collections</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 3: 13-Channel Manifold -->
      <div class="tab-pane" id="tab-manifold">
        <div class="card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <span style="font-weight: 700; font-size: 0.9rem;">13-Channel Spatial Difference Manifold</span>
              <div style="font-size: 0.72rem; color: var(--text-muted);">Base Grayscale + 4 Diagonal Derivatives + 8 Chess Knight-Move Operators</div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem;">
              <span>Colormap:</span>
              <select id="cmapSelect" style="background: var(--bg-card-alt); border: 1px solid var(--border-bright); color: var(--text-primary); padding: 0.25rem 0.5rem; border-radius: 0.35rem; font-size: 0.75rem;">
                <option value="turbo">Turbo (Thermal)</option>
                <option value="neon" selected>Neon Cyan</option>
                <option value="emerald">Emerald</option>
                <option value="gray">Grayscale</option>
              </select>
            </div>
          </div>

          <div class="manifold-grid" id="manifoldGrid">
            <!-- 13 Channel Cards generated dynamically -->
          </div>
        </div>
      </div>

      <!-- Tab 4: Layer Activations -->
      <div class="tab-pane" id="tab-layers">
        <div class="card">
          <div style="font-weight: 700; font-size: 0.9rem;">Feature Hierarchy &amp; Intermediate Activations</div>
          <div style="font-size: 0.75rem; color: var(--text-secondary); background: var(--bg-card-alt); padding: 0.6rem; border-radius: 0.4rem; border: 1px solid var(--border-color); font-family: monospace;">
            [1×28×28] ➔ [13×28×28] ➔ Conv1[16×28×28] ➔ MaxPool[16×14×14] ➔ Conv2[32×14×14] ➔ MaxPool[32×7×7] ➔ Pool[32×4×4=512] ➔ FC1[128] ➔ FC2[K]
          </div>

          <div class="stat-grid-3">
            <div class="stat-box">
              <span class="stat-box-title">Conv1 Activation</span>
              <span class="stat-box-val accent" id="actConv1Mean">&mdash;</span>
              <span class="stat-box-sub" id="actConv1Sparsity">Sparsity: &mdash;%</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Conv2 Activation</span>
              <span class="stat-box-val cyan" id="actConv2Mean">&mdash;</span>
              <span class="stat-box-sub" id="actConv2Sparsity">Sparsity: &mdash;%</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">512-D Latent Norm</span>
              <span class="stat-box-val emerald" id="actPoolNorm">&mdash;</span>
              <span class="stat-box-sub">AdaptiveAvgPool L2</span>
            </div>
          </div>

          <!-- FC1 128-Neuron Vector Visualizer -->
          <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
              <span style="font-weight: 600; font-size: 0.8rem;">FC1 Dense Hidden Vector (128 Dimensions)</span>
              <span style="font-size: 0.72rem; color: var(--text-muted);" id="fc1ActiveLabel">Active: &mdash; / 128</span>
            </div>
            <div class="vector-chart" id="fc1VectorChart">
              <!-- 128 bars generated dynamically -->
            </div>
          </div>
        </div>
      </div>

      <!-- Tab 5: Model Architecture Specs -->
      <div class="tab-pane" id="tab-arch">
        <div class="card">
          <div style="font-weight: 700; font-size: 0.9rem;">DiagonalNet Architecture &amp; Parameter Topology</div>
          
          <div class="stat-grid-3">
            <div class="stat-box">
              <span class="stat-box-title">Total Parameters</span>
              <span class="stat-box-val accent" id="archTotalParams">&mdash;</span>
              <span class="stat-box-sub" id="archWeightsSize">&mdash; KB Memory</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Inference FLOPs</span>
              <span class="stat-box-val emerald">~4.88 MFLOPs</span>
              <span class="stat-box-sub">Sub-8ms CPU Target</span>
            </div>
            <div class="stat-box">
              <span class="stat-box-title">Binary Format</span>
              <span class="stat-box-val purple">DIAGON01</span>
              <span class="stat-box-sub">LittleEndian Float32</span>
            </div>
          </div>

          <table class="data-table" style="margin-top: 0.5rem;">
            <thead>
              <tr>
                <th>Layer Name</th>
                <th>Type</th>
                <th>Input Dim</th>
                <th>Output Dim</th>
                <th>Trainable Parameters</th>
              </tr>
            </thead>
            <tbody id="archTableBody">
              <tr><td>13-Ch Spatial Manifold</td><td>ManifoldCalculus</td><td>1 &times; 28 &times; 28</td><td>13 &times; 28 &times; 28</td><td>0 (Pure Calculus)</td></tr>
              <tr><td>Conv2D Stage 1</td><td>Conv2D (K=3, S=1, P=1)</td><td>13 &times; 28 &times; 28</td><td>16 &times; 28 &times; 28</td><td>1,888 (1,872 W + 16 B)</td></tr>
              <tr><td>ReLU1 + MaxPool1</td><td>ReLU + MaxPool2D(2)</td><td>16 &times; 28 &times; 28</td><td>16 &times; 14 &times; 14</td><td>0</td></tr>
              <tr><td>Conv2D Stage 2</td><td>Conv2D (K=3, S=1, P=1)</td><td>16 &times; 14 &times; 14</td><td>32 &times; 14 &times; 14</td><td>4,640 (4,608 W + 32 B)</td></tr>
              <tr><td>ReLU2 + MaxPool2</td><td>ReLU + MaxPool2D(2)</td><td>32 &times; 14 &times; 14</td><td>32 &times; 7 &times; 7</td><td>0</td></tr>
              <tr><td>Adaptive AvgPool</td><td>AdaptiveAvgPool2D(4x4)</td><td>32 &times; 7 &times; 7</td><td>32 &times; 4 &times; 4 (512)</td><td>0</td></tr>
              <tr><td>Dense Head FC1</td><td>Linear + ReLU + Drop(0.2)</td><td>512</td><td>128</td><td>65,664 (65,536 W + 128 B)</td></tr>
              <tr><td>Classifier Head FC</td><td>Linear + Softmax</td><td>128</td><td id="archOutputClasses">K Classes</td><td id="archFCParams">&mdash;</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- Channel Zoom Modal -->
  <div class="modal-overlay" id="channelModal">
    <div class="modal-box">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700;" id="modalChannelTitle">Channel Details</span>
        <button class="btn-action" id="modalCloseBtn">&times; Close</button>
      </div>
      <div style="display: flex; justify-content: center; padding: 1rem 0;">
        <canvas id="modalCanvas" width="28" height="28" style="width: 224px; height: 224px; image-rendering: pixelated; border-radius: 0.5rem; background: #000; border: 2px solid var(--border-bright);"></canvas>
      </div>
      <div class="stat-grid-3">
        <div class="stat-box">
          <span class="stat-box-title">Mean Energy</span>
          <span class="stat-box-val accent" id="modalEnergy">&mdash;</span>
        </div>
        <div class="stat-box">
          <span class="stat-box-title">Peak Value</span>
          <span class="stat-box-val emerald" id="modalPeak">&mdash;</span>
        </div>
        <div class="stat-box">
          <span class="stat-box-title">Sparsity</span>
          <span class="stat-box-val amber" id="modalSparsity">&mdash;</span>
        </div>
      </div>
    </div>
  </div>

<script>
  const canvas = document.getElementById('paintCanvas');
  const ctx = canvas.getContext('2d');
  const brushSlider = document.getElementById('brushSize');
  const brushVal = document.getElementById('brushVal');
  const chkAuto = document.getElementById('chkAutoPredict');
  const prev28Canvas = document.getElementById('preview28Canvas');
  const prev28Ctx = prev28Canvas ? prev28Canvas.getContext('2d') : null;
  const tempSlider = document.getElementById('tempSlider');
  const tempVal = document.getElementById('tempVal');
  const cmapSelect = document.getElementById('cmapSelect');

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, 400, 400);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 22;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (brushSlider && brushVal) {
    brushSlider.addEventListener('input', () => {
      ctx.lineWidth = parseInt(brushSlider.value);
      brushVal.innerText = brushSlider.value;
    });
  }

  let drawing = false;
  let hasDrawn = false;
  let debounceTimer = null;
  let lastTelemetry = null;
  let rollingLatencies = [];

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  function startDraw(e) {
    drawing = true;
    hasDrawn = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    draw(e);
  }

  function draw(e) {
    if (!drawing) return;
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    if (chkAuto && chkAuto.checked) {
      schedulePredict();
    }
  }

  function endDraw() {
    if (drawing) {
      drawing = false;
      ctx.closePath();
      triggerPredict();
    }
  }

  canvas.addEventListener('mousedown', startDraw);
  canvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', endDraw);
  canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); }, { passive: false });
  canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); }, { passive: false });
  window.addEventListener('touchend', endDraw);

  function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 400, 400);
    hasDrawn = false;
    document.getElementById('topClass').innerText = '—';
    document.getElementById('topConfidence').innerText = '0.0%';
    document.getElementById('latencyBadge').innerText = '— ms';
    document.getElementById('fpsBadge').innerText = '— inf/sec';
    document.getElementById('marginBadge').innerText = 'Margin: —';
    document.getElementById('entropyBadge').innerText = 'Entropy: —';
    if (prev28Ctx) {
      prev28Ctx.fillStyle = '#000000';
      prev28Ctx.fillRect(0, 0, 28, 28);
    }
    document.getElementById('classList').innerHTML = '<div style="color: var(--text-secondary); font-size: 0.85rem; text-align: center; padding: 2rem 0;">Canvas cleared. Draw a sketch.</div>';
    resetStats();
  }

  function resetStats() {
    ['geomBBoxDim', 'geomAspect', 'geomFgCount', 'geomDensity', 'geomCentroid', 'geomCoverage',
     'infoEntropy', 'infoUncertainty', 'infoPerplexity', 'infoGini'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerText = '—';
    });
  }

  document.getElementById('btnClear').addEventListener('click', clearCanvas);
  document.getElementById('btnPredict').addEventListener('click', triggerPredict);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C' || e.key === 'Escape') {
      clearCanvas();
    } else if (e.key === 'Enter') {
      triggerPredict();
    }
  });

  // Presets drawing handler
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const preset = btn.getAttribute('data-preset');
      drawPreset(preset);
    });
  });

  function drawPreset(preset) {
    clearCanvas();
    hasDrawn = true;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = parseInt(brushSlider ? brushSlider.value : 22);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    switch(preset) {
      case '0':
      case 'Circle':
        ctx.ellipse(200, 200, 90, 120, 0, 0, 2 * Math.PI);
        break;
      case '1':
        ctx.moveTo(170, 90); ctx.lineTo(200, 60); ctx.lineTo(200, 340);
        break;
      case '2':
        ctx.moveTo(130, 120);
        ctx.bezierCurveTo(130, 60, 270, 60, 270, 140);
        ctx.bezierCurveTo(270, 210, 130, 270, 130, 340);
        ctx.lineTo(270, 340);
        break;
      case '3':
        ctx.moveTo(130, 80); ctx.lineTo(260, 80); ctx.lineTo(190, 180);
        ctx.bezierCurveTo(260, 180, 270, 320, 130, 320);
        break;
      case '4':
        ctx.moveTo(230, 70); ctx.lineTo(120, 240); ctx.lineTo(270, 240);
        ctx.moveTo(230, 70); ctx.lineTo(230, 330);
        break;
      case '5':
        ctx.moveTo(250, 80); ctx.lineTo(140, 80); ctx.lineTo(130, 180);
        ctx.bezierCurveTo(200, 160, 270, 190, 260, 280);
        ctx.bezierCurveTo(250, 330, 180, 340, 130, 320);
        break;
      case '6':
        ctx.moveTo(240, 90);
        ctx.bezierCurveTo(140, 110, 120, 220, 120, 260);
        ctx.ellipse(200, 260, 80, 75, 0, 0, 2 * Math.PI);
        break;
      case '7':
        ctx.moveTo(130, 80); ctx.lineTo(270, 80); ctx.lineTo(160, 330);
        break;
      case '8':
        ctx.ellipse(200, 130, 60, 55, 0, 0, 2 * Math.PI);
        ctx.ellipse(200, 255, 75, 70, 0, 0, 2 * Math.PI);
        break;
      case '9':
        ctx.ellipse(200, 140, 75, 70, 0, 0, 2 * Math.PI);
        ctx.moveTo(275, 140); ctx.lineTo(275, 270);
        ctx.bezierCurveTo(275, 330, 170, 340, 140, 310);
        break;
      case 'Cross':
        ctx.moveTo(200, 70); ctx.lineTo(200, 330);
        ctx.moveTo(70, 200); ctx.lineTo(330, 200);
        break;
      case 'Box':
        ctx.rect(90, 90, 220, 220);
        break;
      case 'Diagonal':
        ctx.moveTo(80, 80); ctx.lineTo(320, 320);
        break;
    }
    ctx.stroke();
    triggerPredict();
  }

  function schedulePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerPredict, 40);
  }

  async function triggerPredict() {
    if (!hasDrawn) return;
    const dataUrl = canvas.toDataURL('image/png');
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });
      if (!resp.ok) return;
      const data = await resp.json();
      lastTelemetry = data;
      renderPrediction(data);
    } catch (err) {
      console.error('Predict error:', err);
    }
  }

  // Colormap utilities for heatmaps
  function getTurboColor(t) {
    const r = Math.min(255, Math.max(0, Math.round(255 * (0.1357 + 4.5 * t - 14.5 * t * t + 16.5 * t * t * t - 6.5 * t * t * t * t))));
    const g = Math.min(255, Math.max(0, Math.round(255 * (0.0914 + 2.1 * t + 4.8 * t * t - 14.1 * t * t * t + 8.1 * t * t * t * t))));
    const b = Math.min(255, Math.max(0, Math.round(255 * (0.1067 + 12.5 * t - 37.0 * t * t + 36.5 * t * t * t - 12.0 * t * t * t * t))));
    return [r, g, b];
  }

  function getColormapColor(val, cmap) {
    const t = Math.max(0, Math.min(1, val / 255.0));
    if (cmap === 'turbo') return getTurboColor(t);
    if (cmap === 'emerald') return [Math.round(16 * t), Math.round(185 * t + 70 * t * t), Math.round(129 * t)];
    if (cmap === 'gray') return [val, val, val];
    // Neon Cyan (default)
    return [Math.round(6 * t + 30 * t * t), Math.round(182 * t + 73 * t * t), Math.round(212 * t + 43 * t * t)];
  }

  if (cmapSelect) {
    cmapSelect.addEventListener('change', () => {
      if (lastTelemetry && lastTelemetry.stats) {
        renderManifold(lastTelemetry.stats.manifold);
      }
    });
  }

  // Temperature slider recalibration
  if (tempSlider && tempVal) {
    tempSlider.addEventListener('input', () => {
      const T = parseFloat(tempSlider.value);
      tempVal.innerText = T.toFixed(1);
      if (lastTelemetry && lastTelemetry.stats && lastTelemetry.stats.raw_logits) {
        recalculateTemperature(T);
      }
    });
  }

  function recalculateTemperature(T) {
    const logits = lastTelemetry.stats.raw_logits;
    if (!logits || logits.length === 0) return;
    const maxL = Math.max(...logits);
    let sumExp = 0;
    const expVals = logits.map(z => {
      const e = Math.exp((z - maxL) / T);
      sumExp += e;
      return e;
    });
    const newProbs = expVals.map(e => e / sumExp);
    
    // Update confidences in copy
    const confs = lastTelemetry.confidences.map((c, i) => ({
      ...c,
      confidence: newProbs[i]
    }));
    renderProbabilityList(confs);
  }

  function renderPrediction(data) {
    if (data.is_blank) {
      document.getElementById('topClass').innerText = 'Blank';
      document.getElementById('topConfidence').innerText = '0.0%';
      document.getElementById('latencyBadge').innerText = data.latency_ms.toFixed(2) + ' ms';
      return;
    }

    // Rolling latency & FPS
    rollingLatencies.push(data.latency_ms);
    if (rollingLatencies.length > 30) rollingLatencies.shift();
    const avgLat = rollingLatencies.reduce((a, b) => a + b, 0) / rollingLatencies.length;
    const fps = data.latency_ms > 0 ? (1000.0 / data.latency_ms).toFixed(0) : '999';

    // Hero Banner
    document.getElementById('topClass').innerText = data.predicted_class;
    document.getElementById('topConfidence').innerText = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('latencyBadge').innerText = '⚡ ' + data.latency_ms.toFixed(2) + ' ms';
    document.getElementById('fpsBadge').innerText = fps + ' inf/sec';
    const hdrFps = document.getElementById('hdrFps');
    if (hdrFps) hdrFps.innerText = 'FPS: ' + fps;

    // Render Probabilities
    renderProbabilityList(data.confidences);

    // Deep Stats Rendering
    if (data.stats) {
      const s = data.stats;
      
      // Margin & Entropy badges
      const marginEl = document.getElementById('marginBadge');
      if (marginEl) marginEl.innerText = 'Margin: +' + (s.top_margin * 100).toFixed(1) + '%';
      const entEl = document.getElementById('entropyBadge');
      if (entEl) entEl.innerText = 'H(P): ' + s.entropy_bits.toFixed(2) + ' bits';

      // Decision Theory Tab
      document.getElementById('infoEntropy').innerText = s.entropy_bits.toFixed(3);
      document.getElementById('infoEntropyMax').innerText = 'Max: ' + s.max_entropy_bits.toFixed(2) + ' bits';
      document.getElementById('infoUncertainty').innerText = s.uncertainty_pct.toFixed(1) + '%';
      document.getElementById('infoPerplexity').innerText = s.perplexity.toFixed(2);
      document.getElementById('infoGini').innerText = s.gini_impurity.toFixed(3);

      // Logits Table
      renderLogitsTable(data.confidences, s.raw_logits);

      // Geometry & 28x28 Preview
      renderGeometry(s.geometry);

      // Performance Profiler Tab
      renderPerformance(s.timing, s.runtime, avgLat);

      // 13-Channel Manifold
      renderManifold(s.manifold);

      // Layer Activations
      renderLayers(s.layers);

      // Header System Info
      if (s.runtime) {
        const hdrCores = document.getElementById('hdrCores');
        if (hdrCores) hdrCores.innerText = '⚡ CPU Cores: ' + s.runtime.cpu_cores;
        const hdrMem = document.getElementById('hdrMem');
        if (hdrMem) hdrMem.innerText = 'RAM: ' + s.runtime.heap_alloc_mb.toFixed(1) + ' MB';
      }
    }
  }

  function renderProbabilityList(confidences) {
    const list = document.getElementById('classList');
    list.innerHTML = '';
    const sorted = [...confidences].sort((a, b) => b.confidence - a.confidence);

    sorted.forEach((item, idx) => {
      const pct = (item.confidence * 100).toFixed(1);
      const isTop = idx === 0;
      const row = document.createElement('div');
      row.className = 'class-row' + (isTop ? ' top' : '');
      row.innerHTML =
        '<div class="class-info">' +
          '<span style="' + (isTop ? 'color: var(--accent-blue); font-weight:700;' : '') + '">' +
            (idx < 3 ? '<b style="color:var(--accent-cyan);">#' + (idx+1) + '</b> ' : '') + item.class_name +
          '</span>' +
          '<span style="' + (isTop ? 'color: var(--accent-emerald); font-weight:700;' : 'color: var(--text-secondary);') + '">' + pct + '%</span>' +
        '</div>' +
        '<div class="progress-bg">' +
          '<div class="progress-fill" style="width: ' + pct + '%"></div>' +
        '</div>';
      list.appendChild(row);
    });
  }

  function renderLogitsTable(confidences, logits) {
    const tbody = document.getElementById('logitsTableBody');
    if (!tbody || !logits) return;
    tbody.innerHTML = '';
    const sortedIndices = confidences.map((c, i) => i).sort((a, b) => confidences[b].confidence - confidences[a].confidence);

    sortedIndices.slice(0, 8).forEach(idx => {
      const c = confidences[idx];
      const z = logits[idx];
      const ez = Math.exp(z);
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td style="color: var(--accent-blue); font-weight: 600;">' + c.class_name + '</td>' +
        '<td>' + (z >= 0 ? '+' : '') + z.toFixed(4) + '</td>' +
        '<td>' + (ez > 1000 ? ez.toExponential(2) : ez.toFixed(3)) + '</td>' +
        '<td style="color: var(--accent-emerald); font-weight: 600;">' + (c.confidence * 100).toFixed(2) + '%</td>';
      tbody.appendChild(tr);
    });
  }

  function renderGeometry(geom) {
    if (!geom) return;
    document.getElementById('geomBBoxDim').innerText = geom.bbox_width + ' × ' + geom.bbox_height;
    document.getElementById('geomAspect').innerText = geom.aspect_ratio.toFixed(2);
    document.getElementById('geomFgCount').innerText = geom.foreground_pixels + ' px';
    document.getElementById('geomDensity').innerText = geom.stroke_density_pct.toFixed(1) + '%';
    document.getElementById('geomCentroid').innerText = '(' + geom.centroid_x.toFixed(1) + ', ' + geom.centroid_y.toFixed(1) + ')';
    document.getElementById('geomCoverage').innerText = geom.canvas_coverage_pct.toFixed(1) + '%';

    // Render 28x28 Resampled Preview
    if (prev28Ctx && geom.resampled_28x28) {
      const imgData = prev28Ctx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) {
        const v = geom.resampled_28x28[i];
        imgData.data[i * 4 + 0] = v;
        imgData.data[i * 4 + 1] = v;
        imgData.data[i * 4 + 2] = v;
        imgData.data[i * 4 + 3] = 255;
      }
      prev28Ctx.putImageData(imgData, 0, 0);
    }
  }

  function renderPerformance(timing, runtime, avgLat) {
    if (!timing) return;
    const tot = timing.total_us || 1;
    document.getElementById('tPreUs').innerText = timing.preprocess_us.toFixed(1) + ' µs';
    document.getElementById('tPreMs').innerText = (timing.preprocess_us / 1000).toFixed(3) + ' ms';
    document.getElementById('tPrePct').innerText = ((timing.preprocess_us / tot) * 100).toFixed(1) + '%';

    document.getElementById('tManUs').innerText = timing.manifold_us.toFixed(1) + ' µs';
    document.getElementById('tManMs').innerText = (timing.manifold_us / 1000).toFixed(3) + ' ms';
    document.getElementById('tManPct').innerText = ((timing.manifold_us / tot) * 100).toFixed(1) + '%';

    document.getElementById('tConvUs').innerText = timing.conv_us.toFixed(1) + ' µs';
    document.getElementById('tConvMs').innerText = (timing.conv_us / 1000).toFixed(3) + ' ms';
    document.getElementById('tConvPct').innerText = ((timing.conv_us / tot) * 100).toFixed(1) + '%';

    document.getElementById('tDenseUs').innerText = timing.dense_us.toFixed(1) + ' µs';
    document.getElementById('tDenseMs').innerText = (timing.dense_us / 1000).toFixed(3) + ' ms';
    document.getElementById('tDensePct').innerText = ((timing.dense_us / tot) * 100).toFixed(1) + '%';

    document.getElementById('tSoftUs').innerText = timing.softmax_us.toFixed(1) + ' µs';
    document.getElementById('tSoftMs').innerText = (timing.softmax_us / 1000).toFixed(3) + ' ms';
    document.getElementById('tSoftPct').innerText = ((timing.softmax_us / tot) * 100).toFixed(1) + '%';

    document.getElementById('tTotUs').innerText = timing.total_us.toFixed(1) + ' µs';
    document.getElementById('tTotMs').innerText = (timing.total_us / 1000).toFixed(3) + ' ms';

    // Update stacked bar segments
    const segments = document.querySelectorAll('.stage-segment');
    if (segments.length >= 5) {
      segments[0].style.width = ((timing.preprocess_us / tot) * 100) + '%';
      segments[1].style.width = ((timing.manifold_us / tot) * 100) + '%';
      segments[2].style.width = ((timing.conv_us / tot) * 100) + '%';
      segments[3].style.width = ((timing.dense_us / tot) * 100) + '%';
      segments[4].style.width = ((timing.softmax_us / tot) * 100) + '%';
    }

    document.getElementById('perfThroughput').innerText = timing.throughput_fps.toFixed(0);
    document.getElementById('perfAvgLatency').innerText = avgLat.toFixed(2) + ' ms';

    if (runtime) {
      document.getElementById('perfHeapAlloc').innerText = runtime.heap_alloc_mb.toFixed(1) + ' MB';
      document.getElementById('perfNumGC').innerText = runtime.num_gc;
    }
  }

  function renderManifold(manifold) {
    if (!manifold || !manifold.channel_grids) return;
    const grid = document.getElementById('manifoldGrid');
    grid.innerHTML = '';
    const cmap = cmapSelect ? cmapSelect.value : 'neon';

    manifold.channel_names.forEach((name, idx) => {
      const card = document.createElement('div');
      card.className = 'manifold-card';
      const cCanvas = document.createElement('canvas');
      cCanvas.width = 28;
      cCanvas.height = 28;
      const cCtx = cCanvas.getContext('2d');
      const imgData = cCtx.createImageData(28, 28);
      const rawGrid = manifold.channel_grids[idx];

      for (let i = 0; i < 784; i++) {
        const val = rawGrid[i];
        const [r, g, b] = getColormapColor(val, cmap);
        imgData.data[i * 4 + 0] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = 255;
      }
      cCtx.putImageData(imgData, 0, 0);

      const energy = (manifold.channel_energy[idx] * 100).toFixed(1);
      const peak = manifold.channel_max[idx].toFixed(2);
      const sparsity = manifold.channel_sparsity[idx].toFixed(0);

      card.appendChild(cCanvas);
      card.innerHTML += 
        '<div class="manifold-title" title="' + name + '">' + name + '</div>' +
        '<div class="manifold-sub">E: ' + energy + '% | Pk: ' + peak + '</div>';

      card.addEventListener('click', () => {
        openChannelModal(name, rawGrid, energy, peak, sparsity, cmap);
      });

      grid.appendChild(card);
    });
  }

  function openChannelModal(name, rawGrid, energy, peak, sparsity, cmap) {
    const modal = document.getElementById('channelModal');
    document.getElementById('modalChannelTitle').innerText = name;
    document.getElementById('modalEnergy').innerText = energy + '%';
    document.getElementById('modalPeak').innerText = peak;
    document.getElementById('modalSparsity').innerText = sparsity + '%';

    const mCanvas = document.getElementById('modalCanvas');
    const mCtx = mCanvas.getContext('2d');
    const imgData = mCtx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const [r, g, b] = getColormapColor(rawGrid[i], cmap);
      imgData.data[i * 4 + 0] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    mCtx.putImageData(imgData, 0, 0);
    modal.classList.add('active');
  }

  document.getElementById('modalCloseBtn').addEventListener('click', () => {
    document.getElementById('channelModal').classList.remove('active');
  });

  function renderLayers(layers) {
    if (!layers) return;
    document.getElementById('actConv1Mean').innerText = layers.conv1_mean_act.toFixed(3);
    document.getElementById('actConv1Sparsity').innerText = 'Sparsity: ' + layers.conv1_sparsity_pct.toFixed(1) + '%';
    document.getElementById('actConv2Mean').innerText = layers.conv2_mean_act.toFixed(3);
    document.getElementById('actConv2Sparsity').innerText = 'Sparsity: ' + layers.conv2_sparsity_pct.toFixed(1) + '%';
    document.getElementById('actPoolNorm').innerText = layers.pool512_l2_norm.toFixed(2);
    document.getElementById('fc1ActiveLabel').innerText = 'Active: ' + layers.fc1_active_neurons + ' / 128 (' + (100 - layers.fc1_sparsity_pct).toFixed(1) + '%)';

    // Render FC1 Vector Bars
    const chart = document.getElementById('fc1VectorChart');
    if (chart && layers.fc1_hidden_vector) {
      chart.innerHTML = '';
      const maxVal = Math.max(0.1, ...layers.fc1_hidden_vector);
      layers.fc1_hidden_vector.forEach(v => {
        const bar = document.createElement('div');
        bar.className = 'vector-bar';
        const hPct = Math.min(100, Math.max(4, (v / maxVal) * 100));
        bar.style.height = hPct + '%';
        if (v === 0) {
          bar.style.background = '#334155';
          bar.style.height = '3px';
        } else {
          bar.style.background = 'linear-gradient(to top, var(--accent-blue), var(--accent-cyan))';
        }
        chart.appendChild(bar);
      });
    }
  }

  // Export JSON Diagnostics Telemetry
  document.getElementById('btnExportJson').addEventListener('click', () => {
    if (!lastTelemetry) {
      alert('Draw something first to generate diagnostics telemetry!');
      return;
    }
    const jsonStr = JSON.stringify(lastTelemetry, null, 2);
    navigator.clipboard.writeText(jsonStr).then(() => {
      alert('Telemetry JSON copied to clipboard!');
    }).catch(() => {
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'diagonalnet_telemetry.json';
      a.click();
    });
  });

  // Initial metadata query
  fetch('/api/info').then(r => r.json()).then(info => {
    if (info) {
      if (info.classes) {
        const list = document.getElementById('classList');
        list.innerHTML = '';
        info.classes.forEach((c, idx) => {
          const row = document.createElement('div');
          row.className = 'class-row';
          row.innerHTML =
            '<div class="class-info">' +
              '<span>' + c + '</span>' +
              '<span style="color: var(--text-secondary);">0.0%</span>' +
            '</div>' +
            '<div class="progress-bg"><div class="progress-fill" style="width: 0%"></div></div>';
          list.appendChild(row);
        });
      }
      if (info.parameters) {
        document.getElementById('archTotalParams').innerText = info.parameters.toLocaleString();
        document.getElementById('archWeightsSize').innerText = (info.parameters * 4 / 1024).toFixed(1) + ' KB';
      }
      if (info.num_classes) {
        document.getElementById('archOutputClasses').innerText = info.num_classes + ' Classes';
        const fcParams = (128 * info.num_classes + info.num_classes).toLocaleString();
        document.getElementById('archFCParams').innerText = fcParams + ' (' + (128 * info.num_classes) + ' W + ' + info.num_classes + ' B)';
      }
      if (info.cpu_cores) {
        const hdrCores = document.getElementById('hdrCores');
        if (hdrCores) hdrCores.innerText = '⚡ CPU Cores: ' + info.cpu_cores;
      }
    }
  }).catch(() => {});
</script>
</body>
</html>`

// PredictRequest contains the client drawing payload.
type PredictRequest struct {
	Image string `json:"image"`
}

// ClassConfidence contains an individual class prediction score.
type ClassConfidence struct {
	ClassName  string  `json:"class_name"`
	ClassIndex int     `json:"class_index"`
	Confidence float32 `json:"confidence"`
}

// TimingBreakdown contains microsecond profiling for each pipeline stage.
type TimingBreakdown struct {
	PreprocessUs  float64 `json:"preprocess_us"`
	ManifoldUs    float64 `json:"manifold_us"`
	ConvUs        float64 `json:"conv_us"`
	DenseUs       float64 `json:"dense_us"`
	SoftmaxUs     float64 `json:"softmax_us"`
	TotalUs       float64 `json:"total_us"`
	ThroughputFps float64 `json:"throughput_fps"`
}

// GeometryStats contains spatial, bounding box, and morphological measurements of the input stroke.
type GeometryStats struct {
	BBoxMinX         int     `json:"bbox_min_x"`
	BBoxMinY         int     `json:"bbox_min_y"`
	BBoxMaxX         int     `json:"bbox_max_x"`
	BBoxMaxY         int     `json:"bbox_max_y"`
	BBoxWidth        int     `json:"bbox_width"`
	BBoxHeight       int     `json:"bbox_height"`
	AspectRatio      float64 `json:"aspect_ratio"`
	CanvasCoveragePct float64 `json:"canvas_coverage_pct"`
	ForegroundPixels int     `json:"foreground_pixels"`
	StrokeDensityPct float64 `json:"stroke_density_pct"`
	CentroidX        float64 `json:"centroid_x"`
	CentroidY        float64 `json:"centroid_y"`
	Resampled28x28   []int   `json:"resampled_28x28"`
}

// ManifoldStats contains 13-channel spatial difference manifold metrics and visual heatmaps.
type ManifoldStats struct {
	ChannelNames    []string  `json:"channel_names"`
	ChannelEnergy   []float64 `json:"channel_energy"`
	ChannelMax      []float64 `json:"channel_max"`
	ChannelSparsity []float64 `json:"channel_sparsity"`
	ChannelGrids    [][]uint8 `json:"channel_grids"`
}

// LayerActivationStats contains intermediate layer representations and neuron sparsity.
type LayerActivationStats struct {
	Conv1MeanAct     float64   `json:"conv1_mean_act"`
	Conv1SparsityPct float64   `json:"conv1_sparsity_pct"`
	Conv2MeanAct     float64   `json:"conv2_mean_act"`
	Conv2SparsityPct float64   `json:"conv2_sparsity_pct"`
	Pool512L2Norm    float64   `json:"pool512_l2_norm"`
	FC1HiddenVector  []float32 `json:"fc1_hidden_vector"`
	FC1ActiveNeurons int       `json:"fc1_active_neurons"`
	FC1SparsityPct   float64   `json:"fc1_sparsity_pct"`
}

// RuntimeStats contains host Go runtime and hardware execution diagnostics.
type RuntimeStats struct {
	CPUCores        int     `json:"cpu_cores"`
	Goroutines      int     `json:"goroutines"`
	HeapAllocMB     float64 `json:"heap_alloc_mb"`
	SysMemMB        float64 `json:"sys_mem_mb"`
	NumGC           uint32  `json:"num_gc"`
	ModelParameters int     `json:"model_parameters"`
}

// InferenceDeepStats encapsulates comprehensive neural metrics, information theory, profiler, and representations.
type InferenceDeepStats struct {
	EntropyBits    float64              `json:"entropy_bits"`
	MaxEntropyBits float64              `json:"max_entropy_bits"`
	UncertaintyPct float64              `json:"uncertainty_pct"`
	Perplexity     float64              `json:"perplexity"`
	GiniImpurity   float64              `json:"gini_impurity"`
	TopMargin      float32              `json:"top_margin"`
	Top2Class      string               `json:"top2_class"`
	Top2Confidence float32              `json:"top2_confidence"`
	Top3Class      string               `json:"top3_class"`
	Top3Confidence float32              `json:"top3_confidence"`
	RawLogits      []float32            `json:"raw_logits"`
	Timing         TimingBreakdown      `json:"timing"`
	Geometry       GeometryStats        `json:"geometry"`
	Manifold       ManifoldStats        `json:"manifold"`
	Layers         LayerActivationStats `json:"layers"`
	Runtime        RuntimeStats         `json:"runtime"`
}

// PredictResponse contains the model classification results, confidence array, inference latency, and deep stats.
type PredictResponse struct {
	PredictedClass string              `json:"predicted_class"`
	ClassIndex     int                 `json:"class_index"`
	Confidence     float32             `json:"confidence"`
	Confidences    []ClassConfidence   `json:"confidences"`
	LatencyMs      float64             `json:"latency_ms"`
	IsBlank        bool                `json:"is_blank"`
	Stats          *InferenceDeepStats `json:"stats,omitempty"`
}

// WebImagePreprocessingStats contains bounding box, morphology, and resampled 28x28 normalized values.
type WebImagePreprocessingStats struct {
	BBoxMinX        int
	BBoxMinY        int
	BBoxMaxX        int
	BBoxMaxY        int
	BBoxWidth       int
	BBoxHeight      int
	AspectRatio     float64
	CanvasCoverage  float64
	ForegroundCount int
	StrokeDensity   float64
	CentroidX       float64
	CentroidY       float64
	Resampled28x28  []int
}

// PreprocessWebImageDetailed extracts the centered 28x28 Tensor as well as morphological spatial diagnostics.
func PreprocessWebImageDetailed(src image.Image) (*Tensor, bool, WebImagePreprocessingStats) {
	bounds := src.Bounds()
	gray := image.NewGray(bounds)
	draw.Draw(gray, bounds, src, bounds.Min, draw.Src)

	var sumX, sumY, totalLum float64
	var fgCount int
	w, h := bounds.Dx(), bounds.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			lum := float64(gray.GrayAt(x, y).Y)
			if lum > 10 {
				fgCount++
				sumX += float64(x) * lum
				sumY += float64(y) * lum
				totalLum += lum
			}
		}
	}

	bbox := FindBoundingBox(gray, 10)
	if bbox == nil {
		return NewTensor(1, InputSize, InputSize), true, WebImagePreprocessingStats{}
	}

	centered := PadAndCenter(gray, bbox)
	stretched := ContrastStretch(centered)
	resized := ResizeBilinear(stretched, InputSize, InputSize)
	tensor := GrayImageToTensor(resized)

	resampled := make([]int, InputSize*InputSize)
	for i := range tensor.Data {
		v := int(math.Round(float64(tensor.Data[i] * 255.0)))
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		resampled[i] = v
	}

	bboxW := bbox.MaxX - bbox.MinX + 1
	bboxH := bbox.MaxY - bbox.MinY + 1
	aspectRatio := 1.0
	if bboxH > 0 {
		aspectRatio = float64(bboxW) / float64(bboxH)
	}
	coverage := 0.0
	if w > 0 && h > 0 {
		coverage = (float64(bboxW*bboxH) / float64(w*h)) * 100.0
	}
	density := 0.0
	if bboxW*bboxH > 0 {
		density = (float64(fgCount) / float64(bboxW*bboxH)) * 100.0
	}
	var cx, cy float64
	if totalLum > 0 {
		cx = sumX / totalLum
		cy = sumY / totalLum
	}

	stats := WebImagePreprocessingStats{
		BBoxMinX:        bbox.MinX,
		BBoxMinY:        bbox.MinY,
		BBoxMaxX:        bbox.MaxX,
		BBoxMaxY:        bbox.MaxY,
		BBoxWidth:       bboxW,
		BBoxHeight:      bboxH,
		AspectRatio:     aspectRatio,
		CanvasCoverage:  coverage,
		ForegroundCount: fgCount,
		StrokeDensity:   density,
		CentroidX:       cx,
		CentroidY:       cy,
		Resampled28x28:  resampled,
	}

	return tensor, false, stats
}

// PreprocessWebImage takes an arbitrary decoded image, locates the tight bounding box, centers it with scale-invariant padding,
// stretches stroke contrast, and resamples to an InputSize x InputSize Tensor normalized to [0.0, 1.0].
func PreprocessWebImage(src image.Image) (*Tensor, bool) {
	tensor, isBlank, _ := PreprocessWebImageDetailed(src)
	return tensor, isBlank
}

// InferenceServer hosts the HTTP web canvas UI and real-time prediction API.
type InferenceServer struct {
	Model      *DiagonalNetModel
	ClassNames []string
	Port       int
}

// NewInferenceServer constructs a new inference web server.
func NewInferenceServer(model *DiagonalNetModel, classNames []string, port int) *InferenceServer {
	if port <= 0 {
		port = 8081
	}
	if len(classNames) < model.NumClasses {
		classNames = make([]string, model.NumClasses)
		for i := 0; i < model.NumClasses; i++ {
			classNames[i] = fmt.Sprintf("Class_%d", i)
		}
	}
	return &InferenceServer{
		Model:      model,
		ClassNames: classNames,
		Port:       port,
	}
}

// ServeHTTP handles routing for the web application and REST API.
func (s *InferenceServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/", "/index.html":
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(webAppHTML))
	case "/api/predict":
		s.handlePredict(w, r)
	case "/api/info":
		s.handleInfo(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (s *InferenceServer) handleInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	totalParams := CountModelParameters(s.Model.Parameters())

	json.NewEncoder(w).Encode(map[string]interface{}{
		"classes":      s.ClassNames,
		"num_classes":  s.Model.NumClasses,
		"parameters":   totalParams,
		"cpu_cores":    runtime.NumCPU(),
		"goroutines":   runtime.NumGoroutine(),
		"heap_alloc_mb": float64(mem.HeapAlloc) / (1024.0 * 1024.0),
		"sys_mem_mb":   float64(mem.Sys) / (1024.0 * 1024.0),
	})
}

var manifoldChannelDisplayNames = []string{
	"Ch 0: Base Grayscale (I₀)",
	"Ch 1: Diag ↖ (-1,-1)",
	"Ch 2: Diag ↗ (+1,-1)",
	"Ch 3: Diag ↙ (-1,+1)",
	"Ch 4: Diag ↘ (+1,+1)",
	"Ch 5: Knight 1 (-2,-1)",
	"Ch 6: Knight 2 (-2,+1)",
	"Ch 7: Knight 3 (-1,-2)",
	"Ch 8: Knight 4 (-1,+2)",
	"Ch 9: Knight 5 (+1,-2)",
	"Ch 10: Knight 6 (+1,+2)",
	"Ch 11: Knight 7 (+2,-1)",
	"Ch 12: Knight 8 (+2,+1)",
}

func (s *InferenceServer) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON request: %v", err), http.StatusBadRequest)
		return
	}

	rawB64 := req.Image
	if idx := strings.Index(rawB64, ","); idx != -1 {
		rawB64 = rawB64[idx+1:]
	}

	imgBytes, err := base64.StdEncoding.DecodeString(rawB64)
	if err != nil {
		http.Error(w, fmt.Sprintf("Base64 decode error: %v", err), http.StatusBadRequest)
		return
	}

	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		http.Error(w, fmt.Sprintf("Image decode error: %v", err), http.StatusBadRequest)
		return
	}

	tStart := time.Now()

	// 1. Preprocessing stage
	tensor, isBlank, pStats := PreprocessWebImageDetailed(img)
	tPreprocess := float64(time.Since(tStart).Microseconds())

	var predClass int
	var maxProb float32
	confidences := make([]ClassConfidence, s.Model.NumClasses)

	var tManifold, tConv, tDense, tSoftmax float64
	var rawLogits []float32
	var deepStats *InferenceDeepStats

	if isBlank {
		for i := 0; i < s.Model.NumClasses; i++ {
			confidences[i] = ClassConfidence{
				ClassName:  s.ClassNames[i],
				ClassIndex: i,
				Confidence: 0.0,
			}
		}
	} else {
		s.Model.SetTraining(false)

		// 2. 13-Channel Manifold stage
		t1 := time.Now()
		manifold := ComputeManifoldTensor(tensor)
		tManifold = float64(time.Since(t1).Microseconds())

		// Extract 13-Channel Manifold stats and heatmap grids
		hw := InputSize * InputSize
		channelGrids := make([][]uint8, 13)
		channelEnergy := make([]float64, 13)
		channelMax := make([]float64, 13)
		channelSparsity := make([]float64, 13)

		for c := 0; c < 13; c++ {
			grid := make([]uint8, hw)
			var sum float64
			var maxV float32
			var zeroCount int
			offset := c * hw
			for i := 0; i < hw; i++ {
				val := manifold.Data[offset+i]
				if val > maxV {
					maxV = val
				}
				if val < 0.01 {
					zeroCount++
				}
				sum += float64(val)
				v := int(val * 255.0)
				if v < 0 {
					v = 0
				} else if v > 255 {
					v = 255
				}
				grid[i] = uint8(v)
			}
			channelGrids[c] = grid
			channelEnergy[c] = sum / float64(hw)
			channelMax[c] = float64(maxV)
			channelSparsity[c] = (float64(zeroCount) / float64(hw)) * 100.0
		}

		// 3. Convolutional stages
		t2 := time.Now()
		c1 := s.Model.Conv1.Forward(manifold)
		a1 := s.Model.ReLU1.ForwardTensor(c1)
		p1 := s.Model.Pool1.Forward(a1)
		c2 := s.Model.Conv2.Forward(p1)
		a2 := s.Model.ReLU2.ForwardTensor(c2)
		p2 := s.Model.Pool2.Forward(a2)
		tConv = float64(time.Since(t2).Microseconds())

		// Conv Activation Metrics
		var c1Sum float64
		var c1Zero int
		for _, v := range a1.Data {
			if v == 0 {
				c1Zero++
			}
			c1Sum += float64(v)
		}
		conv1Mean := c1Sum / float64(len(a1.Data))
		conv1Sparsity := (float64(c1Zero) / float64(len(a1.Data))) * 100.0

		var c2Sum float64
		var c2Zero int
		for _, v := range a2.Data {
			if v == 0 {
				c2Zero++
			}
			c2Sum += float64(v)
		}
		conv2Mean := c2Sum / float64(len(a2.Data))
		conv2Sparsity := (float64(c2Zero) / float64(len(a2.Data))) * 100.0

		// 4. Dense & Pooling stages
		t3 := time.Now()
		pooled := s.Model.Pool.Forward(p2)
		var poolSqSum float64
		for _, v := range pooled.Data {
			poolSqSum += float64(v * v)
		}
		pool512Norm := math.Sqrt(poolSqSum)

		h := s.Model.FC1.Forward(pooled.Data)
		hAct := s.Model.ReLU3.Forward(h)
		s.Model.Dropout.Training = false
		hDrop := s.Model.Dropout.Forward(hAct)
		logits := s.Model.FC.Forward(hDrop)
		tDense = float64(time.Since(t3).Microseconds())

		rawLogits = make([]float32, len(logits))
		copy(rawLogits, logits)

		fc1Vector := make([]float32, len(hAct))
		copy(fc1Vector, hAct)
		var fc1Active int
		for _, v := range hAct {
			if v > 0 {
				fc1Active++
			}
		}
		fc1Sparsity := (float64(len(hAct)-fc1Active) / float64(len(hAct))) * 100.0

		// 5. Softmax & Decision analytics
		t4 := time.Now()
		probs := Softmax(logits)

		var entropy float64
		var sumSq float64
		for i := 0; i < s.Model.NumClasses; i++ {
			p := probs[i]
			if p > maxProb {
				maxProb = p
				predClass = i
			}
			confidences[i] = ClassConfidence{
				ClassName:  s.ClassNames[i],
				ClassIndex: i,
				Confidence: p,
			}
			if p > 1e-12 {
				entropy -= float64(p) * math.Log2(float64(p))
			}
			sumSq += float64(p * p)
		}

		maxEntropy := math.Log2(float64(s.Model.NumClasses))
		if maxEntropy <= 0 {
			maxEntropy = 1.0
		}
		uncertaintyPct := (entropy / maxEntropy) * 100.0
		perplexity := math.Pow(2.0, entropy)
		gini := 1.0 - sumSq

		type classScore struct {
			name string
			prob float32
		}
		scores := make([]classScore, s.Model.NumClasses)
		for i := 0; i < s.Model.NumClasses; i++ {
			scores[i] = classScore{name: s.ClassNames[i], prob: probs[i]}
		}
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].prob > scores[j].prob
		})

		topMargin := maxProb
		top2Class := ""
		top2Prob := float32(0.0)
		top3Class := ""
		top3Prob := float32(0.0)
		if len(scores) > 1 {
			top2Class = scores[1].name
			top2Prob = scores[1].prob
			topMargin = maxProb - top2Prob
		}
		if len(scores) > 2 {
			top3Class = scores[2].name
			top3Prob = scores[2].prob
		}

		tSoftmax = float64(time.Since(t4).Microseconds())
		totalUs := float64(time.Since(tStart).Microseconds())
		fps := 0.0
		if totalUs > 0 {
			fps = 1000000.0 / totalUs
		}

		var mem runtime.MemStats
		runtime.ReadMemStats(&mem)

		deepStats = &InferenceDeepStats{
			EntropyBits:    entropy,
			MaxEntropyBits: maxEntropy,
			UncertaintyPct: uncertaintyPct,
			Perplexity:     perplexity,
			GiniImpurity:   gini,
			TopMargin:      topMargin,
			Top2Class:      top2Class,
			Top2Confidence: top2Prob,
			Top3Class:      top3Class,
			Top3Confidence: top3Prob,
			RawLogits:      rawLogits,
			Timing: TimingBreakdown{
				PreprocessUs:  tPreprocess,
				ManifoldUs:    tManifold,
				ConvUs:        tConv,
				DenseUs:       tDense,
				SoftmaxUs:     tSoftmax,
				TotalUs:       totalUs,
				ThroughputFps: fps,
			},
			Geometry: GeometryStats{
				BBoxMinX:          pStats.BBoxMinX,
				BBoxMinY:          pStats.BBoxMinY,
				BBoxMaxX:          pStats.BBoxMaxX,
				BBoxMaxY:          pStats.BBoxMaxY,
				BBoxWidth:         pStats.BBoxWidth,
				BBoxHeight:        pStats.BBoxHeight,
				AspectRatio:       pStats.AspectRatio,
				CanvasCoveragePct: pStats.CanvasCoverage,
				ForegroundPixels:  pStats.ForegroundCount,
				StrokeDensityPct:  pStats.StrokeDensity,
				CentroidX:         pStats.CentroidX,
				CentroidY:         pStats.CentroidY,
				Resampled28x28:    pStats.Resampled28x28,
			},
			Manifold: ManifoldStats{
				ChannelNames:    manifoldChannelDisplayNames,
				ChannelEnergy:   channelEnergy,
				ChannelMax:      channelMax,
				ChannelSparsity: channelSparsity,
				ChannelGrids:    channelGrids,
			},
			Layers: LayerActivationStats{
				Conv1MeanAct:     conv1Mean,
				Conv1SparsityPct: conv1Sparsity,
				Conv2MeanAct:     conv2Mean,
				Conv2SparsityPct: conv2Sparsity,
				Pool512L2Norm:    pool512Norm,
				FC1HiddenVector:  fc1Vector,
				FC1ActiveNeurons: fc1Active,
				FC1SparsityPct:   fc1Sparsity,
			},
			Runtime: RuntimeStats{
				CPUCores:        runtime.NumCPU(),
				Goroutines:      runtime.NumGoroutine(),
				HeapAllocMB:     float64(mem.HeapAlloc) / (1024.0 * 1024.0),
				SysMemMB:        float64(mem.Sys) / (1024.0 * 1024.0),
				NumGC:           mem.NumGC,
				ModelParameters: CountModelParameters(s.Model.Parameters()),
			},
		}
	}

	latencyMs := float64(time.Since(tStart).Microseconds()) / 1000.0

	resp := PredictResponse{
		PredictedClass: s.ClassNames[predClass],
		ClassIndex:     predClass,
		Confidence:     maxProb,
		Confidences:    confidences,
		LatencyMs:      latencyMs,
		IsBlank:        isBlank,
		Stats:          deepStats,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// OpenBrowser opens the specified URL in the system's default web browser across Windows, macOS, and Linux.
func OpenBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start()
}

// StartInferenceServer launches the HTTP web server and optionally opens the browser.
func StartInferenceServer(model *DiagonalNetModel, classNames []string, port int, autoOpen bool) error {
	srv := NewInferenceServer(model, classNames, port)
	addr := fmt.Sprintf(":%d", port)
	url := fmt.Sprintf("http://localhost:%d", port)

	fmt.Println("====================================================================================================")
	fmt.Println("                      DIAGONALNET REAL-TIME INFERENCE & WEB RUNTIME")
	fmt.Println("====================================================================================================")
	fmt.Printf(" Server URL         : %s\n", url)
	fmt.Printf(" HTTP Listen Port   : %d\n", port)
	fmt.Printf(" Model Classes (K=%d): %v\n", model.NumClasses, classNames)
	fmt.Printf(" CPU Workers        : %d Logical Cores\n", runtime.NumCPU())
	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Println(" Access the interactive web canvas in your browser. Press Ctrl+C to terminate.")
	fmt.Println("====================================================================================================")

	if autoOpen {
		go func() {
			time.Sleep(300 * time.Millisecond)
			OpenBrowser(url)
		}()
	}

	return http.ListenAndServe(addr, srv)
}

// ============================================================================
// 11. CLI ROUTING & EXECUTION HANDLERS
// ============================================================================

func printHelp() {
	fmt.Print(banner)
	fmt.Println("Usage:")
	fmt.Println("  diagonalnet [mode flag] [configuration flags]")
	fmt.Println("  diagonalnet [subcommand] [configuration flags]")
	fmt.Println()
	fmt.Println("Execution Modes:")
	fmt.Println("  -train          Launch deep learning model training pipeline")
	fmt.Println("  -serve          Start the interactive HTTP inference and dashboard server")
	fmt.Println("  -audit          Run dataset validation, shape verification, and integrity audit")
	fmt.Println()
	fmt.Println("Positional Subcommands:")
	fmt.Println("  train, serve, audit, help")
	fmt.Println()
	fmt.Println("Configuration Flags:")
	fmt.Println("  -data string    Path to dataset samples directory (default \"data\")")
	fmt.Println("  -model string   Path to binary model weights file (default \"weights/diagonalnet_model.bin\")")
	fmt.Println("  -epochs int     Number of training epochs (default 8)")
	fmt.Println("  -lr float       Learning rate for parameter optimization (default 0.002)")
	fmt.Println("  -batch int      Mini-batch size for training (default 64)")
	fmt.Println("  -port int       HTTP server listen port (default 8081)")
	fmt.Println("  -help, -h       Display this help and exit")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  diagonalnet -train -data data -epochs 10 -lr 0.001 -batch 32")
	fmt.Println("  diagonalnet -serve -model weights/diagonalnet_model.bin -port 8081")
	fmt.Println("  diagonalnet -audit -data data")
	fmt.Println()
}

func runAudit(dataDir string) {
	fmt.Println(">>> [Audit Mode] Starting dataset validation, quality analysis, and integrity audit...")
	report, err := AuditDataset(dataDir)
	if err != nil {
		fmt.Printf(">>> [Audit Warning/Error] %v\n", err)
		return
	}
	PrintAuditReport(report)
	fmt.Println(">>> Dataset audit completed.")
}

func runTrain(dataDir string, modelPath string, epochs int, lr float32, batchSize int, profile string) {
	profile = strings.ToLower(strings.TrimSpace(profile))
	profileTitle := "Standard Configuration"
	switch profile {
	case "fast", "quick":
		epochs = 4
		batchSize = 64
		lr = 0.0025
		profileTitle = "Fast Training Profile (4 Epochs, Batch: 64, LR: 0.0025)"
	case "normal", "standard":
		epochs = 12
		batchSize = 32
		lr = 0.0020
		profileTitle = "Normal Standard Profile (12 Epochs, Batch: 32, LR: 0.0020) [Recommended]"
	case "hardcore", "deep":
		epochs = 30
		batchSize = 32
		lr = 0.0020
		profileTitle = "Hardcore Deep Profile (30 Epochs, Batch: 32, LR: 0.0020, 15x Augmentation)"
	case "manual", "custom":
		profileTitle = "Manual Custom Profile"
	}

	fmt.Println("====================================================================================================")
	fmt.Println("                        DIAGONALNET DEEP LEARNING MODEL TRAINING PIPELINE")
	fmt.Println("====================================================================================================")
	fmt.Printf(" Training Profile  : %s\n", profileTitle)
	fmt.Printf(" Dataset Directory : %s\n", dataDir)
	fmt.Printf(" Target Model Path : %s\n", modelPath)
	fmt.Printf(" Training Epochs   : %d\n", epochs)
	fmt.Printf(" Learning Rate     : %.4f\n", lr)
	fmt.Printf(" Mini-Batch Size   : %d\n", batchSize)
	fmt.Printf(" Parallel Workers  : %d Logical CPU Cores\n", NumWorkers())
	fmt.Println("----------------------------------------------------------------------------------------------------")

	// 1. Scan dataset
	ds, err := ScanDataset(dataDir)
	if err != nil {
		fmt.Printf(">>> [Dataset Error] %v\n", err)
		return
	}
	fmt.Printf(" Discovered %d Dynamic Classes (K=%d): %v\n", ds.Metadata.NumClasses, ds.Metadata.NumClasses, ds.Metadata.Classes)
	fmt.Printf(" Total Discovered Image Files : %d\n", len(ds.Samples))

	// 2. Stratified Train / Validation Split (80% Train, 20% Val)
	trainItems, valItems := TrainTestSplit(ds.Samples, 0.20, 42)

	loadAndPreprocess := func(items []ImageItem, label string, augment bool) []Sample {
		augStr := "1x (Raw)"
		if augment {
			augStr = "15x (Rotations, Scale/Aspect, Shears, Dilation, Erosion)"
		}
		fmt.Printf(" Preprocessing %d %s samples [%s]... ", len(items), label, augStr)
		start := time.Now()

		numWorkers := NumWorkers()
		if numWorkers > len(items) {
			numWorkers = len(items)
		}
		if numWorkers <= 0 {
			numWorkers = 1
		}

		type workerResult struct {
			samples []Sample
		}
		results := make([]workerResult, numWorkers)
		chunkSize := (len(items) + numWorkers - 1) / numWorkers
		var wg sync.WaitGroup

		for w := 0; w < numWorkers; w++ {
			s := w * chunkSize
			e := s + chunkSize
			if e > len(items) {
				e = len(items)
			}
			if s >= e {
				continue
			}

			wg.Add(1)
			go func(workerIdx, startIdx, endIdx int) {
				defer wg.Done()
				localSamples := make([]Sample, 0, (endIdx-startIdx)*15)
				for i := startIdx; i < endIdx; i++ {
					it := items[i]
					gray, err := LoadImageFromFile(it.Path)
					if err != nil {
						continue
					}

					var variants []*image.Gray
					if augment {
						variants = AugmentImage(gray)
					} else {
						variants = []*image.Gray{gray}
					}

					for _, v := range variants {
						bbox := FindBoundingBox(v, 10)
						if bbox == nil {
							// Blank variant: no foreground stroke survived. Feeding an all-zero image
							// under a real class label is pure label noise, so drop it.
							continue
						}
						centered := PadAndCenter(v, bbox)
						stretched := ContrastStretch(centered)
						resized := ResizeBilinear(stretched, InputSize, InputSize)
						tensor := GrayImageToTensor(resized)
						localSamples = append(localSamples, Sample{
							Input:       tensor,
							TargetClass: it.ClassIndex,
						})
					}
				}
				results[workerIdx].samples = localSamples
			}(w, s, e)
		}
		wg.Wait()

		var totalCount int
		for _, r := range results {
			totalCount += len(r.samples)
		}
		samples := make([]Sample, 0, totalCount)
		for _, r := range results {
			samples = append(samples, r.samples...)
		}

		fmt.Printf("Done (%.2fs | %d clean samples)\n", time.Since(start).Seconds(), len(samples))
		return samples
	}

	trainSamples := loadAndPreprocess(trainItems, "training", true)
	valSamples := loadAndPreprocess(valItems, "validation", false)

	if len(trainSamples) == 0 || len(valSamples) == 0 {
		fmt.Println(">>> [Error] Insufficient valid training or validation samples.")
		return
	}

	// 3. Initialize Model, Optimizer, BatchTrainer, Checkpoint
	numClasses := ds.Metadata.NumClasses
	rng := rand.New(rand.NewSource(42))
	masterModel := NewDiagonalNetModel(numClasses, rng)
	paramCount := CountModelParameters(masterModel.Parameters())

	optimizer := NewAdamOptimizer(masterModel.Parameters(), AdamOptimizerConfig{
		LearningRate: lr,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  1e-4,
	})

	trainer := NewBatchTrainer(masterModel, optimizer, NumWorkers())
	checkpoint := NewModelCheckpoint()

	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Printf(" Input Resolution  : %dx%d (bbox-cropped, centered, contrast-stretched)\n", InputSize, InputSize)
	fmt.Printf(" Model Architecture: 13-Manifold -> Conv(13->%d,K3,S1) -> ReLU -> MaxPool2 -> Conv(%d->%d,K3,S1) -> ReLU -> MaxPool2\n",
		diagonalConv1Channels, diagonalConv1Channels, diagonalConv2Channels)
	fmt.Printf("                     -> AdaptiveAvgPool(%dx%d) -> Linear(%d->%d) -> ReLU -> Dropout(0.2) -> Linear(%d->%d)\n",
		diagonalPoolTarget, diagonalPoolTarget,
		diagonalConv2Channels*diagonalPoolTarget*diagonalPoolTarget, diagonalHiddenUnits,
		diagonalHiddenUnits, numClasses)
	fmt.Printf(" Trainable Parameters: %d float32 weights and biases\n", paramCount)
	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Println(" Starting Data-Parallel Training Across CPU Cores...")
	fmt.Println("----------------------------------------------------------------------------------------------------")

	N := len(trainSamples)
	indices := make([]int, N)
	for i := 0; i < N; i++ {
		indices[i] = i
	}
	shuffleRng := rand.New(rand.NewSource(42))

	trainStart := time.Now()

	for ep := 1; ep <= epochs; ep++ {
		epStart := time.Now()

		// Dynamic Proportional Step LR schedule (Milestone 1 at ~40%, Milestone 2 at ~75%)
		m1 := epochs * 40 / 100
		if m1 < 4 {
			m1 = 4
		}
		m2 := epochs * 75 / 100
		if m2 <= m1 {
			m2 = m1 + 2
		}

		if ep == m1 {
			optimizer.Config.LearningRate = lr * 0.50
			fmt.Printf(" >>> [LR Scheduler] Epoch %d: Learning rate adjusted to %.6f (50%%)\n", ep, optimizer.Config.LearningRate)
		} else if ep == m2 {
			optimizer.Config.LearningRate = lr * 0.20
			fmt.Printf(" >>> [LR Scheduler] Epoch %d: Learning rate adjusted to %.6f (20%%)\n", ep, optimizer.Config.LearningRate)
		}

		// Shuffle training data
		shuffleRng.Shuffle(N, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		var totalEpochLoss float32
		var totalEpochAcc float32
		numBatches := (N + batchSize - 1) / batchSize

		for b := 0; b < numBatches; b++ {
			start := b * batchSize
			end := start + batchSize
			if end > N {
				end = N
			}
			currBatch := make([]Sample, end-start)
			for i := start; i < end; i++ {
				currBatch[i-start] = trainSamples[indices[i]]
			}

			batchLoss, batchAcc := trainer.TrainBatch(currBatch)
			totalEpochLoss += batchLoss
			totalEpochAcc += batchAcc

			// Real-time batch progress report every 50 batches or at end
			if (b+1)%50 == 0 || b+1 == numBatches {
				pct := float64(b+1) / float64(numBatches) * 100.0
				runningLoss := totalEpochLoss / float32(b+1)
				runningAcc := (totalEpochAcc / float32(b+1)) * 100.0
				elapsedSec := time.Since(epStart).Seconds()
				var speed float64
				if elapsedSec > 0 {
					speed = float64((b + 1) * batchSize) / elapsedSec
				}
				fmt.Printf("\r [Epoch %2d/%2d] Batch [%4d/%4d] (%5.1f%%) | Loss: %.4f (Acc: %5.1f%%) | Speed: %4.0f img/s",
					ep, epochs, b+1, numBatches, pct, runningLoss, runningAcc, speed)
				_ = os.Stdout.Sync()
			}
		}

		avgTrainLoss := totalEpochLoss / float32(numBatches)
		avgTrainAcc := (totalEpochAcc / float32(numBatches)) * 100.0

		// Evaluate on Validation Set
		valLoss, valAcc := trainer.Evaluate(valSamples)
		valAccPct := float64(valAcc) * 100.0

		isBest := checkpoint.Update(masterModel, ep, float64(valAcc))
		bestTag := ""
		if isBest {
			bestTag = " [BEST]"
		}

		epDuration := time.Since(epStart).Seconds()
		fmt.Printf("\r Epoch [%2d/%2d] | Train Loss: %.4f (Acc: %5.1f%%) | Val Loss: %.4f (Acc: %5.1f%%) | Time: %5.2fs%s\n",
			ep, epochs, avgTrainLoss, avgTrainAcc, valLoss, valAccPct, epDuration, bestTag)
	}

	totalTrainDuration := time.Since(trainStart).Seconds()
	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Printf(" Training Completed in %.2f seconds (%.2fs / epoch).\n", totalTrainDuration, totalTrainDuration/float64(epochs))

	// Restore best checkpoint weights
	checkpoint.RestoreBest(masterModel)
	fmt.Printf(" Restored optimal weights from Epoch %d (Best Validation Accuracy: %.2f%%).\n",
		checkpoint.BestEpoch, checkpoint.BestValAcc*100.0)

	// Compute & Print Comprehensive Multi-Class Evaluation Report
	evalReport := ComputeEvaluationMetrics(masterModel, valSamples, ds.Metadata.Classes)
	PrintEvaluationReport(evalReport)

	// Save binary model weights
	if err := SaveModelWeights(modelPath, masterModel.Parameters(), ds.Metadata.Classes); err != nil {
		fmt.Printf(">>> [Save Error] Failed to save weights to %s: %v\n", modelPath, err)
	} else {
		fmt.Printf(">>> Successfully serialized best model weights (DIAGON01) to: %s\n", modelPath)
	}
	fmt.Println("====================================================================================================")
}

func runServer(modelPath string, port int) {
	fmt.Println(">>> [Serve Mode] Initializing interactive HTTP inference runtime...")

	var model *DiagonalNetModel
	var classNames []string

	// 1. Try loading weights from disk if file exists
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Printf("    Loading model weights from: %s\n", modelPath)
		tempModel := NewDiagonalNetModel(2, nil)
		loadedClasses, err := LoadModelWeights(modelPath, tempModel.Parameters())
		if err == nil && len(loadedClasses) >= 2 {
			classNames = loadedClasses
			model = NewDiagonalNetModel(len(classNames), nil)
			_, _ = LoadModelWeights(modelPath, model.Parameters())
			fmt.Printf("    Successfully loaded weights for %d classes: %v\n", len(classNames), classNames)
		} else {
			// A checkpoint written by an older build has a different parameter layout and will
			// fail here. Say so loudly: silently serving random weights looks like a broken
			// model rather than a stale file.
			fmt.Printf("    [WARNING] Could not load %s: %v\n", modelPath, err)
			fmt.Println("    [WARNING] The checkpoint does not match the current network layout.")
			fmt.Println("    [WARNING] Retrain before serving:  diagonalnet -train -data data -profile normal")
		}
	}

	// 2. Fallback to scanning dataset directory if model weights not yet saved
	if model == nil {
		ds, err := ScanDataset("data")
		if err == nil && ds.Metadata.NumClasses >= 2 {
			classNames = ds.Metadata.Classes
			model = NewDiagonalNetModel(len(classNames), rand.New(rand.NewSource(42)))
			fmt.Printf("    [WARNING] Serving UNTRAINED He-initialized weights for discovered classes: %v\n", classNames)
		} else {
			classNames = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
			model = NewDiagonalNetModel(10, rand.New(rand.NewSource(42)))
			fmt.Println("    [WARNING] Serving UNTRAINED He-initialized weights for standard digits (0-9)")
		}
		fmt.Println("    [WARNING] Predictions will be random until the model is trained.")
	}

	if err := StartInferenceServer(model, classNames, port, true); err != nil {
		fmt.Printf(">>> [Server Error] %v\n", err)
	}
}

func main() {
	// 1. Configure multi-core parallel runtime settings
	runtime.GOMAXPROCS(runtime.NumCPU())
	PrintHardwareDiagnostics()

	fs := flag.NewFlagSet("diagonalnet", flag.ContinueOnError)
	fs.Usage = printHelp

	trainFlag := fs.Bool("train", false, "Launch deep learning model training pipeline")
	serveFlag := fs.Bool("serve", false, "Start the interactive HTTP inference and dashboard server")
	auditFlag := fs.Bool("audit", false, "Run dataset validation, shape verification, and integrity audit")

	dataDir := fs.String("data", "data", "Path to dataset directory")
	modelPath := fs.String("model", "weights/diagonalnet_model.bin", "Path to binary model weights")
	epochs := fs.Int("epochs", 8, "Number of training epochs")
	lr := fs.Float64("lr", 0.002, "Learning rate for optimization")
	batchSize := fs.Int("batch", 64, "Mini-batch size for training")
	port := fs.Int("port", 8081, "HTTP server listen port")
	profileFlag := fs.String("profile", "", "Training profile template: fast, normal, hardcore, manual")

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
		runTrain(*dataDir, *modelPath, *epochs, float32(*lr), *batchSize, *profileFlag)
	case *serveFlag:
		runServer(*modelPath, *port)
	default:
		// Positional fallback check from flag.Args() if flags were mixed
		if fs.NArg() > 0 {
			switch strings.ToLower(fs.Arg(0)) {
			case "train":
				runTrain(*dataDir, *modelPath, *epochs, float32(*lr), *batchSize, *profileFlag)
				return
			case "serve":
				runServer(*modelPath, *port)
				return
			case "audit":
				runAudit(*dataDir)
				return
			}
		}
		printHelp()
	}
}
