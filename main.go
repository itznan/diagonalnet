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
// 6. DATASET SCANNER & DYNAMIC CLASS MAPPING (PROMPTS 29 & 30)
// ============================================================================

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
			for dy := -1; dy <= 1; dy++ {
				ny := y + dy
				if ny < 0 || ny >= h {
					minVal = 0
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					nx := x + dx
					if nx < 0 || nx >= w {
						minVal = 0
						continue
					}
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

// AugmentImage generates 15 comprehensive geometric and morphological variants per training image:
// 1. Original image
// 2-5. Rotations: -15 deg, +15 deg, -10 deg, +10 deg
// 6-11. Shifts: (-3, 0), (+3, 0), (0, -3), (0, +3), (+2, +2), (-2, -2)
// 12-13. Horizontal shears: -0.20, +0.20
// 14. Morphological dilation (stroke thickening)
// 15. Morphological erosion (stroke thinning)
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

	// 6-11. Shifts
	variants = append(variants, ShiftImage(src, -3, 0))
	variants = append(variants, ShiftImage(src, 3, 0))
	variants = append(variants, ShiftImage(src, 0, -3))
	variants = append(variants, ShiftImage(src, 0, 3))
	variants = append(variants, ShiftImage(src, 2, 2))
	variants = append(variants, ShiftImage(src, -2, -2))

	// 12-13. Horizontal Shears
	variants = append(variants, ShearImage(src, -0.20))
	variants = append(variants, ShearImage(src, 0.20))

	// 14. Morphological Dilation
	variants = append(variants, MorphDilation(src))

	// 15. Morphological Erosion
	variants = append(variants, MorphErosion(src))

	return variants
}

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
	fmt.Println("                              DIAGONNET DATASET AUDIT & HEALTH REPORT")
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
// 8. END-TO-END MODEL ARCHITECTURE & MULTI-CORE BATCH TRAINER (Prompts 41 & 42)
// ============================================================================

// Sample encapsulates an input feature tensor and integer class target.
type Sample struct {
	Input       *Tensor // [1 x H x W] or [13 x H x W]
	TargetClass int     // Integer label in [0, K-1]
}

// DiagonNetModel represents the complete neural network architecture for DiagonNet:
// 13-Channel Manifold -> Conv2D (13->16, K=3, S=2, P=1) -> ReLU -> AdaptiveAvgPool (4x4) -> Dropout (p=0.2) -> Linear (256->K)
type DiagonNetModel struct {
	NumClasses int
	Conv       *Conv2DLayer
	ReLU       *ReLULayer
	Pool       *AdaptiveAvgPool2DLayer
	Dropout    *DropoutLayer
	FC         *LinearLayer
	LossFn     *CategoricalCrossEntropyLoss
}

// NewDiagonNetModel constructs a DiagonNet classification model configured dynamically for K classes.
func NewDiagonNetModel(numClasses int, rng *rand.Rand) *DiagonNetModel {
	if numClasses < 2 {
		numClasses = 2
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	convRNG := rand.New(rand.NewSource(rng.Int63()))
	fcRNG := rand.New(rand.NewSource(rng.Int63()))
	dropRNG := rand.New(rand.NewSource(rng.Int63()))

	conv := NewConv2DLayer(13, 16, 3, 2, 1, convRNG)
	relu := NewReLULayer()
	pool := NewAdaptiveAvgPool2DLayer(4, 4)
	dropout := NewDropoutLayer(0.2, dropRNG)
	fc := NewLinearLayer(16*4*4, numClasses, fcRNG)
	lossFn := NewCategoricalCrossEntropyLoss()

	return &DiagonNetModel{
		NumClasses: numClasses,
		Conv:       conv,
		ReLU:       relu,
		Pool:       pool,
		Dropout:    dropout,
		FC:         fc,
		LossFn:     lossFn,
	}
}

// Parameters returns all trainable parameter buffers in the model.
func (m *DiagonNetModel) Parameters() []*Parameter {
	return []*Parameter{
		m.Conv.Weights,
		m.Conv.Bias,
		m.FC.Weights,
		m.FC.Biases,
	}
}

// ZeroGrad resets analytical Jacobian gradient buffers for all parameters to zero.
func (m *DiagonNetModel) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

// SetTraining toggles training vs evaluation mode (affecting Dropout regularization).
func (m *DiagonNetModel) SetTraining(training bool) {
	m.Dropout.Training = training
}

// CloneForWorker constructs an isolated model replica for a parallel batch worker,
// with independent gradient and layer state buffers.
func (m *DiagonNetModel) CloneForWorker(workerID int) *DiagonNetModel {
	rng := rand.New(rand.NewSource(int64(1000 + workerID*37)))
	replica := &DiagonNetModel{
		NumClasses: m.NumClasses,
		Conv: &Conv2DLayer{
			InChannels:  m.Conv.InChannels,
			OutChannels: m.Conv.OutChannels,
			KernelSize:  m.Conv.KernelSize,
			Stride:      m.Conv.Stride,
			Padding:     m.Conv.Padding,
			Weights:     NewParameter(m.Conv.Weights.Size()),
			Bias:        NewParameter(m.Conv.Bias.Size()),
		},
		ReLU:    NewReLULayer(),
		Pool:    NewAdaptiveAvgPool2DLayer(m.Pool.TargetH, m.Pool.TargetW),
		Dropout: NewDropoutLayer(m.Dropout.DropRate, rng),
		FC: &LinearLayer{
			Weights:   NewParameter(m.FC.Weights.Size()),
			Biases:    NewParameter(m.FC.Biases.Size()),
			InputDim:  m.FC.InputDim,
			OutputDim: m.FC.OutputDim,
		},
		LossFn: NewCategoricalCrossEntropyLoss(),
	}
	replica.SyncWeightsFrom(m)
	return replica
}

// SyncWeightsFrom copies trainable weight vectors from the master model into the replica.
func (m *DiagonNetModel) SyncWeightsFrom(master *DiagonNetModel) {
	copy(m.Conv.Weights.Data, master.Conv.Weights.Data)
	copy(m.Conv.Bias.Data, master.Conv.Bias.Data)
	copy(m.FC.Weights.Data, master.FC.Weights.Data)
	copy(m.FC.Biases.Data, master.FC.Biases.Data)
}

// SnapshotWeights creates deep copies of all trainable parameter weights in the model.
func (m *DiagonNetModel) SnapshotWeights() [][]float32 {
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
func (m *DiagonNetModel) RestoreWeights(snapshot [][]float32) {
	params := m.Parameters()
	for i, p := range params {
		if p != nil && i < len(snapshot) && snapshot[i] != nil {
			copy(p.Data, snapshot[i])
		}
	}
}

// Forward executes the full model inference forward pass, returning unnormalized class logits.
func (m *DiagonNetModel) Forward(input *Tensor) []float32 {
	var manifold *Tensor
	if input.Channels == 1 {
		manifold = ComputeManifoldTensor(input)
	} else {
		manifold = input
	}

	convOut := m.Conv.Forward(manifold)
	reluOut := m.ReLU.ForwardTensor(convOut)
	poolOut := m.Pool.Forward(reluOut)
	dropOut := m.Dropout.Forward(poolOut.Data)
	logits := m.FC.Forward(dropOut)
	return logits
}

// ForwardBackward executes forward evaluation, cross-entropy loss computation, and full analytical Jacobian
// backpropagation through all layers, accumulating gradients into parameter buffers.
func (m *DiagonNetModel) ForwardBackward(input *Tensor, targetClass int) (float32, []float32) {
	var manifold *Tensor
	if input.Channels == 1 {
		manifold = ComputeManifoldTensor(input)
	} else {
		manifold = input
	}

	// 1. Forward Pass
	convOut := m.Conv.Forward(manifold)
	reluOut := m.ReLU.ForwardTensor(convOut)
	poolOut := m.Pool.Forward(reluOut)
	dropOut := m.Dropout.Forward(poolOut.Data)
	logits := m.FC.Forward(dropOut)

	// 2. Numerically stable Softmax & Categorical Cross-Entropy Loss
	probs := Softmax(logits)
	loss := m.LossFn.Forward(probs, targetClass)

	// 3. Analytical pre-softmax logit gradient: dL/dz_i = p_i - 1(i == target)
	gradLogits := SoftmaxCrossEntropyGrad(probs, targetClass)

	// 4. Dense Head Analytical Backpropagation
	gradDrop := m.FC.Backward(gradLogits)

	// 5. Inverted Dropout Analytical Backpropagation
	gradPool := m.Dropout.Backward(gradDrop)

	// 6. Adaptive Average Pooling Analytical Backpropagation
	poolGradTensor := &Tensor{
		Data:     gradPool,
		Channels: m.Conv.OutChannels,
		Height:   m.Pool.TargetH,
		Width:    m.Pool.TargetW,
	}
	gradReLU := m.Pool.Backward(poolGradTensor)

	// 7. ReLU Activation Analytical Backpropagation
	gradConv := m.ReLU.BackwardTensor(gradReLU)

	// 8. 2D Convolution Analytical Backpropagation (dL/dW, dL/dB, dL/dX)
	m.Conv.Backward(gradConv)

	return loss, probs
}

// BatchTrainer coordinates multi-threaded data-parallel batch training across N worker model replicas.
type BatchTrainer struct {
	MasterModel *DiagonNetModel
	Optimizer   *AdamOptimizer
	NumWorkers  int
	Workers     []*DiagonNetModel
}

// NewBatchTrainer constructs a BatchTrainer coordinating N worker model replicas (scaled to runtime.NumCPU()).
func NewBatchTrainer(master *DiagonNetModel, optimizer *AdamOptimizer, numWorkers int) *BatchTrainer {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
		if numWorkers <= 0 {
			numWorkers = 1
		}
	}

	workers := make([]*DiagonNetModel, numWorkers)
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

	var totalLoss float32
	var correct int

	for _, sample := range samples {
		logits := bt.MasterModel.Forward(sample.Input)
		probs := Softmax(logits)
		loss := bt.MasterModel.LossFn.Forward(probs, sample.TargetClass)
		totalLoss += loss

		predClass := 0
		maxP := probs[0]
		for k := 1; k < len(probs); k++ {
			if probs[k] > maxP {
				maxP = probs[k]
				predClass = k
			}
		}
		if predClass == sample.TargetClass {
			correct++
		}
	}

	return totalLoss / float32(len(samples)), float32(correct) / float32(len(samples))
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
func (cp *ModelCheckpoint) Update(model *DiagonNetModel, epoch int, valAcc float64) bool {
	if valAcc > cp.BestValAcc {
		cp.BestValAcc = valAcc
		cp.BestEpoch = epoch
		cp.BestWeights = model.SnapshotWeights()
		return true
	}
	return false
}

// RestoreBest restores the optimal historical weights into the model.
func (cp *ModelCheckpoint) RestoreBest(model *DiagonNetModel) {
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
func ComputeEvaluationMetrics(model *DiagonNetModel, samples []Sample, classNames []string) EvaluationReport {
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
	fmt.Println("                       DIAGONNET MODEL EVALUATION REPORT                               ")
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

// ============================================================================
// 8. BASELINE ARCHITECTURES & COMPARATIVE BENCHMARK RUNNER (Prompt 45)
// ============================================================================

// TrainableModel defines the common interface for benchmark neural architectures.
type TrainableModel interface {
	Forward(input *Tensor) []float32
	ForwardBackward(input *Tensor, targetClass int) (float32, []float32)
	Parameters() []*Parameter
	SetTraining(training bool)
	SnapshotWeights() [][]float32
	RestoreWeights(snapshot [][]float32)
}

// SimpleCNNModel implements a standard 1-channel convolutional baseline without manifold calculus.
type SimpleCNNModel struct {
	NumClasses int
	Conv       *Conv2DLayer
	ReLU       *ReLULayer
	Pool       *AdaptiveAvgPool2DLayer
	Dropout    *DropoutLayer
	FC         *LinearLayer
	LossFn     *CategoricalCrossEntropyLoss
}

// NewSimpleCNNModel constructs a 1-channel baseline convolutional model.
func NewSimpleCNNModel(numClasses int, rng *rand.Rand) *SimpleCNNModel {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	conv := NewConv2DLayer(1, 16, 3, 2, 1, rng)
	pool := NewAdaptiveAvgPool2DLayer(4, 4)
	dropout := NewDropoutLayer(0.2, rng)
	fc := NewLinearLayer(16*4*4, numClasses, rng)
	lossFn := NewCategoricalCrossEntropyLoss()

	return &SimpleCNNModel{
		NumClasses: numClasses,
		Conv:       conv,
		ReLU:       NewReLULayer(),
		Pool:       pool,
		Dropout:    dropout,
		FC:         fc,
		LossFn:     lossFn,
	}
}

func (m *SimpleCNNModel) Parameters() []*Parameter {
	return []*Parameter{
		m.Conv.Weights,
		m.Conv.Bias,
		m.FC.Weights,
		m.FC.Biases,
	}
}

func (m *SimpleCNNModel) SetTraining(training bool) {
	m.Dropout.Training = training
}

func (m *SimpleCNNModel) SnapshotWeights() [][]float32 {
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

func (m *SimpleCNNModel) RestoreWeights(snapshot [][]float32) {
	params := m.Parameters()
	for i, p := range params {
		if p != nil && i < len(snapshot) && snapshot[i] != nil {
			copy(p.Data, snapshot[i])
		}
	}
}

func (m *SimpleCNNModel) Forward(input *Tensor) []float32 {
	convOut := m.Conv.Forward(input)
	reluOut := m.ReLU.ForwardTensor(convOut)
	poolOut := m.Pool.Forward(reluOut)
	dropOut := m.Dropout.Forward(poolOut.Data)
	logits := m.FC.Forward(dropOut)
	return logits
}

func (m *SimpleCNNModel) ForwardBackward(input *Tensor, targetClass int) (float32, []float32) {
	convOut := m.Conv.Forward(input)
	reluOut := m.ReLU.ForwardTensor(convOut)
	poolOut := m.Pool.Forward(reluOut)
	dropOut := m.Dropout.Forward(poolOut.Data)
	logits := m.FC.Forward(dropOut)

	probs := Softmax(logits)
	loss := m.LossFn.Forward(probs, targetClass)

	gradLogits := SoftmaxCrossEntropyGrad(probs, targetClass)
	gradDrop := m.FC.Backward(gradLogits)
	gradPoolData := m.Dropout.Backward(gradDrop)

	gradPoolTensor := &Tensor{
		Channels: m.Conv.OutChannels,
		Height:   m.Pool.TargetH,
		Width:    m.Pool.TargetW,
		Data:     gradPoolData,
	}
	gradReLU := m.Pool.Backward(gradPoolTensor)
	gradConvOut := m.ReLU.BackwardTensor(gradReLU)
	m.Conv.Backward(gradConvOut)

	return loss, probs
}

// SimpleMLPModel implements a fully-connected baseline neural network.
type SimpleMLPModel struct {
	NumClasses int
	Pool       *AdaptiveAvgPool2DLayer
	FC1        *LinearLayer
	ReLU       *ReLULayer
	Dropout    *DropoutLayer
	FC2        *LinearLayer
	LossFn     *CategoricalCrossEntropyLoss
}

// NewSimpleMLPModel constructs a fully-connected baseline neural network.
func NewSimpleMLPModel(numClasses int, rng *rand.Rand) *SimpleMLPModel {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	pool := NewAdaptiveAvgPool2DLayer(16, 16) // 1x16x16 = 256
	fc1 := NewLinearLayer(256, 64, rng)
	dropout := NewDropoutLayer(0.2, rng)
	fc2 := NewLinearLayer(64, numClasses, rng)
	lossFn := NewCategoricalCrossEntropyLoss()

	return &SimpleMLPModel{
		NumClasses: numClasses,
		Pool:       pool,
		FC1:        fc1,
		ReLU:       NewReLULayer(),
		Dropout:    dropout,
		FC2:        fc2,
		LossFn:     lossFn,
	}
}

func (m *SimpleMLPModel) Parameters() []*Parameter {
	return []*Parameter{
		m.FC1.Weights,
		m.FC1.Biases,
		m.FC2.Weights,
		m.FC2.Biases,
	}
}

func (m *SimpleMLPModel) SetTraining(training bool) {
	m.Dropout.Training = training
}

func (m *SimpleMLPModel) SnapshotWeights() [][]float32 {
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

func (m *SimpleMLPModel) RestoreWeights(snapshot [][]float32) {
	params := m.Parameters()
	for i, p := range params {
		if p != nil && i < len(snapshot) && snapshot[i] != nil {
			copy(p.Data, snapshot[i])
		}
	}
}

func (m *SimpleMLPModel) Forward(input *Tensor) []float32 {
	poolOut := m.Pool.Forward(input)
	fc1Out := m.FC1.Forward(poolOut.Data)
	reluOut := m.ReLU.Forward(fc1Out)
	dropOut := m.Dropout.Forward(reluOut)
	logits := m.FC2.Forward(dropOut)
	return logits
}

func (m *SimpleMLPModel) ForwardBackward(input *Tensor, targetClass int) (float32, []float32) {
	poolOut := m.Pool.Forward(input)
	fc1Out := m.FC1.Forward(poolOut.Data)
	reluOut := m.ReLU.Forward(fc1Out)
	dropOut := m.Dropout.Forward(reluOut)
	logits := m.FC2.Forward(dropOut)

	probs := Softmax(logits)
	loss := m.LossFn.Forward(probs, targetClass)

	gradLogits := SoftmaxCrossEntropyGrad(probs, targetClass)
	gradDrop := m.FC2.Backward(gradLogits)
	gradReLU := m.Dropout.Backward(gradDrop)
	gradFC1 := m.ReLU.Backward(gradReLU)
	gradPoolData := m.FC1.Backward(gradFC1)

	gradPoolTensor := &Tensor{
		Channels: 1,
		Height:   m.Pool.TargetH,
		Width:    m.Pool.TargetW,
		Data:     gradPoolData,
	}
	m.Pool.Backward(gradPoolTensor)

	return loss, probs
}

// BenchmarkResult stores performance metrics for a single architecture run.
type BenchmarkResult struct {
	Architecture string  `json:"architecture"`
	ParamCount   int     `json:"parameters"`
	TrainTimeMs  int64   `json:"train_time_ms"`
	FinalLoss    float32 `json:"final_loss"`
	ValAccuracy  float64 `json:"val_accuracy"`
	ValMacroF1   float64 `json:"val_macro_f1"`
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

// EvaluateModelPredictions runs forward inference across samples and calculates accuracy, Macro-F1, and loss.
func EvaluateModelPredictions(model TrainableModel, samples []Sample, numClasses int, classNames []string) (float32, float64, float64) {
	if len(samples) == 0 {
		return 0, 0, 0
	}
	model.SetTraining(false)
	var totalLoss float32

	matrix := make([][]int, numClasses)
	for i := 0; i < numClasses; i++ {
		matrix[i] = make([]int, numClasses)
	}

	correct := 0
	lossFn := NewCategoricalCrossEntropyLoss()

	for _, s := range samples {
		logits := model.Forward(s.Input)
		probs := Softmax(logits)
		loss := lossFn.Forward(probs, s.TargetClass)
		totalLoss += loss

		pred := 0
		maxP := probs[0]
		for k := 1; k < len(probs); k++ {
			if probs[k] > maxP {
				maxP = probs[k]
				pred = k
			}
		}
		if s.TargetClass >= 0 && s.TargetClass < numClasses && pred >= 0 && pred < numClasses {
			matrix[s.TargetClass][pred]++
			if pred == s.TargetClass {
				correct++
			}
		}
	}

	acc := float64(correct) / float64(len(samples))
	var sumF1 float64
	for c := 0; c < numClasses; c++ {
		tp := matrix[c][c]
		fp := 0
		for a := 0; a < numClasses; a++ {
			if a != c {
				fp += matrix[a][c]
			}
		}
		fn := 0
		for p := 0; p < numClasses; p++ {
			if p != c {
				fn += matrix[c][p]
			}
		}
		var prec, rec, f1 float64
		if tp+fp > 0 {
			prec = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			rec = float64(tp) / float64(tp+fn)
		}
		if prec+rec > 0 {
			f1 = 2.0 * (prec * rec) / (prec + rec)
		}
		sumF1 += f1
	}
	macroF1 := sumF1 / float64(numClasses)
	avgLoss := totalLoss / float32(len(samples))

	return avgLoss, acc, macroF1
}

// TrainBenchmarkModel trains an architecture model on the provided dataset split for the specified epochs.
func TrainBenchmarkModel(model TrainableModel, trainSamples, valSamples []Sample, numClasses int, epochs int, batchSize int, lr float32) BenchmarkResult {
	params := model.Parameters()
	paramCount := CountModelParameters(params)

	opt := NewAdamOptimizer(params, AdamOptimizerConfig{
		LearningRate: lr,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  1e-4,
	})

	bestAcc := -1.0
	var bestWeights [][]float32
	var finalLoss float32

	startTime := time.Now()

	N := len(trainSamples)
	indices := make([]int, N)
	for i := 0; i < N; i++ {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(42))

	for ep := 1; ep <= epochs; ep++ {
		model.SetTraining(true)
		rng.Shuffle(N, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		var epochLoss float32
		numBatches := (N + batchSize - 1) / batchSize

		for b := 0; b < numBatches; b++ {
			start := b * batchSize
			end := start + batchSize
			if end > N {
				end = N
			}
			currBatchSize := end - start
			if currBatchSize <= 0 {
				continue
			}

			opt.ZeroGrad()

			var batchLoss float32
			for i := start; i < end; i++ {
				idx := indices[i]
				loss, _ := model.ForwardBackward(trainSamples[idx].Input, trainSamples[idx].TargetClass)
				batchLoss += loss
			}

			scale := float32(1.0) / float32(currBatchSize)
			for _, p := range params {
				if p != nil {
					for j := range p.Grad {
						p.Grad[j] *= scale
					}
				}
			}

			opt.Step()
			epochLoss += batchLoss / float32(currBatchSize)
		}

		finalLoss = epochLoss / float32(numBatches)

		_, valAcc, _ := EvaluateModelPredictions(model, valSamples, numClasses, nil)
		if valAcc > bestAcc {
			bestAcc = valAcc
			bestWeights = model.SnapshotWeights()
		}
	}

	trainTimeMs := time.Since(startTime).Milliseconds()

	if bestWeights != nil {
		model.RestoreWeights(bestWeights)
	}

	_, valAcc, valMacroF1 := EvaluateModelPredictions(model, valSamples, numClasses, nil)

	return BenchmarkResult{
		ParamCount:  paramCount,
		TrainTimeMs: trainTimeMs,
		FinalLoss:   finalLoss,
		ValAccuracy: valAcc,
		ValMacroF1:  valMacroF1,
	}
}

// GenerateSyntheticBenchmarkDataset creates a deterministic synthetic vision dataset of K classes with geometric shapes.
func GenerateSyntheticBenchmarkDataset(numClasses int, samplesPerClass int, seed int64) []Sample {
	rng := rand.New(rand.NewSource(seed))
	samples := make([]Sample, 0, numClasses*samplesPerClass)

	for c := 0; c < numClasses; c++ {
		for s := 0; s < samplesPerClass; s++ {
			t := NewTensor(1, 100, 100)
			cx := 50 + rng.Intn(11) - 5
			cy := 50 + rng.Intn(11) - 5
			radius := 18 + rng.Intn(9)

			switch c % 5 {
			case 0: // Circle / Disk
				for y := 0; y < 100; y++ {
					for x := 0; x < 100; x++ {
						dx := float64(x - cx)
						dy := float64(y - cy)
						dist := math.Sqrt(dx*dx + dy*dy)
						if math.Abs(dist-float64(radius)) <= 3.0 {
							t.Set(0, y, x, 0.9)
						}
					}
				}
			case 1: // Horizontal stripe
				yMin := cy - radius/2
				yMax := cy + radius/2
				for y := yMin; y <= yMax; y++ {
					for x := 20; x < 80; x++ {
						if y >= 0 && y < 100 {
							t.Set(0, y, x, 0.9)
						}
					}
				}
			case 2: // Vertical stripe
				xMin := cx - radius/2
				xMax := cx + radius/2
				for y := 20; y < 80; y++ {
					for x := xMin; x <= xMax; x++ {
						if x >= 0 && x < 100 {
							t.Set(0, y, x, 0.9)
						}
					}
				}
			case 3: // Cross / Plus
				for y := 20; y < 80; y++ {
					for x := 20; x < 80; x++ {
						if (math.Abs(float64(x-cx)) <= 4 && math.Abs(float64(y-cy)) <= float64(radius)) ||
							(math.Abs(float64(y-cy)) <= 4 && math.Abs(float64(x-cx)) <= float64(radius)) {
							t.Set(0, y, x, 0.9)
						}
					}
				}
			case 4: // Diagonal Line
				for d := -radius; d <= radius; d++ {
					x := cx + d
					y := cy + d
					if x >= 0 && x < 100 && y >= 0 && y < 100 {
						t.Set(0, y, x, 0.9)
					}
				}
			}
			samples = append(samples, Sample{Input: t, TargetClass: c})
		}
	}
	return samples
}

// ExportBenchmarkCSV writes benchmark comparison results to a standard CSV file.
func ExportBenchmarkCSV(path string, results []BenchmarkResult) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create CSV file %s: %w", path, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	writer.WriteString("Architecture,Parameters,TrainTimeMs,FinalLoss,ValAccuracy,ValMacroF1\n")
	for _, r := range results {
		line := fmt.Sprintf("%s,%d,%d,%.6f,%.6f,%.6f\n",
			r.Architecture, r.ParamCount, r.TrainTimeMs, r.FinalLoss, r.ValAccuracy, r.ValMacroF1)
		writer.WriteString(line)
	}
	return writer.Flush()
}

// RunArchitectureBenchmark executes the 3-model benchmark comparison and writes the results CSV.
func RunArchitectureBenchmark(dataDir string, epochs int, batchSize int, lr float32, csvOutPath string) ([]BenchmarkResult, error) {
	fmt.Println("====================================================================================================")
	fmt.Println("                        DIAGONNET ARCHITECTURE BENCHMARK & COMPARISON")
	fmt.Println("====================================================================================================")

	var trainSamples, valSamples []Sample
	var numClasses int

	// 1. Attempt loading from filesystem
	ds, err := ScanDataset(dataDir)
	if err == nil && ds.Metadata.NumClasses >= 2 {
		numClasses = ds.Metadata.NumClasses
		trainItems, valItems := TrainTestSplit(ds.Samples, 0.20, 42)

		loadItems := func(items []ImageItem) []Sample {
			res := make([]Sample, 0, len(items))
			for _, it := range items {
				gray, err := LoadImageFromFile(it.Path)
				if err != nil {
					continue
				}
				bbox := FindBoundingBox(gray, 10)
				centered := PadAndCenter(gray, bbox)
				stretched := ContrastStretch(centered)
				resized := ResizeBilinear(stretched, 100, 100)
				t := GrayImageToTensor(resized)
				res = append(res, Sample{Input: t, TargetClass: it.ClassIndex})
			}
			return res
		}

		trainSamples = loadItems(trainItems)
		valSamples = loadItems(valItems)
		fmt.Printf(" Loaded Dataset from '%s': %d Train / %d Validation samples across %d classes.\n",
			dataDir, len(trainSamples), len(valSamples), numClasses)
	}

	// Fallback to deterministic synthetic vision benchmark if dataset empty or unavailable
	if len(trainSamples) == 0 || len(valSamples) == 0 {
		numClasses = 5
		allSynth := GenerateSyntheticBenchmarkDataset(numClasses, 60, 42)
		trainSamples = allSynth[:200]
		valSamples = allSynth[200:]
		fmt.Printf(" Using Synthetic Benchmark Dataset: %d Train / %d Validation samples across %d classes.\n",
			len(trainSamples), len(valSamples), numClasses)
	}

	if epochs <= 0 {
		epochs = 15
	}
	if batchSize <= 0 {
		batchSize = 32
	}
	if lr <= 0 {
		lr = 0.002
	}
	if csvOutPath == "" {
		csvOutPath = filepath.Join("assets", "comparison_results.csv")
	}

	fmt.Printf(" Config: Epochs=%d | BatchSize=%d | InitialLR=%.4f | Target CSV=%s\n",
		epochs, batchSize, lr, csvOutPath)
	fmt.Println("----------------------------------------------------------------------------------------------------")

	results := make([]BenchmarkResult, 3)

	// 1. DiagonNet Model (13-Channel Manifold CNN)
	fmt.Print(" [1/3] Training DiagonNet (13-Channel Manifold CNN)... ")
	diagonRng := rand.New(rand.NewSource(42))
	diagonModel := NewDiagonNetModel(numClasses, diagonRng)
	res1 := TrainBenchmarkModel(diagonModel, trainSamples, valSamples, numClasses, epochs, batchSize, lr)
	res1.Architecture = "DiagonNet (13-Ch)"
	results[0] = res1
	fmt.Printf("Done (%.2fs | Acc: %.2f%% | Loss: %.4f)\n", float64(res1.TrainTimeMs)/1000.0, res1.ValAccuracy*100.0, res1.FinalLoss)

	// 2. SimpleCNN Model (1-Channel CNN)
	fmt.Print(" [2/3] Training SimpleCNN (1-Channel Standard CNN)...  ")
	cnnRng := rand.New(rand.NewSource(42))
	cnnModel := NewSimpleCNNModel(numClasses, cnnRng)
	res2 := TrainBenchmarkModel(cnnModel, trainSamples, valSamples, numClasses, epochs, batchSize, lr)
	res2.Architecture = "SimpleCNN (1-Ch)"
	results[1] = res2
	fmt.Printf("Done (%.2fs | Acc: %.2f%% | Loss: %.4f)\n", float64(res2.TrainTimeMs)/1000.0, res2.ValAccuracy*100.0, res2.FinalLoss)

	// 3. SimpleMLP Model (Dense Baseline)
	fmt.Print(" [3/3] Training SimpleMLP (Fully-Connected Baseline)... ")
	mlpRng := rand.New(rand.NewSource(42))
	mlpModel := NewSimpleMLPModel(numClasses, mlpRng)
	res3 := TrainBenchmarkModel(mlpModel, trainSamples, valSamples, numClasses, epochs, batchSize, lr)
	res3.Architecture = "SimpleMLP (Dense)"
	results[2] = res3
	fmt.Printf("Done (%.2fs | Acc: %.2f%% | Loss: %.4f)\n", float64(res3.TrainTimeMs)/1000.0, res3.ValAccuracy*100.0, res3.FinalLoss)

	// Print Comparative Summary Table
	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Printf(" %-18s | %10s | %10s | %10s | %12s | %12s | %12s\n",
		"Architecture", "Parameters", "Train Time", "Final Loss", "Val Accuracy", "Val Macro-F1", "Delta vs CNN")
	fmt.Println("----------------------------------------------------------------------------------------------------")

	baselineAcc := results[1].ValAccuracy
	for _, r := range results {
		delta := (r.ValAccuracy - baselineAcc) * 100.0
		deltaStr := fmt.Sprintf("%+6.2f%%", delta)
		if r.Architecture == "SimpleCNN (1-Ch)" {
			deltaStr = "Baseline"
		}
		fmt.Printf(" %-18s | %10d | %9.2fs | %10.4f | %11.2f%% | %11.2f%% | %12s\n",
			r.Architecture, r.ParamCount, float64(r.TrainTimeMs)/1000.0, r.FinalLoss, r.ValAccuracy*100.0, r.ValMacroF1*100.0, deltaStr)
	}

	// Export to CSV
	if err := ExportBenchmarkCSV(csvOutPath, results); err != nil {
		fmt.Printf(" [Warning] Failed to export CSV: %v\n", err)
	} else {
		fmt.Printf("----------------------------------------------------------------------------------------------------\n")
		fmt.Printf(" Exported comparative results CSV to: %s\n", csvOutPath)
	}
	fmt.Println("====================================================================================================")

	return results, nil
}

// ============================================================================
// 9. EMBEDDED HTML5 CANVAS & REAL-TIME WEB INFERENCE ENGINE (Prompts 46 - 48)
// ============================================================================

// webAppHTML embeds the complete single-page interactive drawing canvas application.
const webAppHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DiagonNet | Real-Time Neural Drawing Canvas</title>
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-card: #151e32;
    --border-color: #263554;
    --accent-blue: #38bdf8;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-pink: #ec4899;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  body { background: var(--bg-main); color: var(--text-primary); min-height: 100vh; display: flex; flex-direction: column; }
  header { background: var(--bg-card); border-bottom: 1px solid var(--border-color); padding: 1rem 2rem; display: flex; align-items: center; justify-content: space-between; }
  .logo-title { display: flex; align-items: center; gap: 0.75rem; font-size: 1.25rem; font-weight: 700; background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .badge { background: #1e293b; border: 1px solid var(--border-color); color: var(--accent-blue); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
  .main-container { flex: 1; max-width: 1200px; width: 100%; margin: 0 auto; padding: 2rem; display: grid; grid-template-columns: 440px 1fr; gap: 2rem; }
  @media (max-width: 900px) { .main-container { grid-template-columns: 1fr; } }
  .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 1rem; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); }
  .canvas-wrapper { position: relative; width: 400px; height: 400px; margin: 0 auto; border-radius: 0.75rem; overflow: hidden; border: 2px solid var(--border-color); box-shadow: inset 0 2px 8px rgba(0,0,0,0.5); }
  canvas { width: 400px; height: 400px; background: #000000; cursor: crosshair; touch-action: none; }
  .controls { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
  button { flex: 1; padding: 0.75rem 1rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; transition: all 0.2s; border: none; font-size: 0.9rem; }
  .btn-clear { background: #334155; color: #f8fafc; border: 1px solid #475569; }
  .btn-clear:hover { background: #475569; }
  .btn-predict { background: linear-gradient(135deg, #0284c7, #06b6d4); color: white; }
  .btn-predict:hover { opacity: 0.9; transform: translateY(-1px); }
  .prediction-banner { background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(6, 182, 212, 0.05)); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 0.75rem; padding: 1.25rem; display: flex; align-items: center; justify-content: space-between; }
  .pred-label-group { display: flex; flex-direction: column; gap: 0.25rem; }
  .pred-sub { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
  .pred-name { font-size: 2rem; font-weight: 800; color: var(--accent-blue); }
  .latency-tag { font-family: monospace; font-size: 0.85rem; color: var(--accent-emerald); background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 0.35rem 0.65rem; border-radius: 0.5rem; }
  .class-list { display: flex; flex-direction: column; gap: 0.6rem; max-height: 380px; overflow-y: auto; padding-right: 0.25rem; }
  .class-row { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
  .class-info { display: flex; justify-content: space-between; font-weight: 500; }
  .progress-bg { height: 8px; background: #1e293b; border-radius: 9999px; overflow: hidden; border: 1px solid #334155; }
  .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); border-radius: 9999px; transition: width 0.15s ease-out; }
  .class-row.top .progress-fill { background: linear-gradient(90deg, var(--accent-emerald), #34d399); }
  .shortcut-hint { font-size: 0.75rem; color: var(--text-secondary); text-align: center; }
</style>
</head>
<body>
  <header>
    <div class="logo-title">
      <span>⬡</span>
      <span>DiagonNet 13-Manifold Web Engine</span>
    </div>
    <div class="badge">Pure Go Zero-Dep Runtime</div>
  </header>
  <div class="main-container">
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600;">Interactive Sketch Canvas</span>
        <span style="font-size: 0.8rem; color: var(--text-secondary);">400 &times; 400 px</span>
      </div>
      <div class="canvas-wrapper">
        <canvas id="paintCanvas" width="400" height="400"></canvas>
      </div>
      <div class="controls">
        <button class="btn-clear" id="btnClear">Clear Canvas (C)</button>
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: var(--text-secondary);">
          <span>Brush:</span>
          <input id="brushSize" type="range" min="12" max="36" value="22" style="accent-color: var(--accent-blue); cursor: pointer; width: 80px;">
        </div>
        <button class="btn-predict" id="btnPredict">Predict</button>
      </div>
      <div class="shortcut-hint">Shortcuts: <b>C</b> / <b>Esc</b> to Clear | Real-Time Continuous Autograd</div>
    </div>
    <div class="card">
      <div class="prediction-banner">
        <div class="pred-label-group">
          <span class="pred-sub">Top Predicted Class</span>
          <span class="pred-name" id="topClass">&mdash;</span>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem;">
          <span id="topConfidence" style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">0.0%</span>
          <span class="latency-tag" id="latencyBadge">&mdash; ms</span>
        </div>
      </div>
      <div style="font-weight: 600; font-size: 0.95rem; margin-top: 0.5rem;">Class Probability Distribution</div>
      <div class="class-list" id="classList">
        <div style="color: var(--text-secondary); font-size: 0.85rem; text-align: center; padding: 2rem 0;">Draw on the canvas to evaluate live probabilities.</div>
      </div>
    </div>
  </div>
<script>
  const canvas = document.getElementById('paintCanvas');
  const ctx = canvas.getContext('2d');
  const brushSlider = document.getElementById('brushSize');
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, 400, 400);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 22;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (brushSlider) {
    brushSlider.addEventListener('input', () => {
      ctx.lineWidth = parseInt(brushSlider.value);
    });
  }

  let drawing = false;
  let hasDrawn = false;
  let debounceTimer = null;

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
    schedulePredict();
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
    document.getElementById('classList').innerHTML = '<div style="color: var(--text-secondary); font-size: 0.85rem; text-align: center; padding: 2rem 0;">Canvas cleared. Draw a sketch.</div>';
  }

  document.getElementById('btnClear').addEventListener('click', clearCanvas);
  document.getElementById('btnPredict').addEventListener('click', triggerPredict);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C' || e.key === 'Escape') {
      clearCanvas();
    }
  });

  function schedulePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerPredict, 60);
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
      renderPrediction(data);
    } catch (err) {
      console.error('Predict error:', err);
    }
  }

  function renderPrediction(data) {
    if (data.is_blank) {
      document.getElementById('topClass').innerText = 'Blank';
      document.getElementById('topConfidence').innerText = '0.0%';
      document.getElementById('latencyBadge').innerText = data.latency_ms.toFixed(2) + ' ms';
      return;
    }
    document.getElementById('topClass').innerText = data.predicted_class;
    document.getElementById('topConfidence').innerText = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('latencyBadge').innerText = '⚡ ' + data.latency_ms.toFixed(2) + ' ms';

    const list = document.getElementById('classList');
    list.innerHTML = '';
    const sorted = [...data.confidences].sort((a, b) => b.confidence - a.confidence);

    sorted.forEach((item, idx) => {
      const pct = (item.confidence * 100).toFixed(1);
      const isTop = idx === 0;
      const row = document.createElement('div');
      row.className = 'class-row' + (isTop ? ' top' : '');
      row.innerHTML =
        '<div class="class-info">' +
          '<span style="' + (isTop ? 'color: var(--accent-blue); font-weight:700;' : '') + '">' + item.class_name + '</span>' +
          '<span style="' + (isTop ? 'color: var(--accent-emerald); font-weight:700;' : 'color: var(--text-secondary);') + '">' + pct + '%</span>' +
        '</div>' +
        '<div class="progress-bg">' +
          '<div class="progress-fill" style="width: ' + pct + '%"></div>' +
        '</div>';
      list.appendChild(row);
    });
  }

  // Initial metadata query
  fetch('/api/info').then(r => r.json()).then(info => {
    if (info && info.classes) {
      const list = document.getElementById('classList');
      list.innerHTML = '';
      info.classes.forEach(c => {
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

// PredictResponse contains the model classification results, confidence array, and inference latency.
type PredictResponse struct {
	PredictedClass string            `json:"predicted_class"`
	ClassIndex     int               `json:"class_index"`
	Confidence     float32           `json:"confidence"`
	Confidences    []ClassConfidence `json:"confidences"`
	LatencyMs      float64           `json:"latency_ms"`
	IsBlank        bool              `json:"is_blank"`
}

// PreprocessWebImage takes an arbitrary decoded image, locates the tight bounding box, centers it with scale-invariant padding,
// stretches stroke contrast, and resamples to a 100x100 Tensor normalized to [0.0, 1.0].
func PreprocessWebImage(src image.Image) (*Tensor, bool) {
	bounds := src.Bounds()
	gray := image.NewGray(bounds)
	draw.Draw(gray, bounds, src, bounds.Min, draw.Src)

	bbox := FindBoundingBox(gray, 10)
	if bbox == nil {
		return NewTensor(1, 100, 100), true
	}

	centered := PadAndCenter(gray, bbox)
	stretched := ContrastStretch(centered)
	resized := ResizeBilinear(stretched, 100, 100)
	tensor := GrayImageToTensor(resized)
	return tensor, false
}

// InferenceServer hosts the HTTP web canvas UI and real-time prediction API.
type InferenceServer struct {
	Model      *DiagonNetModel
	ClassNames []string
	Port       int
}

// NewInferenceServer constructs a new inference web server.
func NewInferenceServer(model *DiagonNetModel, classNames []string, port int) *InferenceServer {
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
	json.NewEncoder(w).Encode(map[string]interface{}{
		"classes":     s.ClassNames,
		"num_classes": s.Model.NumClasses,
		"parameters":  CountModelParameters(s.Model.Parameters()),
		"cpu_cores":   runtime.NumCPU(),
	})
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

	start := time.Now()

	tensor, isBlank := PreprocessWebImage(img)

	var predClass int
	var maxProb float32
	confidences := make([]ClassConfidence, s.Model.NumClasses)

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
		logits := s.Model.Forward(tensor)
		probs := Softmax(logits)

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
		}
	}

	latencyMs := float64(time.Since(start).Microseconds()) / 1000.0

	resp := PredictResponse{
		PredictedClass: s.ClassNames[predClass],
		ClassIndex:     predClass,
		Confidence:     maxProb,
		Confidences:    confidences,
		LatencyMs:      latencyMs,
		IsBlank:        isBlank,
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
func StartInferenceServer(model *DiagonNetModel, classNames []string, port int, autoOpen bool) error {
	srv := NewInferenceServer(model, classNames, port)
	addr := fmt.Sprintf(":%d", port)
	url := fmt.Sprintf("http://localhost:%d", port)

	fmt.Println("====================================================================================================")
	fmt.Println("                       DIAGONNET REAL-TIME INFERENCE & WEB RUNTIME")
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
// 10. CLI ROUTING & EXECUTION HANDLERS
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
	fmt.Println(">>> [Audit Mode] Starting dataset validation, quality analysis, and integrity audit...")
	report, err := AuditDataset(dataDir)
	if err != nil {
		fmt.Printf(">>> [Audit Warning/Error] %v\n", err)
		return
	}
	PrintAuditReport(report)
	fmt.Println(">>> Dataset audit completed.")
}

func runTrain(dataDir string, modelPath string, epochs int, lr float32, batchSize int) {
	fmt.Println("====================================================================================================")
	fmt.Println("                         DIAGONNET DEEP LEARNING MODEL TRAINING PIPELINE")
	fmt.Println("====================================================================================================")
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
			augStr = "15x (Rotations, Shifts, Shears, Dilation, Erosion)"
		}
		fmt.Printf(" Preprocessing %d %s samples [%s]... ", len(items), label, augStr)
		start := time.Now()
		samples := make([]Sample, 0, len(items)*15)
		for _, it := range items {
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
				centered := PadAndCenter(v, bbox)
				stretched := ContrastStretch(centered)
				resized := ResizeBilinear(stretched, 100, 100)
				tensor := GrayImageToTensor(resized)
				samples = append(samples, Sample{
					Input:       tensor,
					TargetClass: it.ClassIndex,
				})
			}
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
	masterModel := NewDiagonNetModel(numClasses, rng)
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
	fmt.Printf(" Model Architecture: 13-Manifold -> Conv2D(13->16, K=3, S=2) -> ReLU -> AdaptiveAvgPool(4x4) -> Linear(256->%d)\n", numClasses)
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
		fmt.Printf(" Epoch [%2d/%2d] | Train Loss: %.4f (Acc: %5.1f%%) | Val Loss: %.4f (Acc: %5.1f%%) | Time: %5.2fs%s\n",
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

	var model *DiagonNetModel
	var classNames []string

	// 1. Try loading weights from disk if file exists
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Printf("    Loading model weights from: %s\n", modelPath)
		tempModel := NewDiagonNetModel(2, nil)
		loadedClasses, err := LoadModelWeights(modelPath, tempModel.Parameters())
		if err == nil && len(loadedClasses) >= 2 {
			classNames = loadedClasses
			model = NewDiagonNetModel(len(classNames), nil)
			_, _ = LoadModelWeights(modelPath, model.Parameters())
			fmt.Printf("    Successfully loaded weights for %d classes: %v\n", len(classNames), classNames)
		}
	}

	// 2. Fallback to scanning dataset directory if model weights not yet saved
	if model == nil {
		ds, err := ScanDataset("data")
		if err == nil && ds.Metadata.NumClasses >= 2 {
			classNames = ds.Metadata.Classes
			model = NewDiagonNetModel(len(classNames), rand.New(rand.NewSource(42)))
			fmt.Printf("    Initialized He weights for discovered dataset classes: %v\n", classNames)
		} else {
			classNames = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
			model = NewDiagonNetModel(10, rand.New(rand.NewSource(42)))
			fmt.Printf("    Initialized He weights for standard digits (0-9)\n")
		}
	}

	if err := StartInferenceServer(model, classNames, port, true); err != nil {
		fmt.Printf(">>> [Server Error] %v\n", err)
	}
}

func runBenchmark(dataDir string) {
	fmt.Println(">>> [Benchmark Mode] Initializing comparative architecture benchmark...")
	_, err := RunArchitectureBenchmark(dataDir, 15, 32, 0.002, filepath.Join("assets", "comparison_results.csv"))
	if err != nil {
		fmt.Printf(">>> [Benchmark Error] %v\n", err)
		return
	}
	fmt.Println(">>> Architecture benchmark completed successfully.")
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
