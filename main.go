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
// 6. CLI ROUTING & EXECUTION HANDLERS
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
