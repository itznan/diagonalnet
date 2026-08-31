# Running DiagonalNet

This guide provides step-by-step instructions to build, train, evaluate, test, and serve DiagonalNet.

---

## 1. Prerequisites

- **Go 1.22+** installed on your system.
- **Zero Third-Party Dependencies**: No external C/C++ libraries, PyTorch, TensorFlow, OpenCV, or third-party Go packages required.

Verify your Go installation:
```bash
go version
```

---

## 2. Execution Methods

DiagonalNet can be executed in three ways:

### Method A: Run Directly with Go (Fastest)
You do not need to pre-compile a binary. Run any command directly with `go run .`:
```bash
go run . [command/flags]
```

### Method B: Compile Native Standalone Binary
Compile a single self-contained binary (<3 MB) for your platform:

* **Windows**:
  ```bash
  go build -trimpath -buildvcs=false -ldflags="-s -w" -o bin/diagonalnet.exe .
  .\bin\diagonalnet.exe [command/flags]
  ```

* **Linux / macOS**:
  ```bash
  go build -trimpath -buildvcs=false -ldflags="-s -w" -o bin/diagonalnet .
  ./bin/diagonalnet [command/flags]
  ```

### Method C: Interactive Windows Hub (`diagonalnet.bat`)
On Windows, you can launch the interactive terminal menu:
```cmd
.\diagonalnet.bat
```

---

## 3. Step-by-Step Workflow

### Step 1: Audit Dataset Integrity (`audit`)
Scan dataset directories under `data/`, analyze bounding boxes, check stroke densities, and detect corrupt, blank, or tiny outliers:
```bash
go run . audit -data data
# Or with compiled binary:
# .\bin\diagonalnet.exe audit -data data
```

---

### Step 2: Train the Neural Network (`train`)
Train the 13-channel spatial difference manifold network with lock-free CPU multi-core parallel backpropagation and 15x data augmentation.

* **Fast Profile** (~1–2 mins rapid smoke test):
  ```bash
  go run . train -profile fast -data data -model weights/diagonalnet_model.bin
  ```

* **Normal Profile** (~3–4 mins, **Recommended**, ~94%–96%+ accuracy):
  ```bash
  go run . train -profile normal -data data -model weights/diagonalnet_model.bin
  ```

* **Hardcore Profile** (~8 mins, 30 epochs, ~98%–99%+ max accuracy):
  ```bash
  go run . train -profile hardcore -data data -model weights/diagonalnet_model.bin
  ```

* **Custom Training Configuration**:
  ```bash
  go run . train -data data -model weights/diagonalnet_model.bin -epochs 15 -lr 0.002 -batch 32
  ```

---

### Step 3: Launch Web Drawing Canvas & Inference Server (`serve`)
Start the real-time embedded HTTP server and web app (automatically opens your default browser):
```bash
go run . serve -model weights/diagonalnet_model.bin -port 8081
```

- **Web Dashboard**: Open [http://localhost:8081](http://localhost:8081) in your browser.
- **REST Endpoints**:
  - `POST /api/predict`: Real-time sub-8ms sketch inference with top predictions and manifold activation stats.
  - `GET /api/info`: Metadata introspection (system runtime, CPU cores, active classes, model parameters).

---

### Step 4: Run the Mathematical & Autograd Test Suite
Execute all 54 unit tests, numerical Jacobian gradient verifications, and layer benchmarks:
```bash
go test -v ./...
```

---

### Step 5: Verify Zero-Dependency Compliance
Confirm that the engine imports only the Go standard library with zero external modules:
```bash
go list -m all
```
Or run the dependency audit script:
```cmd
diagonalnet.bat deps
```

---

## 4. CLI Flags & Configuration Reference

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-train` | Launch deep learning training pipeline | `false` |
| `-serve` | Start interactive HTTP inference & dashboard server | `false` |
| `-audit` | Run dataset quality and integrity audit | `false` |
| `-data <path>` | Path to dataset directory | `data` |
| `-model <path>` | Path to binary model weights file (`DIAGON01` format) | `weights/diagonalnet_model.bin` |
| `-profile <name>` | Training profile template (`fast`, `normal`, `hardcore`, `manual`) | `""` |
| `-epochs <int>` | Number of training epochs | `8` |
| `-lr <float>` | Learning rate for Adam optimizer | `0.002` |
| `-batch <int>` | Mini-batch size for training | `64` |
| `-port <int>` | HTTP server listen port | `8081` |
| `-help`, `-h` | Display CLI help menu | `false` |
