# TT Swiss - A swiss army knife for model bringup 🇨🇭

<img src="media/tt-swiss-img.png" alt="TT Swiss" width="600"/>


This repo is a collection of all of the useful tools for enabling models to work on TT hardware. This includes:
1. Memory profiler `ttmem` - useful for look at memory usage of the model. Signs that you need this - errors like `Out of Memory: Not enough space to allocate <nbytes> B DRAM buffer across <nbanks> banks`

2. Model analyzer `ttchop` - analyzes PyTorch models to identify which modules/ops work on TT hardware. Generates interactive HTML report showing pass/fail status for each module.

3. Claude skills and commands - We recommend you copy paste these in your `~/.claude` or `<tt-xla-path>/.claude` for easier debugging of models.

## One click setup
This installs both ttmem and ttchop CLI tools and python packages

```
pip install git+https://github.com/vkovinicTT/tt-swiss.git
```

### Prerequisites

Before using tt-swiss, you need to configure TT-XLA for memory logging and op by op testing:

#### 1. Build `tt-xla` with debug flags and Python bindings enabled

```bash
source venv/activate

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTT_RUNTIME_DEBUG=ON -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build build
```

#### 2. Export runtime logger flag (for op and memory info)

```bash
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
export TT_RUNTIME_MEMORY_LOG_LEVEL=operation
```

#### 3. Initialize TTRT artifacts

```bash
ttrt query --save-artifacts # --disable-eth-dispatch # add this for blackhole qb
```

## Installation

```bash
cd /path/to/tt-xla
source venv/activate
pip install git+https://github.com/vkovinicTT/tt-swiss.git
```

> **Note**: Always activate the tt-xla environment first (`source venv/activate`). This sets up the required paths for the model analyzer to find tt-xla's op-by-op test infrastructure.

## Memory profiler

### Usage

#### Interactive CLI (Recommended for Remote Development)

```bash
ttmem
```

The interactive CLI guides you through the process with prompts:
1. Asks if you have a log file ready (shows prerequisites if not)
2. Prompts for the log file path with autocomplete
3. Parses the log and generates the HTML report
4. Optionally starts an HTTP server for remote viewing

When working on a remote machine via VS Code Remote SSH, the HTTP server option allows you to view the report in your local browser. VS Code automatically forwards the port, so `http://localhost:8000/report.html` will work from your local machine.

#### Command Line Interface

```bash
# Default: run + parse + visualize (recommended)
tt-memory-profiler path/to/your_model.py

# Only capture logs (for later processing)
tt-memory-profiler --log path/to/your_model.py

# Parse existing log file
tt-memory-profiler --analyze logs/your_model_20260122_143957/your_model_profile.log

# Generate visualization from existing run
tt-memory-profiler --visualize logs/your_model_20260122_143957/

# Specify custom output directory
tt-memory-profiler --output-dir /path/to/output path/to/your_model.py
```

### Output Structure

Output is stored in `./logs/` relative to your current working directory (or `--output-dir` if specified):

```
./logs/<script_name>_YYYYMMDD_HHMMSS/
├── <script_name>_memory.json      # Memory stats per operation
├── <script_name>_operations.json  # Operation metadata per operation
├── <script_name>_profile.log      # Raw logs
└── <script_name>_report.html      # Interactive visualization
```

### View Visualization

**Option 1: Using `ttmem` (recommended for remote development)**
- Run `ttmem`, select "Yes" when asked to serve via HTTP
- Open `http://localhost:PORT/report.html` in your browser
- VS Code Remote SSH automatically forwards the port

**Option 2: Using VS Code Live Server**
- Right-click on the HTML file and choose "Open with Live Server"
- Requires the Live Server extension in VS Code

### Features

- Interactive HTML visualization with memory graphs, fragmentation analysis, peak operations
- Synchronized JSON outputs (nth element = same operation)
- Filtered data (excludes deallocate operations)
- Timestamped runs (never overwrites previous data)

## Model Analysis Tool

Analyze PyTorch models to identify which modules/ops work on TT hardware.

### Prerequisites

### Quick Start

```bash
ttchop \
    --model-path path/to/model.py::load_model \
    --inputs-path path/to/model.py::get_inputs
```

### What It Does

1. **Extract Modules**: Identifies all unique modules in the model
2. **Run Op-by-Op Analysis**: Tests each module hierarchically on TT hardware
3. **Generate Report**: Creates interactive HTML visualization showing pass/fail status

### Usage

The tool requires two Python functions:
- `load_model()` - Returns the PyTorch model
- `get_inputs()` - Returns sample input tensors

```bash
# Basic usage
ttchop --model-path model.py::load_model --inputs-path model.py::get_inputs

# Specify output directory
ttchop --model-path model.py::load_model --inputs-path model.py::get_inputs --dir ./output
```

### Output

```
<ModelClass>/
├── unique_modules.json      # Module analysis results with status
├── analysis_report.html     # Interactive tree visualization
└── module_irs/              # IR files for each module
```

### Example

```python
# model.py
import torch
import torch.nn as nn

def load_model():
    # Just return the model on CPU - the tool handles device placement
    return nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def get_inputs():
    # Just return CPU tensors - the tool handles device placement
    return torch.randn(1, 3, 224, 224)
```

> **Note**: Your functions should return CPU models/tensors. The tool automatically handles moving them to the TT device.

```bash
ttchop --model-path model.py::load_model --inputs-path model.py::get_inputs
```
