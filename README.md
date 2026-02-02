# TT Swiss - A swiss knife for model bringup 🇨🇭

This repo is a collection of all of the useful tools for enabling models to work on TT hardware. This includes:
1. Memory profiler `ttmem` - useful for look at memory usage of the model. Signs that you need this - errors like `Out of Memory: Not enough space to allocate <nbytes> B DRAM buffer across <nbanks> banks` 

2. Claude skills and commands - We recommend you copy paste these in your `~/.claude` or `<tt-xla-path>/.claude` for easier debugging of models.

## Memory profiler

### Quick start

```bash
# Install from GitHub
pip install git+https://github.com/vkovinicTT/tt-swiss.git

# Or install locally for development
pip install -e /path/to/tt-swiss
```

### Prerequisites (TT-XLA Setup)

Before using the profiler, you need to configure TT-XLA for memory logging:

#### 1. Build the project with debug flags

```bash
source venv/activate

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTT_RUNTIME_DEBUG=ON

cmake --build build
```

#### 2. Export runtime logger flag (for op and memory info)

```bash
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
export TT_RUNTIME_MEMORY_LOG_LEVEL=operation
```

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

**Option 3: Manual HTTP server**
```bash
cd logs/your_model_YYYYMMDD_HHMMSS/
python -m http.server 8000
# Open http://localhost:8000/your_model_report.html
```

### Features

- Interactive HTML visualization with memory graphs, fragmentation analysis, peak operations
- Synchronized JSON outputs (nth element = same operation)
- Filtered data (excludes deallocate operations)
- Timestamped runs (never overwrites previous data)
