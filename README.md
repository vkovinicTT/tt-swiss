# TT Memory Profiler

Memory profiler for Tenstorrent hardware - extracts per-op memory stats and generates interactive visualizations.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/tt-memory-profiler.git

# Or install locally for development
pip install -e /path/to/tt-memory-profiler
```

## Prerequisites (TT-XLA Setup)

Before using the profiler, you need to configure TT-XLA for memory logging:

### 1. Enable memory logging per operation

Add this line of code at the beginning of the `execute` function in `pjrt_implementation/src/api/flatbuffer_loaded_executable_instance.cc`:

```cpp
tt::runtime::setMemoryLogLevel(tt::runtime::MemoryLogLevel::Operation);
```

### 2. Build the project with debug flags

```bash
source venv/activate

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTT_RUNTIME_DEBUG=ON

cmake --build build
```

### 3. Export runtime logger flag

```bash
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
```

## Usage

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

## Output Structure

Output is stored in `./logs/` relative to your current working directory (or `--output-dir` if specified):

```
./logs/<script_name>_YYYYMMDD_HHMMSS/
├── <script_name>_memory.json      # Memory stats per operation
├── <script_name>_operations.json  # Operation metadata per operation
├── <script_name>_profile.log      # Raw logs
└── <script_name>_report.html      # Interactive visualization
```

## View Visualization

Right-click on the HTML file and choose "Open with Live Server" (requires the Live Server extension in VS Code).
```

## Features

- Interactive HTML visualization with memory graphs, fragmentation analysis, peak operations
- Synchronized JSON outputs (nth element = same operation)
- Filtered data (excludes deallocate operations)
- Timestamped runs (never overwrites previous data)
