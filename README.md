# Memory Profiler

Profiles memory usage during model execution on Tenstorrent hardware. Extracts memory statistics (DRAM, L1, L1_SMALL, TRACE) and operation metadata (MLIR ops, attributes, types) into synchronized JSON files with interactive HTML visualization.

## Setup

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

Export the flag so we can track which operations are being executed alongside their input/output shapes and params:

```bash
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
```

### 4. Get the memory profiler

Cherry-pick the memory profiler commit from `vkovinic/mochi`:

```bash
git fetch origin vkovinic/mochi && git cherry-pick $(git log origin/vkovinic/mochi --oneline --grep="memory profiler with visualizer" -1 --format="%H")
```

### 5. Run full analysis with visualization

```bash
python memory_profiler/run_profiled.py path/to/forward_pass/script.py
```

Output results will be in `memory_profiler/logs/<script_name>_<timestamp>/` containing:
- `<script_name>_profile.log` - Raw log
- `<script_name>_memory.json` - Memory usage by op
- `<script_name>_operations.json` - Op metadata
- `<script_name>_report.html` - Interactive visualization

### 6. View visualization

Right-click on the HTML file and choose "Open with Live Server" (requires the Live Server extension in VS Code).


## Usage

```bash
# Default: run + parse + visualize (recommended)
python memory_profiler/run_profiled.py path/to/your_model.py

# Only capture logs (for later processing)
python memory_profiler/run_profiled.py --log path/to/your_model.py

# Parse existing log file
python memory_profiler/run_profiled.py --analyze memory_profiler/logs/your_model_20260122_143957/your_model_profile.log

# Generate visualization from existing run
python memory_profiler/run_profiled.py --visualize memory_profiler/logs/your_model_20260122_143957/
```

## Output Structure

```
memory_profiler/logs/<script_name>_YYYYMMDD_HHMMSS/
├── <script_name>_memory.json      # Memory stats per operation
├── <script_name>_operations.json  # Operation metadata per operation
├── <script_name>_profile.log      # Raw logs
└── <script_name>_report.html      # Interactive visualization
```

## Features

- Interactive HTML visualization with memory graphs, fragmentation analysis, peak operations
- Synchronized JSON outputs (nth element = same operation)
- Filtered data (excludes deallocate operations)
- Timestamped runs (never overwrites previous data)
