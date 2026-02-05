# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT Memory Profiler - Memory profiler for Tenstorrent hardware that extracts per-operation memory statistics from TT-XLA runtime logs and generates interactive HTML visualizations.

## Installation & Development

```bash
# Install locally for development
pip install -e /path/to/tt-memory-profiler
```

## Commands

```bash
# Interactive CLI (recommended for remote development)
ttmem

# Full profiling pipeline (run + parse + visualize)
tt-memory-profiler path/to/your_model.py

# Capture logs only
tt-memory-profiler --log path/to/your_model.py

# Parse existing log file
tt-memory-profiler --analyze path/to/script_profile.log

# Generate visualization from parsed data
tt-memory-profiler --visualize ~/.ttmem/reports/<report-name>/

# Extract last run from multi-pass logs (removes warmup)
python memory_profiler/extract_last_run.py path/to/script_profile.log

# Standalone visualization generation
python memory_profiler/generate_viz.py ~/.ttmem/reports/<report-name>/
```

## Architecture

### Processing Pipeline

1. **interactive_cli.py** - Interactive CLI (`ttmem`), guides users through log processing with built-in HTTP server for remote access. Includes option to browse and serve existing reports from `~/.ttmem/reports/`
2. **run_profiled.py** - CLI entry point (`tt-memory-profiler`), orchestrates the pipeline with 4 modes: default (full), --log, --analyze, --visualize
3. **parser.py** - Main log parser, coordinates all sub-parsers and produces synchronized JSON outputs
4. **mlir_parser.py** - Extracts MLIR operation details (op names, shapes, dtypes, attributes) using regex
5. **memory_parser.py** - Parses memory state blocks (DRAM, L1, L1_SMALL, TRACE) from log lines
6. **inputs_registry_parser.py** - Extracts function argument registry (parameters/constants/inputs) from MLIR module dumps
7. **visualizer.py** - Generates interactive HTML reports with Plotly.js

### Visualization Sections

The HTML report includes:
- **Memory Usage Over Time**: Tab-selectable graph (DRAM/L1/L1_SMALL buttons) showing allocated memory, weight ops (red markers), and capacity line
- **Tile Padding Overhead**: Graph comparing unpadded (logical) vs padded (tile-aligned) memory in total MB; table of top 10 ops sorted by absolute overhead in MB
- **Peak Memory Analysis**: Cards showing peak usage per memory type with operation details
- **Top Memory Consumers**: Table of operations with highest DRAM allocation

### Key Design Patterns

- **Synchronized JSON outputs**: The nth element in `memory.json` corresponds to the nth element in `operations.json`
- **Import compatibility**: Modules use try/except for relative vs absolute imports to work both as package and standalone scripts
- **const_eval tracking**: Operations are classified as weight vs activation based on const_eval graph detection

### Output Structure

```
~/.ttmem/reports/<report-name>/
├── <report-name>_memory.json         # Memory stats per operation
├── <report-name>_operations.json     # Operation metadata
├── <report-name>_inputs_registry.json # Weights/constants/inputs registry
└── <report-name>_report.html         # Interactive visualization
```

Note: `<report-name>` has underscores replaced with hyphens from the original script name.

## Prerequisites

Requires TT-XLA configured for memory logging:
1. Enable `tt::runtime::setMemoryLogLevel(tt::runtime::MemoryLogLevel::Operation)` in TT-XLA
2. Build TT-XLA with `-DCMAKE_BUILD_TYPE=Debug -DTT_RUNTIME_DEBUG=ON`
3. Export `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG`
