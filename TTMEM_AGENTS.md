# TTMEM Agent Instructions

LLM-friendly memory profiler for Tenstorrent hardware. Generates compact markdown reports from runtime logs.

## Quick Start

```bash
# Generate memory profile report to stdout
ttmem --llm --logfile path/to/profile.log

# Save to file
ttmem --llm --logfile path/to/profile.log -o report.md
```

## Output Format

Reports follow [llms.txt](https://llmstxt.org/) conventions:

```markdown
# Memory Profile: <model-name>

> Peak DRAM: X MB/bank (Y%) | Peak L1: Z MB/bank | Ops: N | Weights: W MB | Padding Overhead: P%

## Configuration
## Peak Memory
## Top 10 Memory Consumers (DRAM)
## Top 10 Padding Overhead
## Model Weights (X MB total)
## Operation Distribution
```

## Key Metrics

| Metric | Description | Optimization Target |
|--------|-------------|---------------------|
| Peak DRAM | Max DRAM per bank during execution | < 80% capacity |
| Peak L1 | Max L1 per bank (fast scratchpad) | < 90% capacity |
| Utilization % | allocated / capacity | Lower is better headroom |
| Padding Overhead | Wasted memory from 32x32 tile alignment | Minimize for small tensors |
| Weight vs Activation ops | Static weights vs dynamic intermediates | Weights load once |

## Interpreting Results

### Memory Pressure Issues
- **DRAM > 90%**: Risk of OOM, reduce batch size or model size
- **L1 > 95%**: Operations may spill to DRAM, performance degrades
- **High padding overhead (>50%)**: Consider reshaping tensors to tile-friendly dimensions (multiples of 32)

### Common Bottlenecks
- `ttnn.matmul` with large shapes: Dominates DRAM, consider tiling
- `ttnn.mean`/`ttnn.softmax`: Often at peak memory due to intermediate buffers
- `ttcore.load_cached`: Weight loading ops, appear early in execution

### Shape Notation
- `1x32x4x262144`: batch x channels x height x width (or similar)
- `(f32)` / `(bf16)`: Data type affects memory by 2x

## Example Queries

**"Why is memory usage high?"**
→ Check Top 10 Memory Consumers table, identify ops at peak

**"Can this model fit on device?"**
→ Compare Peak DRAM to capacity in Configuration section

**"Where is memory wasted?"**
→ Check Padding Overhead table for inefficient tensor shapes

**"What operations dominate?"**
→ Check Operation Distribution for hotspots

## Log File Sources

Log files come from running models with TT-XLA memory logging enabled:

```bash
# Generate logs (requires TT-XLA setup)
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
tt-memory-profiler --log path/to/model.py
# Output: ./logs/<model>_<timestamp>/<model>_profile.log
```

## Report Storage

Parsed reports are cached at `~/.ttmem/reports/<model-name>/` containing:
- `*_memory.json` - Per-op memory snapshots
- `*_operations.json` - Op metadata (shapes, dtypes, locations)
- `*_inputs_registry.json` - Model weights/constants registry
- `*_report.html` - Interactive HTML visualization

## Limitations

- Memory values are per-bank (multiply by bank count for total)
- DRAM has 7-12 banks depending on chip config
- L1 has 64-110 banks
- Padding overhead only tracked for tiled tensors
- Location field may be "N/A" if MLIR debug info unavailable
