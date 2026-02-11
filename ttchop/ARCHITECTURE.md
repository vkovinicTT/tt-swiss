# ttchop — Architecture

## What is ttchop?

`ttchop` is a CLI tool that analyzes PyTorch models to determine which modules and operations run on Tenstorrent hardware. It extracts each unique submodule, exports its MLIR IR via `torch.compile(backend="tt")`, and runs op-by-op tests against the TT backend. Results are presented as an interactive HTML report and a markdown summary.

**Entry point:** `ttchop = "ttchop.cli:main"` (registered in `pyproject.toml`)

```
ttchop --model-path file.py::load_model --inputs-path file.py::get_inputs [--dir output] [--root-only]
```

---

## High-Level Pipeline

```
                            ttchop CLI (cli.py)
                                  |
            +------------+--------+--------+------------+
            |            |                 |            |
      Step 1: Extract  Step 2: Op-by-Op  Step 3: HTML  Step 4: Summary
     (module_extractor) (op_by_op_runner) (visualizer)  (summary)
            |            |                 |            |
     unique_modules.json updated .json  report.html  summary.md
```

### Step 1 — Extract Unique Modules (`module_extractor.py`)

```
load_fn() --> model       inputs_fn() --> sample_input
    |                          |
    +----------+---------------+
               |
     [use_device=True?]
        /           \
      YES            NO
       |              |
  Subprocess       CPU forward pass
  (shape_capture   (ShapeCapture class
   _subprocess.py)  with forward hooks)
       |              |
       +------+-------+
              |
   shapes: {module_path -> {inputs: [...], outputs: [...]}}
              |
   For each named_module in model:
     - Get class_name, shapes, dtypes, parameters
     - Compute uniqueness_key
     - Group by uniqueness key
              |
   unique_modules.json
```

**Uniqueness key formula:**
- Regular modules: `class_name || sorted_input_shapes || sorted_output_shapes || json(params)`
- Containers (Sequential/ModuleList/ModuleDict): `class_name || PATH:actual_path`

**Module ID naming:** `generate_module_id(index, module_path)` extracts the last segment of the dotted path as the directory name. Example: `"res_blocks.0.conv1"` → `mod_004_conv1`, `"full_model"` → `mod_000_full_model`.

**Root module:** The root module (unnamed in `named_modules()`) is assigned the path `"full_model"`.

### Step 2 — Hierarchical Op-by-Op Analysis (`op_by_op_runner.py`)

Builds an N-ary tree from the flat modules list and recursively tests each module with lazy IR export.

```
unique_modules.json
        |
  build_module_tree()         (module_tree.py)
        |
  ModuleNode tree (N-ary)
        |
  run_hierarchical_op_by_op()
        |
  _ensure_system_desc()       generates ttrt-artifacts/system_desc.ttsys if missing
        |
  analyze(root, is_root_call=True)   recursive (see details below)
        |
  _update_container_status(root)     post-order: set container status from children
        |
  Updated unique_modules.json with status per module
```

### Step 3 — HTML Visualization (`visualizer.py`)

Reads the updated `unique_modules.json` and generates a self-contained HTML report:

- **Collapsible tree view** with status-colored dot indicators
- **Module detail panel** — shapes, dtypes, parameters, IR file buttons
- **File viewer overlay** — view MLIR IR files, logs, op-by-op reports, and failed op MLIR inline
- **Collapsible error boxes** — failed ops show truncated preview, click to expand full trace
- **Op-to-TTIR linking** — clicking an op in the report switches to TTIR tab and highlights the corresponding line
- **Failed Ops IR viewer** — browse individual MLIR modules for each failed op (saved by `--failed-ops-folder`)
- **Light/dark theme toggle** with `localStorage` persistence

### Step 4 — Markdown Summary (`summary.py`)

Generates a `summary.md` alongside the HTML report. The summary is **op-centric** — it collects unique failed ops across all modules and presents them in a single table:

- **Header:** `# ModelName — FAILED (N unique ops)` or `# ModelName — PASSED`
- **Metadata:** device architecture, mesh shape, date
- **Failed Ops table** with columns: Op, Inputs, Outputs, Params, Module, Error
- **Detailed Report** per unique op with backtick-wrapped values and collapsible error traces

**Op deduplication:** Failed ops are deduplicated by `(op_name, inputs, outputs, op_params)`. For each unique op, the deepest module (by tree depth) where it appears is shown. Op params are extracted from TTIR MLIR files by matching on op name and input tensor shapes.

**TTIR attribute extraction:** `summary.py` parses TTIR MLIR files to extract the raw `<{...}>` attribute string for each op. Matching uses `(op_name, TTIR-style input types)` with a name-only fallback. TensorDesc strings from the JSON report are converted to TTIR type signatures (e.g., `tensor<1x128x2x4x4xbf16>`) for matching.

---

## The `analyze()` Recursive Function

Located in `op_by_op_runner.py`. For each module node, it performs two subprocess calls:

### Subprocess 1: IR Export (`_export_ir`)

```
Parent process                          Subprocess
     |                                      |
     +--- subprocess.run() --------------> ir_export_single_module.py
          timeout=300s                       |
                                      Load model via load_fn()
                                      Find submodule via get_module_by_path()
                                             |
                                      module_runner.py:run_submodule_for_ir()
                                             |
                                      torch_xla.set_custom_compile_options({
                                        export_path: module_irs/mod_XXX_name/,
                                        export_model_name: mod_XXX_name
                                      })
                                             |
                                      torch.compile(submodule, backend="tt")
                                             |
                                      Forward pass (generates MLIR)
                                             |
                                      Output: module_irs/mod_XXX_name/irs/ttir_*.mlir
```

**Key detail:** Runtime failures during the forward pass are tolerated. MLIR IR files are exported during compilation, before execution. If TTIR files exist on disk despite a non-zero exit code, the export is considered successful.

### Subprocess 2: Op-by-Op Test (`_run_op_by_op`)

```
Parent process                            Subprocess
     |                                        |
     +--- subprocess.run() --------------->  pytest -svv
          timeout=1800s                       tests/op_by_op/op_by_op_test.py::test_op_by_op
          cwd=tt-xla root                     --folder=module_irs/mod_XXX_name
          env:                                --ir-file-prefix=irs/ttir_
            PYTHONPATH += tests/              --json-report
              + tt-mlir python pkgs           --json-report-file=...report.json
            SYSTEM_DESC_PATH=                 --failed-ops-folder=.../failed_ops
              ttrt-artifacts/system_desc          |
                                              Tests each op individually on TT hardware
                                              Writes pytest-json-report
                                              Saves failed op MLIR to failed_ops/
```

After the test completes:
1. The execution log is parsed by `log_parser.py` to extract detailed per-op error traces (TT_FATAL messages with full backtraces). These are saved as `*_op_by_op_parsed.json` and used by both the HTML visualizer and the markdown summary.
2. Individual MLIR modules for each failed op are saved to `failed_ops/` by the `op_by_op_infra` framework (via `--failed-ops-folder`). Files are named `{global_index:04d}_{sanitized_op_name}.mlir` and contain the last successfully compiled module (TTIR, TTNN, or StableHLO depending on the failure point).

---

## Decision Tree: `--root-only` vs Default

### Default (full recursive)

```
analyze(node)
    |
    +-- Is container? (Sequential/ModuleList/ModuleDict)
    |       YES --> status = "skipped", recurse to all children
    |
    +-- Export IR
    |       FAILED --> status = "ir_export_failed"
    |                  recurse to all children (drill down)
    |
    +-- Run op-by-op test
    |       |
    |       +-- No TTIR files? --> status = "skipped", recurse to children
    |       |
    |       +-- SUCCESS --> status = "success"
    |       |               mark ALL descendants = "inherited_success"
    |       |               STOP recursion
    |       |
    |       +-- FAILED --> status = "failed"
    |                      record failed_ops list
    |                      recurse to all children
```

### With `--root-only`

```
analyze(node, is_root_call)
    |
    +-- NOT root call? --> mark entire subtree "skipped"
    |
    +-- Is container? --> status = "skipped", recurse (children hit NOT root check)
    |
    +-- Export IR
    |       FAILED --> status = "ir_export_failed", children = "skipped"
    |
    +-- Run op-by-op
            SUCCESS --> status = "success", children = "skipped"
            FAILED  --> status = "failed", children = "skipped"
```

| Aspect | Default | `--root-only` |
|--------|---------|---------------|
| Modules tested | All, recursively | Root only |
| On success | Children get `inherited_success` | Children get `skipped` |
| On failure | Recurse to isolate failing leaf | Stop, children `skipped` |
| Subprocess count | Up to 2N | 2 |
| Use case | Full compatibility audit | Quick root-level check |

---

## Status Values and Propagation

| Status | Meaning | Set by |
|--------|---------|--------|
| `success` | Module passed op-by-op directly | `analyze()` on test pass |
| `inherited_success` | Parent passed, no need to test | `_mark_subtree_success()` |
| `failed` | Op-by-op test failed (has `failed_ops`) | `analyze()` on test fail |
| `ir_export_failed` | Could not export MLIR IR | `analyze()` on export fail |
| `skipped` | Container, no TTIR files, or `--root-only` | Various |

**Container status** (`_update_container_status`, post-order):
- Any child `failed` or `ir_export_failed` → container = `failed`
- All children `success`/`inherited_success` → container = `inherited_success`
- Mixed → container = `inherited_success`
- No children with status → `skipped`

---

## Log Parser (`log_parser.py`)

Parses raw `op_by_op.log` files into structured per-op execution blocks:

```
op_by_op.log
     |
  parse_op_by_op_log()
     |
  For each "Starting execution of program: main" block:
     - Track sub-program depth (main_const_eval_0 etc.)
     - Record last "Executing operation: ttnn.xxx" at top level
     - Detect TT_FATAL errors → extract error_message + full error_trace
     - Error trace collection stops at non-trace lines
       (timestamps, "Always | DEBUG" lines)
     |
  List of {success, last_ttnn_op, error_message, error_trace}
```

**Key design:** The parser distinguishes top-level program executions from nested sub-programs by tracking depth. Error traces are collected only within the backtrace block — lines starting with `---`, `info:`, `backtrace:`, or lines without timestamp/`Always |` prefix.

---

## Output Directory Structure

```
<output_dir>/                               (default: ./<ModelClassName>/)
  |
  +-- unique_modules.json                   Steps 1+2 output (modules + status)
  |
  +-- module_irs/
  |     +-- mod_000_full_model/
  |     |     +-- irs/
  |     |     |     +-- ttir_0_mod_000_full_model.mlir
  |     |     |     +-- ttnn_0_mod_000_full_model.mlir
  |     |     |     +-- ...
  |     |     +-- failed_ops/               Per-failed-op MLIR modules
  |     |     |     +-- 0005_ttir_conv3d.mlir
  |     |     |     +-- 0012_ttir_conv3d.mlir
  |     |     |     +-- ...
  |     |     +-- run.log                   IR export stdout/stderr
  |     |     +-- op_by_op.log              pytest output
  |     |     +-- mod_000_full_model_op_by_op_report.json
  |     |     +-- mod_000_full_model_op_by_op_parsed.json
  |     +-- mod_001_initial_conv/
  |     |     +-- ...
  |     +-- ...
  |
  +-- analysis_report.html                  Step 3 output
  +-- summary.md                            Step 4 output
```

---

## Full Data Flow

```
User invocation:
  ttchop --model-path m.py::load --inputs-path m.py::inputs [--root-only]
                                    |
                              cli.py:main()
                                    |
                 +------------------+------------------+
                 |                                     |
        load_function_from_path()             get_tt_xla_root()
        (dynamic import via importlib)        (find tt-xla root)
                 |                                     |
                 v                                     v
  ======= STEP 1: EXTRACT ========          project_root (Path)
  extract_unique_modules()
       |
       +-- load_fn() --> model.eval()
       +-- inputs_fn() --> sample_input
       +-- [subprocess] shape_capture_subprocess.py
       |     Runs forward pass on TT device
       |     Returns per-module input/output shapes + device info
       +-- Group by uniqueness key
       +-- generate_module_id(index, module_path) for each group
       +-- Write unique_modules.json
       |
       v
  ======= STEP 2: OP-BY-OP ========
  build_module_tree(result) --> root: ModuleNode
       |
  run_hierarchical_op_by_op(root, ..., root_only)
       |
       +-- _ensure_system_desc()
       |     [subprocess] ttrt query --save-artifacts
       |
       +-- analyze(root, is_root_call=True)
       |     |
       |     +-- [subprocess] ir_export_single_module.py
       |     |     Loads model, finds submodule by path
       |     |     torch.compile(sub, backend="tt") --> exports MLIR
       |     |
       |     +-- [subprocess] pytest op_by_op_test.py
       |     |     Tests each TTIR op on hardware
       |     |     Writes json report + failed op MLIR files
       |     |
       |     +-- parse_op_by_op_log() --> parsed error traces
       |     +-- Mark status, recurse or propagate
       |
       +-- _update_container_status(root)
       +-- update_modules_with_status() --> write updated JSON
       |
       v
  ======= STEP 3: VISUALIZE ========
  generate_visualization(modules_json)
       |
       +-- Read updated unique_modules.json
       +-- Collect all module files (IR, logs, reports, parsed traces, failed op MLIR)
       +-- Build nested tree structure with file data
       +-- Generate self-contained HTML with embedded CSS/JS
       +-- Output: analysis_report.html
       |
       v
  ======= STEP 4: SUMMARY ========
  generate_summary(modules_json)
       |
       +-- Read updated unique_modules.json
       +-- Collect unique failed ops across all modules (dedup by op signature)
       +-- Enrich each op with TTIR attributes and error traces from parsed logs
       +-- Build markdown with header, failed ops table, detailed report
       +-- Output: summary.md
```

---

## Shared Constants (`data_types.py`)

| Constant | Value | Used by |
|----------|-------|---------|
| `CONTAINER_TYPES` | `("Sequential", "ModuleList", "ModuleDict")` | module_extractor, module_runner, op_by_op_runner, ir_export_single_module, summary |
| `STATUS_ORDER` | `["failed", "ir_export_failed", "success", "inherited_success", "skipped", "unknown"]` | (available for consumers) |
| `STATUS_LABELS` | `{status → display name}` | (available for consumers) |
| `MODULE_ATTRS` | List of 21 PyTorch module attribute names | extract_module_parameters() |

---

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `cli.py` | 102 | Entry point, orchestrates 4-step pipeline |
| `module_extractor.py` | 136 | Step 1: Extract unique modules with shape capture |
| `op_by_op_runner.py` | 307 | Step 2: Hierarchical op-by-op with lazy IR export |
| `module_tree.py` | 80 | ModuleNode dataclass, tree builder, status updater |
| `ir_export_single_module.py` | 63 | Subprocess: export IR for one module |
| `module_runner.py` | 91 | torch.compile + forward pass for IR generation |
| `shapes.py` | 72 | ShapeCapture class (forward hooks) |
| `shape_capture_subprocess.py` | 59 | Subprocess: shape capture on TT device |
| `log_parser.py` | 166 | Parse op-by-op execution logs for per-op error traces |
| `visualizer.py` | 635 | Step 3: Interactive HTML report with failed ops IR viewer |
| `summary.py` | 349 | Step 4: Op-centric markdown summary with TTIR attribute extraction |
| `data_types.py` | 69 | ModuleInfo dataclass, shared constants |
| `utils.py` | 155 | Shared utilities (function loading, device setup, path helpers) |

### Subprocess scripts

Two files are invoked as standalone subprocesses (not via package imports):
- `ir_export_single_module.py` — exports IR for a single module
- `shape_capture_subprocess.py` — captures shapes on TT device

These use `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` to enable `from ttchop.xxx import ...` package imports when run outside the package context.

### `__init__.py` exports

```python
from .module_extractor import extract_unique_modules
from .module_tree import ModuleNode, build_module_tree, update_modules_with_status
from .op_by_op_runner import run_hierarchical_op_by_op
from .summary import generate_summary
from .utils import load_function_from_path, setup_tt_device, get_module_by_path
from .visualizer import generate_visualization

__all__ = [
    "extract_unique_modules", "generate_summary", "generate_visualization",
    "ModuleNode", "build_module_tree", "update_modules_with_status",
    "run_hierarchical_op_by_op", "load_function_from_path", "setup_tt_device",
    "get_module_by_path",
]
```
