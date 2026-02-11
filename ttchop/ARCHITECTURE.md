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
- **File viewer overlay** — view MLIR IR files, logs, and op-by-op reports inline
- **Collapsible error boxes** — failed ops show truncated preview, click to expand full trace
- **Op-to-TTIR linking** — clicking an op in the report switches to TTIR tab and highlights the corresponding line
- **Light/dark theme toggle** with `localStorage` persistence

### Step 4 — Markdown Summary (`summary.py`)

Generates a `summary.md` alongside the HTML report:

- Status table with merged success counts (success + inherited_success = "Success")
- Failed modules listed by `ClassName (path)` format (no module IDs)
- Container modules list their failed children instead of individual ops
- Detailed report section with full error traces per failed op (from parsed log blocks)

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
            SYSTEM_DESC_PATH=                     |
              ttrt-artifacts/system_desc      Tests each op individually on TT hardware
                                              Writes pytest-json-report
```

After the test completes, the execution log is parsed by `log_parser.py` to extract detailed per-op error traces (TT_FATAL messages with full backtraces). These are saved as `*_op_by_op_parsed.json` and used by both the HTML visualizer and the markdown summary.

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
       |     |     Writes json report
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
       +-- Collect all module files (IR, logs, reports, parsed traces)
       +-- Build nested tree structure with file data
       +-- Generate self-contained HTML with embedded CSS/JS
       +-- Output: analysis_report.html
       |
       v
  ======= STEP 4: SUMMARY ========
  generate_summary(modules_json)
       |
       +-- Read updated unique_modules.json
       +-- Count statuses (merge success + inherited_success)
       +-- Collect failed modules with enriched error traces
       +-- Build markdown with status table, failed modules, detailed report
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
| `cli.py` | 103 | Entry point, orchestrates 4-step pipeline |
| `module_extractor.py` | 137 | Step 1: Extract unique modules with shape capture |
| `op_by_op_runner.py` | 305 | Step 2: Hierarchical op-by-op with lazy IR export |
| `module_tree.py` | 81 | ModuleNode dataclass, tree builder, status updater |
| `ir_export_single_module.py` | 64 | Subprocess: export IR for one module |
| `module_runner.py` | 92 | torch.compile + forward pass for IR generation |
| `shapes.py` | 72 | ShapeCapture class (forward hooks) |
| `shape_capture_subprocess.py` | 55 | Subprocess: shape capture on TT device |
| `log_parser.py` | 167 | Parse op-by-op execution logs for per-op error traces |
| `visualizer.py` | ~600 | Step 3: Interactive HTML report generation |
| `summary.py` | ~200 | Step 4: Markdown summary generation |
| `data_types.py` | 69 | ModuleInfo dataclass, shared constants |
| `utils.py` | 156 | Shared utilities (function loading, device setup, path helpers) |

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
```
