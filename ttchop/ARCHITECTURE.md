# ttchop - Full Architecture Report

## What is ttchop?

`ttchop` is a CLI tool that analyzes PyTorch models to determine which modules and operations are compatible with Tenstorrent hardware. It does this by extracting each unique submodule from a model, exporting its MLIR IR, and running op-by-op tests against the TT backend.

**Entry point:** `ttchop = "ttchop.cli:main"` (registered in `pyproject.toml`)

```
ttchop --model-path file.py::load_model --inputs-path file.py::get_inputs [--dir output] [--root-only]
```

---

## High-Level Pipeline

```
                            ttchop CLI (cli.py)
                                  |
               +------------------+------------------+
               |                  |                  |
         Step 1: Extract    Step 2: Op-by-Op   Step 3: Visualize
        (module_extractor)  (op_by_op_runner)    (visualizer.py)
               |                  |                  |
        unique_modules.json  updated .json     analysis_report.html
```

### Step 1 - Extract Unique Modules (`module_extractor.py`)

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
     - Compute uniqueness_key = class + shapes + params
       (containers use path as key instead)
     - Group by uniqueness key
              |
   unique_modules.json
   {metadata: {...}, modules: [{id, class_name, module_path, parent, shapes, ...}]}
```

**Uniqueness key formula:**
- Regular modules: `class_name || sorted_input_shapes || sorted_output_shapes || json(params)`
- Containers (Sequential/ModuleList/ModuleDict): `class_name || PATH:actual_path`

### Step 2 - Hierarchical Op-by-Op Analysis (`op_by_op_runner.py`)

This is the core of ttchop. It builds a tree from the flat modules list and recursively tests each module.

```
unique_modules.json
        |
  build_module_tree()    (module_tree.py)
        |
  ModuleNode tree (N-ary)
        |
  run_hierarchical_op_by_op()
        |
  _ensure_system_desc()   <-- generates ttrt-artifacts/system_desc.ttsys if missing
        |
  analyze(root, is_root_call=True)   <-- recursive function (details below)
        |
  _update_container_status(root)     <-- post-order: set container status from children
        |
  Updated unique_modules.json with status per module
```

### Step 3 - Visualization (`visualizer.py`)

Reads the updated `unique_modules.json` and generates a self-contained HTML report with:
- Collapsible tree view with status-colored indicators
- Module detail panels (shapes, dtypes, parameters, failed ops)
- Summary statistics

---

## The `analyze()` Recursive Function - Core Logic

This is the heart of the tool. Located in `op_by_op_runner.py:182-249`.

For each module node, it performs two subprocess calls:

### Subprocess 1: IR Export (`_export_ir`)

```
Parent process                          Subprocess
     |                                      |
     +--- subprocess.run() --------------> ir_export_single_module.py
          timeout=300s                       |
                                      Load model via load_fn()
                                      Find submodule by path
                                             |
                                      module_runner.py:run_submodule_for_ir()
                                             |
                                      torch_xla.set_custom_compile_options({
                                        export_path: module_irs/mod_XXX/,
                                        export_model_name: mod_XXX_ClassName
                                      })
                                             |
                                      torch.compile(submodule, backend="tt")
                                             |
                                      Forward pass (generates MLIR)
                                             |
                                      Output: module_irs/mod_XXX/irs/ttir_*.mlir
```

**Key detail:** Runtime failures during the forward pass are tolerated - the MLIR IR files are exported during compilation, before execution, so they're available even if the run crashes.

### Subprocess 2: Op-by-Op Test (`_run_op_by_op`)

```
Parent process                            Subprocess
     |                                        |
     +--- subprocess.run() --------------->  pytest -svv
          timeout=600s                        tests/op_by_op/op_by_op_test.py::test_op_by_op
          cwd=tt-xla root                     --folder=module_irs/mod_XXX
          env:                                --ir-file-prefix=irs/ttir_
            PYTHONPATH += tests/              --json-report
              + tt-mlir python pkgs           --json-report-file=mod_XXX_op_by_op_report.json
            SYSTEM_DESC_PATH=                     |
              ttrt-artifacts/system_desc      Reads each ttir_*.mlir file
                                              Tests each op individually on TT hardware
                                              Writes pytest-json-report with per-op results
                                                  |
                                              Report parsed by _parse_report()
                                              Extracts: op_name, error_message, inputs, outputs
                                              for each failed op
```

---

## Decision Tree: `--root-only` vs Default (Full Recursive)

### Without `--root-only` (default)

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
    |       |               STOP recursion (children don't need testing)
    |       |
    |       +-- FAILED --> status = "failed"
    |                      record failed_ops list
    |                      recurse to all children (find which sub-parts fail)
```

**Behavior:** Top-down recursive. Success propagates down (children inherit). Failure triggers deeper investigation into submodules. This gives you a full compatibility map.

### With `--root-only`

```
analyze(node, is_root_call)
    |
    +-- NOT root call? --> SKIP entire subtree immediately
    |                      (all descendants = "skipped")
    |
    +-- (root call only below this point)
    |
    +-- Is container? --> status = "skipped", recurse children
    |                     (but children hit the NOT root check above --> skipped)
    |
    +-- Export IR
    |       FAILED --> status = "ir_export_failed"
    |                  all children = "skipped" (no drill-down)
    |
    +-- Run op-by-op test
            |
            +-- SUCCESS --> status = "success"
            |               all children = "skipped" (NOT inherited_success)
            |
            +-- FAILED --> status = "failed"
                           record failed_ops
                           all children = "skipped" (no drill-down)
```

**Behavior:** Only tests the root module. All children are marked "skipped" regardless of outcome. No recursive drill-down on failure.

### Side-by-Side Comparison

```
                    Default (full)              --root-only
                    +-----------+               +-----------+
                    |  Root: A  |               |  Root: A  |
                    |  FAILED   |               |  FAILED   |
                    +-----+-----+               +-----+-----+
                          |                           |
               +----------+----------+     +----------+----------+
               |          |          |     |          |          |
          +----+----+ +---+---+ +---+---+  +----+----+ +---+---+ +---+---+
          | B: test | | C: ok | | D: err|  |B: skip  | |C: skip| |D: skip|
          | FAILED  | |SUCCESS| |IR_FAIL|  |         | |       | |       |
          +----+----+ +---+---+ +---+---+  +---------+ +-------+ +-------+
               |         inh.     drill
          +----+----+  success   deeper
          |E: test  |
          | SUCCESS |
          +---------+
           (found the
            failing leaf)
```

### Comparison Table

| Aspect | Default (full) | `--root-only` |
|--------|---------------|---------------|
| **Modules tested** | All, recursively | Root only |
| **On success** | Children get `inherited_success` | Children get `skipped` |
| **On failure** | Recurse to children to isolate | Stop, children `skipped` |
| **On IR export fail** | Recurse to children | Stop, children `skipped` |
| **Subprocess count** | Up to 2N (N=unique modules) | 2 (1 IR export + 1 test) |
| **Speed** | Slower (comprehensive) | Fast (single module) |
| **Use case** | Full compatibility audit | Quick root-level check |
| **Status values used** | success, inherited_success, failed, ir_export_failed, skipped | success, failed, ir_export_failed, skipped |

---

## Status Values and Propagation

| Status | Meaning | Set by |
|--------|---------|--------|
| `success` | Module passed op-by-op test directly | `analyze()` on test pass |
| `inherited_success` | Parent passed, no need to test | `_mark_subtree_success()` |
| `failed` | Op-by-op test failed (has `failed_ops` list) | `analyze()` on test fail |
| `ir_export_failed` | Could not export MLIR IR | `analyze()` on export fail |
| `skipped` | Container type, no TTIR files, or `--root-only` | Various |

**Post-order container update** (`_update_container_status`):
- If any child is `failed` or `ir_export_failed` --> container = `failed`
- If all children are `success`/`inherited_success` --> container = `inherited_success`
- If some children succeed --> container = `inherited_success`
- Otherwise --> `skipped`

---

## Output Directory Structure

```
<output_dir>/                        (default: ./<ModelClassName>/)
  |
  +-- unique_modules.json            Step 1 output, updated in Step 2 with status
  |
  +-- module_irs/
  |     +-- mod_000/
  |     |     +-- irs/
  |     |     |     +-- ttir_0_mod_000_ClassName.mlir
  |     |     |     +-- ttir_1_mod_000_ClassName.mlir
  |     |     +-- mod_000_op_by_op_report.json   (pytest-json-report)
  |     +-- mod_001/
  |     |     +-- ...
  |     +-- ...
  |
  +-- analysis_report.html           Step 3 output (self-contained HTML)
```

---

## Full Data Flow Diagram

```
User invocation:
  ttchop --model-path m.py::load --inputs-path m.py::inputs [--root-only]
                                    |
                              cli.py:main()
                                    |
                 +------------------+------------------+
                 |                                     |
        load_function_from_path()             get_tt_xla_root()
        (dynamic import of user fns)          (find tt-xla install)
                 |                                     |
                 v                                     v
  ======= STEP 1: EXTRACT ========          project_root (Path)
  extract_unique_modules()
       |
       +-- load_fn() --> model
       +-- inputs_fn() --> sample_input
       +-- [subprocess] shape_capture_subprocess.py
       |     Runs forward pass on TT device
       |     Captures per-module input/output shapes
       +-- Group by uniqueness key
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
       +-- analyze(root)  -- recursive --
       |     |
       |     +-- [subprocess] ir_export_single_module.py
       |     |     Loads model, finds submodule
       |     |     torch.compile(sub, backend="tt")
       |     |     Forward pass --> exports ttir_*.mlir
       |     |
       |     +-- [subprocess] pytest op_by_op_test.py
       |     |     Reads ttir_*.mlir files
       |     |     Tests each MLIR op on hardware
       |     |     Writes json report
       |     |
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
       +-- Build nested tree structure
       +-- Generate self-contained HTML
       +-- Output: analysis_report.html
```

---

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `ttchop/cli.py` | 94 | Entry point, orchestrates 3-step pipeline |
| `ttchop/module_extractor.py` | 138 | Step 1: Extract unique modules with shape capture |
| `ttchop/op_by_op_runner.py` | 250 | Step 2: Hierarchical op-by-op with lazy IR export |
| `ttchop/module_tree.py` | 81 | ModuleNode dataclass, tree builder, status updater |
| `ttchop/ir_export_single_module.py` | 58 | Subprocess: export IR for one module |
| `ttchop/module_runner.py` | 86 | torch.compile + forward pass for IR generation |
| `ttchop/shapes.py` | - | ShapeCapture class (CPU fallback) |
| `ttchop/shape_capture_subprocess.py` | - | Subprocess: shape capture on TT device |
| `ttchop/visualizer.py` | - | Step 3: HTML report generation |
| `ttchop/utils.py` | - | Shared utilities (function loading, device setup) |
| `ttchop/data_types.py` | - | ModuleInfo dataclass |
