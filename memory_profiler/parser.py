# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Main log parser for extracting operation and memory statistics from runtime logs.
"""

import json
import re
import sys
from typing import Dict, List, Optional, Tuple

# Regex for multi-host rank prefix (e.g., "[1,0]<stdout>: ")
_RANK_PREFIX_RE = re.compile(r"^\[(\d+,\d+)\]<(?:stdout|stderr)>:\s*(.*)", re.DOTALL)


def strip_rank_prefix(line: str) -> Tuple[str, Optional[str]]:
    """Strip multi-host rank prefix like '[1,0]<stdout>:' from a line.
    Returns (cleaned_line, rank_id) or (line, None) if no prefix."""
    m = _RANK_PREFIX_RE.match(line)
    if m:
        return m.group(2), m.group(1)
    return line, None

# Handle both package import and direct execution
try:
    from .inputs_registry_parser import parse_inputs_registry, parse_all_inputs_registries
    from .ir_parser import parse_ir_modules, parse_all_ir_modules
    from .memory_parser import parse_memory_stats
    from .mlir_parser import parse_mlir_operation, extract_ir_op_sequence
    from .mem_logger import log_mem
    from .streaming_reader import BufferedLineReader, strip_log_prefix
except ImportError:
    from inputs_registry_parser import parse_inputs_registry, parse_all_inputs_registries
    from ir_parser import parse_ir_modules, parse_all_ir_modules
    from memory_parser import parse_memory_stats
    from mlir_parser import parse_mlir_operation, extract_ir_op_sequence
    from mem_logger import log_mem
    from streaming_reader import BufferedLineReader, strip_log_prefix


def calculate_unpadded_memory_state(live_tensors: Dict[str, Dict]) -> Dict:
    """
    Calculate total unpadded vs padded memory from all live tensors.

    Args:
        live_tensors: Dict mapping SSA names to layout_info dicts

    Returns:
        Dict with DRAM and L1 memory states including unpadded/padded bytes,
        MB values, overhead, and tensor counts.
    """
    result = {
        "DRAM": {
            "unpadded_bytes": 0,
            "padded_bytes": 0,
            "num_tensors": 0,
        },
        "L1": {
            "unpadded_bytes": 0,
            "padded_bytes": 0,
            "num_tensors": 0,
        },
    }

    for ssa, layout in live_tensors.items():
        buf_type = layout.get("buffer_type", "")
        if buf_type:
            buf_type_upper = buf_type.upper()
            if buf_type_upper in result:
                result[buf_type_upper]["unpadded_bytes"] += layout.get(
                    "unpadded_bytes", 0
                )
                result[buf_type_upper]["padded_bytes"] += layout.get("padded_bytes", 0)
                result[buf_type_upper]["num_tensors"] += 1

    # Convert to MB and calculate overhead
    for mem_type in result:
        r = result[mem_type]
        r["unpadded_MB"] = r["unpadded_bytes"] / (1024 * 1024)
        r["padded_MB"] = r["padded_bytes"] / (1024 * 1024)
        r["overhead_MB"] = r["padded_MB"] - r["unpadded_MB"]
        r["overhead_pct"] = (
            (r["overhead_MB"] / r["unpadded_MB"] * 100) if r["unpadded_MB"] > 0 else 0
        )

    return result


def _group_ir_modules_by_program(
    all_ir: List[Dict],
    programs: List[Dict],
) -> List[Dict]:
    """
    Group IR module pairs into programs using greedy assignment, then merge
    text and loc_index for modules assigned to the same program.

    Uses the same greedy logic as _backfill_tile_padding_from_ir: consume
    modules until the IR op count meets or exceeds the program's runtime
    op count.
    """
    if not all_ir:
        return [
            {
                "name": p["name"],
                "ttir": {"text": "", "loc_index": {}},
                "ttnn": {"text": "", "loc_index": {}},
            }
            for p in programs
        ]

    # Count emitted IR ops per module (non-get_device, non-deallocate)
    module_op_counts = []
    for ir_pair in all_ir:
        ttnn_text = ir_pair.get("ttnn", {}).get("text", "")
        if ttnn_text:
            ir_ops = extract_ir_op_sequence(ttnn_text)
            count = sum(
                1
                for op in ir_ops
                if not op["is_get_device"] and not op["is_deallocate"]
            )
        else:
            count = 0
        module_op_counts.append(count)

    # Greedily assign modules to programs
    program_modules: List[List[int]] = []
    mod_idx = 0
    for prog in programs:
        prog_n_ops = prog["end_op_index"] - prog["start_op_index"] + 1
        assigned = []
        ir_count = 0
        while mod_idx < len(all_ir):
            assigned.append(mod_idx)
            ir_count += module_op_counts[mod_idx]
            mod_idx += 1
            if ir_count >= prog_n_ops:
                break
        program_modules.append(assigned)

    # Merge modules per program
    result = []
    for prog_idx, prog in enumerate(programs):
        mod_indices = program_modules[prog_idx] if prog_idx < len(program_modules) else []

        merged = {"name": prog["name"]}
        for ir_type in ("ttir", "ttnn"):
            texts = []
            merged_loc = {}
            cumulative_lines = 0

            for mi in mod_indices:
                ir_data = all_ir[mi].get(ir_type, {})
                text = ir_data.get("text", "")
                loc_index = ir_data.get("loc_index", {})

                if text:
                    texts.append(text)
                    # Offset line numbers by cumulative line count
                    for loc_name, line_num in loc_index.items():
                        if loc_name not in merged_loc:
                            merged_loc[loc_name] = line_num + cumulative_lines
                    cumulative_lines += len(text.split("\n"))

            merged[ir_type] = {
                "text": "\n\n".join(texts),
                "loc_index": merged_loc,
            }

        result.append(merged)

    return result


def _backfill_tile_padding_from_ir(
    operations: List[Dict],
    memory_stats: List[Dict],
    log_path: str,
    programs: List[Dict],
) -> None:
    """
    Backfill output_layout_info and unpadded_memory on secondary-path ops
    by extracting layout data from TTNN IR modules in the same log file.

    TTNN IR ops are high-level and compile to many device operations, so the
    count of IR ops is typically much smaller than runtime ops. We handle this
    by greedily assigning IR modules to programs, then mapping IR snapshots to
    runtime ops proportionally within each program.
    """
    all_ir = parse_all_ir_modules(log_path)
    if not all_ir:
        return

    # Step 1: Build per-module IR snapshots (layout + live tensor state)
    module_snapshots: List[List[Dict]] = []
    for ir_pair in all_ir:
        ttnn_text = ir_pair.get("ttnn", {}).get("text", "")
        if not ttnn_text:
            continue

        ir_ops = extract_ir_op_sequence(ttnn_text)
        live_tensors: Dict[str, Dict] = {}
        snapshots: List[Dict] = []

        for ir_op in ir_ops:
            if ir_op["is_get_device"]:
                continue
            if ir_op["is_deallocate"]:
                live_tensors.pop(ir_op["deallocated_ssa"], None)
                continue

            layout = ir_op["output_layout_info"]
            if layout and ir_op["ssa"]:
                if layout.get("buffer_type") in ("dram", "l1"):
                    live_tensors[ir_op["ssa"]] = layout

            snapshots.append({
                "output_layout_info": layout,
                "unpadded_memory": calculate_unpadded_memory_state(
                    dict(live_tensors)
                ),
                "loc": ir_op["loc"],
            })
        module_snapshots.append(snapshots)

    if not module_snapshots:
        return

    # Step 2: Greedily assign modules to programs by IR op count
    program_snapshots: List[List[Dict]] = []
    mod_idx = 0
    for prog in programs:
        prog_n_ops = prog["end_op_index"] - prog["start_op_index"] + 1
        prog_snaps: List[Dict] = []
        while mod_idx < len(module_snapshots):
            prog_snaps.extend(module_snapshots[mod_idx])
            mod_idx += 1
            if len(prog_snaps) >= prog_n_ops:
                break
        program_snapshots.append(prog_snaps)

    # Step 3: Map IR snapshots to runtime ops proportionally per program
    for prog_idx, prog in enumerate(programs):
        if prog_idx >= len(program_snapshots):
            break
        snaps = program_snapshots[prog_idx]
        if not snaps:
            continue

        start = prog["start_op_index"]
        prog_n_ops = prog["end_op_index"] - start + 1
        n_snaps = len(snaps)

        for i in range(prog_n_ops):
            snap_idx = min(i * n_snaps // prog_n_ops, n_snaps - 1)
            rt_idx = start + i
            if rt_idx < len(operations):
                operations[rt_idx]["output_layout_info"] = snaps[snap_idx][
                    "output_layout_info"
                ]
                memory_stats[rt_idx]["unpadded_memory"] = snaps[snap_idx][
                    "unpadded_memory"
                ]
                if snaps[snap_idx].get("loc"):
                    operations[rt_idx]["loc"] = snaps[snap_idx]["loc"]
                    memory_stats[rt_idx]["loc"] = snaps[snap_idx]["loc"]


def parse_log_file(
    log_path: str,
    mem_output: str,
    ops_output: str,
    registry_output: Optional[str] = None,
    ir_output: Optional[str] = None,
) -> None:
    """
    Parse log file and generate synchronized JSON outputs.

    Args:
        log_path: Path to the log file to parse
        mem_output: Path for memory statistics JSON output
        ops_output: Path for operation details JSON output
        registry_output: Optional path for inputs registry JSON output
        ir_output: Optional path for IR modules JSON output (TTIR/TTNN)
    """
    log_mem("parse_log_file: start")
    operations = []
    memory_stats = []

    try:
        reader = BufferedLineReader(log_path, buffer_size=10)
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return

    log_mem("parse_log_file: streaming reader opened")

    op_index = 0
    skipped_ops = 0
    deallocate_count = 0
    get_device_count = 0

    # Track const_eval graph stack (can be nested)
    const_eval_stack: List[str] = []
    const_eval_cache_misses: set = set()
    const_eval_ops_count: Dict[str, int] = {}

    # Track live tensors by SSA name for unpadded memory analysis
    # Key: SSA name (e.g., "%0"), Value: layout_info dict
    live_tensors: Dict[str, Dict] = {}

    # Multi-program tracking
    programs: List[Dict] = []
    current_program_name = "program_0"
    current_program_start_op = 0
    program_counter = 0

    # Multi-host rank filtering (latch to first rank seen)
    target_rank: Optional[str] = None

    # Log format detection: True once we see "Executing operation:" lines
    has_mlir_ops = False

    while reader.has_lines:
        raw_line = reader.current_line
        line_num = reader.total_consumed + 1

        # Strip multi-host rank prefix (e.g., "[1,0]<stdout>:")
        cleaned_line, rank_id = strip_rank_prefix(raw_line)

        # Rank filtering: latch to first rank seen, skip other ranks
        if rank_id:
            if target_rank is None:
                target_rank = rank_id
            elif rank_id != target_rank:
                reader.advance()
                continue

        # --- Program boundary detection ---
        # Track test names as they appear (before ops start in a program)
        if op_index == current_program_start_op:
            test_match = re.search(r"tests/\S+::(\w+)", cleaned_line)
            if test_match:
                current_program_name = test_match.group(1)

        # Pytest session start
        if "= test session starts =" in cleaned_line:
            # Close current program if we have accumulated operations
            if op_index > current_program_start_op:
                programs.append({
                    "name": current_program_name,
                    "start_op_index": current_program_start_op,
                    "end_op_index": op_index - 1,
                })
                program_counter += 1
                current_program_start_op = op_index
            # Reset name for this new program
            current_program_name = f"program_{program_counter}"
            # Reset per-program state
            const_eval_stack.clear()
            live_tensors.clear()

        # New PJRT runtime session (non-pytest program boundary)
        elif "ClientInstance::PJRT_Client_Create" in cleaned_line:
            # Only close if we have ops AND no pytest session just started on this boundary
            if op_index > current_program_start_op:
                programs.append({
                    "name": current_program_name,
                    "start_op_index": current_program_start_op,
                    "end_op_index": op_index - 1,
                })
                program_counter += 1
                current_program_name = f"program_{program_counter}"
                current_program_start_op = op_index
                # Reset per-program state
                const_eval_stack.clear()
                live_tensors.clear()

        # Check for const_eval cache miss
        cache_miss_match = re.search(
            r"Cache miss or invalid cache for function:\s*(\S+)", cleaned_line
        )
        if cache_miss_match:
            func_name = cache_miss_match.group(1)
            const_eval_cache_misses.add(func_name)

        # Check for starting execution of a const_eval program
        start_match = re.search(r"Starting execution of program:\s*(\S+)", cleaned_line)
        if start_match:
            program_name = start_match.group(1)
            if "const_eval" in program_name:
                const_eval_stack.append(program_name)
                if program_name not in const_eval_ops_count:
                    const_eval_ops_count[program_name] = 0

        # Check for finishing execution of a const_eval program
        finish_match = re.search(r"Finished execution of program:\s*(\S+)", cleaned_line)
        if finish_match:
            program_name = finish_match.group(1)
            if (
                "const_eval" in program_name
                and const_eval_stack
                and const_eval_stack[-1] == program_name
            ):
                const_eval_stack.pop()

        # --- Operation detection ---
        # Primary path: standard logs with MLIR operation details
        if "Executing operation:" in cleaned_line and "RuntimeTTNN" in cleaned_line:
            has_mlir_ops = True
            op_info = parse_mlir_operation(cleaned_line)

            if not op_info:
                print(
                    f"Warning: Could not parse operation at line {line_num}", file=sys.stderr
                )
                reader.advance()
                skipped_ops += 1
                continue

            # Handle deallocate operations - track deallocation for unpadded analysis
            if "deallocate" in op_info["mlir_op"]:
                deallocate_count += 1
                # Extract deallocated tensor SSA and remove from tracking
                if op_info.get("inputs"):
                    deallocated_ssa = op_info["inputs"][0]
                    if deallocated_ssa in live_tensors:
                        del live_tensors[deallocated_ssa]
                reader.advance()
                continue

            # Skip get_device operations (no tensor data)
            if "get_device" in op_info["mlir_op"]:
                get_device_count += 1
                reader.advance()
                continue

            # Look ahead for memory stats using the buffer (no full-file load)
            lookahead = reader.peek_slice(1, 6)
            mem_info = parse_memory_stats(lookahead, 0)

            if mem_info:
                # Add index and cross-reference fields to both outputs
                op_info["index"] = op_index
                op_info["program_index"] = len(programs)
                mem_info["index"] = op_index
                mem_info["mlir_op"] = op_info["mlir_op"]
                mem_info["loc"] = op_info["loc"]

                # Add const_eval info if we're inside a const_eval graph
                # Operations inside const_eval graphs are weight operations since
                # const_eval only processes parameters and constants
                if const_eval_stack:
                    current_const_eval = const_eval_stack[-1]
                    op_info["const_eval_graph"] = current_const_eval
                    mem_info["const_eval_graph"] = current_const_eval
                    # Track if this is a cache miss execution
                    op_info["const_eval_cache_miss"] = (
                        current_const_eval in const_eval_cache_misses
                    )
                    mem_info["const_eval_cache_miss"] = (
                        current_const_eval in const_eval_cache_misses
                    )
                    const_eval_ops_count[current_const_eval] += 1
                    # All const_eval operations are weight operations
                    op_info["is_weight_op"] = True
                    mem_info["is_weight_op"] = True
                else:
                    op_info["const_eval_graph"] = None
                    mem_info["const_eval_graph"] = None
                    op_info["const_eval_cache_miss"] = False
                    mem_info["const_eval_cache_miss"] = False
                    op_info["is_weight_op"] = False
                    mem_info["is_weight_op"] = False

                # Track new tensor allocation from operation output
                if op_info.get("result") and op_info.get("output_layout_info"):
                    layout = op_info["output_layout_info"]
                    if layout and layout.get("buffer_type") in ("dram", "l1"):
                        live_tensors[op_info["result"]] = layout

                # Calculate unpadded memory state at this operation
                unpadded_state = calculate_unpadded_memory_state(live_tensors)
                mem_info["unpadded_memory"] = unpadded_state

                operations.append(op_info)
                memory_stats.append(mem_info)
                op_index += 1
            else:
                print(
                    f"Warning: No memory stats found for operation '{op_info['mlir_op']}' at line {line_num} (loc: {op_info['loc']})",
                    file=sys.stderr,
                )
                skipped_ops += 1

        # Secondary path: memory-state-only logs (multihost) without MLIR op details
        elif not has_mlir_ops and "Device memory state before operation" in cleaned_line:
            # Build 5-line block: header + 4 memory type lines
            # Peek extra lines to handle noise/interleaving between memory lines
            raw_lookahead = reader.peek_slice(1, 8)
            cleaned_block = [cleaned_line]
            lines_consumed = 0
            for raw in raw_lookahead:
                stripped, r = strip_rank_prefix(raw)
                # Skip lines from other ranks
                if r and target_rank and r != target_rank:
                    lines_consumed += 1
                    continue
                # Only include memory state lines
                if "Device" in stripped and "memory state:" in stripped:
                    cleaned_block.append(stripped)
                    lines_consumed += 1
                    if len(cleaned_block) >= 5:
                        break
                else:
                    # Non-memory line = end of this block
                    break

            mem_info = parse_memory_stats(cleaned_block, 0)
            if mem_info:
                op_type = mem_info.get("op_type", "")

                # Skip GetDeviceOp and DeallocateOp in secondary path
                if op_type in ("GetDeviceOp", "DeallocateOp"):
                    if op_type == "GetDeviceOp":
                        get_device_count += 1
                    else:
                        deallocate_count += 1
                    for _ in range(lines_consumed):
                        reader.advance()
                    reader.advance()
                    continue

                # Skip ops with incomplete memory data (missing DRAM)
                if "DRAM" not in mem_info.get("memory", {}):
                    skipped_ops += 1
                    for _ in range(lines_consumed):
                        reader.advance()
                    reader.advance()
                    continue

                # Build minimal op_info (no MLIR details available)
                op_info = {
                    "result": None,
                    "mlir_op": mem_info.get("op_type", "unknown"),
                    "inputs": [],
                    "attributes": None,
                    "input_shapes": [],
                    "input_dtypes": [],
                    "output_shapes": [],
                    "output_dtypes": [],
                    "output_layout_info": None,
                    "loc": None,
                    "index": op_index,
                    "program_index": len(programs),
                    "const_eval_graph": None,
                    "const_eval_cache_miss": False,
                    "is_weight_op": False,
                    "weights": [],
                }
                if rank_id:
                    op_info["rank"] = rank_id

                mem_info["index"] = op_index
                mem_info["mlir_op"] = op_info["mlir_op"]
                mem_info["loc"] = None
                mem_info["const_eval_graph"] = None
                mem_info["const_eval_cache_miss"] = False
                mem_info["is_weight_op"] = False
                mem_info["unpadded_memory"] = calculate_unpadded_memory_state(live_tensors)

                operations.append(op_info)
                memory_stats.append(mem_info)
                op_index += 1

                # Advance past the consumed lookahead lines
                for _ in range(lines_consumed):
                    reader.advance()

        reader.advance()

    total_lines = reader.total_consumed
    reader.close()

    # Close the final program
    if op_index > current_program_start_op:
        programs.append({
            "name": current_program_name,
            "start_op_index": current_program_start_op,
            "end_op_index": op_index - 1,
        })
    # If no boundaries were detected, wrap everything in one program
    if not programs and op_index > 0:
        programs.append({
            "name": "main",
            "start_op_index": 0,
            "end_op_index": op_index - 1,
        })

    log_mem("parse_log_file: main loop done")

    # Backfill tile padding data from IR for secondary-path logs
    if not has_mlir_ops and operations:
        _backfill_tile_padding_from_ir(operations, memory_stats, log_path, programs)
        log_mem("parse_log_file: IR tile padding backfill done")

    # Parse inputs registry from MLIR module
    registry = None
    weight_lookup = {}
    if registry_output:
        registry = parse_inputs_registry(log_path)

        # Build lookup by shape: shape -> registry entry (only for weights)
        # This allows matching based on tensor shape rather than SSA names
        # (SSA names like %arg0, %arg1 can mean different things in different MLIR modules)
        weight_lookup_by_shape = {}
        for entry in registry.get("entries", []):
            if entry["type"] in ("parameter", "constant"):
                weight_lookup_by_shape[entry["shape"]] = entry

        # Track which weights are passed to each const_eval invocation
        # Key: (const_eval_name, op_index) -> list of weight info
        const_eval_weights: Dict[str, List[Dict]] = {}

        # First pass: identify load_cached operations and record which weights they pass
        # Match weights by input_shapes rather than SSA names
        for op_info in operations:
            if "load_cached" in op_info.get("mlir_op", ""):
                weights = []
                const_eval_name = None
                for inp in op_info.get("inputs", []):
                    # Extract const_eval function name (e.g., "@main_const_eval_1")
                    if inp.startswith("@"):
                        const_eval_name = inp[1:]  # Remove @ prefix

                # Match weights by input shape
                for shape in op_info.get("input_shapes", []):
                    if shape and shape in weight_lookup_by_shape:
                        w = weight_lookup_by_shape[shape]
                        weights.append(
                            {
                                "registry_index": w["index"],
                                "name": w["name"],
                                "shape": w["shape"],
                                "dtype": w["dtype"],
                                "bytes": w["bytes"],
                            }
                        )
                op_info["weights"] = weights

                # Record weights for the const_eval function that will be invoked next
                if const_eval_name and weights:
                    const_eval_weights[f"{const_eval_name}_{op_info['index']}"] = (
                        weights
                    )

        # Second pass: for each const_eval operation, find the load_cached that invoked it
        # and inherit those weights
        current_load_cached_idx = None
        for i, op_info in enumerate(operations):
            # Track when we see a load_cached (the next const_eval operations belong to it)
            if "load_cached" in op_info.get("mlir_op", ""):
                current_load_cached_idx = op_info["index"]
                continue

            # For const_eval operations, look up weights from the load_cached that invoked it
            if op_info.get("const_eval_graph"):
                const_eval_name = op_info["const_eval_graph"]
                key = f"{const_eval_name}_{current_load_cached_idx}"
                if key in const_eval_weights:
                    op_info["weights"] = const_eval_weights[key].copy()
                else:
                    op_info["weights"] = []
            elif "weights" not in op_info:
                # Non-const_eval operations: match weights by input shapes
                weights = []
                for shape in op_info.get("input_shapes", []):
                    if shape and shape in weight_lookup_by_shape:
                        w = weight_lookup_by_shape[shape]
                        # Avoid duplicates
                        if not any(
                            existing["registry_index"] == w["index"]
                            for existing in weights
                        ):
                            weights.append(
                                {
                                    "registry_index": w["index"],
                                    "name": w["name"],
                                    "shape": w["shape"],
                                    "dtype": w["dtype"],
                                    "bytes": w["bytes"],
                                }
                            )
                op_info["weights"] = weights

        # Third pass: propagate weights through const_eval chains
        # Track derived weights: SSA values that are outputs of weight operations
        # Key includes load_cached index to properly scope derived weights to each invocation
        derived_weights: Dict[str, List[Dict]] = {}

        changed = True
        while changed:
            changed = False
            current_load_cached_idx = None
            for op_info in operations:
                # Track load_cached index for scoping
                if "load_cached" in op_info.get("mlir_op", ""):
                    current_load_cached_idx = op_info["index"]
                    continue

                # Only process const_eval operations
                if not op_info.get("const_eval_graph"):
                    continue

                # Use load_cached index to scope derived weights
                scope_prefix = (
                    f"{op_info['const_eval_graph']}_{current_load_cached_idx}"
                )

                # Check if any input is a derived weight (within same const_eval invocation)
                for inp in op_info.get("inputs", []):
                    key = f"{scope_prefix}_{inp}"
                    if key in derived_weights:
                        for dw in derived_weights[key]:
                            if not any(
                                w.get("registry_index") == dw.get("registry_index")
                                for w in op_info["weights"]
                            ):
                                op_info["weights"].append(dw.copy())
                                changed = True

                # If this operation has weights, register its output as derived
                if op_info["weights"] and op_info.get("result"):
                    key = f"{scope_prefix}_{op_info['result']}"
                    if key not in derived_weights:
                        derived_weights[key] = op_info["weights"].copy()
                        changed = True

        # Mark operations with weights as weight operations
        for op_info in operations:
            if op_info.get("weights") and not op_info.get("is_weight_op"):
                op_info["is_weight_op"] = True

    log_mem("parse_log_file: registry + IR done")

    # Extract memory configuration from first operation
    memory_config = {}
    if memory_stats:
        first_mem = memory_stats[0]["memory"]
        for mem_type in ["DRAM", "L1", "L1_SMALL", "TRACE"]:
            if mem_type in first_mem:
                memory_config[mem_type] = {
                    "num_banks": first_mem[mem_type].get("numBanks", 0),
                    "total_bytes_per_bank_MB": first_mem[mem_type].get(
                        "totalBytesPerBank_MB", 0
                    ),
                }

    # Write outputs - memory JSON with metadata header
    try:
        mem_output_data = {
            "metadata": {
                "memory_config": memory_config,
                "total_operations": len(memory_stats),
                "programs": programs,
            },
            "operations": memory_stats,
        }
        with open(mem_output, "w", encoding="utf-8") as f:
            json.dump(mem_output_data, f, indent=2)
        print(f"Memory statistics written to: {mem_output}")
    except Exception as e:
        print(f"Error writing memory output: {e}", file=sys.stderr)

    try:
        with open(ops_output, "w", encoding="utf-8") as f:
            json.dump(operations, f, indent=2)
        print(f"Operation details written to: {ops_output}")
    except Exception as e:
        print(f"Error writing operations output: {e}", file=sys.stderr)

    # Write registry output if requested
    if registry_output and registry:
        try:
            with open(registry_output, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            print(f"Inputs registry written to: {registry_output}")
        except Exception as e:
            print(f"Error writing registry output: {e}", file=sys.stderr)

    # Parse and write IR modules if requested
    if ir_output:
        try:
            if len(programs) > 1:
                # Multi-program: group IR modules into programs using
                # greedy assignment (matches _backfill_tile_padding_from_ir)
                all_ir = parse_all_ir_modules(log_path)
                ir_data = {
                    "programs": _group_ir_modules_by_program(all_ir, programs)
                }
            else:
                ir_data = parse_ir_modules(log_path)
            with open(ir_output, "w", encoding="utf-8") as f:
                json.dump(ir_data, f, indent=2)
            print(f"IR modules written to: {ir_output}")
        except Exception as e:
            print(f"Error writing IR output: {e}", file=sys.stderr)

    # Count const_eval operations
    total_const_eval_ops = sum(const_eval_ops_count.values())

    # Print summary
    print(f"\nParsing Summary:")
    print(f"  Total operations parsed: {op_index}")
    print(f"  Deallocate operations excluded: {deallocate_count}")
    print(f"  Get_device operations excluded: {get_device_count}")
    print(f"  Operations skipped (no memory stats): {skipped_ops}")
    print(f"  Total log lines processed: {total_lines}")
    if target_rank:
        print(f"  Multi-host rank used: [{target_rank}]")
    if len(programs) > 1:
        print(f"\nProgram Boundaries ({len(programs)} programs detected):")
        for i, prog in enumerate(programs):
            n_ops = prog["end_op_index"] - prog["start_op_index"] + 1
            print(f"  [{i}] {prog['name']}: ops {prog['start_op_index']}-{prog['end_op_index']} ({n_ops} ops)")

    if const_eval_ops_count:
        print(f"\nConst Eval Summary:")
        print(f"  Total const_eval operations: {total_const_eval_ops}")
        print(f"  Const_eval graphs with cache miss: {len(const_eval_cache_misses)}")
        for graph_name, count in sorted(const_eval_ops_count.items()):
            cache_status = (
                "(cache miss)" if graph_name in const_eval_cache_misses else "(cached)"
            )
            print(f"    {graph_name}: {count} ops {cache_status}")

    if registry and registry.get("entries"):
        print(f"\nInputs Registry Summary:")
        print(f"  Total entries: {registry['metadata'].get('total_entries', 0)}")
        print(
            f"  Weights (parameter/constant): {registry['metadata'].get('total_weights', 0)}"
        )
        print(f"  Activations (input): {registry['metadata'].get('total_inputs', 0)}")
        print(
            f"  Total weight size: {registry['metadata'].get('total_weight_MB', 0):.2f} MB"
        )
        # Count operations with weights (direct reference)
        ops_with_weights = sum(1 for op in operations if op.get("weights"))
        print(f"  Operations with direct weight inputs: {ops_with_weights}")
        # Count all weight operations (const_eval + direct weight refs)
        weight_ops = sum(1 for op in operations if op.get("is_weight_op"))
        activation_ops = sum(1 for op in operations if not op.get("is_weight_op"))
        print(f"  Weight operations (const_eval + direct): {weight_ops}")
        print(f"  Activation operations: {activation_ops}")


def validate_outputs(mem_file: str, ops_file: str) -> bool:
    """
    Validate that the two output files are properly aligned.

    Args:
        mem_file: Path to memory statistics JSON
        ops_file: Path to operations JSON

    Returns:
        True if validation passes, False otherwise
    """
    try:
        with open(mem_file, "r") as f:
            mem_json = json.load(f)
        with open(ops_file, "r") as f:
            ops_data = json.load(f)

        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(mem_json, dict) and "operations" in mem_json:
            mem_data = mem_json["operations"]
        else:
            mem_data = mem_json

        if len(mem_data) != len(ops_data):
            print(
                f"Error: Mismatched lengths - memory: {len(mem_data)}, operations: {len(ops_data)}",
                file=sys.stderr,
            )
            return False

        missing_shapes = 0
        for i in range(len(mem_data)):
            if mem_data[i]["index"] != ops_data[i]["index"]:
                print(f"Error: Index mismatch at position {i}", file=sys.stderr)
                return False
            if mem_data[i]["loc"] != ops_data[i]["loc"]:
                print(
                    f"Error: Location mismatch at index {i}: {mem_data[i]['loc']} != {ops_data[i]['loc']}",
                    file=sys.stderr,
                )
                return False
            # Check for missing shapes
            op = ops_data[i]
            if not op.get("input_shapes") and not op.get("output_shapes"):
                missing_shapes += 1

        print(f"Validation passed: {len(mem_data)} operations properly aligned")
        if missing_shapes > 0:
            print(
                f"Warning: {missing_shapes} operations have no input/output shapes parsed"
            )

        return True

    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return False


def validate_log_content(log_path: str) -> Optional[str]:
    """
    Fast-scan a log file for required data markers.
    Returns None if valid, or a diagnostic message string if incomplete.
    Accepts both standard logs (Executing operation + memory states)
    and multihost logs (memory states only).
    """
    has_operations = False
    has_memory_states = False

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = strip_log_prefix(raw_line)
                if not has_operations and "Executing operation:" in line and "RuntimeTTNN" in line:
                    has_operations = True
                if not has_memory_states and "Device memory state before operation" in line:
                    has_memory_states = True
                # Valid if we have memory states (with or without MLIR op lines)
                if has_memory_states:
                    return None
    except Exception as e:
        return f"Error reading log file: {e}"

    return (
        "Hm, it seems I don't have all the information to make a report. One of the most likely reasons:\n"
        "1. You didn't build tt-xla in debug mode:\n"
        "    Solution: Use `cmake --preset debug && cmake --build build` to solve this.\n"
        "2. You don't have all of the required log levels enabled\n"
        "    Solution: Set the following environment variables:\n"
        "        export TTXLA_LOGGER_LEVEL=DEBUG\n"
        "        export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG\n"
        "        export TT_RUNTIME_MEMORY_LOG_LEVEL=operation\n"
        "Let's fix this and get the log running \U0001f919"
    )
