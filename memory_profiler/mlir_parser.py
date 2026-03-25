# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR operation parser for extracting operation details from runtime logs.
"""

import re
import sys
from typing import Dict, List, Optional, Tuple


# Dtype sizes in bytes
DTYPE_SIZES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "si8": 1,
    "si16": 2,
    "si32": 4,
    "si64": 8,
    "ui8": 1,
    "ui16": 2,
    "ui32": 4,
    "ui64": 8,
    "bool": 1,
}


def parse_tensor_type(type_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract shape and dtype from a tensor type string.

    Handles formats like:
        - 'tensor<1x2x3xf32>'
        - 'tensor<768xbf16, #ttnn.ttnn_layout<...>>'
        - '1x2x3xf32'

    Args:
        type_str: Tensor type string

    Returns:
        Tuple of (shape, dtype) e.g., ('1x2x3', 'f32')
    """
    if not type_str:
        return None, None

    # Extract the shape and dtype part from tensor<shape_dtype, ...> or tensor<shape_dtype>
    # The shape/dtype is always at the beginning, before any comma or '>'
    tensor_match = re.search(r"tensor<([\dx]+[a-zA-Z]\w*)", type_str)
    if tensor_match:
        inner = tensor_match.group(1)
    else:
        inner = type_str.strip()

    # Match dimensions and dtype: e.g., "1x2x3xf32" or "1x2x3xbf16" or "768xbf16"
    # Dimensions are numbers separated by 'x', dtype is letters/numbers at the end
    match = re.match(r"^((?:\d+x)*\d+)x([a-zA-Z]\w*)$", inner)
    if match:
        return match.group(1), match.group(2)

    # Handle scalar types like just "f32"
    if re.match(r"^[a-zA-Z]\w*$", inner):
        return None, inner

    return None, None


def parse_tensor_layout_info(type_str: str) -> Optional[Dict]:
    """
    Extract layout information from a tensor type string.

    Example input:
    tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>>

    Returns:
        {
            "logical_shape": [64, 128],
            "dtype": "f32",
            "is_tiled": True,
            "memref_shape": [2, 4],  # Number of tiles
            "tile_size": [32, 32],
            "padded_shape": [64, 128],  # memref * tile_size
            "buffer_type": "dram",
            "unpadded_bytes": 32768,  # 64*128*4
            "padded_bytes": 32768,    # 64*128*4 (no overhead here)
            "overhead_pct": 0.0
        }
        Returns None if parsing fails.
    """
    if not type_str:
        return None

    result = {}

    # Extract logical shape and dtype from tensor<NxMx...dtype, ...>
    shape_str, dtype = parse_tensor_type(type_str)
    if not shape_str or not dtype:
        return None

    # Parse logical shape dimensions
    try:
        logical_shape = [int(d) for d in shape_str.split("x")]
    except ValueError:
        return None

    result["logical_shape"] = logical_shape
    result["dtype"] = dtype

    # Get dtype size
    dtype_size = DTYPE_SIZES.get(dtype, 4)

    # Calculate unpadded bytes
    unpadded_elements = 1
    for d in logical_shape:
        unpadded_elements *= d
    result["unpadded_bytes"] = unpadded_elements * dtype_size

    # Check if this is a tiled layout by looking for memref and tile patterns
    # Pattern: memref<AxBx!ttcore.tile<H,W, dtype>...> or memref<AxBx!tt.tile<H,W, dtype>...>
    memref_match = re.search(
        r"memref<([\dx]+)x!(?:ttcore|tt)\.tile<(\d+)x(\d+),\s*(\w+)>", type_str
    )

    if memref_match:
        result["is_tiled"] = True

        # Parse memref shape (tile counts)
        memref_shape_str = memref_match.group(1)
        tile_h = int(memref_match.group(2))
        tile_w = int(memref_match.group(3))
        memref_dtype = memref_match.group(4)

        try:
            memref_shape = [int(d) for d in memref_shape_str.split("x")]
        except ValueError:
            memref_shape = []

        result["memref_shape"] = memref_shape
        result["tile_size"] = [tile_h, tile_w]

        # Calculate padded shape: for each tile dimension, multiply by tile size
        # The physical shape is memref_shape * tile_size for the last two dims
        # For 1D tensors: memref is 1xN tiles of 32x32, so padded is 32 x (N*32)
        if len(memref_shape) >= 2:
            # Last two memref dims correspond to tile grid
            padded_shape = memref_shape[:-2] + [
                memref_shape[-2] * tile_h,
                memref_shape[-1] * tile_w,
            ]
        elif len(memref_shape) == 1:
            # Single dim - interpret as Nx1 tile grid
            padded_shape = [tile_h, memref_shape[0] * tile_w]
        else:
            padded_shape = [tile_h, tile_w]

        result["padded_shape"] = padded_shape

        # Calculate padded bytes
        padded_elements = 1
        for d in padded_shape:
            padded_elements *= d
        result["padded_bytes"] = padded_elements * dtype_size
    else:
        # Not a tiled layout (row-major or scalar)
        result["is_tiled"] = False
        result["memref_shape"] = None
        result["tile_size"] = None
        result["padded_shape"] = logical_shape.copy()
        result["padded_bytes"] = result["unpadded_bytes"]

    # Extract buffer type from #ttnn.buffer_type<...>
    buffer_match = re.search(r"#ttnn\.buffer_type<(\w+)>", type_str)
    if buffer_match:
        result["buffer_type"] = buffer_match.group(1).lower()
    else:
        result["buffer_type"] = None

    # Calculate overhead percentage
    if result["padded_bytes"] > 0:
        overhead_bytes = result["padded_bytes"] - result["unpadded_bytes"]
        result["overhead_pct"] = (overhead_bytes / result["unpadded_bytes"]) * 100
    else:
        result["overhead_pct"] = 0.0

    return result


def parse_type_string(type_str: str) -> List[Dict]:
    """
    Parse a type string that may contain multiple tensor types.

    Handles complex TTNN tensor formats like:
        tensor<768xbf16, #ttnn.ttnn_layout<...>>

    Args:
        type_str: String containing one or more tensor types

    Returns:
        List of dicts with 'shape' and 'dtype' keys
    """
    if not type_str:
        return []

    results = []
    # Find all tensor< patterns and extract shape/dtype
    # Pattern matches 'tensor<' followed by dimensions and dtype
    for match in re.finditer(r"tensor<([\dx]+[a-zA-Z]\w*)", type_str):
        shape, dtype = parse_tensor_type(f"tensor<{match.group(1)}>")
        results.append({"shape": shape, "dtype": dtype})

    return results


def find_top_level_arrow(s: str) -> int:
    """
    Find the position of ' -> ' that is not inside angle brackets or parentheses.

    This handles cases like:
        (tensor<..., #ttnn.ttnn_layout<(d0) -> (0, d0), ...>>) -> tensor<...>

    Args:
        s: String to search in

    Returns:
        Position of the top-level ' -> ', or -1 if not found
    """
    depth_angle = 0
    depth_paren = 0
    i = 0
    while i < len(s) - 3:
        c = s[i]
        if c == "<":
            depth_angle += 1
        elif c == ">":
            # Don't count '>' that's part of '->'
            if i > 0 and s[i - 1] == "-":
                pass  # Skip, this is part of '->'
            else:
                depth_angle -= 1
        elif c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1

        if depth_angle == 0 and depth_paren == 0 and s[i : i + 4] == " -> ":
            return i
        i += 1
    return -1


def parse_mlir_operation(line: str) -> Optional[Dict]:
    """
    Extract MLIR operation details from a log line.

    Expected formats:
        %4 = "ttnn.typecast"(%3) <{dtype = ...}> : (...) -> ... loc("convert.80")
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<...>) -> () loc("convert.80")

    Args:
        line: Log line containing MLIR operation

    Returns:
        Dictionary with operation details, or None if parsing fails
    """
    # Extract operation with result variable
    # Handles both quoted ("ttnn.typecast") and unquoted (ttcore.load_cached) operations
    # Pattern: %N = "op.name"(...) or %N = op.name(...)
    match = re.search(r'(%\d+)\s*=\s*"?([\w.]+)"?\(([^)]*)\)', line)

    if not match:
        # Handle operations without result (like deallocate)
        # Pattern: "op.name"(...) or op.name(...)
        match = re.search(r'"?([\w.]+)"?\(([^)]*)\)', line)
        if not match:
            return None
        result = None
        mlir_op = match.group(1)
        inputs = match.group(2)
    else:
        result = match.group(1)
        mlir_op = match.group(2)
        inputs = match.group(3)

    # Extract attributes (e.g., <{dtype = #ttcore.supportedDataTypes<f32>}>)
    attrs_match = re.search(r"<\{([^}]+)\}>", line)
    attributes = attrs_match.group(1) if attrs_match else None

    # Extract location (e.g., loc("convert.80") or loc("reduce.864_mean"("reduce.864")))
    loc_match = re.search(r'loc\("([^"]+)"', line)
    location = loc_match.group(1) if loc_match else None

    # Fallback for load_cached ops with loc(unknown): use @function_name as synthetic location
    if location is None and "load_cached" in line:
        func_match = re.search(r'load_cached\((@[\w.]+)', line)
        if func_match:
            location = func_match.group(1)  # e.g., "@main_const_eval_0"

    # Extract type signatures
    # Format: : (input_types) -> output_type loc(...)
    # Need to handle nested parentheses in ttnn layouts like:
    #   tensor<768xbf16, #ttnn.ttnn_layout<(d0) -> (0, d0), ...>>
    # Strategy: Find the type signature between "}> :" and " loc("
    input_types_str = None
    output_type_str = None

    # Try pattern with attributes first: }> : ... loc(
    type_sig_match = re.search(r"\}>\s*:\s*(.+)\s+loc\(", line)
    if not type_sig_match:
        # Try pattern without attributes: ) : ... loc(  or > : ... loc(
        type_sig_match = re.search(r"[)>]\s*:\s*(.+)\s+loc\(", line)

    if type_sig_match:
        type_sig = type_sig_match.group(1)
        # Split on " -> " to separate input and output types
        # But need to be careful of " -> " inside layouts
        # Find the last " -> " that's not inside angle brackets
        arrow_pos = find_top_level_arrow(type_sig)
        if arrow_pos != -1:
            input_types_str = type_sig[:arrow_pos].strip()
            output_type_str = type_sig[arrow_pos + 4 :].strip()
            # Remove outer parentheses from input types if present
            if input_types_str.startswith("(") and input_types_str.endswith(")"):
                input_types_str = input_types_str[1:-1]

    # Parse input operands into a list
    input_list = []
    if inputs.strip():
        # Split by commas, but be careful of nested parentheses in type signatures
        input_list = [i.strip() for i in inputs.split(",") if i.strip()]

    # Parse input and output types to extract shapes and dtypes
    input_tensors = parse_type_string(input_types_str)
    output_tensors = parse_type_string(output_type_str)

    # Parse output layout info for unpadded memory analysis
    output_layout_info = parse_tensor_layout_info(output_type_str)

    return {
        "result": result,
        "mlir_op": mlir_op,
        "inputs": input_list,
        "attributes": attributes,
        "input_types": input_types_str,
        "output_type": output_type_str,
        "input_shapes": [t["shape"] for t in input_tensors],
        "input_dtypes": [t["dtype"] for t in input_tensors],
        "output_shapes": [t["shape"] for t in output_tensors],
        "output_dtypes": [t["dtype"] for t in output_tensors],
        "output_layout_info": output_layout_info,
        "loc": location,
    }


def extract_ir_op_sequence(ttnn_text: str) -> List[Dict]:
    """
    Parse TTNN IR module text to extract ordered operations with layout info.

    Resolves layout/buffer aliases, then walks each operation line to produce
    a sequence usable for tile padding backfill on secondary-path logs.

    Returns list of dicts for ALL ops (including deallocate/get_device):
        {
            "ssa": "%1" or None,
            "op": "mesh_shard",
            "inputs": ["%arg0", "%0"],
            "output_layout_info": {...} or None,
            "is_deallocate": bool,
            "is_get_device": bool,
            "deallocated_ssa": "%arg0" or None,
        }
    """
    if not ttnn_text:
        return []

    lines = ttnn_text.split("\n")

    # --- Pass 1: collect aliases ---
    layout_aliases = {}   # "#ttnn_layout1" -> "#ttnn.ttnn_layout<...>"
    buffer_aliases = {}   # "#dram" -> "#ttnn.buffer_type<dram>"

    layout_re = re.compile(r'^(#ttnn_layout\w*)\s*=\s*(#ttnn\.ttnn_layout<.+>)\s*$')
    buffer_re = re.compile(r'^(#\w+)\s*=\s*(#ttnn\.buffer_type<\w+>)\s*$')

    for line in lines:
        stripped = line.strip()
        m = layout_re.match(stripped)
        if m:
            layout_aliases[m.group(1)] = m.group(2)
            continue
        m = buffer_re.match(stripped)
        if m:
            buffer_aliases[m.group(1)] = m.group(2)

    def resolve_aliases(type_str: str) -> str:
        """Replace layout and buffer aliases with their full definitions."""
        s = type_str
        # Resolve layout aliases (longest first to avoid partial matches)
        for alias in sorted(layout_aliases, key=len, reverse=True):
            if alias in s:
                expanded = layout_aliases[alias]
                # Also resolve buffer aliases inside the expanded layout
                for ba, be in buffer_aliases.items():
                    expanded = expanded.replace(ba, be)
                s = s.replace(alias, expanded)
        # Resolve any remaining buffer aliases in the string
        for alias, expansion in buffer_aliases.items():
            s = s.replace(alias, expansion)
        return s

    # --- Pass 2: walk operations ---
    # Op with SSA result: %N = "ttnn.xxx"(inputs) ...
    op_re = re.compile(r'(%\d+)\s*=\s*"ttnn\.(\w+)"\(([^)]*)\)')
    # Op without result (deallocate): "ttnn.deallocate"(%x) ...
    dealloc_re = re.compile(r'"ttnn\.deallocate"\((%[\w]+)\)')

    result = []
    for line in lines:
        stripped = line.strip()

        # Check deallocate first (no SSA result)
        dm = dealloc_re.search(stripped)
        if dm:
            result.append({
                "ssa": None,
                "op": "deallocate",
                "inputs": [dm.group(1)],
                "output_layout_info": None,
                "is_deallocate": True,
                "is_get_device": False,
                "deallocated_ssa": dm.group(1),
            })
            continue

        # Check ops with SSA result
        om = op_re.search(stripped)
        if not om:
            continue

        ssa = om.group(1)
        op_name = om.group(2)
        inputs_str = om.group(3)
        input_list = [i.strip() for i in inputs_str.split(",") if i.strip()]

        if op_name == "get_device":
            result.append({
                "ssa": ssa,
                "op": "get_device",
                "inputs": input_list,
                "output_layout_info": None,
                "is_deallocate": False,
                "is_get_device": True,
                "deallocated_ssa": None,
            })
            continue

        # Extract output type: find type signature, split at top-level arrow
        output_layout = None
        type_sig_match = re.search(r'[>)]\s*:\s*(.+)\s+loc\(', stripped)
        if type_sig_match:
            type_sig = type_sig_match.group(1)
            arrow_pos = find_top_level_arrow(type_sig)
            if arrow_pos != -1:
                output_type_str = type_sig[arrow_pos + 4:].strip()
                resolved = resolve_aliases(output_type_str)
                output_layout = parse_tensor_layout_info(resolved)

        result.append({
            "ssa": ssa,
            "op": op_name,
            "inputs": input_list,
            "output_layout_info": output_layout,
            "is_deallocate": False,
            "is_get_device": False,
            "deallocated_ssa": None,
        })

    return result
