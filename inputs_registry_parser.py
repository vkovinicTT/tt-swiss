# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR module parser for extracting function argument registry from runtime logs.

Parses the MLIR module dumps to extract function arguments with their:
- ttcore.argument_type (parameter/constant/input)
- ttir.name (human-readable name)
- tensor shape and dtype
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
    "ui8": 1,
    "ui16": 2,
    "ui32": 4,
    "ui64": 8,
    "bool": 1,
}


def calculate_tensor_bytes(shape: str, dtype: str) -> int:
    """
    Calculate tensor size in bytes from shape string and dtype.

    Args:
        shape: Shape string like "768x768x3x3x3" or "768"
        dtype: Data type string like "bf16", "f32"

    Returns:
        Size in bytes
    """
    if not shape:
        return 0

    dtype_size = DTYPE_SIZES.get(dtype, 4)  # Default to 4 bytes if unknown

    try:
        dims = [int(d) for d in shape.split("x")]
        num_elements = 1
        for d in dims:
            num_elements *= d
        return num_elements * dtype_size
    except (ValueError, AttributeError):
        return 0


def parse_tensor_type(type_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract shape and dtype from a tensor type string.

    Args:
        type_str: String like "tensor<768x768x3x3x3xbf16>"

    Returns:
        Tuple of (shape, dtype) e.g., ("768x768x3x3x3", "bf16")
    """
    # Match tensor<shape_dtype> pattern
    match = re.search(r"tensor<([\dx]+)x([a-zA-Z]\w*)(?:[,>])", type_str)
    if match:
        return match.group(1), match.group(2)

    # Try simpler pattern for scalar-like tensors
    match = re.search(r"tensor<(\d+)x([a-zA-Z]\w*)(?:[,>])", type_str)
    if match:
        return match.group(1), match.group(2)

    return None, None


def parse_argument(arg_str: str, index: int) -> Optional[Dict]:
    """
    Parse a single function argument definition.

    Args:
        arg_str: String like "%arg0: tensor<768xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = \"l__self___conv1_conv_bias\"}"
        index: Index to assign to this argument

    Returns:
        Dictionary with argument metadata or None if parsing fails
    """
    # Extract SSA name (%argN)
    ssa_match = re.search(r"(%arg\d+)", arg_str)
    if not ssa_match:
        return None
    ssa = ssa_match.group(1)

    # Extract tensor type
    tensor_match = re.search(r"tensor<([^>]+)>", arg_str)
    if not tensor_match:
        return None

    tensor_content = tensor_match.group(1)

    # Parse shape and dtype from tensor content
    # Handle formats like "768xbf16" or "768x768x3x3x3xbf16"
    shape_dtype_match = re.match(r"^((?:\d+x)*\d+)x([a-zA-Z]\w*)", tensor_content)
    if shape_dtype_match:
        shape = shape_dtype_match.group(1)
        dtype = shape_dtype_match.group(2)
    else:
        # Scalar case
        dtype_match = re.match(r"^([a-zA-Z]\w*)", tensor_content)
        if dtype_match:
            shape = None
            dtype = dtype_match.group(1)
        else:
            return None

    # Extract argument type (parameter/constant/input)
    arg_type_match = re.search(
        r"ttcore\.argument_type\s*=\s*#ttcore\.argument_type<(\w+)>", arg_str
    )
    arg_type = arg_type_match.group(1) if arg_type_match else "unknown"

    # Extract name from ttir.name
    name_match = re.search(r'ttir\.name\s*=\s*"([^"]+)"', arg_str)
    name = name_match.group(1) if name_match else ssa

    # Calculate bytes
    bytes_size = (
        calculate_tensor_bytes(shape, dtype) if shape else DTYPE_SIZES.get(dtype, 4)
    )

    return {
        "index": index,
        "ssa": ssa,
        "name": name,
        "type": arg_type,
        "shape": shape,
        "dtype": dtype,
        "bytes": bytes_size,
        "bytes_MB": bytes_size / (1024 * 1024),
    }


def find_mlir_module_section(
    lines: List[str], section_name: str = "shlo_frontend"
) -> Tuple[int, int]:
    """
    Find the start and end indices of an MLIR module section in log lines.

    Args:
        lines: List of log lines
        section_name: Name of the section to find (e.g., 'shlo_frontend', 'shlo_compiler')

    Returns:
        Tuple of (start_index, end_index) or (-1, -1) if not found
    """
    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if f"MLIR Module {section_name}:" in line:
            start_idx = i
        elif start_idx >= 0 and "END OF MLIR MODULE" in line:
            end_idx = i
            break

    return start_idx, end_idx


def parse_func_signature(lines: List[str], start_idx: int, end_idx: int) -> str:
    """
    Extract the func.func @main(...) signature from MLIR module lines.

    The signature may span multiple lines, so we need to collect until we find
    the closing parenthesis followed by return type.

    Args:
        lines: List of log lines
        start_idx: Start of MLIR module section
        end_idx: End of MLIR module section

    Returns:
        Complete function signature string
    """
    signature_parts = []
    in_signature = False
    paren_depth = 0

    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i]

        if "func.func @main(" in line:
            in_signature = True

        if in_signature:
            signature_parts.append(line)
            paren_depth += line.count("(") - line.count(")")

            # Check if we've closed the main function signature
            # Look for pattern like ") -> tensor<" or ") {" after balanced parens
            if paren_depth <= 0 and (") ->" in line or ") {" in line):
                break

    return " ".join(signature_parts)


def parse_inputs_registry(log_path: str) -> Dict:
    """
    Parse MLIR module from log to extract function argument registry.

    Searches for 'func.func @main(' in the shlo_frontend section and parses
    argument definitions with:
    - ttcore.argument_type (parameter/constant/input)
    - ttir.name (human-readable name)
    - tensor shape and dtype

    Args:
        log_path: Path to the log file

    Returns:
        Dictionary with 'metadata' and 'entries' keys
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return {"metadata": {}, "entries": []}
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return {"metadata": {}, "entries": []}

    # Try to find shlo_frontend section first (has ttcore.argument_type markers)
    start_idx, end_idx = find_mlir_module_section(lines, "shlo_frontend")

    if start_idx < 0:
        # Fallback to shlo_compiler section
        start_idx, end_idx = find_mlir_module_section(lines, "shlo_compiler")

    if start_idx < 0:
        # Try any MLIR module section
        start_idx, end_idx = find_mlir_module_section(lines, "shlo")

    if start_idx < 0:
        print("Warning: Could not find MLIR module section in log", file=sys.stderr)
        return {"metadata": {}, "entries": []}

    # Extract function signature
    signature = parse_func_signature(lines, start_idx, end_idx)

    if not signature:
        print("Warning: Could not find func.func @main signature", file=sys.stderr)
        return {"metadata": {}, "entries": []}

    # Parse individual arguments
    # Find the arguments section between @main( and the return type
    args_match = re.search(r"@main\((.+?)\)\s*(?:->|{)", signature, re.DOTALL)
    if not args_match:
        print("Warning: Could not parse function arguments", file=sys.stderr)
        return {"metadata": {}, "entries": []}

    args_str = args_match.group(1)

    # Split arguments - they're separated by commas, but we need to handle
    # nested braces in attributes
    entries = []
    current_arg = []
    brace_depth = 0

    for char in args_str:
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "," and brace_depth == 0:
            # End of argument
            arg_str = "".join(current_arg).strip()
            if arg_str:
                entry = parse_argument(arg_str, len(entries))
                if entry:
                    entries.append(entry)
            current_arg = []
            continue
        current_arg.append(char)

    # Don't forget the last argument
    arg_str = "".join(current_arg).strip()
    if arg_str:
        entry = parse_argument(arg_str, len(entries))
        if entry:
            entries.append(entry)

    # Calculate metadata
    total_weights = sum(1 for e in entries if e["type"] in ("parameter", "constant"))
    total_inputs = sum(1 for e in entries if e["type"] == "input")
    total_weight_bytes = sum(
        e["bytes"] for e in entries if e["type"] in ("parameter", "constant")
    )

    metadata = {
        "total_entries": len(entries),
        "total_weights": total_weights,
        "total_inputs": total_inputs,
        "total_weight_bytes": total_weight_bytes,
        "total_weight_MB": total_weight_bytes / (1024 * 1024),
    }

    return {"metadata": metadata, "entries": entries}


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python inputs_registry_parser.py <log_path>")
        sys.exit(1)

    log_path = sys.argv[1]
    registry = parse_inputs_registry(log_path)

    print(json.dumps(registry, indent=2))
