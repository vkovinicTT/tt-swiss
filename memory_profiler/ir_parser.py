# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
IR parser for extracting TTIR and TTNN intermediate representations from runtime logs.

Parses MLIR module dumps to extract:
- TTIR module (contains ttir.* operations)
- TTNN module (contains ttnn.* operations)
- Location-to-line-number index for linking operations to IR lines
"""

import json
import re
import sys
from typing import Dict, List, Optional, Tuple


def find_ir_module_boundaries(
    lines: List[str], module_type: str
) -> Tuple[int, int]:
    """
    Find the start and end indices of an IR module section in log lines.

    Looks for exact module name match: "MLIR Module ttir:" or "MLIR Module ttnn:"
    This avoids incorrectly matching shlo_frontend or shlo_compiler modules.

    Args:
        lines: List of log lines
        module_type: Type of module to find ('ttir' or 'ttnn')

    Returns:
        Tuple of (start_index, end_index) or (-1, -1) if not found
    """
    start_idx = -1
    target_marker = f"MLIR Module {module_type}:"

    for i, line in enumerate(lines):
        if target_marker in line:
            start_idx = i
        elif start_idx >= 0 and "END OF MLIR MODULE" in line:
            return start_idx, i

    return -1, -1


def extract_module_text(lines: List[str], start_idx: int, end_idx: int) -> str:
    """
    Extract the IR module text from log lines.

    Args:
        lines: List of log lines
        start_idx: Start index of module
        end_idx: End index of module

    Returns:
        Module text as a single string
    """
    if start_idx < 0 or end_idx < 0:
        return ""

    # Skip the header line and extract just the module content
    module_lines = []
    for i in range(start_idx + 1, end_idx):
        line = lines[i]
        # Remove common log prefixes (timestamps, log levels, etc.)
        # Pattern: optional timestamp, optional log level, then content
        cleaned = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+", "", line)
        cleaned = re.sub(r"^(DEBUG|INFO|WARNING|ERROR)\s+", "", cleaned)
        # Remove RuntimeTTNN prefix if present
        cleaned = re.sub(r"^RuntimeTTNN:\s*", "", cleaned)
        module_lines.append(cleaned)

    return "".join(module_lines)


def build_loc_line_index(ir_text: str) -> Dict[str, int]:
    """
    Build a mapping from location names to operation line numbers.

    Two-step process to handle MLIR location alias definitions:
    1. Parse alias definitions: #loc56 = loc("multiply.3545") -> alias_to_name map
    2. Find operations with loc(#locN) -> map name to operation line number

    This ensures we map to the actual operation line (e.g., line 300 with
    "ttnn.reshape" ... loc(#loc56)) rather than the alias definition line
    (e.g., line 56 with #loc56 = loc("multiply.3545")).

    Args:
        ir_text: The IR module text

    Returns:
        Dictionary mapping location identifiers to line numbers
    """
    lines = ir_text.split("\n")
    loc_index = {}

    # Step 1: Build alias -> name mapping from definitions
    # Pattern: #loc56 = loc("multiply.3545")
    alias_to_name = {}
    alias_def_pattern = re.compile(r'(#loc\d+)\s*=\s*loc\("([^"]+)"\)')

    for line in lines:
        match = alias_def_pattern.search(line)
        if match:
            alias_to_name[match.group(1)] = match.group(2)  # #loc56 -> "multiply.3545"

    # Step 2: Find operations that reference loc(#locN)
    # Pattern: anything ... loc(#loc56)
    op_loc_pattern = re.compile(r'loc\((#loc\d+)\)')

    for line_num, line in enumerate(lines, start=1):
        match = op_loc_pattern.search(line)
        if match:
            alias = match.group(1)  # #loc56
            if alias in alias_to_name:
                name = alias_to_name[alias]  # "multiply.3545"
                if name not in loc_index:
                    loc_index[name] = line_num

    # Step 3: Also handle inline loc("name") patterns for ops without aliases
    # Pattern: loc("something") directly in operation lines
    inline_loc_pattern = re.compile(r'loc\("([^"]+)"\)')

    for line_num, line in enumerate(lines, start=1):
        # Skip alias definition lines (they start with #loc)
        if re.match(r'\s*#loc\d+\s*=', line):
            continue
        matches = inline_loc_pattern.findall(line)
        for loc_id in matches:
            if loc_id not in loc_index:
                loc_index[loc_id] = line_num

    return loc_index


def parse_ir_modules(log_path: str) -> Dict:
    """
    Parse IR modules from a log file.

    Extracts TTIR and TTNN modules along with their location indices
    for linking operations to specific IR lines.

    Args:
        log_path: Path to the log file

    Returns:
        Dictionary with structure:
        {
            "ttir": {"text": "...", "loc_index": {"loc_id": line_num, ...}},
            "ttnn": {"text": "...", "loc_index": {"loc_id": line_num, ...}}
        }
        Returns empty dict entries if modules are not found.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return {"ttir": {"text": "", "loc_index": {}}, "ttnn": {"text": "", "loc_index": {}}}
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return {"ttir": {"text": "", "loc_index": {}}, "ttnn": {"text": "", "loc_index": {}}}

    result = {
        "ttir": {"text": "", "loc_index": {}},
        "ttnn": {"text": "", "loc_index": {}},
    }

    # Find and parse TTIR module
    ttir_start, ttir_end = find_ir_module_boundaries(lines, "ttir")
    if ttir_start >= 0:
        ttir_text = extract_module_text(lines, ttir_start, ttir_end)
        result["ttir"]["text"] = ttir_text
        result["ttir"]["loc_index"] = build_loc_line_index(ttir_text)
        print(f"Found TTIR module: {len(ttir_text)} chars, {len(result['ttir']['loc_index'])} locations")

    # Find and parse TTNN module
    ttnn_start, ttnn_end = find_ir_module_boundaries(lines, "ttnn")
    if ttnn_start >= 0:
        ttnn_text = extract_module_text(lines, ttnn_start, ttnn_end)
        result["ttnn"]["text"] = ttnn_text
        result["ttnn"]["loc_index"] = build_loc_line_index(ttnn_text)
        print(f"Found TTNN module: {len(ttnn_text)} chars, {len(result['ttnn']['loc_index'])} locations")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ir_parser.py <log_path> [output_path]")
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    ir_data = parse_ir_modules(log_path)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ir_data, f, indent=2)
        print(f"IR data written to: {output_path}")
    else:
        # Print summary
        print("\nIR Parsing Summary:")
        print(f"  TTIR: {len(ir_data['ttir']['text'])} chars, {len(ir_data['ttir']['loc_index'])} locations")
        print(f"  TTNN: {len(ir_data['ttnn']['text'])} chars, {len(ir_data['ttnn']['loc_index'])} locations")
