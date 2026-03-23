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

try:
    from .mem_logger import log_mem
    from .multihost import line_matches_host, strip_mpi_prefix
except ImportError:
    from mem_logger import log_mem
    from multihost import line_matches_host, strip_mpi_prefix


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
        # Remove MPI host prefix if present (e.g., [1,0]<stdout>:)
        cleaned = strip_mpi_prefix(line)
        # Remove common log prefixes (timestamps, log levels, etc.)
        # Pattern: optional timestamp, optional log level, then content
        cleaned = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+", "", cleaned)
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
    alias_def_pattern = re.compile(r'(#loc\d+)\s*=\s*loc\("([^"]+)"')

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
    inline_loc_pattern = re.compile(r'loc\("([^"]+)"')

    for line_num, line in enumerate(lines, start=1):
        # Skip alias definition lines (they start with #loc)
        if re.match(r'\s*#loc\d+\s*=', line):
            continue
        matches = inline_loc_pattern.findall(line)
        for loc_id in matches:
            if loc_id not in loc_index:
                loc_index[loc_id] = line_num

    # Step 4: Handle ttcore.load_cached operations with loc(unknown)
    # Map @function_name to the line where load_cached appears
    load_cached_pattern = re.compile(r'load_cached\((@[\w.]+)')
    for line_num, line in enumerate(lines, start=1):
        match = load_cached_pattern.search(line)
        if match:
            func_name = match.group(1)  # "@main_const_eval_0"
            if func_name not in loc_index:
                loc_index[func_name] = line_num

    return loc_index


def _stream_extract_ir_sections(
    log_path: str, host_filter: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Stream through a log file and extract only the TTIR and TTNN module
    sections, without loading the entire file into memory.

    Args:
        log_path: Path to the log file
        host_filter: Optional MPI host to filter to (e.g., '1,0')

    Returns:
        Dict mapping module type ('ttir', 'ttnn') to list of section lines.
    """
    sections: Dict[str, List[str]] = {}
    current_type = None
    current_lines: List[str] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if host_filter is not None:
                if not line_matches_host(line, host_filter):
                    continue
                line = strip_mpi_prefix(line)
            if current_type is None:
                # Check for section start - only exact ttir: or ttnn: matches
                for mod_type in ("ttir", "ttnn"):
                    if f"MLIR Module {mod_type}:" in line:
                        current_type = mod_type
                        current_lines = [line]
                        break
            else:
                current_lines.append(line)
                if "END OF MLIR MODULE" in line:
                    sections[current_type] = current_lines
                    current_type = None
                    current_lines = []
                    # Stop early if we found both
                    if "ttir" in sections and "ttnn" in sections:
                        break

    return sections


def parse_ir_modules(log_path: str, host_filter: Optional[str] = None) -> Dict:
    """
    Parse IR modules from a log file.

    Extracts TTIR and TTNN modules along with their location indices
    for linking operations to specific IR lines.

    Args:
        log_path: Path to the log file
        host_filter: Optional MPI host to filter to (e.g., '1,0')

    Returns:
        Dictionary with structure:
        {
            "ttir": {"text": "...", "loc_index": {"loc_id": line_num, ...}},
            "ttnn": {"text": "...", "loc_index": {"loc_id": line_num, ...}}
        }
        Returns empty dict entries if modules are not found.
    """
    log_mem("parse_ir_modules: start")
    try:
        sections = _stream_extract_ir_sections(log_path, host_filter=host_filter)
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return {"ttir": {"text": "", "loc_index": {}}, "ttnn": {"text": "", "loc_index": {}}}
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return {"ttir": {"text": "", "loc_index": {}}, "ttnn": {"text": "", "loc_index": {}}}

    log_mem("parse_ir_modules: sections extracted (streaming)")

    result = {
        "ttir": {"text": "", "loc_index": {}},
        "ttnn": {"text": "", "loc_index": {}},
    }

    # Process TTIR section if found
    if "ttir" in sections:
        ttir_lines = sections["ttir"]
        ttir_text = extract_module_text(ttir_lines, 0, len(ttir_lines) - 1)
        result["ttir"]["text"] = ttir_text
        result["ttir"]["loc_index"] = build_loc_line_index(ttir_text)
        print(f"Found TTIR module: {len(ttir_text)} chars, {len(result['ttir']['loc_index'])} locations")

    # Process TTNN section if found
    if "ttnn" in sections:
        ttnn_lines = sections["ttnn"]
        ttnn_text = extract_module_text(ttnn_lines, 0, len(ttnn_lines) - 1)
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
