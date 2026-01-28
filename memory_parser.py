# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Memory statistics parser for extracting device memory state from runtime logs.
"""

import re
import sys
from typing import Dict, List, Optional


def parse_memory_stats(lines: List[str], start_idx: int) -> Optional[Dict]:
    """
    Parse memory statistics block starting from start_idx.

    Expected format:
        Always |     INFO | Device memory state before operation TypecastOp
        Always |     INFO | Device DRAM memory state: MemoryView{numBanks: 12, ...}
        Always |     INFO | Device L1 memory state: MemoryView{...}
        Always |     INFO | Device L1 SMALL memory state: MemoryView{...}
        Always |     INFO | Device TRACE memory state: MemoryView{...}

    Args:
        lines: List of log lines
        start_idx: Index to start searching for memory stats

    Returns:
        Dictionary with 'op_type' and 'memory' fields, or None if not found
    """
    if start_idx >= len(lines):
        return None

    if "Device memory state before operation" not in lines[start_idx]:
        return None

    # Extract operation type from header line
    op_type_match = re.search(r"before operation (\w+)", lines[start_idx])
    op_type = op_type_match.group(1) if op_type_match else None

    # Parse next 4 lines for DRAM, L1, L1_SMALL, TRACE
    memory_data = {}
    memory_types = ["DRAM", "L1", "L1_SMALL", "TRACE"]

    for i, mem_type in enumerate(memory_types):
        line_idx = start_idx + i + 1
        if line_idx >= len(lines):
            break

        line = lines[line_idx]
        # Handle both "L1_SMALL" and "L1 SMALL" formats
        search_type = mem_type.replace("_", " ")

        if (
            f"Device {mem_type} memory state" in line
            or f"Device {search_type} memory state" in line
        ):
            mem_stats = parse_memory_view(line)
            if mem_stats:
                memory_data[mem_type] = mem_stats

    if not memory_data:
        return None

    return {"op_type": op_type, "memory": memory_data}


def parse_memory_view(line: str) -> Optional[Dict]:
    """
    Parse MemoryView{...} format from a log line.

    Expected format:
        Device DRAM memory state: MemoryView{numBanks: 12, totalBytesPerBank: 1024.000 MB, ...}

    Args:
        line: Log line containing MemoryView

    Returns:
        Dictionary with parsed memory statistics, or None if parsing fails
    """
    match = re.search(r"MemoryView\{([^}]+)\}", line)
    if not match:
        return None

    content = match.group(1)
    stats = {}

    # Parse key-value pairs separated by commas
    for pair in content.split(","):
        key_val = pair.split(":")
        if len(key_val) == 2:
            key = key_val[0].strip()
            val = key_val[1].strip()

            # Convert MB values to float
            if "MB" in val:
                try:
                    val = float(val.replace("MB", "").strip())
                    key = key + "_MB"
                except ValueError:
                    print(f"Warning: Could not parse MB value: {val}", file=sys.stderr)
                    continue
            elif key == "numBanks":
                try:
                    val = int(val)
                except ValueError:
                    print(
                        f"Warning: Could not parse numBanks value: {val}",
                        file=sys.stderr,
                    )
                    continue

            stats[key] = val

    return stats if stats else None
