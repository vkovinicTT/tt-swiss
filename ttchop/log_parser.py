# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parse op-by-op execution logs to extract per-op traces."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_RE.sub("", text)


def parse_op_by_op_log(log_path: Path) -> List[Dict[str, Any]]:
    """Parse op-by-op.log into per-op execution blocks.

    Each block corresponds to one op test (1:1 with report JSON by index).
    Returns list of dicts with keys:
      - success: bool
      - last_ttnn_op: str or None (last "Executing operation" before crash)
      - error_message: str or None (short TT_FATAL message)
      - error_trace: str or None (full error output including backtrace)
    """
    content = log_path.read_text(errors="replace")
    lines = [strip_ansi(line) for line in content.split("\n")]

    blocks: List[Dict[str, Any]] = []
    current_block: Optional[Dict[str, Any]] = None
    sub_depth = 0
    collecting_error = False

    for line in lines:
        stripped = line.strip()

        # Detect "Starting execution of program: main" (top-level only)
        if "Starting execution of program: main" in stripped:
            if "main_" in stripped:
                # Sub-program (e.g., main_const_eval_0) - track depth
                sub_depth += 1
                continue
            # New top-level block - finalize previous
            collecting_error = False
            if current_block is not None:
                if not current_block.pop("_finished"):
                    current_block["success"] = False
                _finalize_error_trace(current_block)
                blocks.append(current_block)
            current_block = {
                "success": True,
                "last_ttnn_op": None,
                "error_message": None,
                "_error_lines": [],
                "_finished": False,
            }
            sub_depth = 0
            continue

        if current_block is None:
            continue

        # Track sub-program exits
        if "Finished execution of program: main_" in stripped:
            sub_depth = max(0, sub_depth - 1)
            continue

        # Skip lines inside sub-programs
        if sub_depth > 0:
            continue

        # Detect top-level "Finished execution"
        if "Finished execution of program: main" in stripped and "main_" not in stripped:
            current_block["_finished"] = True
            collecting_error = False
            continue

        # Once finished, ignore remaining lines until next block starts
        if current_block["_finished"]:
            continue

        # If we're collecting error trace lines, keep appending until backtrace ends
        if collecting_error:
            if _is_error_trace_line(stripped):
                current_block["_error_lines"].append(stripped)
            else:
                collecting_error = False
            continue

        # Track executing operations (only at top level)
        if "Executing operation:" in stripped:
            current_block["last_ttnn_op"] = _extract_op_name(stripped)
            continue

        # Detect TT_FATAL - start collecting full error trace
        if "TT_FATAL" in stripped:
            current_block["success"] = False
            fatal_match = re.search(
                r"TT_FATAL:\s*(.+?)(?:\s*\(assert\.hpp:\d+\))?$", stripped
            )
            current_block["error_message"] = (
                fatal_match.group(1).strip() if fatal_match else stripped
            )
            current_block["_error_lines"].append(stripped)
            collecting_error = True
            continue

    # Finalize last block
    if current_block is not None:
        if not current_block.pop("_finished") and current_block["success"]:
            current_block["success"] = False
        _finalize_error_trace(current_block)
        blocks.append(current_block)

    return blocks


def _finalize_error_trace(block: Dict[str, Any]) -> None:
    """Convert collected error lines into error_trace string and clean up."""
    error_lines = block.pop("_error_lines", [])
    if error_lines:
        block["error_trace"] = "\n".join(error_lines)
    else:
        block["error_trace"] = None


def _is_error_trace_line(line: str) -> bool:
    """Check if a line is part of a TT_FATAL error trace (header, message, or backtrace)."""
    if not line:
        return True
    if line.startswith("---"):
        return True
    if line in ("info:", "backtrace:"):
        return True
    if "TT_FATAL" in line:
        return True
    # Error message lines between "info:" and "backtrace:" (e.g. "Padding must be ...")
    # These don't match any log format prefix like timestamps or "Always |"
    if not re.match(r"^\d{4}-\d{2}-\d{2}", line) and "Always |" not in line:
        return True
    return False


def _extract_op_name(line: str) -> str:
    """Extract TTNN op name from 'Executing operation: ...' line."""
    match = re.search(r'"(ttnn\.\w+)"', line)
    return match.group(1) if match else line.split("Executing operation:")[-1].strip()[:80]


def save_parsed_log(blocks: List[Dict[str, Any]], output_path: Path) -> None:
    """Save parsed log blocks to JSON."""
    compact = [
        {
            "success": b["success"],
            "last_ttnn_op": b.get("last_ttnn_op"),
            "error_message": b.get("error_message"),
            "error_trace": b.get("error_trace"),
        }
        for b in blocks
    ]
    with open(output_path, "w") as f:
        json.dump(compact, f, indent=2)
