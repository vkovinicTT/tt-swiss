#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Extract the last run from a log file that contains multiple forward passes.

This script finds the first occurrence of "Got output shape:" and keeps
everything after that point, effectively removing warmup runs.

Usage:
    python extract_last_run.py <log_file>

Example:
    python extract_last_run.py logs/decoder_20260122_153451/decoder_profile.log
"""

import sys
from pathlib import Path


def extract_last_run(log_file_path: Path) -> None:
    """
    Extract the last run from a log file by finding the first occurrence
    of "Got output shape:" and keeping everything after it.
    """
    if not log_file_path.exists():
        print(f"Error: Log file not found: {log_file_path}")
        sys.exit(1)

    print(f"Processing: {log_file_path}")

    # Read the entire log file
    with open(log_file_path, "r") as f:
        lines = f.readlines()

    print(f"Total lines in log: {len(lines)}")

    # Find all occurrences of "Got output shape:"
    marker = "Got output shape:"
    marker_indices = []

    for i, line in enumerate(lines):
        if marker in line:
            marker_indices.append(i)

    if not marker_indices:
        print(f"Warning: No '{marker}' found in log file. Keeping entire log.")
        return

    print(f"Found {len(marker_indices)} occurrence(s) of '{marker}'")

    # Get the first occurrence
    first_marker_index = marker_indices[0]
    print(f"First occurrence at line {first_marker_index + 1}")

    # Keep everything after the first marker (excluding the marker line itself)
    lines_to_keep = lines[first_marker_index + 1 :]
    print(f"Keeping {len(lines_to_keep)} lines from after first occurrence")

    # Write back to the same file
    with open(log_file_path, "w") as f:
        f.writelines(lines_to_keep)

    print(f"\n✓ Successfully extracted content after first '{marker}'")
    print(
        f"✓ Removed {len(lines) - len(lines_to_keep)} lines (warmup + first run marker)"
    )
    print(f"✓ Updated: {log_file_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_last_run.py <log_file>")
        print("\nExample:")
        print(
            "  python extract_last_run.py logs/decoder_20260122_153451/decoder_profile.log"
        )
        sys.exit(1)

    log_file = Path(sys.argv[1])
    extract_last_run(log_file)


if __name__ == "__main__":
    main()
