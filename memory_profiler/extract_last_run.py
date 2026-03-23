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

import os
import sys
import tempfile
import shutil
from pathlib import Path

try:
    from .multihost import strip_mpi_prefix
except ImportError:
    from multihost import strip_mpi_prefix


def extract_last_run(log_file_path: Path) -> None:
    """
    Extract the last run from a log file by finding the first occurrence
    of "Got output shape:" and keeping everything after it.

    Uses streaming I/O to avoid loading the entire file into memory.
    """
    if not log_file_path.exists():
        print(f"Error: Log file not found: {log_file_path}")
        sys.exit(1)

    print(f"Processing: {log_file_path}")

    marker = "Got output shape:"
    marker_found = False
    marker_line = 0
    total_lines = 0
    lines_kept = 0

    # Stream through the file: skip until marker, write the rest to a temp file
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=log_file_path.parent, suffix=".tmp", prefix=".extract_"
    )
    try:
        with open(log_file_path, "r") as fin, open(tmp_fd, "w") as fout:
            for line_num, line in enumerate(fin, 1):
                total_lines = line_num
                if not marker_found:
                    if marker in strip_mpi_prefix(line):
                        marker_found = True
                        marker_line = line_num
                        continue  # skip the marker line itself
                else:
                    fout.write(line)
                    lines_kept += 1

        if not marker_found:
            print(f"Warning: No '{marker}' found in log file. Keeping entire log.")
            os.unlink(tmp_path)
            return

        print(f"Total lines in log: {total_lines}")
        print(f"First occurrence at line {marker_line}")
        print(f"Keeping {lines_kept} lines from after first occurrence")

        # Atomically replace the original file
        shutil.move(tmp_path, str(log_file_path))

        print(f"\n✓ Successfully extracted content after first '{marker}'")
        print(
            f"✓ Removed {total_lines - lines_kept} lines (warmup + first run marker)"
        )
        print(f"✓ Updated: {log_file_path}")
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


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
