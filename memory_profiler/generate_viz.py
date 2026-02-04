#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone script to generate visualization from existing profiler output.

Usage:
    python generate_viz.py <run_directory>
    python generate_viz.py <run_directory> --name <script_name>
    python generate_viz.py                  # Uses latest run
"""

import argparse
import sys
from pathlib import Path

from memory_profiler.visualizer import MemoryVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization from existing profiler output"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to run directory containing JSON files",
    )
    parser.add_argument(
        "--name",
        metavar="SCRIPT_NAME",
        help="Explicit script name override (used for file naming)",
    )

    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Find latest run
        log_dir = Path(__file__).parent / "logs"
        if not log_dir.exists():
            print("Error: No logs directory found")
            sys.exit(1)

        runs = sorted(log_dir.glob("*_*"))
        if not runs:
            print("Error: No profiling runs found in logs directory")
            sys.exit(1)

        run_dir = runs[-1]
        print(f"Using latest run: {run_dir.name}")

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Generating visualization for: {run_dir.name}")
    viz = MemoryVisualizer(run_dir, script_name=args.name)
    report_path = viz.generate_report()

    print(f"\nVisualization generated: {report_path}")
    print(f"\nOpen in browser:")
    print(f"  file://{report_path.absolute()}")


if __name__ == "__main__":
    main()
