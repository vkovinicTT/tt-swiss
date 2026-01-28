#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Memory profiler wrapper for running and profiling model execution on Tenstorrent hardware.

Modes:
1. Default: Run script + parse + visualize
2. --log: Only capture logs (no parsing or visualization)
3. --analyze LOG_FILE: Parse existing log file
4. --visualize RUN_DIR: Generate visualization from existing run directory

Usage:
    # Default: Full profiling with visualization
    tt-memory-profiler path/to/your_model.py

    # Only capture logs
    tt-memory-profiler --log path/to/your_model.py

    # Analyze existing log
    tt-memory-profiler --analyze logs/your_model_20260122_143957/your_model_profile.log

    # Visualize existing run
    tt-memory-profiler --visualize logs/your_model_20260122_143957/

    # Specify custom output directory
    tt-memory-profiler --output-dir /path/to/output path/to/your_model.py
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_and_capture(target_script: Path, output_dir: Path, script_name: str) -> tuple:
    """Run the target script and capture logs"""
    log_file = output_dir / f"{script_name}_profile.log"

    print("=" * 70)
    print("Memory Profiler")
    print("=" * 70)
    print(f"Target script: {target_script}")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file.name}")
    print("=" * 70)

    # Set environment variables for profiling
    env = os.environ.copy()
    env["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"

    # Run target script with tee to capture logs
    print(f"\nRunning {target_script.name}...")
    print("(This may take several minutes...)\n")

    # Use -u flag to disable Python output buffering for proper log capture
    cmd = f"python -u {target_script} 2>&1 | tee {log_file}"
    result = subprocess.run(cmd, shell=True, env=env)

    if result.returncode != 0:
        print(f"\nWarning: Script exited with code {result.returncode}")

    return log_file, result.returncode


def analyze_log(log_file: Path, output_dir: Path, script_name: str):
    """Parse log file and generate JSON outputs"""
    mem_json = output_dir / f"{script_name}_memory.json"
    ops_json = output_dir / f"{script_name}_operations.json"
    registry_json = output_dir / f"{script_name}_inputs_registry.json"

    try:
        from parser import parse_log_file, validate_outputs

        # Parse the log file
        print("\n" + "=" * 70)
        print("Parsing logs...")
        print("=" * 70)
        parse_log_file(str(log_file), str(mem_json), str(ops_json), str(registry_json))

        # Validate outputs
        print("\n" + "=" * 70)
        print("Validating outputs...")
        print("=" * 70)
        validate_outputs(str(mem_json), str(ops_json))

        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)
        print(f"\nGenerated files:")
        print(f"  - {mem_json}")
        print(f"  - {ops_json}")
        print(f"  - {registry_json}")

        return mem_json, ops_json, registry_json

    except ImportError as e:
        print(f"Error: Could not import parser module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during parsing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def generate_visualization(run_dir: Path):
    """Generate visualization from existing run directory"""
    # Determine script name from directory
    dir_name = run_dir.name
    script_name = "_".join(dir_name.split("_")[:-2]) if "_" in dir_name else "decoder"

    try:
        from visualizer import MemoryVisualizer

        print("=" * 70)
        print("Generating visualization...")
        print("=" * 70)
        print(f"Run directory: {run_dir}")
        print("=" * 70)

        viz = MemoryVisualizer(run_dir)
        report_path = viz.generate_report()

        print("\n" + "=" * 70)
        print("Visualization complete!")
        print("=" * 70)
        print(f"\nGenerated: {report_path}")

    except Exception as e:
        print(f"Error generating visualization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Memory profiler for model execution on Tenstorrent hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: run + parse + visualize
  tt-memory-profiler path/to/your_model.py

  # Only capture logs
  tt-memory-profiler --log path/to/your_model.py

  # Analyze existing log
  tt-memory-profiler --analyze logs/your_model_20260122_143957/your_model_profile.log

  # Visualize existing run
  tt-memory-profiler --visualize logs/your_model_20260122_143957/

  # Specify custom output directory
  tt-memory-profiler --output-dir /path/to/output path/to/your_model.py
        """,
    )

    parser.add_argument(
        "script_path", nargs="?", help="Path to the Python script to profile"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Only capture logs without parsing or visualization",
    )
    parser.add_argument(
        "--analyze",
        metavar="LOG_FILE",
        help="Parse existing log file (outputs JSON files)",
    )
    parser.add_argument(
        "--visualize",
        metavar="RUN_DIR",
        help="Generate visualization from existing run directory",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Output directory for logs (default: ./logs in current working directory)",
    )

    args = parser.parse_args()

    # Mode 1: Visualize existing run
    if args.visualize:
        run_dir = Path(args.visualize)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
        generate_visualization(run_dir)

    # Mode 2: Analyze existing log
    elif args.analyze:
        log_file = Path(args.analyze)
        if not log_file.exists():
            print(f"Error: Log file not found: {log_file}")
            sys.exit(1)

        # Determine output directory and script name from log file path
        output_dir = log_file.parent
        script_name = log_file.stem.replace("_profile", "")

        print("=" * 70)
        print("Memory Profiler - Analyze Mode")
        print("=" * 70)
        print(f"Log file: {log_file}")
        print(f"Output directory: {output_dir}")
        print("=" * 70)

        analyze_log(log_file, output_dir, script_name)

    # Mode 3 & 4: Run script (with or without --log flag)
    elif args.script_path:
        target_script = Path(args.script_path)

        if not target_script.exists():
            print(f"Error: Script not found: {target_script}")
            sys.exit(1)

        # Extract script name and create output directory
        script_name = target_script.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use --output-dir if specified, otherwise use ./logs in current working directory
        if args.output_dir:
            base_dir = Path(args.output_dir)
        else:
            base_dir = Path.cwd() / "logs"
        output_dir = base_dir / f"{script_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run and capture logs
        log_file, return_code = run_and_capture(target_script, output_dir, script_name)

        if return_code != 0:
            print("Attempting to parse logs anyway...\n")

        # If --log flag, stop here
        if args.log:
            print("\n" + "=" * 70)
            print("Log capture complete!")
            print("=" * 70)
            print(f"\nLog file: {log_file}")
            print(f"\nTo analyze later, run:")
            print(f"  tt-memory-profiler --analyze {log_file}")
            print(f"\nTo visualize later, run:")
            print(f"  tt-memory-profiler --visualize {output_dir}")
            return

        # Default: parse and visualize
        analyze_log(log_file, output_dir, script_name)
        generate_visualization(output_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
