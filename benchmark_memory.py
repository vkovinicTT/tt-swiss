#!/usr/bin/env python3
"""Benchmark memory usage and wall time for the log parsing pipeline."""

import os
import sys
import time
import resource
import tracemalloc
import tempfile
import shutil
from pathlib import Path

LOG_FILE = os.path.expanduser("~/tt-xla/vae.log")

def get_rss_mb():
    """Get current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB

def benchmark_parse_log_file():
    """Benchmark the main parse_log_file function."""
    from memory_profiler.parser import parse_log_file

    with tempfile.TemporaryDirectory() as tmpdir:
        mem_out = os.path.join(tmpdir, "mem.json")
        ops_out = os.path.join(tmpdir, "ops.json")
        reg_out = os.path.join(tmpdir, "reg.json")
        ir_out = os.path.join(tmpdir, "ir.json")

        tracemalloc.start()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()

        parse_log_file(LOG_FILE, mem_out, ops_out, reg_out, ir_out)

        t1 = time.perf_counter()
        rss_after = get_rss_mb()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "wall_time_s": t1 - t0,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "peak_rss_mb": rss_after,  # max RSS is cumulative
            "tracemalloc_peak_mb": peak / (1024 * 1024),
            "tracemalloc_current_mb": current / (1024 * 1024),
        }

def benchmark_inputs_registry():
    """Benchmark inputs_registry_parser standalone."""
    from memory_profiler.inputs_registry_parser import parse_inputs_registry

    tracemalloc.start()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()

    parse_inputs_registry(LOG_FILE)

    t1 = time.perf_counter()
    rss_after = get_rss_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "wall_time_s": t1 - t0,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "tracemalloc_peak_mb": peak / (1024 * 1024),
    }

def benchmark_ir_parser():
    """Benchmark ir_parser standalone."""
    from memory_profiler.ir_parser import parse_ir_modules

    tracemalloc.start()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()

    parse_ir_modules(LOG_FILE)

    t1 = time.perf_counter()
    rss_after = get_rss_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "wall_time_s": t1 - t0,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "tracemalloc_peak_mb": peak / (1024 * 1024),
    }

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found: {LOG_FILE}")
        sys.exit(1)

    file_size_mb = os.path.getsize(LOG_FILE) / (1024 * 1024)
    print(f"Log file: {LOG_FILE}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"{'=' * 60}")

    # Run benchmarks in separate sub-measurements
    print("\n--- Full pipeline (parse_log_file) ---")
    results = benchmark_parse_log_file()
    print(f"  Wall time:        {results['wall_time_s']:.2f} s")
    print(f"  RSS before:       {results['rss_before_mb']:.1f} MB")
    print(f"  RSS after:        {results['rss_after_mb']:.1f} MB")
    print(f"  tracemalloc peak: {results['tracemalloc_peak_mb']:.1f} MB")
    print(f"  tracemalloc curr: {results['tracemalloc_current_mb']:.1f} MB")

    # Note: RSS is cumulative max, so these show incremental tracemalloc only
    print("\n--- inputs_registry_parser (standalone) ---")
    results2 = benchmark_inputs_registry()
    print(f"  Wall time:        {results2['wall_time_s']:.2f} s")
    print(f"  tracemalloc peak: {results2['tracemalloc_peak_mb']:.1f} MB")

    print("\n--- ir_parser (standalone) ---")
    results3 = benchmark_ir_parser()
    print(f"  Wall time:        {results3['wall_time_s']:.2f} s")
    print(f"  tracemalloc peak: {results3['tracemalloc_peak_mb']:.1f} MB")

    print(f"\n{'=' * 60}")
    print(f"Peak RSS (process-wide): {get_rss_mb():.1f} MB")

if __name__ == "__main__":
    main()
