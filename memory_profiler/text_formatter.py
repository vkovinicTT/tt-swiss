# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LLM-friendly text report generator for memory profiling data.

Generates compact, markdown-formatted reports designed for LLM consumption,
following the llms.txt standard (https://llmstxt.org/).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class LLMTextFormatter:
    """Generate LLM-friendly text reports from memory profiler output"""

    def __init__(self, run_dir: Path, script_name: str = None):
        """
        Initialize formatter with a profiler run directory.

        Args:
            run_dir: Path to profiler output directory containing JSON files
            script_name: Optional explicit script name. If not provided, inferred from directory name.
        """
        self.run_dir = Path(run_dir)

        # Use provided script_name or infer from directory name
        if script_name is not None:
            self.script_name = script_name
        else:
            parts = self.run_dir.name.split("_")
            if len(parts) >= 3:
                self.script_name = "_".join(parts[:-2])
            else:
                self.script_name = parts[0]

        self.mem_file = self.run_dir / f"{self.script_name}_memory.json"
        self.ops_file = self.run_dir / f"{self.script_name}_operations.json"
        self.registry_file = self.run_dir / f"{self.script_name}_inputs_registry.json"

        # Validate required files exist
        if not self.mem_file.exists():
            raise FileNotFoundError(
                f"Memory file not found: {self.mem_file}\n"
                f"Hint: Run 'tt-memory-profiler --analyze <log_file>' first to generate JSON files."
            )
        if not self.ops_file.exists():
            raise FileNotFoundError(
                f"Operations file not found: {self.ops_file}\n"
                f"Hint: Run 'tt-memory-profiler --analyze <log_file>' first to generate JSON files."
            )

        # Load data
        with open(self.mem_file) as f:
            mem_json = json.load(f)
        with open(self.ops_file) as f:
            self.ops_data = json.load(f)

        # Load registry if it exists
        self.registry = None
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)

        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(mem_json, dict) and "metadata" in mem_json:
            self.mem_metadata = mem_json["metadata"]
            self.mem_data = mem_json["operations"]
        else:
            self.mem_metadata = None
            self.mem_data = mem_json

        # Detect available memory types
        self.available_memory_types = []
        if self.mem_data:
            for mt in ["DRAM", "L1", "L1_SMALL", "TRACE"]:
                if all(mt in op.get("memory", {}) for op in self.mem_data):
                    self.available_memory_types.append(mt)

    def generate_report(self, output_file: Path = None) -> str:
        """
        Generate LLM-friendly text report.

        Args:
            output_file: Optional output file path. If provided, writes report to file.

        Returns:
            The generated report as a string
        """
        if not self.mem_data:
            report = f"# Memory Profile: {self.script_name}\n\n> No operations recorded.\n"
            if output_file:
                output_file.write_text(report)
            return report

        sections = [
            self._format_header(),
            self._format_configuration(),
            self._format_peak_memory_table(),
            self._format_top_consumers_table(n=10),
            self._format_padding_overhead_table(n=10),
            self._format_weights_registry(),
            self._format_operation_distribution(),
        ]

        report = "\n".join(s for s in sections if s)

        if output_file:
            output_file.write_text(report)

        return report

    def _truncate_loc(self, loc: str, max_len: int = 50) -> str:
        """Truncate location string if too long"""
        if not loc:
            return "N/A"
        if len(loc) <= max_len:
            return loc
        return loc[:max_len - 3] + "..."

    def _format_header(self) -> str:
        """Format H1 title and blockquote summary"""
        summary_stats = self._compute_summary_stats()
        peak_analysis = self._analyze_peaks()

        # Build summary line
        parts = []

        # Peak DRAM
        if "DRAM" in summary_stats["memory_types"]:
            dram = summary_stats["memory_types"]["DRAM"]
            peak_mb = dram["peak"]
            capacity = dram["capacity"]
            utilization = (peak_mb / capacity * 100) if capacity > 0 else 0
            parts.append(f"Peak DRAM: {peak_mb:.1f} MB/bank ({utilization:.1f}%)")

        # Peak L1
        if "L1" in summary_stats["memory_types"]:
            l1 = summary_stats["memory_types"]["L1"]
            parts.append(f"Peak L1: {l1['peak']:.2f} MB/bank")

        # Total ops
        parts.append(f"Ops: {summary_stats['total_ops']:,}")

        # Total weights
        if self.registry and self.registry.get("metadata"):
            total_weight_mb = self.registry["metadata"].get("total_weight_MB", 0)
            if total_weight_mb > 0:
                parts.append(f"Weights: {total_weight_mb:.1f} MB")

        # Padding overhead
        peak_overhead = self._calculate_peak_padding_overhead()
        if peak_overhead.get("has_data") and peak_overhead.get("dram_pct", 0) > 0:
            parts.append(f"Padding Overhead: {peak_overhead['dram_pct']:.1f}%")

        summary = " | ".join(parts)

        return f"# Memory Profile: {self.script_name}\n\n> {summary}\n"

    def _format_configuration(self) -> str:
        """Format configuration section"""
        lines = ["## Configuration"]

        # Memory configuration
        if self.mem_metadata and "memory_config" in self.mem_metadata:
            config = self.mem_metadata["memory_config"]
            mem_parts = []
            for mem_type in ["DRAM", "L1", "L1_SMALL", "TRACE"]:
                if mem_type in config:
                    num_banks = config[mem_type].get("num_banks", 0)
                    capacity = config[mem_type].get("total_bytes_per_bank_MB", 0)
                    mem_parts.append(f"{mem_type} {num_banks} x {capacity:.2f} MB/bank")
            if mem_parts:
                lines.append(f"- Memory: {' | '.join(mem_parts)}")

        # Operation counts
        total_ops = len(self.mem_data)
        weight_ops = sum(
            1 for i, op in enumerate(self.mem_data)
            if op.get("is_weight_op", False) or
               (i < len(self.ops_data) and self.ops_data[i].get("is_weight_op", False))
        )
        activation_ops = total_ops - weight_ops
        lines.append(f"- Operations: {total_ops:,} total ({weight_ops:,} weight, {activation_ops:,} activation)")

        return "\n".join(lines) + "\n"

    def _format_peak_memory_table(self) -> str:
        """Format peak memory analysis as markdown table"""
        peak_analysis = self._analyze_peaks()
        if not peak_analysis:
            return ""

        lines = ["## Peak Memory"]
        lines.append("| Type | Op# | Operation | Location | Allocated | Capacity | Utilization |")
        lines.append("|------|-----|-----------|----------|-----------|----------|-------------|")

        for mem_type in ["DRAM", "L1", "L1_SMALL", "TRACE"]:
            if mem_type not in peak_analysis:
                continue
            data = peak_analysis[mem_type]
            op = data["operation"]
            mem = data["memory"]
            idx = data["index"]

            allocated = mem["totalBytesAllocatedPerBank_MB"]
            capacity = mem["totalBytesPerBank_MB"]
            utilization = (allocated / capacity * 100) if capacity > 0 else 0

            loc = self._truncate_loc(op.get("loc", ""))
            mlir_op = op.get("mlir_op", "unknown")

            lines.append(
                f"| {mem_type} | {idx} | {mlir_op} | {loc} | "
                f"{allocated:.2f} MB | {capacity:.2f} MB | {utilization:.1f}% |"
            )

        return "\n".join(lines) + "\n"

    def _format_top_consumers_table(self, n: int = 10) -> str:
        """Format top memory consumers as markdown table"""
        top_ops = self._get_top_operations(n=n)
        if not top_ops:
            return ""

        lines = [f"## Top {n} Memory Consumers (DRAM)"]
        lines.append("| Rank | Op# | Operation | Location | DRAM (MB) | Input | Output |")
        lines.append("|------|-----|-----------|----------|-----------|-------|--------|")

        for rank, item in enumerate(top_ops, 1):
            op = item["operation"]
            dram = item["dram"]
            idx = item["index"]

            loc = self._truncate_loc(op.get("loc", ""))
            mlir_op = op.get("mlir_op", "unknown")

            # Format shapes
            input_shapes = op.get("input_shapes", [])
            output_shapes = op.get("output_shapes", [])
            input_str = ", ".join(s for s in input_shapes if s) if input_shapes else "N/A"
            output_str = ", ".join(s for s in output_shapes if s) if output_shapes else "N/A"

            lines.append(
                f"| {rank} | {idx} | {mlir_op} | {loc} | {dram:.2f} | {input_str} | {output_str} |"
            )

        return "\n".join(lines) + "\n"

    def _format_padding_overhead_table(self, n: int = 10) -> str:
        """Format top padding overhead operations as markdown table"""
        top_padding_ops = self._get_top_padding_overhead_ops(n=n)
        if not top_padding_ops:
            return ""

        lines = [f"## Top {n} Padding Overhead"]
        lines.append("| Rank | Op# | Operation | Logical | Padded | Overhead |")
        lines.append("|------|-----|-----------|---------|--------|----------|")

        for rank, item in enumerate(top_padding_ops, 1):
            op = item["operation"]
            layout = item["layout_info"]
            idx = item["index"]

            mlir_op = op.get("mlir_op", "unknown")
            logical_shape = "x".join(str(d) for d in layout.get("logical_shape", []))
            padded_shape = "x".join(str(d) for d in layout.get("padded_shape", []))
            dtype = layout.get("dtype", "?")

            unpadded_bytes = layout.get("unpadded_bytes", 0)
            padded_bytes = layout.get("padded_bytes", 0)
            overhead_pct = layout.get("overhead_pct", 0)
            absolute_overhead = padded_bytes - unpadded_bytes
            overhead_mb = absolute_overhead / (1024 * 1024)

            lines.append(
                f"| {rank} | {idx} | {mlir_op} | {logical_shape} ({dtype}) | "
                f"{padded_shape} | {overhead_mb:.2f} MB ({overhead_pct:.1f}%) |"
            )

        return "\n".join(lines) + "\n"

    def _format_weights_registry(self) -> str:
        """Format model weights registry as markdown table"""
        if not self.registry or not self.registry.get("entries"):
            return ""

        # Filter to parameters and constants only
        weight_entries = [
            e for e in self.registry["entries"]
            if e.get("type") in ("parameter", "constant")
        ]
        if not weight_entries:
            return ""

        # Sort by size descending
        weight_entries.sort(key=lambda e: e.get("bytes", 0), reverse=True)

        total_mb = self.registry.get("metadata", {}).get("total_weight_MB", 0)
        lines = [f"## Model Weights ({total_mb:.1f} MB total)"]
        lines.append("| Name | Shape | Type | Size |")
        lines.append("|------|-------|------|------|")

        for entry in weight_entries[:20]:  # Limit to top 20 weights
            name = entry.get("name", "unknown")
            shape = "x".join(str(d) for d in entry.get("shape", []))
            dtype = entry.get("dtype", "?")
            size_bytes = entry.get("bytes", 0)

            if size_bytes >= 1024 * 1024:
                size_str = f"{size_bytes / (1024*1024):.2f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"

            lines.append(f"| {name} | {shape} | {dtype} | {size_str} |")

        if len(weight_entries) > 20:
            lines.append(f"\n*... and {len(weight_entries) - 20} more weights*")

        return "\n".join(lines) + "\n"

    def _format_operation_distribution(self) -> str:
        """Format operation type distribution as bullet list"""
        op_distribution = self._get_op_distribution()
        if not op_distribution:
            return ""

        total_ops = sum(op_distribution.values())
        # Sort by count descending
        sorted_ops = sorted(op_distribution.items(), key=lambda x: x[1], reverse=True)

        lines = ["## Operation Distribution"]
        for op_name, count in sorted_ops[:15]:  # Top 15 op types
            pct = (count / total_ops * 100) if total_ops > 0 else 0
            lines.append(f"- {op_name}: {count:,} ({pct:.1f}%)")

        if len(sorted_ops) > 15:
            remaining = sum(c for _, c in sorted_ops[15:])
            lines.append(f"- *other*: {remaining:,}")

        return "\n".join(lines) + "\n"

    # === Analysis methods (adapted from visualizer.py) ===

    def _compute_summary_stats(self) -> Dict:
        """Compute summary statistics"""
        stats = {"total_ops": len(self.mem_data), "memory_types": {}}

        for mem_type in self.available_memory_types:
            allocated_values = [
                op["memory"][mem_type]["totalBytesAllocatedPerBank_MB"]
                for op in self.mem_data
            ]

            stats["memory_types"][mem_type] = {
                "peak": max(allocated_values),
                "min": min(allocated_values),
                "avg": sum(allocated_values) / len(allocated_values),
                "capacity": self.mem_data[0]["memory"][mem_type]["totalBytesPerBank_MB"],
            }

        return stats

    def _analyze_peaks(self) -> Dict:
        """Analyze peak memory usage for each type"""
        peaks = {}

        for mem_type in self.available_memory_types:
            peak_idx = max(
                range(len(self.mem_data)),
                key=lambda i: self.mem_data[i]["memory"][mem_type][
                    "totalBytesAllocatedPerBank_MB"
                ],
            )

            peaks[mem_type] = {
                "index": peak_idx,
                "memory": self.mem_data[peak_idx]["memory"][mem_type],
                "operation": self.ops_data[peak_idx] if peak_idx < len(self.ops_data) else {},
            }

        return peaks

    def _get_top_operations(self, n: int = 10) -> List[Dict]:
        """Get top N memory-consuming operations (by DRAM)"""
        if "DRAM" not in self.available_memory_types:
            return []

        ops_with_mem = [
            {
                "index": i,
                "dram": self.mem_data[i]["memory"]["DRAM"]["totalBytesAllocatedPerBank_MB"],
                "operation": self.ops_data[i] if i < len(self.ops_data) else {},
                "memory": self.mem_data[i],
            }
            for i in range(len(self.mem_data))
        ]

        ops_with_mem.sort(key=lambda x: x["dram"], reverse=True)
        return ops_with_mem[:n]

    def _get_top_padding_overhead_ops(self, n: int = 10) -> List[Dict]:
        """Get top N operations with highest absolute tile padding overhead"""
        ops_with_overhead = []

        for i, op in enumerate(self.ops_data):
            layout_info = op.get("output_layout_info")
            if not layout_info:
                continue

            overhead_pct = layout_info.get("overhead_pct", 0)
            if overhead_pct <= 0:
                continue

            padded_bytes = layout_info.get("padded_bytes", 0)
            unpadded_bytes = layout_info.get("unpadded_bytes", 0)
            absolute_overhead = padded_bytes - unpadded_bytes

            ops_with_overhead.append({
                "index": i,
                "operation": op,
                "layout_info": layout_info,
                "overhead_pct": overhead_pct,
                "absolute_overhead_bytes": absolute_overhead,
            })

        ops_with_overhead.sort(key=lambda x: x["absolute_overhead_bytes"], reverse=True)
        return ops_with_overhead[:n]

    def _get_op_distribution(self) -> Dict:
        """Get operation type distribution"""
        op_counts = {}
        for op in self.ops_data:
            op_name = op.get("mlir_op", "unknown").split(".")[-1]
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
        return op_counts

    def _calculate_peak_padding_overhead(self) -> Dict:
        """Calculate peak tile padding overhead from memory data"""
        if not self.mem_data or not self.mem_data[0].get("unpadded_memory"):
            return {"dram_pct": 0, "l1_pct": 0, "has_data": False}

        peak_dram_pct = 0
        peak_l1_pct = 0

        for op in self.mem_data:
            unpadded = op.get("unpadded_memory", {})
            if unpadded:
                dram = unpadded.get("DRAM", {})
                l1 = unpadded.get("L1", {})
                peak_dram_pct = max(peak_dram_pct, dram.get("overhead_pct", 0))
                peak_l1_pct = max(peak_l1_pct, l1.get("overhead_pct", 0))

        return {
            "dram_pct": peak_dram_pct,
            "l1_pct": peak_l1_pct,
            "has_data": True,
        }
