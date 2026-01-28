# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive HTML memory visualization generator using Plotly.

Generates comprehensive memory profiling reports with:
- Memory usage over time graphs (const_eval ops shown in red)
- Fragmentation analysis
- Peak memory analysis
- Top memory-consuming operations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class MemoryVisualizer:
    """Generate interactive HTML visualization reports from memory profiler output"""

    def __init__(self, run_dir: Path):
        """
        Initialize visualizer with a profiler run directory.

        Args:
            run_dir: Path to profiler output directory containing JSON files
        """
        self.run_dir = Path(run_dir)

        # Determine script name from directory name
        # Format is: {script_name}_{timestamp} where timestamp is YYYYMMDD_HHMMSS
        # So we join all parts except the last two (date and time)
        parts = self.run_dir.name.split("_")
        if len(parts) >= 3:
            script_name = "_".join(parts[:-2])
        else:
            script_name = parts[0]

        self.mem_file = self.run_dir / f"{script_name}_memory.json"
        self.ops_file = self.run_dir / f"{script_name}_operations.json"
        self.registry_file = self.run_dir / f"{script_name}_inputs_registry.json"

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
            # Old format - list of operations
            self.mem_metadata = None
            self.mem_data = mem_json

        # Detect available memory types - only include types present in ALL operations
        self.available_memory_types = []
        if self.mem_data:
            for mt in ["DRAM", "L1", "L1_SMALL", "TRACE"]:
                if all(mt in op.get("memory", {}) for op in self.mem_data):
                    self.available_memory_types.append(mt)

    def generate_report(self, output_path: Path = None) -> Path:
        """
        Generate complete HTML visualization report.

        Args:
            output_path: Optional custom output path. Defaults to <script>_report.html

        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            # Determine script name from directory name
            parts = self.run_dir.name.split("_")
            if len(parts) >= 3:
                script_name = "_".join(parts[:-2])
            else:
                script_name = parts[0]
            output_path = self.run_dir / f"{script_name}_report.html"

        # Generate all components
        summary_stats = self.compute_summary_stats()
        peak_analysis = self.analyze_peaks()
        top_ops = self.get_top_operations(n=10)

        # Build HTML
        html = self._build_html(
            summary_stats=summary_stats, peak_analysis=peak_analysis, top_ops=top_ops
        )

        output_path.write_text(html)
        return output_path

    def _build_html(
        self, summary_stats: Dict, peak_analysis: Dict, top_ops: List[Dict]
    ) -> str:
        """Build complete HTML document with embedded Plotly graphs"""

        # Prepare data for JavaScript
        memory_graph_data = self._prepare_memory_graph_data()
        fragmentation_data = self._prepare_fragmentation_data()
        weight_activation_data = self._prepare_weight_activation_data()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Profile: {self.run_dir.name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .metadata {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            color: #666;
            font-size: 14px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        }}
        .summary-card.green {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .summary-card.blue {{
            background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        }}
        .summary-card.orange {{
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }}
        .summary-card .label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .graph-container {{
            margin: 20px 0;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .peak-card {{
            background: #fff;
            border-left: 4px solid #4CAF50;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .peak-card h3 {{
            margin-top: 0;
            color: #4CAF50;
        }}
        .peak-card table {{
            width: 100%;
            margin-top: 10px;
        }}
        .peak-card td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
        }}
        .peak-card td:first-child {{
            font-weight: bold;
            width: 200px;
            color: #666;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        table.data-table th {{
            background: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        table.data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        table.data-table tr:hover {{
            background: #f5f5f5;
        }}
        table.data-table tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .code {{
            font-family: 'Courier New', monospace;
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .badge.dram {{ background: #ff9800; color: white; }}
        .badge.l1 {{ background: #2196F3; color: white; }}
        .badge.l1-small {{ background: #9C27B0; color: white; }}
        .badge.trace {{ background: #607D8B; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Memory Profiling Report</h1>
        <div class="metadata">
            <strong>Run:</strong> {self.run_dir.name}<br>
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Total Operations:</strong> {len(self.mem_data)} (deallocations excluded)<br>
            {self._format_memory_config()}
        </div>

        <!-- Summary Statistics -->
        <h2>Summary Statistics</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">Total Operations</div>
                <div class="value">{summary_stats['total_ops']}</div>
            </div>
            <div class="summary-card green">
                <div class="label">Peak DRAM Usage</div>
                <div class="value">{summary_stats['memory_types']['DRAM']['peak']:.1f} MB</div>
            </div>
            <div class="summary-card blue">
                <div class="label">Peak L1 Usage</div>
                <div class="value">{summary_stats['memory_types']['L1']['peak']:.2f} MB</div>
            </div>
            <div class="summary-card orange">
                <div class="label">Avg DRAM Usage</div>
                <div class="value">{summary_stats['memory_types']['DRAM']['avg']:.1f} MB</div>
            </div>
            {self._format_weight_summary_card()}
        </div>

        <!-- Memory Usage Over Time -->
        <h2>Memory Usage Over Time</h2>
        <div class="graph-container">
            <div id="memory-graphs"></div>
        </div>

        <!-- Memory Fragmentation -->
        <h2>Memory Fragmentation Analysis</h2>
        <div class="graph-container">
            <div id="fragmentation-graph"></div>
        </div>

        {self._format_weight_activation_section()}

        <!-- Peak Memory Analysis -->
        <h2>Peak Memory Analysis</h2>
        {self._generate_peak_cards_html(peak_analysis)}

        <!-- Top Memory Consumers -->
        <h2>Top 10 Memory-Consuming Operations</h2>
        {self._generate_top_ops_table_html(top_ops)}

    </div>

    <script>
        // Memory usage graphs data
        const memoryData = {json.dumps(memory_graph_data)};
        const fragmentationData = {json.dumps(fragmentation_data)};
        const weightActivationData = {json.dumps(weight_activation_data)};

        // Create memory usage over time graphs
        Plotly.newPlot('memory-graphs', memoryData.traces, memoryData.layout, {{responsive: true}});

        // Create fragmentation graphs
        Plotly.newPlot('fragmentation-graph', fragmentationData.traces, fragmentationData.layout, {{responsive: true}});

        // Create weight/activation breakdown graph if data available
        if (weightActivationData && weightActivationData.traces && weightActivationData.traces.length > 0) {{
            Plotly.newPlot('weight-activation-graph', weightActivationData.traces, weightActivationData.layout, {{responsive: true}});
        }}
    </script>
</body>
</html>"""
        return html

    def _prepare_memory_graph_data(self) -> Dict:
        """Prepare data for memory usage over time graphs"""
        memory_types = self.available_memory_types
        traces = []

        # Collect all data points with their weight op status and op details
        all_indices = []
        all_allocated = {mt: [] for mt in memory_types}
        weight_op_flags = []
        op_names = []
        input_shapes_list = []
        output_shapes_list = []

        for i, op in enumerate(self.mem_data):
            # Use is_weight_op field (includes const_eval and direct weight inputs)
            is_weight_op = op.get("is_weight_op", False)
            # Fallback: also check ops_data for is_weight_op if not in mem_data
            if not is_weight_op and i < len(self.ops_data):
                is_weight_op = self.ops_data[i].get("is_weight_op", False)
            idx = op["index"]
            all_indices.append(idx)
            weight_op_flags.append(is_weight_op)
            for mt in memory_types:
                all_allocated[mt].append(
                    op["memory"][mt]["totalBytesAllocatedPerBank_MB"]
                )

            # Get op name and shapes from ops_data
            if i < len(self.ops_data):
                op_info = self.ops_data[i]
                op_names.append(op_info.get("mlir_op", "unknown"))
                in_shapes = op_info.get("input_shapes", [])
                out_shapes = op_info.get("output_shapes", [])
                input_shapes_list.append(
                    ", ".join(s for s in in_shapes if s) if in_shapes else "N/A"
                )
                output_shapes_list.append(
                    ", ".join(s for s in out_shapes if s) if out_shapes else "N/A"
                )
            else:
                op_names.append(op.get("mlir_op", "unknown"))
                input_shapes_list.append("N/A")
                output_shapes_list.append("N/A")

        # Separate weight operation indices for red markers overlay
        weight_op_indices = [
            idx for idx, flag in zip(all_indices, weight_op_flags) if flag
        ]
        weight_op_allocated = {mt: [] for mt in memory_types}
        weight_op_names = []
        weight_input_shapes = []
        weight_output_shapes = []
        for i, flag in enumerate(weight_op_flags):
            if flag:
                for mt in memory_types:
                    weight_op_allocated[mt].append(all_allocated[mt][i])
                weight_op_names.append(op_names[i])
                weight_input_shapes.append(input_shapes_list[i])
                weight_output_shapes.append(output_shapes_list[i])

        capacity = {
            mt: self.mem_data[0]["memory"][mt]["totalBytesPerBank_MB"]
            for mt in memory_types
        }

        for mem_type in memory_types:
            axis_idx = memory_types.index(mem_type) + 1

            # Main line connecting all points (blue)
            traces.append(
                {
                    "x": all_indices,
                    "y": all_allocated[mem_type],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": f"{mem_type}",
                    "line": {"width": 2, "color": "#1f77b4"},
                    "marker": {"size": 3, "color": "#1f77b4"},
                    "xaxis": f"x{axis_idx}" if axis_idx > 1 else "x",
                    "yaxis": f"y{axis_idx}" if axis_idx > 1 else "y",
                    "showlegend": (mem_type == "DRAM"),
                    "legendgroup": "main",
                    "customdata": list(
                        zip(op_names, input_shapes_list, output_shapes_list)
                    ),
                    "hovertemplate": f"{mem_type}<br>Op %{{x}}: %{{customdata[0]}}<br>Allocated: %{{y:.2f}} MB<br>Input: %{{customdata[1]}}<br>Output: %{{customdata[2]}}<extra></extra>",
                }
            )

            # Weight operations overlay (red markers on top)
            # Includes const_eval operations and operations with direct weight inputs
            if weight_op_indices:
                traces.append(
                    {
                        "x": weight_op_indices,
                        "y": weight_op_allocated[mem_type],
                        "type": "scatter",
                        "mode": "markers",
                        "name": "weight_ops",
                        "marker": {"size": 5, "color": "red", "symbol": "circle"},
                        "xaxis": f"x{axis_idx}" if axis_idx > 1 else "x",
                        "yaxis": f"y{axis_idx}" if axis_idx > 1 else "y",
                        "showlegend": (mem_type == "DRAM"),
                        "legendgroup": "weight_ops",
                        "customdata": list(
                            zip(
                                weight_op_names,
                                weight_input_shapes,
                                weight_output_shapes,
                            )
                        ),
                        "hovertemplate": f"{mem_type} (weight op)<br>Op %{{x}}: %{{customdata[0]}}<br>Allocated: %{{y:.2f}} MB<br>Input: %{{customdata[1]}}<br>Output: %{{customdata[2]}}<extra></extra>",
                    }
                )

            # Capacity line
            traces.append(
                {
                    "x": [all_indices[0], all_indices[-1]],
                    "y": [capacity[mem_type], capacity[mem_type]],
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"{mem_type} Capacity",
                    "line": {"dash": "dash", "color": "gray", "width": 1},
                    "xaxis": f"x{axis_idx}" if axis_idx > 1 else "x",
                    "yaxis": f"y{axis_idx}" if axis_idx > 1 else "y",
                    "showlegend": False,
                    "hovertemplate": f"Capacity: %{{y:.2f}} MB<extra></extra>",
                }
            )

        # Build dynamic layout based on available memory types
        num_types = len(memory_types)
        row_height = 0.225
        gap = 0.05
        layout = {
            "height": 300 * num_types,
            "showlegend": True,
            "title": {
                "text": "Memory Usage Across Operation Execution",
                "font": {"size": 18},
            },
            "grid": {"rows": num_types, "columns": 1, "pattern": "independent"},
        }

        for i, mem_type in enumerate(memory_types):
            axis_idx = i + 1
            xkey = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
            ykey = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
            # Calculate domain from top to bottom
            top = 1.0 - i * (row_height + gap)
            bottom = top - row_height
            layout[xkey] = {
                "title": "Operation Index",
                "anchor": f"y{axis_idx}" if axis_idx > 1 else "y",
            }
            layout[ykey] = {
                "title": f"{mem_type} (MB/bank)",
                "domain": [max(0, bottom), top],
            }

        return {"traces": traces, "layout": layout}

    def _prepare_fragmentation_data(self) -> Dict:
        """Prepare data for fragmentation visualization"""
        memory_types = self.available_memory_types
        traces = []

        for idx, mem_type in enumerate(memory_types):
            indices = [op["index"] for op in self.mem_data]
            allocated = [
                op["memory"][mem_type]["totalBytesAllocatedPerBank_MB"]
                for op in self.mem_data
            ]
            free = [
                op["memory"][mem_type]["totalBytesFreePerBank_MB"]
                for op in self.mem_data
            ]

            axis_idx = idx + 1
            # Allocated (filled area)
            traces.append(
                {
                    "x": indices,
                    "y": allocated,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Allocated",
                    "stackgroup": f"one{mem_type}",
                    "fillcolor": "rgba(255, 87, 87, 0.7)",
                    "line": {"width": 0},
                    "xaxis": f"x{axis_idx}" if axis_idx > 1 else "x",
                    "yaxis": f"y{axis_idx}" if axis_idx > 1 else "y",
                    "showlegend": (mem_type == memory_types[0]),
                    "hovertemplate": f"Allocated: %{{y:.2f}} MB<extra></extra>",
                }
            )

            # Free (filled area)
            traces.append(
                {
                    "x": indices,
                    "y": free,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Free",
                    "stackgroup": f"one{mem_type}",
                    "fillcolor": "rgba(76, 175, 80, 0.5)",
                    "line": {"width": 0},
                    "xaxis": f"x{axis_idx}" if axis_idx > 1 else "x",
                    "yaxis": f"y{axis_idx}" if axis_idx > 1 else "y",
                    "showlegend": (mem_type == memory_types[0]),
                    "hovertemplate": f"Free: %{{y:.2f}} MB<extra></extra>",
                }
            )

        # Build dynamic layout
        num_types = len(memory_types)
        row_height = 0.225
        gap = 0.05
        layout = {
            "height": 250 * num_types,
            "showlegend": True,
            "title": {"text": "Memory Allocation vs Free Space", "font": {"size": 18}},
            "grid": {"rows": num_types, "columns": 1, "pattern": "independent"},
        }

        for i, mem_type in enumerate(memory_types):
            axis_idx = i + 1
            xkey = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
            ykey = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
            top = 1.0 - i * (row_height + gap)
            bottom = top - row_height
            layout[xkey] = {
                "title": "Operation Index",
                "anchor": f"y{axis_idx}" if axis_idx > 1 else "y",
            }
            layout[ykey] = {
                "title": f"{mem_type} (MB/bank)",
                "domain": [max(0, bottom), top],
            }

        return {"traces": traces, "layout": layout}

    def get_op_distribution(self) -> Dict:
        """Get operation type distribution"""
        op_counts = {}
        for op in self.ops_data:
            op_name = op["mlir_op"].split(".")[-1]
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

        return op_counts

    def _format_memory_config(self) -> str:
        """Format memory configuration for display in metadata section"""
        if not self.mem_metadata or "memory_config" not in self.mem_metadata:
            return ""

        config = self.mem_metadata["memory_config"]
        parts = []
        for mem_type in ["DRAM", "L1", "L1_SMALL", "TRACE"]:
            if mem_type in config:
                num_banks = config[mem_type].get("num_banks", 0)
                capacity = config[mem_type].get("total_bytes_per_bank_MB", 0)
                parts.append(f"{mem_type}: {num_banks} banks × {capacity:.2f} MB/bank")

        if parts:
            return f"<strong>Memory Configuration:</strong> {' | '.join(parts)}"
        return ""

    def _format_shapes_with_dtypes(self, shapes: List, dtypes: List) -> str:
        """Format shapes with their dtypes for display"""
        if not shapes:
            return "N/A"
        parts = []
        for i, shape in enumerate(shapes):
            dtype = dtypes[i] if i < len(dtypes) else "?"
            if shape:
                parts.append(f"{shape} ({dtype})")
            else:
                parts.append(f"scalar ({dtype})")
        return ", ".join(parts)

    def _generate_peak_cards_html(self, peak_analysis: Dict) -> str:
        """Generate HTML for peak memory analysis cards"""
        html_parts = []

        colors = {
            "DRAM": "#ff9800",
            "L1": "#2196F3",
            "L1_SMALL": "#9C27B0",
            "TRACE": "#607D8B",
        }

        for mem_type, data in peak_analysis.items():
            color = colors.get(mem_type, "#999999")
            op = data["operation"]
            mem = data["memory"]
            peak_val = mem["totalBytesAllocatedPerBank_MB"]

            # Format input and output shapes with dtypes
            input_shapes = op.get("input_shapes", [])
            input_dtypes = op.get("input_dtypes", [])
            output_shapes = op.get("output_shapes", [])
            output_dtypes = op.get("output_dtypes", [])

            input_str = self._format_shapes_with_dtypes(input_shapes, input_dtypes)
            output_str = self._format_shapes_with_dtypes(output_shapes, output_dtypes)

            html_parts.append(
                f"""
        <div class="peak-card" style="border-left-color: {color};">
            <h3><span class="badge {mem_type.lower().replace('_', '-')}">{mem_type}</span> Peak: {peak_val:.2f} MB/bank at Operation #{data['index']}</h3>
            <table>
                <tr><td>Operation</td><td><span class="code">{op['mlir_op']}</span></td></tr>
                <tr><td>Location</td><td><span class="code">{op['loc']}</span></td></tr>
                <tr><td>Input Shapes</td><td><span class="code">{input_str}</span></td></tr>
                <tr><td>Output Shapes</td><td><span class="code">{output_str}</span></td></tr>
                <tr><td>Attributes</td><td><span class="code">{op['attributes'] if op['attributes'] else 'None'}</span></td></tr>
                <tr><td>Free Space</td><td>{mem['totalBytesFreePerBank_MB']:.2f} MB/bank</td></tr>
                <tr><td>Largest Contiguous Free</td><td>{mem['largestContiguousBytesFreePerBank_MB']:.2f} MB/bank</td></tr>
            </table>
        </div>"""
            )

        return "\n".join(html_parts)

    def _generate_top_ops_table_html(self, top_ops: List[Dict]) -> str:
        """Generate HTML table for top operations"""
        rows = []
        for rank, item in enumerate(top_ops, 1):
            op = item["operation"]
            dram = item["dram"]
            idx = item["index"]

            # Format input and output shapes with dtypes
            input_shapes = op.get("input_shapes", [])
            input_dtypes = op.get("input_dtypes", [])
            output_shapes = op.get("output_shapes", [])
            output_dtypes = op.get("output_dtypes", [])

            input_str = self._format_shapes_with_dtypes(input_shapes, input_dtypes)
            output_str = self._format_shapes_with_dtypes(output_shapes, output_dtypes)

            rows.append(
                f"""
            <tr>
                <td>{rank}</td>
                <td>{idx}</td>
                <td><span class="code">{op['mlir_op']}</span></td>
                <td><span class="code">{op['loc']}</span></td>
                <td>{dram:.2f}</td>
                <td><span class="code">{input_str}</span></td>
                <td><span class="code">{output_str}</span></td>
            </tr>"""
            )

        return f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Index</th>
                    <th>Operation</th>
                    <th>Location</th>
                    <th>DRAM (MB)</th>
                    <th>Input Shapes</th>
                    <th>Output Shapes</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>"""

    def compute_summary_stats(self) -> Dict:
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
                "capacity": self.mem_data[0]["memory"][mem_type][
                    "totalBytesPerBank_MB"
                ],
            }

        return stats

    def analyze_peaks(self) -> Dict:
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
                "operation": self.ops_data[peak_idx],
            }

        return peaks

    def get_top_operations(self, n: int = 10) -> List[Dict]:
        """Get top N memory-consuming operations (by DRAM)"""
        ops_with_mem = [
            {
                "index": i,
                "dram": self.mem_data[i]["memory"]["DRAM"][
                    "totalBytesAllocatedPerBank_MB"
                ],
                "operation": self.ops_data[i],
                "memory": self.mem_data[i],
            }
            for i in range(len(self.mem_data))
        ]

        ops_with_mem.sort(key=lambda x: x["dram"], reverse=True)
        return ops_with_mem[:n]

    def _format_weight_summary_card(self) -> str:
        """Format weight summary card if registry is available"""
        if not self.registry or not self.registry.get("metadata"):
            return ""

        total_weight_mb = self.registry["metadata"].get("total_weight_MB", 0)
        return f"""
            <div class="summary-card" style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);">
                <div class="label">Model Weights</div>
                <div class="value">{total_weight_mb:.2f} MB</div>
            </div>"""

    def _format_weight_activation_section(self) -> str:
        """Format the weight/activation breakdown section if registry is available"""
        if not self.registry or not self.registry.get("entries"):
            return ""

        return """
        <!-- Weight vs Activation Memory Breakdown -->
        <h2>Weight vs Activation Memory Breakdown (DRAM)</h2>
        <div class="graph-container">
            <div id="weight-activation-graph"></div>
        </div>"""

    def _prepare_weight_activation_data(self) -> Dict:
        """Prepare stacked area chart: weights (blue) + activations (orange) + free (green)

        This visualization shows memory breakdown based on operation type:
        - Weight operations (is_weight_op=True): Memory used for processing model weights
        - Activation operations: Memory used for processing activations/intermediate tensors

        Note: Weight operations include const_eval and operations with direct weight inputs.
        The memory shown is the DRAM allocation at each step, categorized by operation type.
        """
        if not self.registry or not self.registry.get("entries"):
            return {"traces": [], "layout": {}}

        # Handle empty mem_data
        if not self.mem_data:
            return {"traces": [], "layout": {}}

        # Calculate total declared weight memory from registry (for display)
        weight_bytes = sum(
            e["bytes"]
            for e in self.registry["entries"]
            if e["type"] in ("parameter", "constant")
        )
        total_weight_MB = weight_bytes / (1024 * 1024)

        # Get DRAM capacity from first operation
        capacity_MB = self.mem_data[0]["memory"]["DRAM"]["totalBytesPerBank_MB"]

        # For each operation, track allocated memory
        indices = [op["index"] for op in self.mem_data]
        total_allocated = [
            op["memory"]["DRAM"]["totalBytesAllocatedPerBank_MB"]
            for op in self.mem_data
        ]

        # Track memory by operation type
        weight_op_memory = []
        activation_op_memory = []

        # First pass: find all weight operations and their indices
        weight_op_indices = set()
        for i, op in enumerate(self.mem_data):
            is_weight_op = op.get("is_weight_op", False)
            if not is_weight_op and i < len(self.ops_data):
                is_weight_op = self.ops_data[i].get("is_weight_op", False)
            if is_weight_op:
                weight_op_indices.add(i)

        # Find the memory baseline after all weight loading completes
        # This is the memory at the first non-weight operation
        weight_baseline = 0
        for i in range(len(self.mem_data)):
            if i not in weight_op_indices:
                # First activation op - memory here is what remains after weight loading
                weight_baseline = total_allocated[i]
                break

        # Second pass: categorize memory and collect op info for hover
        op_names = []
        for i, op in enumerate(self.mem_data):
            is_weight_op = i in weight_op_indices
            alloc = total_allocated[i]

            if is_weight_op:
                # During weight operation, all current memory is for weight processing
                weight_op_memory.append(alloc)
                activation_op_memory.append(0)
            else:
                # For activation ops:
                # - Weight baseline is what remains loaded after weight ops finish
                # - Everything above that is activation memory
                weight_op_memory.append(min(weight_baseline, alloc))
                activation_op_memory.append(max(0, alloc - weight_baseline))

            # Get op name for hover
            if i < len(self.ops_data):
                op_names.append(self.ops_data[i].get("mlir_op", "unknown"))
            else:
                op_names.append(op.get("mlir_op", "unknown"))

        free_values = [capacity_MB - alloc for alloc in total_allocated]

        # Build customdata for hover: [op_name, total_alloc, weight_mem, act_mem, free_mem, is_weight_op]
        is_weight_flags = [i in weight_op_indices for i in range(len(self.mem_data))]
        customdata = list(
            zip(
                op_names,
                total_allocated,
                weight_op_memory,
                activation_op_memory,
                free_values,
                is_weight_flags,
            )
        )

        traces = [
            {
                "x": indices,
                "y": weight_op_memory,
                "type": "scatter",
                "mode": "lines",
                "name": f"Persistent Weights ({weight_baseline:.1f} MB baseline)",
                "stackgroup": "one",
                "fillcolor": "rgba(30, 60, 114, 0.7)",  # Blue
                "line": {"width": 0},
                "customdata": customdata,
                "hovertemplate": "Op %{x}: %{customdata[0]}<br>Total Allocated: %{customdata[1]:.2f} MB<br>Persistent Weights: %{customdata[2]:.2f} MB<br>Activations: %{customdata[3]:.2f} MB<br>Free: %{customdata[4]:.2f} MB<extra></extra>",
            },
            {
                "x": indices,
                "y": activation_op_memory,
                "type": "scatter",
                "mode": "lines",
                "name": "Activations (above baseline)",
                "stackgroup": "one",
                "fillcolor": "rgba(255, 165, 0, 0.7)",  # Orange
                "line": {"width": 0},
                "hoverinfo": "skip",  # Skip hover for this trace since first trace shows all info
            },
            {
                "x": indices,
                "y": free_values,
                "type": "scatter",
                "mode": "lines",
                "name": "Free",
                "stackgroup": "one",
                "fillcolor": "rgba(76, 175, 80, 0.5)",  # Green
                "line": {"width": 0},
                "hoverinfo": "skip",  # Skip hover for this trace since first trace shows all info
            },
        ]

        layout = {
            "height": 400,
            "showlegend": True,
            "title": {
                "text": f"DRAM Memory Breakdown (Declared Weights: {total_weight_MB:.1f} MB, Persistent Baseline: {weight_baseline:.1f} MB)",
                "font": {"size": 16},
            },
            "xaxis": {"title": "Operation Index"},
            "yaxis": {"title": "Memory (MB/bank)", "rangemode": "tozero"},
            "hovermode": "x",
        }

        return {"traces": traces, "layout": layout}
