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

    def __init__(self, run_dir: Path, script_name: str = None):
        """
        Initialize visualizer with a profiler run directory.

        Args:
            run_dir: Path to profiler output directory containing JSON files
            script_name: Optional explicit script name. If not provided, inferred from directory name.
        """
        self.run_dir = Path(run_dir)

        # Use provided script_name or infer from directory name
        # Format is: {script_name}_{timestamp} where timestamp is YYYYMMDD_HHMMSS
        # So we join all parts except the last two (date and time)
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
        self.ir_file = self.run_dir / f"{self.script_name}_ir.json"

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

        # Load IR data if it exists
        self.ir_data = None
        if self.ir_file.exists():
            with open(self.ir_file) as f:
                self.ir_data = json.load(f)

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
            output_path = self.run_dir / f"{self.script_name}_report.html"

        # Generate all components
        summary_stats = self.compute_summary_stats()
        peak_analysis = self.analyze_peaks()
        top_ops = self.get_top_operations(n=10)
        top_padding_ops = self.get_top_padding_overhead_ops(n=10)
        peak_padding_overhead = self._calculate_peak_padding_overhead()

        # Build HTML
        html = self._build_html(
            summary_stats=summary_stats,
            peak_analysis=peak_analysis,
            top_ops=top_ops,
            top_padding_ops=top_padding_ops,
            peak_padding_overhead=peak_padding_overhead,
        )

        output_path.write_text(html)
        return output_path

    def _has_ir_data(self) -> bool:
        """Check if IR data is available and non-empty."""
        if not self.ir_data:
            return False
        ttir = self.ir_data.get("ttir", {})
        ttnn = self.ir_data.get("ttnn", {})
        return bool(ttir.get("text") or ttnn.get("text"))

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _format_op_link(self, mlir_op: str, loc: str) -> str:
        """Format an operation name as a clickable link to IR if loc is available."""
        if not loc or not self._has_ir_data():
            return f'<span class="code">{self._escape_html(mlir_op)}</span>'

        # Make the operation clickable - links to TTNN by default (most useful)
        return f'<a href="#" class="op-link code" data-loc="{self._escape_html(loc)}" onclick="navigateToIR(\'{self._escape_html(loc)}\', \'ttnn\'); return false;">{self._escape_html(mlir_op)}</a>'

    def _generate_ir_html(self, ir_name: str) -> str:
        """Generate HTML for displaying an IR module with line numbers."""
        if not self.ir_data:
            return '<div class="ir-empty">No IR data available</div>'

        ir_info = self.ir_data.get(ir_name, {})
        ir_text = ir_info.get("text", "")

        if not ir_text:
            return f'<div class="ir-empty">No {ir_name.upper()} IR data available</div>'

        lines = ir_text.split("\n")
        html_lines = []

        for line_num, line in enumerate(lines, start=1):
            escaped_line = self._escape_html(line)
            # Add id for scrolling to specific lines
            html_lines.append(
                f'<div class="ir-line" id="{ir_name}-line-{line_num}">'
                f'<span class="line-num">{line_num}</span>'
                f'<span class="line-content">{escaped_line}</span>'
                f'</div>'
            )

        return "\n".join(html_lines)

    def _build_html(
        self,
        summary_stats: Dict,
        peak_analysis: Dict,
        top_ops: List[Dict],
        top_padding_ops: List[Dict] = None,
        peak_padding_overhead: Dict = None,
    ) -> str:
        """Build complete HTML document with embedded Plotly graphs and IR viewer"""

        # Prepare data for JavaScript
        memory_graph_data = self._prepare_memory_graph_data()
        unpadded_comparison_data = self._prepare_unpadded_comparison_data()

        # Prepare IR location indices for JavaScript
        ir_loc_index = {"ttir": {}, "ttnn": {}}
        if self.ir_data:
            ir_loc_index["ttir"] = self.ir_data.get("ttir", {}).get("loc_index", {})
            ir_loc_index["ttnn"] = self.ir_data.get("ttnn", {}).get("loc_index", {})

        has_ir = self._has_ir_data()
        irs_tab_style = "" if has_ir else "display: none;"

        # Build per-operation data for the detail popup
        ops_for_js = []
        for i, op in enumerate(self.ops_data):
            weights = []
            if op.get("weights"):
                for w in op["weights"]:
                    weights.append({
                        "name": w.get("name", ""),
                        "shape": w.get("shape", ""),
                        "dtype": w.get("dtype", ""),
                    })
            ops_for_js.append({
                "index": i,
                "mlir_op": op.get("mlir_op", "unknown"),
                "loc": op.get("loc", ""),
                "inputs": op.get("inputs", []),
                "input_shapes": op.get("input_shapes", []),
                "input_dtypes": op.get("input_dtypes", []),
                "output_shapes": op.get("output_shapes", []),
                "output_dtypes": op.get("output_dtypes", []),
                "attributes": op.get("attributes", ""),
                "is_weight_op": op.get("is_weight_op", False),
                "weights": weights,
                "source": "Consteval" if op.get("const_eval_graph") else "Main",
            })

        mem_for_js = []
        for entry in self.mem_data:
            mem_entry = {}
            for mt in ["DRAM", "L1", "L1_SMALL"]:
                if mt in entry.get("memory", {}):
                    mem_entry[mt] = entry["memory"][mt].get("totalBytesAllocatedPerBank_MB", 0)
            unpadded = entry.get("unpadded_memory")
            if unpadded:
                mem_entry["unpadded"] = {}
                for mt in ["DRAM", "L1"]:
                    um = unpadded.get(mt)
                    if um:
                        mem_entry["unpadded"][mt] = {
                            "unpadded_MB": um.get("unpadded_MB", 0),
                            "padded_MB": um.get("padded_MB", 0),
                            "overhead_pct": um.get("overhead_pct", 0),
                        }
            mem_for_js.append(mem_entry)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Profile: {self.run_dir.name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            /* Background colors */
            --bg-canvas: #111217;
            --bg-primary: #181b1f;
            --bg-secondary: #22252b;
            --bg-tertiary: #2a2d33;

            /* Text colors */
            --text-primary: rgb(204, 204, 220);
            --text-secondary: rgba(204, 204, 220, 0.65);
            --text-disabled: rgba(204, 204, 220, 0.40);
            --text-link: #6e9fff;

            /* Borders */
            --border-weak: rgba(204, 204, 220, 0.07);
            --border-medium: rgba(204, 204, 220, 0.12);
            --border-strong: rgba(204, 204, 220, 0.20);

            /* Accents */
            --accent-primary: #3d71d9;
            --accent-success: #1a7f4b;
            --accent-warning: #ff9900;
            --accent-error: #d10e5c;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-canvas);
            color: var(--text-primary);
        }}

        /* App layout with sidebar */
        .app-container {{
            display: flex;
            min-height: 100vh;
        }}

        /* Sidebar styles */
        .sidebar {{
            width: 220px;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px 0;
            flex-shrink: 0;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            border-right: 1px solid var(--border-medium);
        }}
        .sidebar-header {{
            padding: 0 20px 20px;
            border-bottom: 1px solid var(--border-medium);
            margin-bottom: 20px;
        }}
        .sidebar-header h2 {{
            font-size: 18px;
            color: var(--accent-primary);
            margin: 0;
        }}
        .sidebar-nav {{
            list-style: none;
        }}
        .sidebar-nav li {{
            margin: 5px 0;
        }}
        .sidebar-nav a {{
            display: block;
            padding: 12px 20px;
            color: var(--text-secondary);
            text-decoration: none;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }}
        .sidebar-nav a:hover {{
            background: rgba(255,255,255,0.05);
            color: var(--text-primary);
        }}
        .sidebar-nav a.active {{
            background: rgba(61, 113, 217, 0.15);
            color: var(--accent-primary);
            border-left-color: var(--accent-primary);
        }}

        /* Main content area */
        .main-content {{
            flex: 1;
            margin-left: 220px;
            padding: 20px;
        }}

        /* View containers */
        .view {{
            display: none;
        }}
        .view.active {{
            display: block;
        }}

        /* Summary view container */
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: var(--bg-primary);
            padding: 30px;
            border-radius: 10px;
            border: 1px solid var(--border-medium);
        }}
        h1 {{
            color: var(--text-primary);
            margin-bottom: 10px;
            border-bottom: 3px solid var(--accent-primary);
            padding-bottom: 10px;
        }}
        h2 {{
            color: var(--text-primary);
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid var(--border-medium);
            padding-bottom: 8px;
        }}
        h3 {{
            color: var(--text-secondary);
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .metadata {{
            background: var(--bg-secondary);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            color: var(--text-secondary);
            font-size: 14px;
            border: 1px solid var(--border-weak);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid var(--accent-primary);
            border-top: 1px solid var(--border-weak);
            border-right: 1px solid var(--border-weak);
            border-bottom: 1px solid var(--border-weak);
        }}
        .summary-card.green {{
            border-left-color: var(--accent-success);
        }}
        .summary-card.blue {{
            border-left-color: #2196F3;
        }}
        .summary-card.orange {{
            border-left-color: var(--accent-warning);
        }}
        .summary-card .label {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        .graph-container {{
            margin: 20px 0;
            background: var(--bg-primary);
            padding: 10px;
            border-radius: 5px;
            border: 1px solid var(--border-medium);
        }}
        .peak-card {{
            background: var(--bg-secondary);
            border-left: 4px solid var(--accent-success);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            border-top: 1px solid var(--border-weak);
            border-right: 1px solid var(--border-weak);
            border-bottom: 1px solid var(--border-weak);
        }}
        .peak-card h3 {{
            margin-top: 0;
            color: var(--text-primary);
        }}
        .peak-card table {{
            width: 100%;
            margin-top: 10px;
        }}
        .peak-card td {{
            padding: 8px;
            border-bottom: 1px solid var(--border-weak);
            color: var(--text-primary);
        }}
        .peak-card td:first-child {{
            font-weight: 600;
            width: 200px;
            color: var(--text-secondary);
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border: 1px solid var(--border-medium);
            border-radius: 5px;
            overflow: hidden;
        }}
        table.data-table th {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid var(--border-medium);
        }}
        table.data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-weak);
            color: var(--text-primary);
        }}
        table.data-table tr:hover {{
            background: var(--bg-tertiary);
        }}
        table.data-table tr:nth-child(even) {{
            background: var(--bg-secondary);
        }}
        .code {{
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            color: var(--text-primary);
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .badge.dram {{ background: var(--accent-warning); color: #111; }}
        .badge.l1 {{ background: #2196F3; color: #111; }}
        .badge.l1-small {{ background: #9C27B0; color: white; }}
        .badge.trace {{ background: #607D8B; color: white; }}

        /* Operation link styles */
        .op-link {{
            color: var(--text-link);
            text-decoration: none;
            cursor: pointer;
        }}
        .op-link:hover {{
            text-decoration: underline;
            color: #8ab4ff;
        }}

        /* IR View styles */
        .ir-view-container {{
            background: var(--bg-primary);
            border-radius: 10px;
            border: 1px solid var(--border-medium);
            overflow: hidden;
        }}
        .ir-tabs {{
            display: flex;
            background: var(--bg-secondary);
            padding: 0;
            border-bottom: 1px solid var(--border-medium);
        }}
        .ir-tab {{
            padding: 15px 30px;
            color: var(--text-secondary);
            cursor: pointer;
            border: none;
            background: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            font-family: 'Inter', sans-serif;
        }}
        .ir-tab:hover {{
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }}
        .ir-tab.active {{
            color: var(--accent-primary);
            background: var(--bg-primary);
        }}
        .ir-content {{
            display: none;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            max-height: calc(100vh - 150px);
            overflow: auto;
        }}
        .ir-content.active {{
            display: block;
        }}
        .ir-line {{
            display: flex;
            padding: 0 15px;
        }}
        .ir-line:hover {{
            background: var(--bg-secondary);
        }}
        .ir-line.highlighted {{
            background: rgba(61, 113, 217, 0.3);
            animation: highlight-pulse 1s ease-out;
        }}
        @keyframes highlight-pulse {{
            0% {{ background: rgba(61, 113, 217, 0.5); }}
            100% {{ background: rgba(61, 113, 217, 0.3); }}
        }}
        .line-num {{
            color: var(--text-disabled);
            min-width: 50px;
            text-align: right;
            padding-right: 15px;
            user-select: none;
            border-right: 1px solid var(--border-medium);
            margin-right: 15px;
        }}
        .line-content {{
            white-space: pre;
        }}
        .ir-empty {{
            padding: 40px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 16px;
        }}

        /* Operation detail popup */
        .op-popup-overlay {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            z-index: 9998;
        }}
        .op-popup {{
            display: none;
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: var(--bg-primary);
            border: 1px solid var(--border-strong);
            border-radius: 10px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 9999;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }}
        .op-popup-header {{
            position: sticky;
            top: 0;
            background: var(--bg-secondary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-medium);
            border-radius: 10px 10px 0 0;
            z-index: 1;
        }}
        .op-popup-header h3 {{
            margin: 0;
            font-size: 15px;
            color: var(--text-primary);
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            word-break: break-all;
        }}
        .op-popup-close {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
            padding: 0 0 0 12px;
            line-height: 1;
            flex-shrink: 0;
        }}
        .op-popup-close:hover {{
            color: var(--text-primary);
        }}
        .op-popup-body {{
            padding: 20px;
        }}
        .op-popup-section {{
            margin-bottom: 16px;
        }}
        .op-popup-label {{
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-disabled);
            margin-bottom: 4px;
        }}
        .op-popup-value {{
            font-size: 14px;
            color: var(--text-primary);
        }}
        .op-popup-footer {{
            padding: 12px 20px;
            border-top: 1px solid var(--border-medium);
            text-align: right;
        }}
        .op-popup-footer button {{
            background: var(--accent-primary);
            color: #fff;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
        }}
        .op-popup-footer button:hover:not(:disabled) {{
            filter: brightness(1.15);
        }}
        .op-popup-footer button:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        .op-popup-io-item {{
            display: inline-block;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-weak);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            margin: 2px 4px 2px 0;
            color: var(--text-primary);
        }}
        .op-popup-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
            vertical-align: middle;
        }}
        .op-popup-badge.weight {{
            background: rgba(209, 14, 92, 0.25);
            color: #ff6b8a;
        }}
        .op-popup-badge.activation {{
            background: rgba(61, 113, 217, 0.25);
            color: #6e9fff;
        }}
        .op-popup-mem-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
        }}
        .op-popup-mem-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-weak);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }}
        .op-popup-mem-card .mem-type {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        .op-popup-mem-card .mem-value {{
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }}
        .op-popup-mem-card .mem-unit {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <nav class="sidebar">
            <div class="sidebar-header">
                <h2>Memory Profiler</h2>
            </div>
            <ul class="sidebar-nav">
                <li><a href="#" class="active" onclick="showView('summary'); return false;">Summary</a></li>
                <li style="{irs_tab_style}"><a href="#" onclick="showView('irs'); return false;">IRs</a></li>
            </ul>
        </nav>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Summary View -->
            <div id="summary-view" class="view active">
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
                        {self._format_padding_overhead_card(peak_padding_overhead)}
                    </div>

                    <!-- Memory Usage Over Time -->
                    <h2>Memory Usage Over Time</h2>
                    <div class="graph-container">
                        <div id="memory-graphs"></div>
                    </div>

                    {self._format_tile_padding_section(top_padding_ops)}

                    <!-- Peak Memory Analysis -->
                    <h2>Peak Memory Analysis</h2>
                    {self._generate_peak_cards_html(peak_analysis)}

                    <!-- Top Memory Consumers -->
                    <h2>Top 10 Memory-Consuming Operations</h2>
                    {self._generate_top_ops_table_html(top_ops)}
                </div>
            </div>

            <!-- IRs View -->
            <div id="irs-view" class="view">
                <div class="ir-view-container">
                    <div class="ir-tabs">
                        <button class="ir-tab active" onclick="showIRTab('ttir')">TTIR</button>
                        <button class="ir-tab" onclick="showIRTab('ttnn')">TTNN</button>
                    </div>
                    <div id="ttir-content" class="ir-content active">
                        {self._generate_ir_html('ttir')}
                    </div>
                    <div id="ttnn-content" class="ir-content">
                        {self._generate_ir_html('ttnn')}
                    </div>
                </div>
            </div>
        </main>

        <!-- Operation detail popup -->
        <div id="op-popup-overlay" class="op-popup-overlay" onclick="closeOpPopup()"></div>
        <div id="op-popup" class="op-popup">
            <div class="op-popup-header">
                <h3 id="op-popup-title">Operation Details</h3>
                <button class="op-popup-close" onclick="closeOpPopup()">&times;</button>
            </div>
            <div class="op-popup-body" id="op-popup-body"></div>
            <div class="op-popup-footer">
                <button id="op-popup-ir-btn" onclick="jumpToIRFromPopup()" disabled>Jump to op in IR</button>
            </div>
        </div>
    </div>

    <script>
        // Memory usage graphs data
        const memoryData = {json.dumps(memory_graph_data)};
        const unpaddedComparisonData = {json.dumps(unpadded_comparison_data)};

        // IR location indices for navigation
        const irLocIndex = {json.dumps(ir_loc_index)};

        // Per-operation data for detail popup
        const opsData = {json.dumps(ops_for_js)};
        const memData = {json.dumps(mem_for_js)};
        const hasIRData = {'true' if has_ir else 'false'};

        // Track current highlighted line
        let currentHighlightedLine = null;

        // View switching
        function showView(viewName) {{
            // Hide all views
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            // Show selected view
            document.getElementById(viewName + '-view').classList.add('active');
            // Update nav
            document.querySelectorAll('.sidebar-nav a').forEach(a => a.classList.remove('active'));
            event.target.classList.add('active');

            // Resize plots when switching to summary view
            if (viewName === 'summary') {{
                setTimeout(() => {{
                    Plotly.Plots.resize('memory-graphs');
                    const unpaddedGraph = document.getElementById('unpadded-comparison-graph');
                    if (unpaddedGraph) {{
                        Plotly.Plots.resize('unpadded-comparison-graph');
                    }}
                }}, 100);
            }}
        }}

        // IR tab switching
        function showIRTab(irType) {{
            // Update tabs
            document.querySelectorAll('.ir-tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            // Update content
            document.querySelectorAll('.ir-content').forEach(c => c.classList.remove('active'));
            document.getElementById(irType + '-content').classList.add('active');
        }}

        // Navigate to specific line in IR
        function navigateToIR(loc, preferredIR) {{
            // Remove previous highlight
            if (currentHighlightedLine) {{
                currentHighlightedLine.classList.remove('highlighted');
            }}

            // Try to find the line in preferred IR first, then fall back to other
            let irType = preferredIR;
            let lineNum = irLocIndex[irType][loc];

            if (!lineNum) {{
                // Try the other IR type
                irType = preferredIR === 'ttnn' ? 'ttir' : 'ttnn';
                lineNum = irLocIndex[irType][loc];
            }}

            if (!lineNum) {{
                console.warn('Location not found in IR:', loc);
                return;
            }}

            // Switch to IRs view
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById('irs-view').classList.add('active');
            document.querySelectorAll('.sidebar-nav a').forEach(a => a.classList.remove('active'));
            document.querySelectorAll('.sidebar-nav a')[1].classList.add('active');

            // Switch to correct IR tab
            document.querySelectorAll('.ir-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.ir-tab')[irType === 'ttir' ? 0 : 1].classList.add('active');
            document.querySelectorAll('.ir-content').forEach(c => c.classList.remove('active'));
            document.getElementById(irType + '-content').classList.add('active');

            // Scroll to and highlight the line
            const lineElement = document.getElementById(irType + '-line-' + lineNum);
            if (lineElement) {{
                const container = lineElement.closest('.ir-content');
                container.scrollTop = lineElement.offsetTop - container.offsetTop - container.clientHeight / 2;
                container.scrollLeft = 0;
                lineElement.classList.add('highlighted');
                currentHighlightedLine = lineElement;
            }}
        }}

        // --- Operation detail popup ---
        let popupCurrentLoc = null;

        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function openOpPopup(opIndex) {{
            if (opIndex < 0 || opIndex >= opsData.length) return;
            const op = opsData[opIndex];
            const mem = opIndex < memData.length ? memData[opIndex] : {{}};
            popupCurrentLoc = op.loc || null;

            // Header: op name + badge
            const badge = op.is_weight_op
                ? '<span class="op-popup-badge weight">Weight Op</span>'
                : '<span class="op-popup-badge activation">Activation</span>';
            document.getElementById('op-popup-title').innerHTML = escapeHtml(op.mlir_op) + badge;

            // Body
            let html = '';

            // Op index
            html += '<div class="op-popup-section">';
            html += '<div class="op-popup-label">Operation Index</div>';
            html += '<div class="op-popup-value">#' + op.index + '</div>';
            html += '</div>';

            // Source
            html += '<div class="op-popup-section">';
            html += '<div class="op-popup-label">Source</div>';
            html += '<div class="op-popup-value">' + escapeHtml(op.source) + '</div>';
            html += '</div>';

            // Inputs
            html += '<div class="op-popup-section">';
            html += '<div class="op-popup-label">Inputs</div>';
            html += '<div class="op-popup-value">';
            if (op.input_shapes && op.input_shapes.length > 0) {{
                op.input_shapes.forEach(function(shape, i) {{
                    const dtype = (op.input_dtypes && op.input_dtypes[i]) || '?';
                    const label = shape ? shape : 'scalar';
                    html += '<span class="op-popup-io-item">' + escapeHtml(label) + ' ' + escapeHtml(dtype) + '</span>';
                }});
            }} else {{
                html += '<em style="color:var(--text-disabled)">None</em>';
            }}
            html += '</div></div>';

            // Outputs
            html += '<div class="op-popup-section">';
            html += '<div class="op-popup-label">Outputs</div>';
            html += '<div class="op-popup-value">';
            if (op.output_shapes && op.output_shapes.length > 0) {{
                op.output_shapes.forEach(function(shape, i) {{
                    const dtype = (op.output_dtypes && op.output_dtypes[i]) || '?';
                    const label = shape ? shape : 'scalar';
                    html += '<span class="op-popup-io-item">' + escapeHtml(label) + ' ' + escapeHtml(dtype) + '</span>';
                }});
            }} else {{
                html += '<em style="color:var(--text-disabled)">None</em>';
            }}
            html += '</div></div>';

            // Attributes
            html += '<div class="op-popup-section">';
            html += '<div class="op-popup-label">Attributes</div>';
            html += '<div class="op-popup-value">';
            if (op.attributes) {{
                html += '<span class="code" style="white-space:pre-wrap;word-break:break-all;">' + escapeHtml(op.attributes) + '</span>';
            }} else {{
                html += '<em style="color:var(--text-disabled)">None</em>';
            }}
            html += '</div></div>';

            // Weights
            if (op.weights && op.weights.length > 0) {{
                html += '<div class="op-popup-section">';
                html += '<div class="op-popup-label">Weights</div>';
                html += '<div class="op-popup-value">';
                op.weights.forEach(function(w) {{
                    html += '<span class="op-popup-io-item">' + escapeHtml(w.name) + ' ' + escapeHtml(w.shape) + ' ' + escapeHtml(w.dtype) + '</span>';
                }});
                html += '</div></div>';
            }}

            // Memory stats
            const memTypes = ['DRAM', 'L1', 'L1_SMALL'];
            const hasAnyMem = memTypes.some(function(mt) {{ return mem[mt] !== undefined; }});
            if (hasAnyMem) {{
                html += '<div class="op-popup-section">';
                html += '<div class="op-popup-label">Memory at This Operation</div>';
                html += '<div class="op-popup-mem-grid">';
                memTypes.forEach(function(mt) {{
                    if (mem[mt] !== undefined) {{
                        html += '<div class="op-popup-mem-card">';
                        html += '<div class="mem-type">' + mt + '</div>';
                        html += '<div class="mem-value">' + mem[mt].toFixed(2) + '</div>';
                        html += '<div class="mem-unit">MB/bank</div>';
                        html += '</div>';
                    }}
                }});
                html += '</div></div>';
            }}

            // Tile padding overhead
            if (mem.unpadded) {{
                html += '<div class="op-popup-section">';
                html += '<div class="op-popup-label">Tile Padding Overhead</div>';
                html += '<div class="op-popup-value">';
                ['DRAM', 'L1'].forEach(function(mt) {{
                    var u = mem.unpadded[mt];
                    if (u && (u.unpadded_MB > 0 || u.padded_MB > 0)) {{
                        html += '<div style="margin-bottom:4px;">';
                        html += '<span style="color:var(--text-secondary);font-size:12px;">' + mt + ':</span> ';
                        html += '<span class="code">' + u.unpadded_MB.toFixed(2) + ' MB</span>';
                        html += ' <span style="color:var(--text-disabled);">&rarr;</span> ';
                        html += '<span class="code">' + u.padded_MB.toFixed(2) + ' MB</span>';
                        if (u.overhead_pct > 0) {{
                            var color = u.overhead_pct > 100 ? '#ff6b6b' : u.overhead_pct > 50 ? '#ff9900' : 'var(--text-secondary)';
                            html += ' <span style="color:' + color + ';font-weight:600;font-size:12px;">(+' + u.overhead_pct.toFixed(1) + '%)</span>';
                        }}
                        html += '</div>';
                    }}
                }});
                html += '</div></div>';
            }}

            document.getElementById('op-popup-body').innerHTML = html;

            // IR button
            const irBtn = document.getElementById('op-popup-ir-btn');
            if (hasIRData && popupCurrentLoc && (irLocIndex.ttnn[popupCurrentLoc] || irLocIndex.ttir[popupCurrentLoc])) {{
                irBtn.disabled = false;
                irBtn.title = '';
            }} else {{
                irBtn.disabled = true;
                irBtn.title = popupCurrentLoc ? 'Location not found in IR data' : 'No location available for this operation';
            }}

            // Show
            document.getElementById('op-popup-overlay').style.display = 'block';
            document.getElementById('op-popup').style.display = 'block';
        }}

        function closeOpPopup() {{
            document.getElementById('op-popup-overlay').style.display = 'none';
            document.getElementById('op-popup').style.display = 'none';
            popupCurrentLoc = null;
        }}

        function jumpToIRFromPopup() {{
            const loc = popupCurrentLoc;
            closeOpPopup();
            if (loc) navigateToIR(loc, 'ttnn');
        }}

        // Dismiss popup on Escape
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape' && document.getElementById('op-popup').style.display === 'block') {{
                closeOpPopup();
            }}
        }});

        // Initialize plots
        document.addEventListener('DOMContentLoaded', function() {{
            // Create memory usage over time graphs
            Plotly.newPlot('memory-graphs', memoryData.traces, memoryData.layout, {{responsive: true}});

            // Create unpadded comparison graph if data available
            if (unpaddedComparisonData && unpaddedComparisonData.traces && unpaddedComparisonData.traces.length > 0) {{
                Plotly.newPlot('unpadded-comparison-graph', unpaddedComparisonData.traces, unpaddedComparisonData.layout, {{responsive: true}});
            }}

            // Click handler for memory graph
            document.getElementById('memory-graphs').on('plotly_click', function(data) {{
                if (!data.points || !data.points.length) return;
                var point = data.points[0];
                if (!point.customdata) return;  // skip capacity line
                var opIndex = point.x;
                if (opIndex >= 0 && opIndex < opsData.length) openOpPopup(opIndex);
            }});

            // Click handler for tile padding graph
            var unpaddedEl = document.getElementById('unpadded-comparison-graph');
            if (unpaddedEl && unpaddedEl.data) {{
                unpaddedEl.on('plotly_click', function(data) {{
                    if (!data.points || !data.points.length) return;
                    var opIndex = data.points[0].x;
                    if (opIndex >= 0 && opIndex < opsData.length) openOpPopup(opIndex);
                }});
            }}
        }});
    </script>
</body>
</html>"""
        return html

    def _prepare_memory_graph_data(self) -> Dict:
        """Prepare data for memory usage over time graph with tab selection"""
        # Exclude TRACE, only show DRAM, L1, L1_SMALL
        display_types = [mt for mt in self.available_memory_types if mt != "TRACE"]
        if not display_types:
            return {"traces": [], "layout": {}}

        traces = []
        trace_mem_type = []  # Track which memory type each trace belongs to

        # Collect all data points with their weight op status and op details
        all_indices = []
        all_allocated = {mt: [] for mt in display_types}
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
            for mt in display_types:
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
        weight_op_allocated = {mt: [] for mt in display_types}
        weight_op_names = []
        weight_input_shapes = []
        weight_output_shapes = []
        for i, flag in enumerate(weight_op_flags):
            if flag:
                for mt in display_types:
                    weight_op_allocated[mt].append(all_allocated[mt][i])
                weight_op_names.append(op_names[i])
                weight_input_shapes.append(input_shapes_list[i])
                weight_output_shapes.append(output_shapes_list[i])

        capacity = {
            mt: self.mem_data[0]["memory"][mt]["totalBytesPerBank_MB"]
            for mt in display_types
        }

        # Create traces for each memory type (all on same axes)
        for mem_type in display_types:
            # Main line connecting all points (blue)
            traces.append(
                {
                    "x": all_indices,
                    "y": all_allocated[mem_type],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Main",
                    "line": {"width": 2, "color": "#1f77b4"},
                    "marker": {"size": 3, "color": "#1f77b4"},
                    "visible": (mem_type == "DRAM"),  # Only DRAM visible by default
                    "showlegend": True,
                    "legendgroup": "main",
                    "customdata": list(
                        zip(op_names, input_shapes_list, output_shapes_list)
                    ),
                    "hovertemplate": f"{mem_type}<br>Op %{{x}}: %{{customdata[0]}}<br>Allocated: %{{y:.2f}} MB/bank<br>Input: %{{customdata[1]}}<br>Output: %{{customdata[2]}}<extra></extra>",
                }
            )
            trace_mem_type.append(mem_type)

            # Weight operations overlay (red markers on top)
            if weight_op_indices:
                traces.append(
                    {
                        "x": weight_op_indices,
                        "y": weight_op_allocated[mem_type],
                        "type": "scatter",
                        "mode": "markers",
                        "name": "Consteval",
                        "marker": {"size": 5, "color": "red", "symbol": "circle"},
                        "visible": (mem_type == "DRAM"),
                        "showlegend": True,
                        "legendgroup": "weight_ops",
                        "customdata": list(
                            zip(
                                weight_op_names,
                                weight_input_shapes,
                                weight_output_shapes,
                            )
                        ),
                        "hovertemplate": f"{mem_type} (weight op)<br>Op %{{x}}: %{{customdata[0]}}<br>Allocated: %{{y:.2f}} MB/bank<br>Input: %{{customdata[1]}}<br>Output: %{{customdata[2]}}<extra></extra>",
                    }
                )
                trace_mem_type.append(mem_type)

            # Capacity line
            traces.append(
                {
                    "x": [all_indices[0], all_indices[-1]],
                    "y": [capacity[mem_type], capacity[mem_type]],
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Capacity",
                    "line": {"dash": "dash", "color": "gray", "width": 1},
                    "visible": (mem_type == "DRAM"),
                    "showlegend": True,
                    "legendgroup": "capacity",
                    "hovertemplate": f"{mem_type} Capacity: %{{y:.2f}} MB/bank<extra></extra>",
                }
            )
            trace_mem_type.append(mem_type)

        # Build visibility arrays for each button
        buttons = []
        for mem_type in display_types:
            visibility = [mt == mem_type for mt in trace_mem_type]
            buttons.append(
                {
                    "label": mem_type,
                    "method": "update",
                    "args": [
                        {"visible": visibility},
                        {"yaxis.title": f"{mem_type} (MB/bank)"},
                    ],
                }
            )

        layout = {
            "height": 450,
            "showlegend": True,
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "title": {
                "text": "Memory Usage Across Operation Execution",
                "font": {"size": 18, "color": "rgb(204, 204, 220)"},
            },
            "xaxis": {
                "title": {"text": "Operation Index", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
                "zerolinecolor": "rgba(204, 204, 220, 0.20)",
            },
            "yaxis": {
                "title": {"text": "DRAM (MB/bank)", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
                "zerolinecolor": "rgba(204, 204, 220, 0.20)",
            },
            "updatemenus": [
                {
                    "type": "buttons",
                    "direction": "right",
                    "active": 0,  # DRAM is default
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.15,
                    "yanchor": "top",
                    "buttons": buttons,
                    "showactive": True,
                    "bgcolor": "#22252b",
                    "bordercolor": "rgba(204, 204, 220, 0.20)",
                    "font": {"size": 12, "color": "rgb(204, 204, 220)"},
                }
            ],
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"color": "rgb(204, 204, 220)"},
            },
            "hoverlabel": {
                "bgcolor": "#22252b",
                "bordercolor": "rgba(204, 204, 220, 0.20)",
                "font": {"color": "rgb(204, 204, 220)"},
            },
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
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "title": {"text": "Memory Allocation vs Free Space", "font": {"size": 18, "color": "rgb(204, 204, 220)"}},
            "grid": {"rows": num_types, "columns": 1, "pattern": "independent"},
            "legend": {"font": {"color": "rgb(204, 204, 220)"}},
            "hoverlabel": {
                "bgcolor": "#22252b",
                "bordercolor": "rgba(204, 204, 220, 0.20)",
                "font": {"color": "rgb(204, 204, 220)"},
            },
        }

        for i, mem_type in enumerate(memory_types):
            axis_idx = i + 1
            xkey = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
            ykey = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
            top = 1.0 - i * (row_height + gap)
            bottom = top - row_height
            layout[xkey] = {
                "title": {"text": "Operation Index", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
                "anchor": f"y{axis_idx}" if axis_idx > 1 else "y",
            }
            layout[ykey] = {
                "title": {"text": f"{mem_type} (MB/bank)", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
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

            # Format operation as clickable link
            op_link = self._format_op_link(op['mlir_op'], op.get('loc'))

            html_parts.append(
                f"""
        <div class="peak-card" style="border-left-color: {color};">
            <h3><span class="badge {mem_type.lower().replace('_', '-')}">{mem_type}</span> Peak: {peak_val:.2f} MB/bank at Operation #{data['index']}</h3>
            <table>
                <tr><td>Operation</td><td>{op_link}</td></tr>
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

            # Format operation as clickable link
            op_link = self._format_op_link(op['mlir_op'], op.get('loc'))

            rows.append(
                f"""
            <tr>
                <td>{rank}</td>
                <td>{idx}</td>
                <td>{op_link}</td>
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
            <div class="summary-card" style="border-left-color: #2196F3;">
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
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "title": {
                "text": f"DRAM Memory Breakdown (Declared Weights: {total_weight_MB:.1f} MB, Persistent Baseline: {weight_baseline:.1f} MB)",
                "font": {"size": 16, "color": "rgb(204, 204, 220)"},
            },
            "xaxis": {
                "title": {"text": "Operation Index", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
            },
            "yaxis": {
                "title": {"text": "Memory (MB/bank)", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
                "rangemode": "tozero",
            },
            "hovermode": "x",
            "legend": {"font": {"color": "rgb(204, 204, 220)"}},
            "hoverlabel": {
                "bgcolor": "#22252b",
                "bordercolor": "rgba(204, 204, 220, 0.20)",
                "font": {"color": "rgb(204, 204, 220)"},
            },
        }

        return {"traces": traces, "layout": layout}

    def _prepare_unpadded_comparison_data(self) -> Dict:
        """Prepare data for unpadded vs padded memory comparison graph.

        Shows three lines:
        - Blue: Unpadded (logical) memory - theoretical minimum
        - Orange: Padded (tile-aligned) memory - calculated from tensor layouts
        - Green dashed: Actual allocated memory - from runtime
        """
        # Check if unpadded_memory data is available
        if not self.mem_data or not self.mem_data[0].get("unpadded_memory"):
            return {"traces": [], "layout": {}}

        indices = []
        unpadded_dram = []
        padded_dram = []
        op_names = []

        for i, op in enumerate(self.mem_data):
            unpadded = op.get("unpadded_memory", {})
            if not unpadded:
                continue

            indices.append(op["index"])

            # Get DRAM values
            dram_unpadded = unpadded.get("DRAM", {})
            unpadded_dram.append(dram_unpadded.get("unpadded_MB", 0))
            padded_dram.append(dram_unpadded.get("padded_MB", 0))

            # Get op name for hover
            if i < len(self.ops_data):
                op_names.append(self.ops_data[i].get("mlir_op", "unknown"))
            else:
                op_names.append(op.get("mlir_op", "unknown"))

        if not indices:
            return {"traces": [], "layout": {}}

        # Build customdata for hover
        customdata = list(zip(op_names, unpadded_dram, padded_dram))

        traces = [
            {
                "x": indices,
                "y": unpadded_dram,
                "type": "scatter",
                "mode": "lines",
                "name": "Unpadded (Logical)",
                "line": {"width": 2, "color": "#1f77b4"},  # Blue
                "customdata": customdata,
                "hovertemplate": "Op %{x}: %{customdata[0]}<br>Unpadded: %{customdata[1]:.2f} MB<br>Padded: %{customdata[2]:.2f} MB<extra></extra>",
            },
            {
                "x": indices,
                "y": padded_dram,
                "type": "scatter",
                "mode": "lines",
                "name": "Padded (Tile-Aligned)",
                "line": {"width": 2, "color": "#ff7f0e"},  # Orange
                "hoverinfo": "skip",
            },
        ]

        # Calculate peak overhead for title
        peak_overhead_pct = 0
        for i in range(len(unpadded_dram)):
            if unpadded_dram[i] > 0:
                overhead = (padded_dram[i] - unpadded_dram[i]) / unpadded_dram[i] * 100
                peak_overhead_pct = max(peak_overhead_pct, overhead)

        layout = {
            "height": 400,
            "showlegend": True,
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "title": {
                "text": f"Tile Padding Memory Overhead (DRAM) - Peak: {peak_overhead_pct:.1f}%",
                "font": {"size": 16, "color": "rgb(204, 204, 220)"},
            },
            "xaxis": {
                "title": {"text": "Operation Index", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
            },
            "yaxis": {
                "title": {"text": "Total Memory (MB)", "font": {"color": "rgb(204, 204, 220)"}},
                "tickfont": {"color": "rgb(204, 204, 220)"},
                "gridcolor": "rgba(204, 204, 220, 0.08)",
                "linecolor": "rgba(204, 204, 220, 0.20)",
                "rangemode": "tozero",
            },
            "hovermode": "x",
            "legend": {"font": {"color": "rgb(204, 204, 220)"}},
            "hoverlabel": {
                "bgcolor": "#22252b",
                "bordercolor": "rgba(204, 204, 220, 0.20)",
                "font": {"color": "rgb(204, 204, 220)"},
            },
        }

        return {"traces": traces, "layout": layout}

    def _calculate_peak_padding_overhead(self) -> Dict:
        """Calculate peak tile padding overhead from memory data."""
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

    def _format_padding_overhead_card(self, peak_padding_overhead: Dict) -> str:
        """Format padding overhead summary card."""
        if not peak_padding_overhead or not peak_padding_overhead.get("has_data"):
            return ""

        dram_pct = peak_padding_overhead.get("dram_pct", 0)
        return f"""
            <div class="summary-card" style="border-left-color: var(--accent-error);">
                <div class="label">Peak Tile Padding Overhead</div>
                <div class="value">{dram_pct:.1f}%</div>
            </div>"""

    def _format_tile_padding_section(self, top_padding_ops: List[Dict]) -> str:
        """Format the tile padding analysis section."""
        if not self.mem_data or not self.mem_data[0].get("unpadded_memory"):
            return ""

        table_html = self._generate_top_padding_ops_table_html(top_padding_ops)

        return f"""
        <!-- Tile Padding Memory Overhead -->
        <h2>Tile Padding Memory Overhead (DRAM)</h2>
        <p style="color: var(--text-secondary); margin-bottom: 15px;">
            Shows actual allocated memory vs theoretical minimum without 32x32 tile alignment.
        </p>
        <div class="graph-container">
            <div id="unpadded-comparison-graph"></div>
        </div>

        <h3>Top 10 Operations by Absolute Padding Overhead</h3>
        {table_html}
        """

    def get_top_padding_overhead_ops(self, n: int = 10) -> List[Dict]:
        """Get top N operations with highest absolute tile padding overhead.

        Returns operations where the output tensor has significant padding overhead,
        sorted by absolute overhead in bytes (padded - unpadded).
        """
        ops_with_overhead = []

        for i, op in enumerate(self.ops_data):
            layout_info = op.get("output_layout_info")
            if not layout_info:
                continue

            overhead_pct = layout_info.get("overhead_pct", 0)
            if overhead_pct <= 0:
                continue

            # Calculate absolute overhead in bytes
            padded_bytes = layout_info.get("padded_bytes", 0)
            unpadded_bytes = layout_info.get("unpadded_bytes", 0)
            absolute_overhead = padded_bytes - unpadded_bytes

            ops_with_overhead.append(
                {
                    "index": i,
                    "operation": op,
                    "layout_info": layout_info,
                    "overhead_pct": overhead_pct,
                    "absolute_overhead_bytes": absolute_overhead,
                }
            )

        # Sort by absolute overhead in bytes descending
        ops_with_overhead.sort(key=lambda x: x["absolute_overhead_bytes"], reverse=True)
        return ops_with_overhead[:n]

    def _generate_top_padding_ops_table_html(self, top_ops: List[Dict]) -> str:
        """Generate HTML table for top padding overhead operations."""
        if not top_ops:
            return "<p>No operations with tile padding overhead found.</p>"

        rows = []
        for rank, item in enumerate(top_ops, 1):
            op = item["operation"]
            layout = item["layout_info"]
            idx = item["index"]

            logical_shape = "x".join(str(d) for d in layout.get("logical_shape", []))
            padded_shape = "x".join(str(d) for d in layout.get("padded_shape", []))
            dtype = layout.get("dtype", "?")
            unpadded_bytes = layout.get("unpadded_bytes", 0)
            padded_bytes = layout.get("padded_bytes", 0)
            overhead_pct = layout.get("overhead_pct", 0)
            absolute_overhead = padded_bytes - unpadded_bytes

            # Format absolute overhead (always in MB for consistency)
            overhead_mb = absolute_overhead / (1024 * 1024)
            overhead_mb_str = f"{overhead_mb:.2f} MB"

            # Format sizes
            if unpadded_bytes >= 1024 * 1024:
                unpadded_str = f"{unpadded_bytes / (1024*1024):.2f} MB"
            elif unpadded_bytes >= 1024:
                unpadded_str = f"{unpadded_bytes / 1024:.1f} KB"
            else:
                unpadded_str = f"{unpadded_bytes} B"

            if padded_bytes >= 1024 * 1024:
                padded_str = f"{padded_bytes / (1024*1024):.2f} MB"
            elif padded_bytes >= 1024:
                padded_str = f"{padded_bytes / 1024:.1f} KB"
            else:
                padded_str = f"{padded_bytes} B"

            # Format operation as clickable link
            op_link = self._format_op_link(op['mlir_op'], op.get('loc'))

            rows.append(
                f"""
            <tr>
                <td>{rank}</td>
                <td>{idx}</td>
                <td>{op_link}</td>
                <td><span class="code">{logical_shape} ({dtype})</span></td>
                <td><span class="code">{padded_shape}</span></td>
                <td>{unpadded_str}</td>
                <td>{padded_str}</td>
                <td style="font-weight: bold;">{overhead_mb_str}</td>
                <td style="color: {'#ff6b6b' if overhead_pct > 100 else '#ff9900' if overhead_pct > 50 else 'inherit'};">{overhead_pct:.1f}%</td>
            </tr>"""
            )

        return f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Index</th>
                    <th>Operation</th>
                    <th>Logical Shape</th>
                    <th>Padded Shape</th>
                    <th>Unpadded</th>
                    <th>Padded</th>
                    <th>Overhead (MB)</th>
                    <th>Overhead (%)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>"""
