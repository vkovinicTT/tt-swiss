# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate compact markdown summary of model analysis results."""

import json
from pathlib import Path
from typing import Any, Dict, List

from .data_types import CONTAINER_TYPES


def generate_summary(modules_json_path: Path) -> Path:
    """Generate markdown summary from unique_modules.json.

    Reads the modules JSON and parsed op-by-op JSONs to produce a compact
    summary showing overall status, module counts, and failed op details.
    Returns path to generated summary file.
    """
    with open(modules_json_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    modules = data["modules"]
    output_dir = modules_json_path.parent

    # Build lookup for finding children of containers
    modules_by_path = {m["module_path"]: m for m in modules}

    # Count statuses (merge success + inherited_success)
    raw_counts: Dict[str, int] = {}
    for m in modules:
        s = m.get("status", "unknown")
        raw_counts[s] = raw_counts.get(s, 0) + 1
    merged_counts: Dict[str, int] = {}
    merged_counts["success"] = raw_counts.get("success", 0) + raw_counts.get("inherited_success", 0)
    for s in ("failed", "ir_export_failed", "skipped", "unknown"):
        if s in raw_counts:
            merged_counts[s] = raw_counts[s]

    has_failures = any(m.get("status") in ("failed", "ir_export_failed") for m in modules)
    overall = "FAILED" if has_failures else "PASSED"

    # Collect failed modules with enriched error messages
    failed_modules = []
    for m in modules:
        if m.get("status") not in ("failed", "ir_export_failed"):
            continue
        is_container = m["class_name"] in CONTAINER_TYPES
        parsed_blocks = _load_parsed_blocks(output_dir, m["id"])
        enriched_ops = _enrich_failed_ops(m.get("failed_ops", []), parsed_blocks)

        # For containers, find failed children
        failed_children = []
        if is_container:
            for child_m in modules:
                if child_m.get("parent") == m["module_path"] and child_m.get("status") in ("failed", "ir_export_failed"):
                    failed_children.append({"class_name": child_m["class_name"], "path": child_m["module_path"]})

        failed_modules.append({
            "class_name": m["class_name"],
            "module_path": m["module_path"],
            "status": m["status"],
            "ops": enriched_ops,
            "is_container": is_container,
            "failed_children": failed_children,
        })

    lines = _build_markdown(meta, merged_counts, overall, failed_modules)

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    return summary_path


def _load_parsed_blocks(output_dir: Path, module_id: str) -> List[Dict[str, Any]]:
    """Load parsed op-by-op JSON for a module."""
    parsed_path = output_dir / "module_irs" / module_id / f"{module_id}_op_by_op_parsed.json"
    if not parsed_path.exists():
        return []
    try:
        with open(parsed_path) as f:
            return json.load(f)
    except Exception:
        return []


def _enrich_failed_ops(
    failed_ops: List[Dict[str, Any]], parsed_blocks: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Enrich failed ops with error_message and error_trace from parsed blocks."""
    failed_parsed = [b for b in parsed_blocks if not b.get("success")]

    enriched = []
    for i, op in enumerate(failed_ops):
        parsed = failed_parsed[i] if i < len(failed_parsed) else {}
        error = parsed.get("error_message") or op.get("error_message", "Unknown error")
        trace = parsed.get("error_trace") or ""
        enriched.append({
            "op_name": op.get("op_name", "Unknown"),
            "error": error,
            "error_trace": trace,
        })
    return enriched


_MERGED_LABELS = {
    "success": "Success",
    "failed": "Failed",
    "ir_export_failed": "IR Export Failed",
    "skipped": "Skipped",
    "unknown": "Unknown",
}
_MERGED_ORDER = ["failed", "ir_export_failed", "success", "skipped", "unknown"]


def _build_markdown(
    meta: Dict[str, Any],
    counts: Dict[str, int],
    overall: str,
    failed_modules: List[Dict[str, Any]],
) -> List[str]:
    """Build markdown lines."""
    lines: List[str] = []

    # Header
    lines.append(f"# {meta['model_class']} — Analysis Summary")
    lines.append("")

    # Metadata line
    device = meta.get("device_arch", "unknown")
    mesh = meta.get("device_mesh", "")
    date = meta.get("timestamp", "")[:10]
    lines.append(
        f"**Device:** {device} ({mesh}) | "
        f"**Modules:** {meta['total_modules']} total, "
        f"{meta['unique_modules']} unique | **Date:** {date}"
    )
    lines.append("")

    # Overall status
    lines.append(f"## Status: {overall}")
    lines.append("")

    # Status table
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for s in _MERGED_ORDER:
        if counts.get(s, 0) > 0:
            lines.append(f"| {_MERGED_LABELS.get(s, s)} | {counts[s]} |")
    lines.append("")

    # Failed modules (compact overview)
    if failed_modules:
        lines.append("## Failed Modules")
        lines.append("")
        for fm in failed_modules:
            path_display = fm["module_path"] if fm["module_path"] not in ("(root)", "full_model") else "root"
            lines.append(f"### {fm['class_name']} (`{path_display}`)")

            if fm["is_container"] and fm["failed_children"]:
                lines.append("Failed children:")
                for child in fm["failed_children"]:
                    lines.append(f"- {child['class_name']} (`{child['path']}`)")
            elif fm["ops"]:
                lines.append("| Op | Error |")
                lines.append("|----|-------|")
                for op in fm["ops"]:
                    err = op["error"].replace("|", "\\|")
                    lines.append(f"| {op['op_name']} | {err} |")
            else:
                msg = "IR export failed" if fm["status"] == "ir_export_failed" else "No op details"
                lines.append(f"*{msg}*")
            lines.append("")

        # Detailed report with full error traces
        non_container_failed = [fm for fm in failed_modules if not fm["is_container"] and fm["ops"]]
        if non_container_failed:
            lines.append("## Detailed Report")
            lines.append("")
            for fm in non_container_failed:
                path_display = fm["module_path"] if fm["module_path"] not in ("(root)", "full_model") else "root"
                lines.append(f"### {fm['class_name']} (`{path_display}`)")
                lines.append("")
                for op in fm["ops"]:
                    lines.append(f"#### {op['op_name']}")
                    if op["error_trace"]:
                        lines.append("```")
                        lines.append(op["error_trace"])
                        lines.append("```")
                    else:
                        lines.append(f"*{op['error']}*")
                    lines.append("")
    else:
        lines.append("All modules passed successfully.")
        lines.append("")

    return lines
