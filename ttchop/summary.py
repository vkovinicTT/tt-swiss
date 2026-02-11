# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate compact markdown summary of model analysis results."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .data_types import CONTAINER_TYPES

# --- TensorDesc parsing ---

_TENSORDESC_RE = re.compile(r"shape=\[([^\]]+)\].*?data_type='([^']+)'")
_DTYPE_TO_MLIR = {"bf16": "bf16", "f32": "f32", "f16": "f16", "si32": "si32", "i32": "i32"}


def _parse_tensordesc_shapes(desc_str: str) -> List[Tuple[str, str]]:
    """Parse TensorDesc string into list of (shape, dtype) tuples.

    Input:  "[TensorDesc(shape=[1, 128, 2, 4, 4], data_type='bf16', ...)]"
    Output: [("1,128,2,4,4", "bf16"), ("1024,128,3,3,3", "bf16")]
    """
    if not desc_str or desc_str in ("[]", "None"):
        return []
    return [
        (m.group(1).replace(" ", ""), m.group(2))
        for m in _TENSORDESC_RE.finditer(desc_str)
    ]


def _tensordesc_compact(desc_str: str) -> str:
    """Format TensorDesc as compact "[shape] dtype" for display.

    Input:  "[TensorDesc(shape=[1, 128, 2, 4, 4], data_type='bf16', ...)]"
    Output: "[1,128,2,4,4] bf16, [1024,128,3,3,3] bf16"
    """
    parts = _parse_tensordesc_shapes(desc_str)
    if not parts:
        return desc_str if desc_str else ""
    return ", ".join(f"[{shape}] {dtype}" for shape, dtype in parts)


def _tensordesc_to_ttir_types(desc_str: str) -> str:
    """Convert TensorDesc string to TTIR-style type signature for matching.

    Input:  "[TensorDesc(shape=[1, 128, 2, 4, 4], data_type='bf16', ...)]"
    Output: "tensor<1x128x2x4x4xbf16>, tensor<1024x128x3x3x3xbf16>"
    """
    parts = _parse_tensordesc_shapes(desc_str)
    if not parts:
        return ""
    return ", ".join(
        f"tensor<{'x'.join(shape.split(','))}x{_DTYPE_TO_MLIR.get(dtype, dtype)}>"
        for shape, dtype in parts
    )


# --- TTIR attribute extraction ---

_TTIR_OP_RE = re.compile(r'"(ttir\.\w+)"')
_TTIR_ATTRS_RE = re.compile(r"<\{([^}]+)\}>")
_TTIR_TYPE_SIG_RE = re.compile(r"\)\s*(?:<\{[^}]*\}>)?\s*:\s*\(([^)]+)\)\s*->")


def _build_ttir_attrs_lookup(
    output_dir: Path, module_id: str
) -> Dict[Tuple[str, str], str]:
    """Build a lookup from (op_name, input_shapes) -> raw attrs from TTIR files.

    Parses TTIR MLIR files to extract each op's raw attribute string (the
    content inside <{...}>). The lookup allows matching report ops to their
    TTIR definitions by op name and input tensor shapes, with a name-only
    fallback.
    """
    irs_dir = output_dir / "module_irs" / module_id / "irs"
    if not irs_dir.exists():
        return {}

    ttir_files = sorted(irs_dir.glob("ttir_*.mlir"))
    if not ttir_files:
        return {}

    lookup: Dict[Tuple[str, str], str] = {}
    name_only: Dict[str, str] = {}
    for ttir_path in ttir_files:
        try:
            content = ttir_path.read_text(errors="replace")
        except Exception:
            continue
        for line in content.split("\n"):
            stripped = line.strip()
            op_match = _TTIR_OP_RE.search(stripped)
            if not op_match or op_match.group(1) == "ttir.constant":
                continue

            attrs_match = _TTIR_ATTRS_RE.search(stripped)
            if not attrs_match:
                continue
            raw_attrs = attrs_match.group(1)

            input_shapes = ""
            type_match = _TTIR_TYPE_SIG_RE.search(stripped)
            if type_match:
                input_shapes = type_match.group(1).strip()

            op_name = op_match.group(1)
            lookup.setdefault((op_name, input_shapes), raw_attrs)
            name_only.setdefault(op_name, raw_attrs)

    # Name-only entries serve as fallback when input shapes don't match
    for name, attrs in name_only.items():
        lookup.setdefault((name, ""), attrs)

    return lookup


def _match_ttir_attrs(
    op_name: str, inputs_desc: str, lookup: Dict[Tuple[str, str], str]
) -> str:
    """Find TTIR attributes for a report op by matching on op_name and input shapes.

    First tries exact match on (op_name, TTIR-style input types).
    Falls back to name-only match.
    """
    ttir_types = _tensordesc_to_ttir_types(inputs_desc)
    if ttir_types:
        result = lookup.get((op_name, ttir_types))
        if result:
            return result
    return lookup.get((op_name, ""), "")


# --- Failed ops collection ---


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
    failed_ops: List[Dict[str, Any]],
    parsed_blocks: List[Dict[str, Any]],
    ttir_lookup: Dict[Tuple[str, str], str],
) -> List[Dict[str, str]]:
    """Enrich failed ops with error traces from parsed blocks and params from TTIR."""
    failed_parsed = [b for b in parsed_blocks if not b.get("success")]

    enriched = []
    for i, op in enumerate(failed_ops):
        parsed = failed_parsed[i] if i < len(failed_parsed) else {}
        op_name = op.get("op_name", "Unknown")
        inputs = op.get("inputs", "")

        enriched.append({
            "op_name": op_name,
            "inputs": inputs,
            "outputs": op.get("outputs", ""),
            "op_params": _match_ttir_attrs(op_name, inputs, ttir_lookup) or op.get("op_params", ""),
            "error": parsed.get("error_message") or op.get("error_message", "Unknown error"),
            "error_trace": parsed.get("error_trace") or "",
        })
    return enriched


def _build_depth_map(modules: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build module_path -> tree depth map using parent chain.

    Root module (parent=None) gets depth 0, its children get depth 1, etc.
    """
    # Build parent lookup: module_path -> parent_path
    parent_of: Dict[str, str] = {}
    for m in modules:
        path = m.get("module_path", "")
        parent = m.get("parent")
        if parent:
            parent_of[path] = parent

    depths: Dict[str, int] = {}

    def get_depth(path: str) -> int:
        if path in depths:
            return depths[path]
        parent = parent_of.get(path)
        if not parent:
            depths[path] = 0
            return 0
        d = get_depth(parent) + 1
        depths[path] = d
        return d

    for m in modules:
        get_depth(m.get("module_path", ""))

    return depths


def _collect_unique_failed_ops(
    modules: List[Dict[str, Any]], output_dir: Path
) -> List[Dict[str, str]]:
    """Collect unique failed ops across all non-container modules.

    Deduplicates by (op_name, inputs, outputs, op_params).
    For each unique op, tracks the deepest module (by tree depth) where it appears.
    """
    depth_map = _build_depth_map(modules)
    seen: Dict[Tuple, int] = {}
    unique_ops: List[Dict[str, str]] = []

    for m in modules:
        if m.get("status") not in ("failed", "ir_export_failed"):
            continue
        if m["class_name"] in CONTAINER_TYPES:
            continue

        module_id = m["id"]
        module_path = m.get("module_path", "")
        module_class = m.get("class_name", "")
        depth = depth_map.get(module_path, 0)

        parsed_blocks = _load_parsed_blocks(output_dir, module_id)
        ttir_lookup = _build_ttir_attrs_lookup(output_dir, module_id)

        for op in _enrich_failed_ops(m.get("failed_ops", []), parsed_blocks, ttir_lookup):
            key = (op["op_name"], op["inputs"], op["outputs"], op["op_params"])
            if key not in seen:
                seen[key] = len(unique_ops)
                op["module"] = f"{module_class} ({module_path})"
                op["_depth"] = depth
                unique_ops.append(op)
            else:
                idx = seen[key]
                if depth > unique_ops[idx]["_depth"]:
                    unique_ops[idx]["module"] = f"{module_class} ({module_path})"
                    unique_ops[idx]["_depth"] = depth

    # Clean up internal field
    for op in unique_ops:
        op.pop("_depth", None)

    return unique_ops


# --- Markdown generation ---


def _build_markdown(
    meta: Dict[str, Any],
    overall: str,
    unique_ops: List[Dict[str, str]],
) -> List[str]:
    """Build markdown lines."""
    lines: List[str] = []

    # Header
    device = meta.get("device_arch", "unknown")
    mesh = meta.get("device_mesh", "")
    date = meta.get("timestamp", "")[:10]
    n_failed = len(unique_ops)
    status_line = f"PASSED" if overall == "PASSED" else f"FAILED ({n_failed} unique op{'s' if n_failed != 1 else ''})"

    lines.append(f"# {meta['model_class']} — {status_line}")
    lines.append("")
    lines.append(f"**Device:** {device} ({mesh}) | **Date:** {date}")
    lines.append("")

    if not unique_ops:
        lines.append("All operations passed successfully.")
        lines.append("")
        return lines

    # Failed ops summary table
    lines.append("## Failed Ops")
    lines.append("")
    lines.append("| # | Op | Inputs | Outputs | Params | Module | Error |")
    lines.append("|---|-----|--------|---------|--------|--------|-------|")
    for i, op in enumerate(unique_ops, 1):
        inputs = _tensordesc_compact(op.get("inputs", ""))
        outputs = _tensordesc_compact(op.get("outputs", ""))
        error = op["error"].replace("|", "\\|").replace("\n", " ")
        params = op.get("op_params", "")
        module = op.get("module", "")
        # Wrap params and error in scrollable divs for table readability
        if params:
            params_cell = f'<div style="min-width:300px;max-height:8em;overflow-y:auto;white-space:pre-wrap;font-size:12px">{params}</div>'
        else:
            params_cell = "-"
        if error:
            error_cell = f'<div style="min-width:300px;max-height:8em;overflow-y:auto;white-space:pre-wrap;font-size:12px">{error}</div>'
        else:
            error_cell = "-"
        lines.append(
            f"| {i} | `{op['op_name']}` | `{inputs}` | `{outputs}` "
            f"| {params_cell} | {module} | {error_cell} |"
        )
    lines.append("")

    # Detailed report with full error traces
    lines.append("## Detailed Report")
    lines.append("")
    for i, op in enumerate(unique_ops, 1):
        params = op.get("op_params", "")
        module = op.get("module", "")
        lines.append(f"### {i}. `{op['op_name']}`")
        lines.append("")
        lines.append(f"- **Inputs:** `{_tensordesc_compact(op.get('inputs', ''))}`")
        lines.append(f"- **Outputs:** `{_tensordesc_compact(op.get('outputs', ''))}`")
        if params:
            lines.append(f"- **Params:** `{params}`")
        if module:
            lines.append(f"- **Module:** {module}")
        lines.append(f"- **Error:** {op['error']}")
        lines.append("")
        if op["error_trace"]:
            lines.append("<details>")
            lines.append("<summary>Full error trace</summary>")
            lines.append("")
            lines.append("```")
            lines.append(op["error_trace"])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    return lines


# --- Public API ---


def generate_summary(modules_json_path: Path) -> Path:
    """Generate markdown summary from unique_modules.json.

    Returns path to generated summary.md file.
    """
    with open(modules_json_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    modules = data["modules"]
    output_dir = modules_json_path.parent

    has_failures = any(m.get("status") in ("failed", "ir_export_failed") for m in modules)
    unique_ops = _collect_unique_failed_ops(modules, output_dir)
    lines = _build_markdown(meta, "FAILED" if has_failures else "PASSED", unique_ops)

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    return summary_path
