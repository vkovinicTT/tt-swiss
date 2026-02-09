# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Op-by-op analysis runner with lazy IR export."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .module_tree import ModuleNode

CONTAINER_TYPES = ("Sequential", "ModuleList", "ModuleDict")


def _ensure_system_desc(project_root: Path) -> Path:
    """Ensure system descriptor exists, generating it if needed."""
    desc_path = project_root / "ttrt-artifacts" / "system_desc.ttsys"
    if desc_path.exists():
        return desc_path

    print(f"Generating system descriptor at {desc_path}...")
    try:
        result = subprocess.run(
            ["ttrt", "query", "--save-artifacts"],
            cwd=str(project_root),
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            print(f"Failed to generate system descriptor at {desc_path}: {stderr}")
            raise SystemExit(1)
    except FileNotFoundError:
        print(f"Failed to generate system descriptor at {desc_path}: 'ttrt' command not found")
        raise SystemExit(1)

    print(f"Successfully generated system descriptor at {desc_path}")
    return desc_path


def _export_ir(module_id: str, modules_json: Path, model_path: str, inputs_path: str, output_dir: Path) -> bool:
    """Export IR for a single module via subprocess."""
    script = Path(__file__).parent / "ir_export_single_module.py"
    cmd = [sys.executable, str(script), "--module-id", module_id, "--modules-json", str(modules_json),
           "--model-path", model_path, "--inputs-path", inputs_path, "--output-dir", str(output_dir)]

    print(f"    Exporting IR for {module_id}...", end=" ", flush=True)
    try:
        result = subprocess.run(cmd, timeout=300, capture_output=True)
        if result.returncode == 0:
            print("OK")
            return True
        print("FAILED")
        if result.stderr:
            for line in result.stderr.decode('utf-8', errors='replace').strip().split('\n')[-3:]:
                print(f"      {line}")
        return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def _run_op_by_op(module_id: str, module_irs_dir: Path, project_root: Path) -> Dict[str, Any]:
    """Run op-by-op analysis on module's TTIR files."""
    irs_dir = module_irs_dir / "irs"
    if not irs_dir.exists() or not list(irs_dir.glob("ttir_*.mlir")):
        return {"success": True, "failed_ops": [], "report_path": None, "skipped": True}

    report_path = module_irs_dir / f"{module_id}_op_by_op_report.json"
    cmd = ["pytest", "-svv", "tests/op_by_op/op_by_op_test.py::test_op_by_op",
           f"--folder={module_irs_dir}", "--ir-file-prefix=irs/ttir_",
           "--json-report", f"--json-report-file={report_path}"]

    print(f"    Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(project_root / "tests"),
        str(project_root / "third_party/tt-mlir/src/tt-mlir/build/python_packages"),
        env.get("PYTHONPATH", ""),
    ])
    env["SYSTEM_DESC_PATH"] = str(project_root / "ttrt-artifacts" / "system_desc.ttsys")

    try:
        result = subprocess.run(cmd, cwd=str(project_root), env=env, timeout=600)
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        return {"success": False, "failed_ops": [{"op_name": "TIMEOUT", "error_message": "10min"}],
                "report_path": None, "skipped": False}
    except Exception as e:
        return {"success": False, "failed_ops": [{"op_name": "ERROR", "error_message": str(e)}],
                "report_path": None, "skipped": False}

    failed_ops = _parse_report(report_path) if report_path.exists() else []
    if returncode != 0 and not failed_ops:
        failed_ops = [{"op_name": "PYTEST_FAILED", "error_message": f"Exit code {returncode}"}]

    return {"success": returncode == 0, "failed_ops": failed_ops,
            "report_path": str(report_path) if report_path.exists() else None, "skipped": False}


def _parse_report(report_path: Path) -> List[Dict[str, Any]]:
    """Parse pytest-json-report for failed ops."""
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    failed_ops = []
    for test in report.get("tests", []):
        for prop in test.get("user_properties", []):
            if not isinstance(prop, dict):
                continue
            for key, value in prop.items():
                if key.startswith("OpTest model for:") and isinstance(value, dict):
                    if value.get("success") == "False":
                        failed_ops.append({
                            "op_name": value.get("op_name"),
                            "error_message": value.get("error_message"),
                            "inputs": value.get("inputs", ""),
                            "outputs": value.get("outputs", ""),
                        })
    return failed_ops


def _mark_subtree_success(node: ModuleNode):
    """Mark node and descendants as success."""
    if node.status is None:
        node.status = "success"
    for child in node.children:
        child.status = "inherited_success"
        _mark_subtree_success(child)


def _mark_subtree_skipped(node: ModuleNode):
    """Mark node and all descendants as skipped."""
    node.status = "skipped"
    for child in node.children:
        _mark_subtree_skipped(child)


def _update_container_status(node: ModuleNode) -> str:
    """Update container status based on children (post-order)."""
    for child in node.children:
        _update_container_status(child)

    if node.class_name not in CONTAINER_TYPES or not node.children:
        return node.status or "unknown"
    if node.status == "inherited_success":
        return node.status

    statuses = [c.status for c in node.children if c.status]
    if not statuses:
        return node.status or "skipped"

    if "failed" in statuses or "ir_export_failed" in statuses:
        node.status = "failed"
    elif all(s in ("success", "inherited_success") for s in statuses):
        node.status = "inherited_success"
    elif any(s in ("success", "inherited_success") for s in statuses):
        node.status = "inherited_success"
    else:
        node.status = "skipped"

    return node.status


def run_hierarchical_op_by_op(root: ModuleNode, module_irs_base: Path, project_root: Path,
                              modules_json_path: Path, model_path: str, inputs_path: str,
                              output_dir: Path, root_only: bool = False) -> None:
    """Run hierarchical op-by-op analysis with lazy IR export."""
    _ensure_system_desc(project_root)
    exported = set()

    def analyze(node: ModuleNode, is_root_call: bool = False):
        if root_only and not is_root_call:
            print(f"  {node.module_id} ({node.class_name}): Skipped (--root-only)")
            _mark_subtree_skipped(node)
            return

        module_irs_dir = module_irs_base / node.module_id

        if node.class_name in CONTAINER_TYPES:
            print(f"  {node.module_id} ({node.class_name}): Skipped (container)")
            node.status = "skipped"
            for child in node.children:
                analyze(child, is_root_call=False)
            return

        print(f"  {node.module_id} ({node.class_name}):")

        if node.module_id not in exported:
            if not _export_ir(node.module_id, modules_json_path, model_path, inputs_path, output_dir):
                node.status = "ir_export_failed"
                if root_only:
                    for child in node.children:
                        _mark_subtree_skipped(child)
                else:
                    for child in node.children:
                        analyze(child, is_root_call=False)
                return
            exported.add(node.module_id)

        if not module_irs_dir.exists():
            node.status = "skipped"
            for child in node.children:
                analyze(child, is_root_call=False)
            return

        print(f"    Running op-by-op...")
        result = _run_op_by_op(node.module_id, module_irs_dir, project_root)

        if result.get("skipped"):
            node.status = "skipped"
            for child in node.children:
                analyze(child, is_root_call=False)
            return

        if result["success"]:
            if root_only:
                print(f"    SUCCESS")
                node.status = "success"
                for child in node.children:
                    _mark_subtree_skipped(child)
            else:
                print(f"    SUCCESS - marking subtree")
                _mark_subtree_success(node)
        else:
            print(f"    FAILED - {len(result['failed_ops'])} ops")
            node.status = "failed"
            node.failed_ops = result["failed_ops"]
            node.op_by_op_report_path = result["report_path"]
            if root_only:
                for child in node.children:
                    _mark_subtree_skipped(child)
            else:
                for child in node.children:
                    analyze(child, is_root_call=False)

    if root:
        analyze(root, is_root_call=True)
        _update_container_status(root)
