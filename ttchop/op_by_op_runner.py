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

from .data_types import CONTAINER_TYPES
from .log_parser import parse_op_by_op_log, save_parsed_log
from .module_tree import ModuleNode


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

    module_dir = output_dir / "module_irs" / module_id
    module_dir.mkdir(parents=True, exist_ok=True)
    log_file = module_dir / "run.log"

    print(f"    Exporting IR for {module_id}...", end=" ", flush=True)
    try:
        result = subprocess.run(cmd, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8', errors='replace')

        with open(log_file, "w") as f:
            f.write(f"=== Run Log for {module_id} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n\n")
            if output:
                f.write(f"=== OUTPUT ===\n{output}\n")

        if result.returncode == 0:
            print("OK")
            return True
        # Subprocess failed but IR files might still be on disk
        # (e.g., torch_xla atexit crash after successful export)
        irs_dir = output_dir / "module_irs" / module_id / "irs"
        if irs_dir.exists() and list(irs_dir.glob("ttir_*.mlir")):
            print("OK (subprocess exit error ignored)")
            return True
        print(f"FAILED (see {log_file})")
        if output:
            for line in output.strip().split('\n')[-3:]:
                print(f"      {line}")
        return False
    except subprocess.TimeoutExpired:
        with open(log_file, "w") as f:
            f.write(f"=== Run Log for {module_id} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"TIMEOUT after 300 seconds\n")
        print(f"TIMEOUT (see {log_file})")
        return False
    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"=== Run Log for {module_id} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"ERROR: {e}\n")
        print(f"ERROR: {e} (see {log_file})")
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

    log_file = module_irs_dir / "op_by_op.log"

    try:
        result = subprocess.run(cmd, cwd=str(project_root), env=env, timeout=1800,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8', errors='replace')
        returncode = result.returncode

        with open(log_file, "w") as f:
            f.write(f"=== Op-by-Op Log for {module_id} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {returncode}\n\n")
            if output:
                f.write(f"=== OUTPUT ===\n{output}\n")

        # Parse execution log for detailed per-op traces
        parsed_log_path = module_irs_dir / f"{module_id}_op_by_op_parsed.json"
        try:
            blocks = parse_op_by_op_log(log_file)
            save_parsed_log(blocks, parsed_log_path)
        except Exception:
            pass  # Non-critical, don't fail the pipeline

    except subprocess.TimeoutExpired:
        with open(log_file, "w") as f:
            f.write(f"=== Op-by-Op Log for {module_id} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"TIMEOUT after 1800 seconds\n")
        return {"success": False, "failed_ops": [{"op_name": "TIMEOUT", "error_message": "30min"}],
                "report_path": None, "skipped": False}
    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"=== Op-by-Op Log for {module_id} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"ERROR: {e}\n")
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
