# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for model analysis."""

import importlib.util
import os
import socket
import sys
from pathlib import Path
from typing import Callable, Optional
import torch.nn as nn


def get_tt_xla_root() -> Path:
    """
    Find tt-xla root directory.

    Searches in order:
    1. TT_XLA_ROOT environment variable
    2. Infer from TTMLIR_TOOLCHAIN_DIR (set by venv/activate)
    3. Relative path from this file (works with -e install)
    4. Search upward from cwd for tests/op_by_op marker
    """
    # 1. Explicit env var
    if root := os.environ.get("TT_XLA_ROOT"):
        path = Path(root)
        if (path / "tests/op_by_op").exists():
            return path

    # 2. Infer from TTMLIR_TOOLCHAIN_DIR
    if toolchain := os.environ.get("TTMLIR_TOOLCHAIN_DIR"):
        # TTMLIR_TOOLCHAIN_DIR is typically <tt-xla>/third_party/tt-mlir/src/tt-mlir/build
        candidate = Path(toolchain).parent.parent.parent.parent.parent
        if (candidate / "tests/op_by_op").exists():
            return candidate

    # 3. Relative path from package (works with pip install -e)
    candidate = Path(__file__).parent.parent.parent
    if (candidate / "tests/op_by_op").exists():
        return candidate

    # 4. Search upward from cwd
    for candidate in [Path.cwd()] + list(Path.cwd().parents):
        if (candidate / "tests/op_by_op/op_by_op_test.py").exists():
            return candidate

    raise RuntimeError(
        "Could not find tt-xla root directory.\n"
        "Either:\n"
        "  1. Run from within tt-xla directory, or\n"
        "  2. Set TT_XLA_ROOT environment variable, or\n"
        "  3. Activate tt-xla venv (source venv/activate)"
    )


def load_function_from_path(path: str) -> Callable:
    """Load a function from 'file.py::function_name' format."""
    if "::" not in path:
        raise ValueError(f"Invalid path format: '{path}'. Expected 'file.py::function_name'")

    file_path, func_name = path.rsplit("::", 1)
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    if str(file_path.parent) not in sys.path:
        sys.path.insert(0, str(file_path.parent))
    spec.loader.exec_module(module)

    if not hasattr(module, func_name):
        raise AttributeError(f"'{func_name}' not found in {file_path}")
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"'{func_name}' is not callable")
    return func


def setup_tt_device():
    """Set up and return TT device."""
    import torch_xla
    import torch_xla.runtime as xr
    xr.set_device_type("TT")
    return torch_xla.device()


def get_device_info() -> dict:
    """Get TT device info (arch, mesh, hostname)."""
    info = {"arch": "unknown", "mesh_shape": "unknown", "hostname": socket.gethostname()}
    try:
        import torch_xla
        import torch_xla.runtime as xr
        attrs = xr.runtime_device_attributes(str(torch_xla.device()))
        arch = attrs.get("device_arch", "unknown")
        info["arch"] = "Wormhole B0" if arch == "Wormhole_b0" else arch
        count = xr.global_device_count()
        info["mesh_shape"] = {1: "n150 (1x1)", 2: "n300 (1x2)"}.get(count, f"{count} chips")
    except Exception:
        pass
    return info


def get_module_by_path(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Get submodule by dot-separated path (e.g., 'block.resnets[0].conv1')."""
    if path in ("(root)", "full_model", ""):
        return model
    try:
        parts, current, i = [], "", 0
        while i < len(path):
            c = path[i]
            if c == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif c == "[":
                if current:
                    parts.append(current)
                    current = ""
                j = path.index("]", i)
                parts.append(int(path[i + 1:j]))
                i = j
            else:
                current += c
            i += 1
        if current:
            parts.append(current)

        module = model
        for p in parts:
            module = module[p] if isinstance(p, int) else getattr(module, p)
        return module
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def get_parent_path(module_path: str) -> Optional[str]:
    """Get parent module path. Returns None for root, 'full_model' for top-level."""
    if module_path in ("(root)", "full_model", ""):
        return None
    last_dot, bracket_depth = -1, 0
    for i, c in enumerate(module_path):
        if c == "[":
            bracket_depth += 1
        elif c == "]":
            bracket_depth -= 1
        elif c == "." and bracket_depth == 0:
            last_dot = i
    return "full_model" if last_dot == -1 else module_path[:last_dot]
