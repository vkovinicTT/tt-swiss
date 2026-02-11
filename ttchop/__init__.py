# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Model analysis package for PyTorch model structure extraction and op-by-op testing."""

from .module_extractor import extract_unique_modules
from .module_tree import ModuleNode, build_module_tree, update_modules_with_status
from .op_by_op_runner import run_hierarchical_op_by_op
from .summary import generate_summary
from .utils import load_function_from_path, setup_tt_device, get_module_by_path
from .visualizer import generate_visualization

__all__ = [
    "extract_unique_modules",
    "generate_summary",
    "generate_visualization",
    "ModuleNode",
    "build_module_tree",
    "update_modules_with_status",
    "run_hierarchical_op_by_op",
    "load_function_from_path",
    "setup_tt_device",
    "get_module_by_path",
]
