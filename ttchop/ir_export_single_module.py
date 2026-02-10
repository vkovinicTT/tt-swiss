# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Export IR for a single module (subprocess entry point)."""

import argparse
import json
import os
import sys
from pathlib import Path

from utils import load_function_from_path, setup_tt_device, get_module_by_path


def main():
    parser = argparse.ArgumentParser(description="Export IR for a single module")
    parser.add_argument("--module-id", required=True)
    parser.add_argument("--modules-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inputs-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.modules_json) as f:
        modules_data = json.load(f)

    module_info = next((m for m in modules_data["modules"] if m["id"] == args.module_id), None)
    if not module_info:
        print(f"Module {args.module_id} not found")
        os._exit(1)

    if module_info["class_name"] in ("Sequential", "ModuleList", "ModuleDict"):
        os._exit(0)

    device = setup_tt_device()
    load_fn = load_function_from_path(args.model_path)
    inputs_fn = load_function_from_path(args.inputs_path)

    model = load_fn()
    model.eval()

    submodule = get_module_by_path(model, module_info["module_path"])
    if submodule is None:
        print(f"Could not find module at path '{module_info['module_path']}'")
        os._exit(1)

    from module_runner import run_submodule_for_ir
    try:
        success = run_submodule_for_ir(submodule, module_info, Path(args.output_dir), device, inputs_fn)
        os._exit(0 if success else 1)
    except Exception as e:
        print(f"Error exporting IR: {e}")
        os._exit(1)


if __name__ == "__main__":
    main()
