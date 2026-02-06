# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extract unique modules from PyTorch models."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .shapes import ShapeCapture
from .data_types import ModuleInfo, extract_module_parameters, generate_module_id
from .utils import get_parent_path

CONTAINER_TYPES = ("ModuleList", "ModuleDict", "Sequential")


def _uniqueness_key(class_name: str, input_shapes: List[str], output_shapes: List[str],
                    parameters: Dict[str, Any], path: str) -> str:
    """Build uniqueness key. Containers use path to stay unique."""
    if class_name in CONTAINER_TYPES:
        return f"{class_name}||PATH:{path}"
    return "||".join([class_name, "|".join(sorted(input_shapes)),
                      "|".join(sorted(output_shapes)), json.dumps(parameters, sort_keys=True)])


def _run_shape_subprocess(model_path: str, inputs_path: str, output_file: Path) -> Dict[str, Any]:
    """Run shape capture in subprocess on TT device."""
    script = Path(__file__).parent / "shape_capture_subprocess.py"
    cmd = [sys.executable, str(script), "--model-path", model_path,
           "--inputs-path", inputs_path, "--output-file", str(output_file)]

    print("  Running shape capture in subprocess on TT device...")
    result = subprocess.run(cmd, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"Shape capture failed with code {result.returncode}")

    with open(output_file) as f:
        data = json.load(f)
    return data if "shapes" in data else {"shapes": data, "device_info": {}}


def extract_unique_modules(
    load_fn: Callable[[], nn.Module],
    get_sample_input: Callable[[], Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, Any]]],
    output_path: Optional[str] = "unique_modules.json",
    model_path: Optional[str] = None,
    inputs_path: Optional[str] = None,
    use_device: bool = True,
) -> Dict[str, Any]:
    """Extract unique modules from a model based on class, shapes, and parameters."""
    print(f"Loading model using {load_fn.__name__}...")
    model = load_fn()
    model.eval()

    print("Getting sample input...")
    sample_input = get_sample_input()

    # Capture shapes
    print("Running forward pass to capture shapes...")
    device_info = {}
    if use_device and model_path and inputs_path:
        shapes_file = Path(output_path).parent / ".shapes_temp.json" if output_path else Path(".shapes_temp.json")
        shapes_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_shape_subprocess(model_path, inputs_path, shapes_file)
            shapes, device_info = result["shapes"], result.get("device_info", {})
        finally:
            shapes_file.unlink(missing_ok=True)
    else:
        print("  (Running on CPU...)")
        shapes = ShapeCapture(model).run(sample_input)

    # Group modules by uniqueness
    print("Extracting module information...")
    groups: Dict[str, List[Tuple[str, Dict]]] = {}

    for name, module in model.named_modules():
        path = name or "(root)"
        mod_shapes = shapes.get(path, {"inputs": [], "outputs": []})
        params = extract_module_parameters(module)
        class_name = type(module).__name__

        data = {
            "module_path": path,
            "parent": get_parent_path(path),
            "class_name": class_name,
            "input_shapes": [s["shape"] for s in mod_shapes.get("inputs", [])],
            "output_shapes": [s["shape"] for s in mod_shapes.get("outputs", [])],
            "input_dtypes": [s["dtype"] for s in mod_shapes.get("inputs", [])],
            "output_dtypes": [s["dtype"] for s in mod_shapes.get("outputs", [])],
            "parameters": params,
        }
        key = _uniqueness_key(class_name, data["input_shapes"], data["output_shapes"], params, path)
        groups.setdefault(key, []).append((path, data))

    # Build unique modules list
    print("Building unique modules list...")
    unique_modules = []
    for i, (_, group) in enumerate(groups.items()):
        first_path, first_data = group[0]
        unique_modules.append(ModuleInfo(
            id=generate_module_id(i), class_name=first_data["class_name"],
            module_path=first_path, parent=first_data["parent"],
            input_shapes=first_data["input_shapes"], output_shapes=first_data["output_shapes"],
            input_dtypes=first_data["input_dtypes"], output_dtypes=first_data["output_dtypes"],
            parameters=first_data["parameters"], occurrences=[p for p, _ in group],
        ).to_dict())

    total = sum(len(g) for g in groups.values())
    result = {
        "metadata": {
            "load_fn_name": load_fn.__name__,
            "model_class": type(model).__name__,
            "total_modules": total,
            "unique_modules": len(unique_modules),
            "timestamp": datetime.now().isoformat(),
            "hostname": device_info.get("hostname", "unknown"),
            "device_arch": device_info.get("arch", "unknown"),
            "device_mesh": device_info.get("mesh_shape", "unknown"),
        },
        "modules": unique_modules,
    }

    if output_path:
        print(f"Saving to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {len(unique_modules)} unique modules (from {total} total)")

    return result
