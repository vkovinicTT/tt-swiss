# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Module runner for exporting IRs from unique modules."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

CONTAINER_TYPES = ("Sequential", "ModuleList", "ModuleDict")


def generate_input_for_module(module_info: Dict[str, Any], device: Optional[Any] = None) -> torch.Tensor:
    """Generate input tensor for a module based on its info."""
    input_shapes = module_info.get("input_shapes", [])
    input_dtypes = module_info.get("input_dtypes", [])

    if not input_shapes:
        raise ValueError(f"No input shapes found for module {module_info.get('id')}")

    shape = [int(d) for d in input_shapes[0].split("x")]
    dtype = getattr(torch, input_dtypes[0] if input_dtypes else "float32", torch.float32)
    tensor = torch.randn(*shape, dtype=dtype)
    return tensor.to(device) if device else tensor


def run_submodule_for_ir(
    submodule: nn.Module,
    module_info: Dict[str, Any],
    output_dir: Path,
    device: Any,
    get_sample_input: Optional[Callable[[], Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, Any]]]] = None,
) -> bool:
    """Run a submodule through TT backend to export IRs."""
    import torch_xla

    module_id = module_info["id"]
    module_path = module_info["module_path"]
    module_class = module_info["class_name"]

    if module_class in CONTAINER_TYPES:
        return False

    # Setup output directory and configure IR export
    module_dir = output_dir / "module_irs" / module_id
    module_dir.mkdir(parents=True, exist_ok=True)
    torch_xla.set_custom_compile_options({
        "export_path": str(module_dir),
        "export_model_name": f"{module_id}_{module_class}",
    })

    # Compile module
    submodule = submodule.to(device).eval()
    compiled = torch.compile(submodule, backend="tt")

    # Generate input
    if module_path == "(root)" and get_sample_input:
        input_data = get_sample_input()
        if isinstance(input_data, dict):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
        elif isinstance(input_data, (tuple, list)):
            inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in input_data)
        else:
            inputs = (input_data.to(device),)
    else:
        inputs = (generate_input_for_module(module_info, device),)

    # Run forward pass - IRs are exported during compilation, so runtime failures are OK
    try:
        with torch.no_grad():
            compiled(*inputs.values()) if isinstance(inputs, dict) else compiled(*inputs)
    except Exception as e:
        # IRs should still be exported even if runtime fails
        print(f"Note: Runtime failed for {module_id} (IRs still exported): {type(e).__name__}")

    # Verify TTIR files were actually exported (op-by-op needs ttir_*.mlir)
    irs_dir = module_dir / "irs"
    if irs_dir.exists() and list(irs_dir.glob("ttir_*.mlir")):
        return True

    print(f"Warning: No TTIR files found in {irs_dir}")
    return False
