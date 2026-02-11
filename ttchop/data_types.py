# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data types and shared constants for model analysis."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

CONTAINER_TYPES = ("Sequential", "ModuleList", "ModuleDict")

STATUS_ORDER = ["failed", "ir_export_failed", "success", "inherited_success", "skipped", "unknown"]
STATUS_LABELS = {
    "failed": "Failed",
    "ir_export_failed": "IR Export Failed",
    "success": "Success",
    "inherited_success": "Inherited Success",
    "skipped": "Skipped",
    "unknown": "Unknown",
}

MODULE_ATTRS = [
    "in_channels", "out_channels", "in_features", "out_features", "kernel_size",
    "stride", "padding", "dilation", "groups", "bias", "padding_mode", "num_features",
    "eps", "momentum", "affine", "track_running_stats", "num_heads", "embed_dim",
    "kdim", "vdim", "batch_first", "normalized_shape", "elementwise_affine",
]


@dataclass
class ModuleInfo:
    """Unique module information."""
    id: str
    class_name: str
    module_path: str
    parent: Optional[str]
    input_shapes: List[str]
    output_shapes: List[str]
    input_dtypes: List[str]
    output_dtypes: List[str]
    parameters: Dict[str, Any]
    occurrences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_module_id(index: int, module_path: str = "") -> str:
    import re
    # Use last segment of the path as the name (e.g., "res_blocks.0.conv1" -> "conv1")
    name = module_path.rsplit(".", 1)[-1] if module_path else ""
    # Clean bracket notation: "blocks[0]" -> "blocks_0"
    name = re.sub(r"\[(\d+)\]", r"_\1", name)
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name)[:60]
    return f"mod_{index:03d}_{sanitized}" if sanitized else f"mod_{index:03d}"


def extract_module_parameters(module) -> Dict[str, Any]:
    """Extract relevant parameters from a PyTorch module."""
    params = {}
    for attr in MODULE_ATTRS:
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, tuple):
                val = list(val)
            if attr == "bias" and hasattr(module, "bias"):
                val = module.bias is not None
            params[attr] = val
    return params
