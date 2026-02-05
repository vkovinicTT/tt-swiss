# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data types for model analysis."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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


def generate_module_id(index: int) -> str:
    return f"mod_{index:03d}"


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
