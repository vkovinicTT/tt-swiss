# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shape capture via forward hooks."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ShapeCapture:
    """Captures input/output shapes for all modules via forward hooks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.shapes: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _tensor_info(self, value: Any) -> List[Dict[str, str]]:
        if isinstance(value, torch.Tensor):
            return [{"shape": "x".join(str(d) for d in value.shape),
                     "dtype": str(value.dtype).replace("torch.", "")}]
        if isinstance(value, (tuple, list)):
            return [i for v in value for i in self._tensor_info(v)]
        if isinstance(value, dict):
            return [i for v in value.values() for i in self._tensor_info(v)]
        return []

    def run(self, sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, Any]],
            device: Optional[Any] = None) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        self.shapes.clear()

        for name, module in self.model.named_modules():
            path = name or "(root)"

            def pre_hook(p):
                def hook(mod, inputs):
                    self.shapes.setdefault(p, {"inputs": [], "outputs": []})["inputs"] = self._tensor_info(inputs)
                return hook

            def post_hook(p):
                def hook(mod, inputs, outputs):
                    self.shapes.setdefault(p, {"inputs": [], "outputs": []})["outputs"] = self._tensor_info(outputs)
                return hook

            self._hooks.append(module.register_forward_pre_hook(pre_hook(path)))
            self._hooks.append(module.register_forward_hook(post_hook(path)))

        model = self.model.to(device) if device else self.model
        if device:
            if isinstance(sample_input, dict):
                sample_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_input.items()}
            elif isinstance(sample_input, tuple):
                sample_input = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in sample_input)
            elif isinstance(sample_input, torch.Tensor):
                sample_input = sample_input.to(device)

        try:
            with torch.no_grad():
                if isinstance(sample_input, dict):
                    model(*sample_input.values())
                elif isinstance(sample_input, tuple):
                    model(*sample_input)
                else:
                    model(sample_input)
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

        return self.shapes
