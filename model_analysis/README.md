# Model Analysis

A PyTorch model analysis tool for extracting unique modules and operations from neural networks.

## Overview

This package provides tools to:
- **Extract unique modules** from a model based on class name, input/output shapes, and parameters
- **Extract PyTorch operations** from modules with their configurations (stride, padding, etc.)
- **Generate JSON outputs** for further analysis and tooling

The tool is designed to help understand model structure, identify duplicate modules, and analyze operation configurations.

## Installation

From the `tt-swiss` directory:

```bash
pip install -e .
```

This installs the `tt-model-analysis` CLI command.

## CLI Usage

The simplest way to use this tool is via the command line:

```bash
tt-model-analysis --model-path path/to/file.py::load_model_fn \
                  --inputs-path path/to/file.py::get_inputs_fn \
                  [--dir output_directory]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model-path` | Yes | Path to model loader function in format `file.py::function_name` |
| `--inputs-path` | Yes | Path to inputs generator function in format `file.py::function_name` |
| `--dir` | No | Output directory (default: `./<ModelClassName>/`) |

### Example

1. Create a file with your model loading functions:

```python
# my_model.py
import torch
import torch.nn as nn

def load_model():
    """Return the model to analyze."""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    return model

def get_inputs():
    """Return sample input tensor(s)."""
    return torch.randn(1, 3, 224, 224)
```

2. Run the analysis:

```bash
tt-model-analysis --model-path my_model.py::load_model --inputs-path my_model.py::get_inputs
```

3. Check the output:
```
Sequential/
├── unique_modules.json
└── pytorch_ops.json
```

## Python API Usage

You can also use the package programmatically:

```python
from model_analysis import extract_unique_modules, extract_pytorch_ops
import torch
import torch.nn as nn

# Define your model and inputs
def load_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )

def get_inputs():
    return torch.randn(1, 10)

# Step 1: Extract unique modules
modules = extract_unique_modules(
    load_fn=load_model,
    get_sample_input=get_inputs,
    output_path="unique_modules.json"
)

# Step 2: Extract ops from modules
ops = extract_pytorch_ops(
    modules_json_path="unique_modules.json",
    load_fn=load_model,
    get_sample_input=get_inputs,
    output_path="pytorch_ops.json"
)
```

## Output Format

### unique_modules.json

Contains metadata and a list of unique modules:

```json
{
  "metadata": {
    "load_fn_location": "my_model.py:load_model",
    "load_fn_name": "load_model",
    "model_class": "Sequential",
    "total_modules": 7,
    "unique_modules": 4,
    "timestamp": "2026-02-02T10:30:00"
  },
  "modules": [
    {
      "id": "mod_000",
      "class_name": "Sequential",
      "module_path": "(root)",
      "parent": null,
      "input_shapes": ["1x3x224x224"],
      "output_shapes": ["1x64x224x224"],
      "input_dtypes": ["float32"],
      "output_dtypes": ["float32"],
      "parameters": {},
      "occurrences": ["(root)"]
    },
    {
      "id": "mod_001",
      "class_name": "Conv2d",
      "module_path": "0",
      "parent": "(root)",
      "input_shapes": ["1x3x224x224"],
      "output_shapes": ["1x64x224x224"],
      "input_dtypes": ["float32"],
      "output_dtypes": ["float32"],
      "parameters": {
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1]
      },
      "occurrences": ["0"]
    }
  ]
}
```

### pytorch_ops.json

Contains metadata and a list of unique operations:

```json
{
  "metadata": {
    "source_modules_json": "unique_modules.json",
    "total_ops": 12,
    "unique_ops": 6,
    "timestamp": "2026-02-02T10:31:00"
  },
  "ops": [
    {
      "id": "op_000",
      "op_name": "torch.conv2d",
      "source_module_id": "mod_001",
      "source_module_class": "Conv2d",
      "attributes": {
        "stride": [1, 1],
        "padding": [1, 1],
        "dilation": [1, 1],
        "groups": 1
      },
      "tensor_types": {
        "output": {
          "shape": "1x64x224x224",
          "dtype": "float32"
        }
      },
      "occurrences": [
        {"source_module_id": "mod_001", "node_name": "conv2d"}
      ]
    }
  ]
}
```

## Key Features

### Module Uniqueness

Modules are considered unique based on:
- Class name (e.g., `Conv2d`, `Linear`)
- Input/output shapes
- Parameter values (kernel_size, stride, padding, etc.)

This means two `Conv2d` layers with the same configuration will be deduplicated.

### Hierarchical Structure

Each module includes a `parent` field pointing to its parent module path:
- `null` for the root module
- `"(root)"` for top-level children
- Full path like `"block1.layer2"` for nested modules

This allows reconstruction of the full model hierarchy.

### All Module Levels

The tool extracts ALL modules - both container modules (Sequential, ModuleList) and leaf modules (Conv2d, Linear). This provides complete visibility into the model structure.

## Module Structure

```
model_analysis/
├── __init__.py           # Public API exports
├── cli.py                # CLI entry point
├── types.py              # Data classes (ModuleInfo, OpInfo)
├── shapes.py             # Shape capture via forward hooks
├── module_extractor.py   # extract_unique_modules()
├── ops_extractor.py      # extract_pytorch_ops()
├── tracer.py             # FX tracing utilities
└── README.md             # This file
```
