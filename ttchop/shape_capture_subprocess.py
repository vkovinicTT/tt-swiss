# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess script for shape capture on TT device."""

import argparse
import json
import sys
from pathlib import Path

# This script runs as a standalone subprocess. Add the parent directory
# so Python can resolve the ttchop package for normal imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttchop.utils import load_function_from_path, setup_tt_device, get_device_info


def main():
    parser = argparse.ArgumentParser(description="Capture shapes (subprocess)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inputs-path", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    import torch_xla

    print("Setting up TT device...")
    device = setup_tt_device()
    print(f"Using device: {device}")

    load_fn = load_function_from_path(args.model_path)
    inputs_fn = load_function_from_path(args.inputs_path)

    print(f"Loading model using {load_fn.__name__}...")
    model = load_fn()
    model.eval()

    print("Getting sample input...")
    sample_input = inputs_fn()

    from ttchop.shapes import ShapeCapture
    print("Capturing shapes on TT device...")
    shapes = ShapeCapture(model).run(sample_input, device=device)
    torch_xla.sync()

    device_info = get_device_info()
    print(f"Device: {device_info}")

    output_path = Path(args.output_file)
    print(f"Saving shapes to {output_path}...")
    with open(output_path, "w") as f:
        json.dump({"shapes": shapes, "device_info": device_info}, f, indent=2)

    print(f"Captured shapes for {len(shapes)} modules")


if __name__ == "__main__":
    main()
