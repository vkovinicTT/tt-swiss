# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI for model analysis: ttchop --model-path file.py::fn --inputs-path file.py::fn"""

import argparse
import json
import sys
from pathlib import Path

from .module_extractor import extract_unique_modules
from .module_tree import build_module_tree, update_modules_with_status
from .op_by_op_runner import run_hierarchical_op_by_op
from .utils import load_function_from_path, get_tt_xla_root


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch model and run op-by-op tests")
    parser.add_argument("--model-path", required=True, help="Model loader (file.py::function)")
    parser.add_argument("--inputs-path", required=True, help="Inputs function (file.py::function)")
    parser.add_argument("--dir", default=None, help="Output directory")
    args = parser.parse_args()

    try:
        load_fn = load_function_from_path(args.model_path)
        inputs_fn = load_function_from_path(args.inputs_path)
    except Exception as e:
        print(f"Error loading functions: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup output directory
    model_class = type(load_fn()).__name__
    output_dir = Path(args.dir) if args.dir else Path.cwd() / model_class
    output_dir.mkdir(parents=True, exist_ok=True)
    modules_json = output_dir / "unique_modules.json"
    project_root = get_tt_xla_root()

    # Step 1: Extract modules
    print(f"\n{'='*60}\nStep 1: Extracting unique modules\n{'='*60}")
    try:
        result = extract_unique_modules(
            load_fn=load_fn, get_sample_input=inputs_fn,
            output_path=str(modules_json), model_path=args.model_path,
            inputs_path=args.inputs_path, use_device=True,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Hierarchical op-by-op
    print(f"\n{'='*60}\nStep 2: Running hierarchical op-by-op analysis\n{'='*60}")
    try:
        root = build_module_tree(result)
        if root:
            print(f"Root: {root.module_id} ({root.class_name})\n")
            run_hierarchical_op_by_op(
                root=root, module_irs_base=output_dir / "module_irs",
                project_root=project_root, modules_json_path=modules_json,
                model_path=args.model_path, inputs_path=args.inputs_path,
                output_dir=output_dir,
            )
            result = update_modules_with_status(result, root)
            with open(modules_json, "w") as f:
                json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

    # Summary
    print(f"\n{'='*60}\nAnalysis Complete!\n{'='*60}")
    meta = result["metadata"]
    print(f"Model: {meta['model_class']}, Modules: {meta['total_modules']} total / {meta['unique_modules']} unique")

    counts = {}
    for m in result.get("modules", []):
        s = m.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    if counts:
        print("Results: " + ", ".join(f"{s}: {c}" for s, c in sorted(counts.items())))

    # Step 3: Visualization
    print(f"\n{'='*60}\nStep 3: Generating HTML visualization\n{'='*60}")
    try:
        from .visualizer import generate_visualization
        report = generate_visualization(modules_json)
        print(f"\nOutputs: {modules_json}, {output_dir / 'module_irs'}, {report}")
    except Exception as e:
        print(f"Warning: {e}")


if __name__ == "__main__":
    main()
