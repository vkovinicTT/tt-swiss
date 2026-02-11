# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Module tree structure for hierarchical analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModuleNode:
    """Node in the module tree."""
    module_id: str
    class_name: str
    module_path: str
    parent_path: Optional[str]
    children: List["ModuleNode"] = field(default_factory=list)
    status: Optional[str] = None
    failed_ops: List[Dict[str, Any]] = field(default_factory=list)
    op_by_op_report_path: Optional[str] = None


def build_module_tree(modules_data: Dict) -> Optional[ModuleNode]:
    """Build N-ary tree from unique_modules.json data."""
    modules = modules_data.get("modules", [])
    if not modules:
        return None

    nodes: Dict[str, ModuleNode] = {
        m["module_path"]: ModuleNode(
            module_id=m["id"], class_name=m["class_name"],
            module_path=m["module_path"], parent_path=m.get("parent"),
        ) for m in modules
    }

    root = None
    for path, node in nodes.items():
        if node.parent_path is None:
            root = node
        elif node.parent_path in ("(root)", "full_model") and node.parent_path in nodes:
            nodes[node.parent_path].children.append(node)
        elif node.parent_path in nodes:
            nodes[node.parent_path].children.append(node)
        else:
            # Orphan: find ancestor
            parts = path.split(".")
            for i in range(len(parts) - 1, 0, -1):
                ancestor = ".".join(parts[:i])
                if ancestor in nodes:
                    nodes[ancestor].children.append(node)
                    break
            else:
                if root:
                    root.children.append(node)

    return root


def update_modules_with_status(modules_data: Dict, root: ModuleNode) -> Dict:
    """Update modules_data with status from tree."""
    status_map: Dict[str, Dict[str, Any]] = {}

    def collect(node: ModuleNode):
        status_map[node.module_path] = {
            "status": node.status,
            "failed_ops": node.failed_ops,
            "op_by_op_report_path": node.op_by_op_report_path,
        }
        for child in node.children:
            collect(child)

    if root:
        collect(root)

    for mod in modules_data.get("modules", []):
        if mod["module_path"] in status_map:
            mod.update(status_map[mod["module_path"]])

    return modules_data
