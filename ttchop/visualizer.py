# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive HTML visualization for model analysis results."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import get_parent_path

# Status colors for CSS
STATUS_COLORS = {
    "success": "#22c55e", "failed": "#ef4444", "inherited_success": "#10b981",
    "skipped": "#9ca3af", "unknown": "#d1d5db", "ir_export_failed": "#f59e0b",
}
BADGE_COLORS = {
    "success": ("dcfce7", "166534"), "failed": ("fee2e2", "991b1b"),
    "inherited_success": ("d1fae5", "065f46"), "skipped": ("f3f4f6", "4b5563"),
    "unknown": ("f3f4f6", "6b7280"), "ir_export_failed": ("fef3c7", "92400e"),
}


def generate_visualization(modules_json_path: Path, output_path: Optional[Path] = None) -> Path:
    """Generate HTML visualization from unique_modules.json."""
    modules_json_path = Path(modules_json_path)
    output_path = output_path or modules_json_path.parent / "analysis_report.html"

    with open(modules_json_path) as f:
        data = json.load(f)

    tree = _build_tree(data, modules_json_path.parent)
    html = _generate_html(data, tree)
    output_path.write_text(html)
    print(f"Generated visualization: {output_path}")
    return output_path


def _build_tree(data: Dict, output_dir: Path) -> Dict[str, Any]:
    """Convert flat module list to nested tree structure."""
    modules = data.get("modules", [])
    by_path: Dict[str, Dict] = {}

    for mod in modules:
        original_path = mod["module_path"]
        for occ_path in mod.get("occurrences", [original_path]):
            is_copy = occ_path != original_path
            ir_dir = output_dir / "module_irs" / mod["id"] / "irs"
            by_path[occ_path] = {
                "id": mod["id"], "class_name": mod["class_name"], "module_path": occ_path,
                "parent": get_parent_path(occ_path), "status": mod.get("status", "unknown"),
                "input_shapes": mod.get("input_shapes", []), "output_shapes": mod.get("output_shapes", []),
                "input_dtypes": mod.get("input_dtypes", []), "output_dtypes": mod.get("output_dtypes", []),
                "parameters": mod.get("parameters", {}), "occurrences": mod.get("occurrences", [original_path]),
                "failed_ops": mod.get("failed_ops", []), "op_by_op_report_path": mod.get("op_by_op_report_path"),
                "ir_dir_path": str(ir_dir) if ir_dir.exists() else None,
                "is_copy": is_copy, "original_path": original_path if is_copy else None, "children": [],
            }

    root = None
    for path, node in by_path.items():
        parent = node["parent"]
        if parent is None:
            root = node
        elif parent in by_path:
            by_path[parent]["children"].append(node)

    return root or {"id": "empty", "class_name": "Empty", "children": []}


def _generate_html(data: Dict, tree: Dict) -> str:
    """Generate self-contained HTML."""
    meta = data.get("metadata", {})
    model_class = meta.get("model_class", "Unknown")
    counts = {}
    for m in data.get("modules", []):
        s = m.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1

    # Format timestamp
    ts = meta.get("timestamp", "")
    try:
        from datetime import datetime
        run_date = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"
    except Exception:
        run_date = ts[:16] if len(ts) > 16 else (ts or "Unknown")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Analysis: {model_class}</title>
<style>{_get_css()}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Model Analysis: {model_class}</h1>
    <div class="meta-info">
      <span><strong>Host:</strong> {meta.get("hostname", "unknown")}</span>
      <span><strong>Arch:</strong> {meta.get("device_arch", "unknown")}</span>
      <span><strong>Device:</strong> {meta.get("device_mesh", "unknown")}</span>
      <span><strong>Date:</strong> {run_date}</span>
    </div>
    <div class="stats">
      <span>Total: {meta.get("total_modules", 0)} | Unique: {meta.get("unique_modules", 0)}</span>
      <span class="dot" style="background:{STATUS_COLORS['success']}"></span>Success: {counts.get("success", 0)}
      <span class="dot" style="background:{STATUS_COLORS['failed']}"></span>Failed: {counts.get("failed", 0)}
      <span class="dot" style="background:{STATUS_COLORS['inherited_success']}"></span>Inherited: {counts.get("inherited_success", 0)}
      <span class="dot" style="background:{STATUS_COLORS['skipped']}"></span>Skipped: {counts.get("skipped", 0)}
    </div>
  </div>
  <div class="main">
    <div class="tree" id="tree"></div>
    <div class="details" id="details"><h3>Module Details</h3><p class="empty">Click a module to see details</p></div>
  </div>
</div>
<script>
const treeData = {json.dumps(tree)};
{_get_javascript()}
</script>
</body>
</html>"""


def _get_css() -> str:
    """Return compact CSS styles."""
    status_dots = "\n".join(f".status-{k}{{background:{v}}}" for k, v in STATUS_COLORS.items())
    badges = "\n".join(f".badge.{k}{{background:#{bg};color:#{fg}}}" for k, (bg, fg) in BADGE_COLORS.items())
    return f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333;line-height:1.5}}
.container{{max-width:1400px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:24px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:24px;margin-bottom:8px}}
.meta-info,.stats{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin-top:8px;opacity:0.9}}
.meta-info span,.stats span{{display:flex;align-items:center;gap:6px}}
.dot{{width:10px;height:10px;border-radius:50%}}
.main{{display:flex;gap:20px}}
.tree{{flex:1;background:white;border-radius:12px;padding:20px;min-height:500px;overflow:auto}}
.details{{width:400px;background:white;border-radius:12px;padding:20px;position:sticky;top:20px;max-height:calc(100vh - 40px);overflow-y:auto}}
.details h3{{font-size:16px;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #eee}}
.empty{{color:#999;font-style:italic}}
.row{{margin-bottom:12px}}.lbl{{font-size:11px;text-transform:uppercase;color:#666;margin-bottom:2px}}
.val{{font-size:13px;word-break:break-all}}.val.mono{{font-family:Monaco,Menlo,monospace;font-size:12px;background:#f5f5f5;padding:4px 8px;border-radius:4px}}
.failed-ops{{margin-top:16px;padding-top:16px;border-top:1px solid #fee}}
.failed-ops h4{{color:#dc2626;font-size:14px;margin-bottom:8px}}
.fail-op{{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;margin-bottom:8px}}
.fail-op .op{{font-weight:600;color:#dc2626;margin-bottom:4px}}
.fail-op .err{{font-size:12px;color:#666;font-family:monospace;white-space:pre-wrap;word-break:break-all}}
.node{{margin-left:24px}}.node.root{{margin-left:0}}
.node-row{{display:flex;align-items:center;padding:6px 8px;border-radius:6px;cursor:pointer;transition:background .15s}}
.node-row:hover{{background:#f0f0f0}}.node-row.sel{{background:#e0e7ff}}
.toggle{{width:20px;height:20px;display:flex;align-items:center;justify-content:center;color:#666;font-size:10px}}
.toggle.has{{cursor:pointer}}.toggle.has:hover{{color:#333}}
.sdot{{width:12px;height:12px;border-radius:50%;margin-right:8px;flex-shrink:0}}
{status_dots}
.nlbl{{flex:1;font-size:14px}}.nlbl .n{{font-weight:500}}.nlbl .c{{color:#666;font-size:12px;margin-left:8px}}
.badge{{font-size:10px;padding:2px 8px;border-radius:10px;text-transform:uppercase;font-weight:600}}
{badges}
.copy{{font-size:9px;padding:1px 5px;border-radius:8px;background:#dbeafe;color:#1e40af;margin-left:6px;text-transform:uppercase;font-weight:600}}
.path{{font-family:Monaco,Menlo,monospace;font-size:11px;color:#2563eb;word-break:break-all}}
.children{{display:none}}.children.open{{display:block}}
.ptbl{{width:100%;font-size:12px;border-collapse:collapse}}.ptbl td{{padding:4px 8px;border-bottom:1px solid #eee}}.ptbl td:first-child{{color:#666;width:40%}}
@media(max-width:900px){{.main{{flex-direction:column}}.details{{width:100%;position:static}}}}
"""


def _get_javascript() -> str:
    """Return compact JavaScript."""
    return """
let sel=null;
function render(n,c,root=false){
  const el=document.createElement('div');el.className='node'+(root?' root':'');el.dataset.path=n.module_path;
  const row=document.createElement('div');row.className='node-row';
  const tog=document.createElement('span');tog.className='toggle';
  if(n.children&&n.children.length){tog.className+=' has';tog.innerHTML='&#9654;';tog.onclick=e=>{e.stopPropagation();toggle(el)}}
  row.appendChild(tog);
  const dot=document.createElement('span');dot.className='sdot status-'+n.status;row.appendChild(dot);
  const lbl=document.createElement('span');lbl.className='nlbl';
  let nm=n.module_path==='(root)'?n.class_name:n.module_path.split('.').pop();
  const lp=n.module_path.split('.').pop();if(lp&&lp.includes('['))nm=lp;
  let h='<span class="n">'+esc(nm)+'</span><span class="c">('+esc(n.class_name)+')</span>';
  if(n.is_copy)h+='<span class="copy">copy</span>';
  lbl.innerHTML=h;row.appendChild(lbl);
  const bd=document.createElement('span');bd.className='badge '+n.status;bd.textContent=n.status.replace('_',' ');row.appendChild(bd);
  row.onclick=()=>select(n,row);el.appendChild(row);
  if(n.children&&n.children.length){const ch=document.createElement('div');ch.className='children';n.children.forEach(x=>render(x,ch));el.appendChild(ch)}
  c.appendChild(el);if(root)toggle(el);
}
function toggle(el){const ch=el.querySelector(':scope>.children'),tog=el.querySelector(':scope>.node-row>.toggle');if(ch){ch.classList.toggle('open');tog.innerHTML=ch.classList.contains('open')?'&#9660;':'&#9654;'}}
function select(n,row){document.querySelectorAll('.node-row.sel').forEach(e=>e.classList.remove('sel'));row.classList.add('sel');sel=n;details(n)}
function details(n){
  const p=document.getElementById('details');
  if(!n){p.innerHTML='<h3>Module Details</h3><p class="empty">Click a module to see details</p>';return}
  let h='<h3>Module Details</h3>'+r('Module ID',n.id)+r('Class',n.class_name)+r('Path',n.module_path,1);
  h+=r('Status','<span class="badge '+n.status+'">'+n.status.replace('_',' ')+'</span>');
  if(n.input_shapes.length)h+=r('Input Shapes',n.input_shapes.join(', '),1);
  if(n.output_shapes.length)h+=r('Output Shapes',n.output_shapes.join(', '),1);
  if(n.input_dtypes.length)h+=r('Input Dtypes',n.input_dtypes.join(', '));
  if(Object.keys(n.parameters).length){h+='<div class="row"><div class="lbl">Parameters</div><table class="ptbl">';for(const[k,v]of Object.entries(n.parameters))h+='<tr><td>'+esc(k)+'</td><td>'+esc(JSON.stringify(v))+'</td></tr>';h+='</table></div>'}
  if(n.is_copy)h+=r('Copy Of',n.original_path,1);
  if(n.occurrences.length>1&&!n.is_copy)h+=r('Occurrences ('+n.occurrences.length+')',n.occurrences.join('<br>'),1);
  if(n.ir_dir_path)h+='<div class="row"><div class="lbl">IR Directory</div><div class="val path">'+esc(n.ir_dir_path)+'</div></div>';
  if(n.op_by_op_report_path)h+='<div class="row"><div class="lbl">Op-by-Op Report</div><div class="val path">'+esc(n.op_by_op_report_path)+'</div></div>';
  if(n.failed_ops&&n.failed_ops.length){h+='<div class="failed-ops"><h4>Failed Operations ('+n.failed_ops.length+')</h4>';n.failed_ops.forEach(o=>{h+='<div class="fail-op"><div class="op">'+esc(o.op_name||'Unknown')+'</div>';if(o.error_message)h+='<div class="err">'+esc(o.error_message)+'</div>';h+='</div>'});h+='</div>'}
  p.innerHTML=h;
}
function r(l,v,m=0){return'<div class="row"><div class="lbl">'+esc(l)+'</div><div class="val'+(m?' mono':'')+'">'+v+'</div></div>'}
function esc(s){if(s==null)return'';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
document.addEventListener('DOMContentLoaded',()=>{const c=document.getElementById('tree');if(treeData)render(treeData,c,true);details(null)});
"""
