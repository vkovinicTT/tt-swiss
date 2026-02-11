# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive HTML visualization for model analysis results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import get_parent_path

# Status colors for CSS
STATUS_COLORS = {
    "success": "#22c55e", "failed": "#ef4444", "inherited_success": "#10b981",
    "skipped": "#9ca3af", "unknown": "#d1d5db", "ir_export_failed": "#ef4444",
}
BADGE_COLORS = {
    "success": ("dcfce7", "166534"), "failed": ("fee2e2", "991b1b"),
    "inherited_success": ("d1fae5", "065f46"), "skipped": ("f3f4f6", "4b5563"),
    "unknown": ("f3f4f6", "6b7280"), "ir_export_failed": ("fee2e2", "991b1b"),
}
STATUS_DISPLAY = {
    "success": "success", "failed": "failed", "inherited_success": "inherited success",
    "skipped": "skipped", "unknown": "unknown", "ir_export_failed": "failed",
}

# IR file display order and labels
IR_FILE_ORDER = ["vhlo", "shlo", "shlo_frontend", "shlo_compiler", "ttir", "ttnn"]
IR_FILE_LABELS = {
    "vhlo": "VHLO", "shlo": "StableHLO", "shlo_frontend": "StableHLO Frontend",
    "shlo_compiler": "StableHLO Compiler", "ttir": "TTIR", "ttnn": "TTNN",
}


def _read_file_safe(path: Path) -> Optional[str]:
    """Read file contents safely, return None on failure."""
    try:
        return path.read_text(errors="replace")
    except Exception:
        return None


def _collect_module_files(module_dir: Path, module_id: str) -> Dict[str, Any]:
    """Collect all files for a module (log, IRs, op-by-op report)."""
    files = {"ir_files": {}, "log": None, "op_by_op_report": None, "op_by_op_log": None, "op_by_op_parsed": None}

    log_file = module_dir / "run.log"
    if log_file.exists():
        files["log"] = _read_file_safe(log_file)

    op_log_file = module_dir / "op_by_op.log"
    if op_log_file.exists():
        files["op_by_op_log"] = _read_file_safe(op_log_file)

    parsed_file = module_dir / f"{module_id}_op_by_op_parsed.json"
    if parsed_file.exists():
        files["op_by_op_parsed"] = _read_file_safe(parsed_file)

    report_file = module_dir / f"{module_id}_op_by_op_report.json"
    if report_file.exists():
        files["op_by_op_report"] = _read_file_safe(report_file)

    irs_dir = module_dir / "irs"
    if irs_dir.exists():
        for mlir_file in sorted(irs_dir.glob("*.mlir")):
            name = mlir_file.stem
            ir_type = _classify_ir_file(name)
            files["ir_files"][ir_type] = {
                "name": mlir_file.name,
                "content": _read_file_safe(mlir_file),
            }

    return files


def _classify_ir_file(stem: str) -> str:
    """Classify IR file type from its stem name."""
    stem_lower = stem.lower()
    if "shlo_compiler" in stem_lower:
        return "shlo_compiler"
    if "shlo_frontend" in stem_lower:
        return "shlo_frontend"
    if "shlo" in stem_lower:
        return "shlo"
    if "vhlo" in stem_lower:
        return "vhlo"
    if "ttnn" in stem_lower:
        return "ttnn"
    if "ttir" in stem_lower:
        return "ttir"
    return stem


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
            module_dir = output_dir / "module_irs" / mod["id"]
            ir_dir = module_dir / "irs"

            module_files = _collect_module_files(module_dir, mod["id"]) if module_dir.exists() else {
                "ir_files": {}, "log": None, "op_by_op_report": None,
            }

            by_path[occ_path] = {
                "id": mod["id"], "class_name": mod["class_name"], "module_path": occ_path,
                "parent": get_parent_path(occ_path), "status": mod.get("status", "unknown"),
                "input_shapes": mod.get("input_shapes", []), "output_shapes": mod.get("output_shapes", []),
                "input_dtypes": mod.get("input_dtypes", []), "output_dtypes": mod.get("output_dtypes", []),
                "parameters": mod.get("parameters", {}), "occurrences": mod.get("occurrences", [original_path]),
                "failed_ops": mod.get("failed_ops", []), "op_by_op_report_path": mod.get("op_by_op_report_path"),
                "ir_dir_path": str(ir_dir) if ir_dir.exists() else None,
                "files": module_files,
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
    <button class="theme-toggle" onclick="toggleTheme()"><span class="theme-icon" id="theme-icon">&#9788;</span><span id="theme-label">Light</span></button>
    <div class="meta-info">
      <span><strong>Host:</strong> {meta.get("hostname", "unknown")}</span>
      <span><strong>Arch:</strong> {meta.get("device_arch", "unknown")}</span>
      <span><strong>Device:</strong> {meta.get("device_mesh", "unknown")}</span>
      <span><strong>Date:</strong> {run_date}</span>
    </div>
    <div class="stats">
      <span>Total: {meta.get("total_modules", 0)} | Unique: {meta.get("unique_modules", 0)}</span>
      <span class="dot" style="background:{STATUS_COLORS['success']}"></span>Success: {counts.get("success", 0)}
      <span class="dot" style="background:{STATUS_COLORS['failed']}"></span>Failed: {counts.get("failed", 0) + counts.get("ir_export_failed", 0)}
      <span class="dot" style="background:{STATUS_COLORS['inherited_success']}"></span>Inherited: {counts.get("inherited_success", 0)}
      <span class="dot" style="background:{STATUS_COLORS['skipped']}"></span>Skipped: {counts.get("skipped", 0)}
    </div>
  </div>
  <div class="main">
    <div class="tree" id="tree"></div>
    <div class="details" id="details"><h3>Module Details</h3><p class="empty">Click a module to see details</p></div>
  </div>
</div>
<div class="viewer-overlay" id="viewer" style="display:none">
  <div class="viewer">
    <div class="viewer-header">
      <div class="viewer-title" id="viewer-title"></div>
      <button class="viewer-close" onclick="closeViewer()">&times;</button>
    </div>
    <div class="viewer-tabs" id="viewer-tabs"></div>
    <div class="viewer-body" id="viewer-body"></div>
  </div>
</div>
<script>
const treeData = {json.dumps(tree)};
const statusDisplay = {json.dumps(STATUS_DISPLAY)};
const irOrder = {json.dumps(IR_FILE_ORDER)};
const irLabels = {json.dumps(IR_FILE_LABELS)};
{_get_javascript()}
</script>
</body>
</html>"""


def _get_css() -> str:
    """Return CSS styles."""
    status_dots = "\n".join(f".status-{k}{{background:{v}}}" for k, v in STATUS_COLORS.items())
    badges = "\n".join(f".badge.{k}{{background:#{bg};color:#{fg}}}" for k, (bg, fg) in BADGE_COLORS.items())
    return f"""
*{{box-sizing:border-box;margin:0;padding:0}}

/* --- Theme variables --- */
:root{{
  --bg-page:#0f172a;--bg-panel:#1e293b;--bg-panel-alt:#162032;--bg-hover:#334155;--bg-selected:#312e81;
  --text-primary:#e2e8f0;--text-secondary:#94a3b8;--text-muted:#64748b;
  --border:#334155;--border-light:#475569;
  --mono-bg:#162032;
  --fail-bg:#3b1219;--fail-border:#7f1d1d;--fail-err-text:#cbd5e1;
  --copy-bg:#1e3a5f;--copy-text:#93c5fd;
  --path-color:#818cf8;
  --file-btn-bg:#1e293b;--file-btn-border:#475569;--file-btn-text:#94a3b8;--file-btn-hover-bg:#334155;--file-btn-hover-text:#e2e8f0;
  --ptbl-border:#334155;--ptbl-label:#94a3b8;
}}
body.light{{
  --bg-page:#f5f5f5;--bg-panel:#ffffff;--bg-panel-alt:#f8fafc;--bg-hover:#f0f0f0;--bg-selected:#e0e7ff;
  --text-primary:#333333;--text-secondary:#666666;--text-muted:#999999;
  --border:#e5e7eb;--border-light:#d1d5db;
  --mono-bg:#f5f5f5;
  --fail-bg:#fef2f2;--fail-border:#fecaca;--fail-err-text:#666666;
  --copy-bg:#dbeafe;--copy-text:#1e40af;
  --path-color:#2563eb;
  --file-btn-bg:#f8fafc;--file-btn-border:#e2e8f0;--file-btn-text:#475569;--file-btn-hover-bg:#e2e8f0;--file-btn-hover-text:#1e293b;
  --ptbl-border:#eeeeee;--ptbl-label:#666666;
}}

body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg-page);color:var(--text-primary);line-height:1.5;transition:background .2s,color .2s}}
.container{{max-width:1400px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:24px;border-radius:12px;margin-bottom:20px;position:relative}}
.header h1{{font-size:24px;margin-bottom:8px}}
.meta-info,.stats{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin-top:8px;opacity:0.9}}
.meta-info span,.stats span{{display:flex;align-items:center;gap:6px}}
.dot{{width:10px;height:10px;border-radius:50%}}
.theme-toggle{{position:absolute;top:16px;right:16px;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);color:white;border-radius:8px;padding:6px 12px;font-size:12px;cursor:pointer;transition:background .15s;display:flex;align-items:center;gap:6px}}
.theme-toggle:hover{{background:rgba(255,255,255,.25)}}
.theme-icon{{font-size:16px}}
.main{{display:flex;gap:20px}}
.tree{{flex:1;background:var(--bg-panel);border-radius:12px;padding:20px;min-height:500px;overflow:auto;border:1px solid var(--border);transition:background .2s,border-color .2s}}
.details{{width:400px;background:var(--bg-panel);border-radius:12px;padding:20px;position:sticky;top:20px;max-height:calc(100vh - 40px);overflow-y:auto;border:1px solid var(--border);transition:background .2s,border-color .2s}}
.details h3{{font-size:16px;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.empty{{color:var(--text-muted);font-style:italic}}
.row{{margin-bottom:12px}}.lbl{{font-size:11px;text-transform:uppercase;color:var(--text-secondary);margin-bottom:2px}}
.val{{font-size:13px;word-break:break-all}}.val.mono{{font-family:Monaco,Menlo,monospace;font-size:12px;background:var(--mono-bg);padding:4px 8px;border-radius:4px}}
.failed-ops{{margin-top:16px;padding-top:16px;border-top:1px solid var(--fail-border)}}
.failed-ops h4{{color:#ef4444;font-size:14px;margin-bottom:8px}}
.fail-op{{background:var(--fail-bg);border:1px solid var(--fail-border);border-radius:8px;margin-bottom:6px;overflow:hidden}}
.fail-op-summary{{display:flex;align-items:center;gap:8px;padding:8px 12px;cursor:pointer;transition:background .15s}}
.fail-op-summary:hover{{background:rgba(255,255,255,.03)}}
.fail-op-summary .chevron{{color:var(--text-muted);font-size:9px;transition:transform .2s;flex-shrink:0}}
.fail-op-summary .chevron.open{{transform:rotate(90deg)}}
.fail-op-summary .op{{font-weight:600;color:#ef4444;font-size:13px;flex-shrink:0}}
.fail-op-summary .err-preview{{font-size:11px;color:var(--fail-err-text);font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}}
.fail-op-detail{{display:none;padding:0 12px 10px 28px}}
.fail-op-detail.open{{display:block}}
.fail-op-detail .err{{font-size:11px;color:var(--fail-err-text);font-family:monospace;white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto;background:rgba(0,0,0,.15);padding:8px;border-radius:4px;line-height:1.5}}
.node{{margin-left:24px}}.node.root{{margin-left:0}}
.node-row{{display:flex;align-items:center;padding:6px 8px;border-radius:6px;cursor:pointer;transition:background .15s}}
.node-row:hover{{background:var(--bg-hover)}}.node-row.sel{{background:var(--bg-selected)}}
.toggle{{width:20px;height:20px;display:flex;align-items:center;justify-content:center;color:var(--text-secondary);font-size:10px}}
.toggle.has{{cursor:pointer}}.toggle.has:hover{{color:var(--text-primary)}}
.sdot{{width:12px;height:12px;border-radius:50%;margin-right:8px;flex-shrink:0}}
{status_dots}
.nlbl{{flex:1;font-size:14px}}.nlbl .n{{font-weight:500}}.nlbl .c{{color:var(--text-secondary);font-size:12px;margin-left:8px}}
.badge{{font-size:10px;padding:2px 8px;border-radius:10px;text-transform:uppercase;font-weight:600}}
{badges}
.copy{{font-size:9px;padding:1px 5px;border-radius:8px;background:var(--copy-bg);color:var(--copy-text);margin-left:6px;text-transform:uppercase;font-weight:600}}
.path{{font-family:Monaco,Menlo,monospace;font-size:11px;color:var(--path-color);word-break:break-all}}
.children{{display:none}}.children.open{{display:block}}
.ptbl{{width:100%;font-size:12px;border-collapse:collapse}}.ptbl td{{padding:4px 8px;border-bottom:1px solid var(--ptbl-border)}}.ptbl td:first-child{{color:var(--ptbl-label);width:40%}}
.file-btn{{display:inline-flex;align-items:center;gap:5px;padding:5px 10px;margin:3px;border-radius:6px;border:1px solid var(--file-btn-border);background:var(--file-btn-bg);color:var(--file-btn-text);font-size:11px;font-weight:600;cursor:pointer;transition:all .15s}}
.file-btn:hover{{background:var(--file-btn-hover-bg);color:var(--file-btn-hover-text)}}
.file-btn .icon{{font-size:13px}}
.files-row{{margin-top:12px;display:flex;flex-wrap:wrap;gap:2px}}

.viewer-overlay{{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:1000;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(2px)}}
.viewer{{width:90vw;height:90vh;background:#1e293b;border-radius:12px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 25px 50px rgba(0,0,0,.3)}}
.viewer-header{{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:#0f172a;border-bottom:1px solid #334155}}
.viewer-title{{color:#e2e8f0;font-size:14px;font-weight:600}}
.viewer-close{{background:none;border:none;color:#94a3b8;font-size:24px;cursor:pointer;padding:0 4px;line-height:1}}.viewer-close:hover{{color:#f1f5f9}}
.viewer-tabs{{display:flex;background:#0f172a;padding:0 16px;border-bottom:1px solid #334155;overflow-x:auto}}
.viewer-tabs::-webkit-scrollbar{{height:3px}}.viewer-tabs::-webkit-scrollbar-thumb{{background:#475569;border-radius:2px}}
.vtab{{padding:8px 16px;color:#94a3b8;font-size:12px;font-weight:500;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s}}
.vtab:hover{{color:#e2e8f0}}.vtab.active{{color:#818cf8;border-bottom-color:#818cf8}}
.viewer-body{{flex:1;overflow:auto;padding:0}}
.viewer-body::-webkit-scrollbar{{width:8px}}.viewer-body::-webkit-scrollbar-track{{background:#1e293b}}.viewer-body::-webkit-scrollbar-thumb{{background:#475569;border-radius:4px}}

.code-view{{display:flex;font-family:Monaco,Menlo,'Courier New',monospace;font-size:12px;line-height:1.7}}
.line-nums{{padding:16px 0;text-align:right;user-select:none;color:#475569;background:#162032;min-width:50px}}
.line-nums span{{display:block;padding:0 12px 0 16px}}
.code-text{{padding:16px;color:#e2e8f0;white-space:pre;flex:1;overflow-x:auto}}
.code-text .code-line{{display:inline}}
.line-highlight{{background:rgba(129,140,248,.2) !important}}

.report-view{{padding:20px;color:#e2e8f0}}
.report-view h4{{font-size:14px;margin-bottom:12px;color:#818cf8}}
.report-summary{{display:flex;gap:16px;margin-bottom:20px}}
.report-stat{{background:#162032;padding:12px 20px;border-radius:8px;text-align:center}}
.report-stat .num{{font-size:24px;font-weight:700}}.report-stat .lbl2{{font-size:11px;color:#94a3b8;margin-top:2px}}
.report-stat.pass .num{{color:#22c55e}}.report-stat.fail .num{{color:#ef4444}}.report-stat.total .num{{color:#818cf8}}
.ops-table{{width:100%;border-collapse:collapse;font-size:12px}}
.ops-table th{{text-align:left;padding:8px 12px;background:#162032;color:#94a3b8;font-weight:600;text-transform:uppercase;font-size:10px;position:sticky;top:0}}
.ops-table td{{padding:8px 12px;border-bottom:1px solid #2d3a4d}}
.ops-table tr:hover td{{background:#162032;cursor:pointer}}
.op-pass{{color:#22c55e}}.op-fail{{color:#ef4444}}
.op-err-msg{{font-size:11px;color:#94a3b8;max-width:400px;word-break:break-all}}
.detail-trace{{color:#fbbf24;font-size:10px;font-family:Monaco,Menlo,monospace;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;background:rgba(0,0,0,.2);padding:6px 8px;border-radius:4px;line-height:1.4}}
.detail-op{{color:#94a3b8;font-size:10px;font-style:italic}}

@media(max-width:900px){{.main{{flex-direction:column}}.details{{width:100%;position:static}}}}
"""


def _get_javascript() -> str:
    """Return JavaScript."""
    return """
let sel=null,viewerNode=null,viewerTab=null;

function render(n,c,root=false){
  const el=document.createElement('div');el.className='node'+(root?' root':'');el.dataset.path=n.module_path;
  const row=document.createElement('div');row.className='node-row';
  const tog=document.createElement('span');tog.className='toggle';
  if(n.children&&n.children.length){tog.className+=' has';tog.innerHTML='&#9654;';tog.onclick=e=>{e.stopPropagation();toggle(el)}}
  row.appendChild(tog);
  const dot=document.createElement('span');dot.className='sdot status-'+n.status;row.appendChild(dot);
  const lbl=document.createElement('span');lbl.className='nlbl';
  let nm=(n.module_path==='(root)'||n.module_path==='full_model')?n.class_name:n.module_path.split('.').pop();
  const lp=n.module_path.split('.').pop();if(lp&&lp.includes('['))nm=lp;
  let h='<span class="n">'+esc(nm)+'</span><span class="c">('+esc(n.class_name)+')</span>';
  if(n.is_copy)h+='<span class="copy">copy</span>';
  lbl.innerHTML=h;row.appendChild(lbl);
  const bd=document.createElement('span');bd.className='badge '+n.status;bd.textContent=statusDisplay[n.status]||n.status.replace('_',' ');row.appendChild(bd);
  row.onclick=()=>select(n,row);el.appendChild(row);
  if(n.children&&n.children.length){const ch=document.createElement('div');ch.className='children';n.children.forEach(x=>render(x,ch));el.appendChild(ch)}
  c.appendChild(el);if(root)toggle(el);
}

function toggle(el){const ch=el.querySelector(':scope>.children'),tog=el.querySelector(':scope>.node-row>.toggle');if(ch){ch.classList.toggle('open');tog.innerHTML=ch.classList.contains('open')?'&#9660;':'&#9654;'}}
function select(n,row){document.querySelectorAll('.node-row.sel').forEach(e=>e.classList.remove('sel'));row.classList.add('sel');sel=n;details(n)}

function details(n){
  const p=document.getElementById('details');
  if(!n){p.innerHTML='<h3>Module Details</h3><p class="empty">Click a module to see details</p>';return}
  let ds=statusDisplay[n.status]||n.status.replace('_',' ');
  let h='<h3>Module Details</h3>'+r('Module ID',n.id)+r('Class',n.class_name)+r('Module Name',n.module_path,1);
  h+=r('Status','<span class="badge '+n.status+'">'+ds+'</span>');
  if(n.status==='ir_export_failed')h+=r('Failure Reason','IR export failed for full module');
  if(n.input_shapes.length)h+=r('Input Shapes',n.input_shapes.join(', '),1);
  if(n.output_shapes.length)h+=r('Output Shapes',n.output_shapes.join(', '),1);
  if(n.input_dtypes.length)h+=r('Input Dtypes',n.input_dtypes.join(', '));
  if(Object.keys(n.parameters).length){h+='<div class="row"><div class="lbl">Parameters</div><table class="ptbl">';for(const[k,v]of Object.entries(n.parameters))h+='<tr><td>'+esc(k)+'</td><td>'+esc(JSON.stringify(v))+'</td></tr>';h+='</table></div>'}
  if(n.is_copy)h+=r('Copy Of',n.original_path,1);
  if(n.occurrences.length>1&&!n.is_copy)h+=r('Occurrences ('+n.occurrences.length+')',n.occurrences.join('<br>'),1);

  // File viewer buttons
  const f=n.files||{};
  const hasFiles=f.log||f.op_by_op_log||f.op_by_op_report||Object.keys(f.ir_files||{}).length;
  if(hasFiles){
    h+='<div class="row"><div class="lbl">Files</div><div class="files-row">';
    if(f.op_by_op_report)h+='<button class="file-btn" onclick="openViewer(sel,\\'report\\')"><span class="icon">&#128202;</span>Op-by-Op Report</button>';
    if(f.log)h+='<button class="file-btn" onclick="openViewer(sel,\\'log\\')"><span class="icon">&#128196;</span>Run Log</button>';
    if(f.op_by_op_log)h+='<button class="file-btn" onclick="openViewer(sel,\\'op_log\\')"><span class="icon">&#128196;</span>Op-by-Op Log</button>';
    const irs=f.ir_files||{};
    const ordered=[...irOrder.filter(k=>irs[k]),...Object.keys(irs).filter(k=>!irOrder.includes(k))];
    ordered.forEach(k=>{
      const label=irLabels[k]||k.toUpperCase();
      h+='<button class="file-btn" onclick="openViewer(sel,\\'ir_'+k+'\\')"><span class="icon">&#128209;</span>'+esc(label)+'</button>';
    });
    h+='</div></div>';
  }

  if(n.failed_ops&&n.failed_ops.length){
    let parsedBlocks=[];
    try{if(f.op_by_op_parsed)parsedBlocks=JSON.parse(f.op_by_op_parsed)}catch(e){}
    const failedParsed=parsedBlocks.filter(b=>!b.success);
    h+='<div class="failed-ops"><h4>Failed Operations ('+n.failed_ops.length+')</h4>';
    n.failed_ops.forEach((o,i)=>{
      const pd=failedParsed[i]||{};
      const errFull=pd.error_trace||o.error_message||'';
      const errShort=pd.error_message||o.error_message||'';
      const preview=errShort.length>80?errShort.slice(0,80)+'...':errShort;
      h+='<div class="fail-op">';
      h+='<div class="fail-op-summary" onclick="toggleErr('+i+')">';
      h+='<span class="chevron" id="err-chev-'+i+'">&#9654;</span>';
      h+='<span class="op">'+esc(o.op_name||'Unknown')+'</span>';
      h+='<span class="err-preview">'+esc(preview)+'</span>';
      h+='</div>';
      if(errFull)h+='<div class="fail-op-detail" id="err-detail-'+i+'"><div class="err">'+esc(errFull)+'</div></div>';
      h+='</div>';
    });
    h+='</div>';
  }
  p.innerHTML=h;
}

function openViewer(n,tab){
  viewerNode=n;viewerTab=tab;
  document.getElementById('viewer').style.display='flex';
  document.getElementById('viewer-title').textContent=n.id+' ('+n.class_name+')';
  renderViewerTabs(n,tab);
  renderViewerContent(n,tab);
}

function closeViewer(){
  document.getElementById('viewer').style.display='none';
  viewerNode=null;viewerTab=null;
}

function renderViewerTabs(n,activeTab){
  const f=n.files||{};
  let h='';
  if(f.op_by_op_report)h+='<div class="vtab'+(activeTab==='report'?' active':'')+'" onclick="switchTab(\\'report\\')">Op-by-Op Report</div>';
  if(f.log)h+='<div class="vtab'+(activeTab==='log'?' active':'')+'" onclick="switchTab(\\'log\\')">Run Log</div>';
  if(f.op_by_op_log)h+='<div class="vtab'+(activeTab==='op_log'?' active':'')+'" onclick="switchTab(\\'op_log\\')">Op-by-Op Log</div>';
  const irs=f.ir_files||{};
  const ordered=[...irOrder.filter(k=>irs[k]),...Object.keys(irs).filter(k=>!irOrder.includes(k))];
  ordered.forEach(k=>{
    const key='ir_'+k;
    const label=irLabels[k]||k.toUpperCase();
    h+='<div class="vtab'+(activeTab===key?' active':'')+'" onclick="switchTab(\\''+key+'\\')">'+esc(label)+'</div>';
  });
  document.getElementById('viewer-tabs').innerHTML=h;
}

function switchTab(tab){
  viewerTab=tab;
  renderViewerTabs(viewerNode,tab);
  renderViewerContent(viewerNode,tab);
}

function renderViewerContent(n,tab){
  const body=document.getElementById('viewer-body');
  const f=n.files||{};
  if(tab==='log'){
    body.innerHTML=renderCode(f.log||'No log available');
  }else if(tab==='op_log'){
    body.innerHTML=renderCode(f.op_by_op_log||'No log available');
  }else if(tab==='report'){
    body.innerHTML=renderReport(f.op_by_op_report, f.op_by_op_parsed);
  }else if(tab.startsWith('ir_')){
    const key=tab.slice(3);
    const ir=(f.ir_files||{})[key];
    body.innerHTML=ir?renderCode(ir.content||'Empty file',key==='ttir'):renderCode('File not found');
  }else{
    body.innerHTML='<div style="padding:20px;color:#94a3b8">Unknown tab</div>';
  }
}

function renderCode(text,isTtir){
  const lines=text.split('\\n');
  let nums='',codeLines='';
  lines.forEach((l,i)=>{
    const ln=i+1;
    const id=isTtir?' id="ttir-ln-'+ln+'"':'';
    nums+='<span'+id+'>'+(ln)+'</span>';
    codeLines+='<span class="code-line"'+(isTtir?' data-ln="'+ln+'"':'')+'>'+esc(l)+'</span>\\n';
  });
  return '<div class="code-view"><div class="line-nums">'+nums+'</div><div class="code-text">'+codeLines+'</div></div>';
}

function renderReport(raw, parsedRaw){
  if(!raw)return '<div class="report-view"><p style="color:#94a3b8">No report available</p></div>';
  let data;
  try{data=JSON.parse(raw)}catch(e){return renderCode(raw)}

  let parsed=[];
  if(parsedRaw){try{parsed=JSON.parse(parsedRaw)}catch(e){}}

  const tests=data.tests||[];
  const ops=[];
  tests.forEach(t=>{
    (t.user_properties||[]).forEach(p=>{
      if(typeof p!=='object')return;
      Object.entries(p).forEach(([k,v])=>{
        if(k.startsWith('OpTest model for:')&&typeof v==='object'){
          ops.push(v);
        }
      });
    });
  });

  if(!ops.length)return renderCode(raw);

  const passed=ops.filter(o=>o.success==='True').length;
  const failed=ops.filter(o=>o.success==='False').length;
  const duration=data.duration?data.duration.toFixed(1)+'s':'?';
  const hasParsed=parsed.length>0;

  let h='<div class="report-view">';
  h+='<div class="report-summary">';
  h+='<div class="report-stat total"><div class="num">'+ops.length+'</div><div class="lbl2">Total Ops</div></div>';
  h+='<div class="report-stat pass"><div class="num">'+passed+'</div><div class="lbl2">Passed</div></div>';
  h+='<div class="report-stat fail"><div class="num">'+failed+'</div><div class="lbl2">Failed</div></div>';
  h+='<div class="report-stat"><div class="num" style="color:#818cf8">'+duration+'</div><div class="lbl2">Duration</div></div>';
  h+='</div>';

  h+='<table class="ops-table"><thead><tr><th>Status</th><th>Op Name</th><th>Inputs</th><th>Outputs</th><th>Details</th>';
  h+='</tr></thead><tbody>';

  // Check if TTIR tab is available for linking
  const hasTtir=viewerNode&&viewerNode.files&&viewerNode.files.ir_files&&viewerNode.files.ir_files.ttir;

  // Keep original indices before sorting so we can look up parsed blocks
  const indexed=ops.map((o,i)=>({...o,_idx:i}));
  indexed.sort((a,b)=>(a.success==='True'?1:0)-(b.success==='True'?1:0));
  indexed.forEach(o=>{
    const ok=o.success==='True';
    const detail=hasParsed?(parsed[o._idx]||{}):{};
    const clickAttr=hasTtir?' style="cursor:pointer" onclick="jumpToTtirOp(\\''+esc(o.op_name||'')+'\\','+o._idx+')"':'';
    h+='<tr'+clickAttr+'><td class="'+(ok?'op-pass':'op-fail')+'">'+(ok?'PASS':'FAIL')+'</td>';
    h+='<td><strong>'+esc(o.op_name||'?')+'</strong></td>';
    h+='<td style="font-size:11px;color:#94a3b8">'+esc(formatTensors(o.inputs))+'</td>';
    h+='<td style="font-size:11px;color:#94a3b8">'+esc(formatTensors(o.outputs))+'</td>';
    h+='<td>';
    if(detail.error_trace){
      h+='<div class="detail-trace">'+esc(detail.error_trace)+'</div>';
    }else if(o.error_message){
      h+='<div class="op-err-msg">'+esc(o.error_message)+'</div>';
    }else if(detail.last_ttnn_op){
      h+='<div class="detail-op">Last op: '+esc(detail.last_ttnn_op)+'</div>';
    }else{
      h+=ok?'-':'Unknown error';
    }
    h+='</td>';
    h+='</tr>';
  });
  h+='</tbody></table></div>';
  return h;
}

function formatTensors(s){
  if(!s||s==='[]')return '-';
  return s.replace(/TensorDesc\\(/g,'').replace(/\\)/g,'').replace(/,\\s*buffer_type=None/g,'').replace(/,\\s*layout=None/g,'').replace(/,\\s*grid_shape=None/g,'').replace(/data_type=/g,'').replace(/shape=\\[/g,'[');
}

function toggleErr(i){const d=document.getElementById('err-detail-'+i),c=document.getElementById('err-chev-'+i);if(d){d.classList.toggle('open');c.classList.toggle('open')}}

function jumpToTtirOp(opName,opIdx){
  // Switch to TTIR tab
  switchTab('ir_ttir');
  // Wait for render, then find the Nth occurrence of the op in the code
  setTimeout(()=>{
    const codeEl=document.querySelector('.code-text');
    if(!codeEl)return;
    const codeLines=codeEl.querySelectorAll('.code-line');
    // Find the opIdx-th ttir op line (matching sequential op order)
    let found=0;
    for(let i=0;i<codeLines.length;i++){
      const txt=codeLines[i].textContent;
      if(txt.match(/"ttir\\./)){
        if(found===opIdx){
          // Remove previous highlights
          document.querySelectorAll('.line-highlight').forEach(el=>el.classList.remove('line-highlight'));
          codeLines[i].classList.add('line-highlight');
          // Also highlight the line number
          const lnEl=document.getElementById('ttir-ln-'+(i+1));
          if(lnEl)lnEl.classList.add('line-highlight');
          codeLines[i].scrollIntoView({behavior:'smooth',block:'center'});
          return;
        }
        found++;
      }
    }
  },50);
}

function r(l,v,m=0){return '<div class="row"><div class="lbl">'+esc(l)+'</div><div class="val'+(m?' mono':'')+'">'+v+'</div></div>'}
function esc(s){if(s==null)return '';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

document.addEventListener('DOMContentLoaded',()=>{
  const c=document.getElementById('tree');if(treeData)render(treeData,c,true);details(null);
});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeViewer()});

function toggleTheme(){
  document.body.classList.toggle('light');
  const isLight=document.body.classList.contains('light');
  document.getElementById('theme-icon').innerHTML=isLight?'&#9790;':'&#9788;';
  document.getElementById('theme-label').textContent=isLight?'Dark':'Light';
  try{localStorage.setItem('ttchop-theme',isLight?'light':'dark')}catch(e){}
}
(function(){
  try{if(localStorage.getItem('ttchop-theme')==='light'){document.body.classList.add('light');document.getElementById('theme-icon').innerHTML='&#9790;';document.getElementById('theme-label').textContent='Dark'}}catch(e){}
})();
"""
