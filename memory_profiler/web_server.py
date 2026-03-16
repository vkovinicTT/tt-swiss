#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Web server for tt-swiss memory profiler.

Provides a browser-based UI to upload log files and generate interactive HTML reports.

Usage:
    ttmem-web              # Start on 0.0.0.0:8001
    ttmem-web --port 9000  # Custom port
"""

import argparse
import json
import mimetypes
import threading
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from memory_profiler.run_profiled import sanitize_report_name, get_reports_dir
from memory_profiler.parser import parse_log_file, validate_outputs
from memory_profiler.visualizer import MemoryVisualizer

# Shared state: UUID -> {state, report_url, error, filename}
jobs = {}

UPLOADS_DIR = Path.home() / ".ttmem" / "uploads"


def process_job(job_id, upload_path, filename):
    """Process an uploaded log file in a background thread."""
    try:
        jobs[job_id]["state"] = "processing"

        log_name = Path(filename).stem
        report_name = sanitize_report_name(log_name)
        short_id = job_id[:8]
        report_dir = get_reports_dir() / f"{report_name}-{short_id}"
        report_dir.mkdir(parents=True, exist_ok=True)

        mem_json = report_dir / f"{report_name}_memory.json"
        ops_json = report_dir / f"{report_name}_operations.json"
        registry_json = report_dir / f"{report_name}_inputs_registry.json"
        ir_json = report_dir / f"{report_name}_ir.json"

        parse_log_file(
            str(upload_path),
            str(mem_json),
            str(ops_json),
            str(registry_json),
            str(ir_json),
        )
        validate_outputs(str(mem_json), str(ops_json))

        viz = MemoryVisualizer(report_dir, script_name=report_name)
        report_path = viz.generate_report()

        report_rel = f"/reports/{report_dir.name}/{report_path.name}"
        jobs[job_id]["state"] = "done"
        jobs[job_id]["report_url"] = report_rel

    except Exception as e:
        jobs[job_id]["state"] = "error"
        jobs[job_id]["error"] = str(e)

    finally:
        # Clean up upload
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass


def parse_multipart(body, content_type):
    """Extract the first file from a multipart/form-data body.

    Returns (filename, file_bytes) or raises ValueError.
    """
    # Extract boundary from Content-Type header
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):]
            break
    if not boundary:
        raise ValueError("No boundary in Content-Type")

    delimiter = f"--{boundary}".encode()
    parts = body.split(delimiter)

    for part in parts:
        if b"Content-Disposition:" not in part:
            continue
        # Split headers from body
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_raw = part[:header_end].decode(errors="replace")
        file_data = part[header_end + 4:]
        # Strip trailing \r\n or -- before next boundary
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        if file_data.endswith(b"--"):
            file_data = file_data[:-2]
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]

        # Extract filename
        filename = None
        for line in headers_raw.split("\r\n"):
            if "filename=" in line:
                # filename="foo.log" or filename=foo.log
                idx = line.index("filename=")
                rest = line[idx + len("filename="):]
                filename = rest.strip('" ')
                break
        if filename:
            return filename, file_data

    raise ValueError("No file found in multipart body")


UPLOAD_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TT Memory Profiler</title>
<style>
:root {
  --bg-canvas: #111217;
  --bg-primary: #181b1f;
  --bg-secondary: #1e2228;
  --border-primary: #2a2e35;
  --text-primary: #e4e4e7;
  --text-secondary: #a0a0ab;
  --accent-primary: #3d71d9;
  --accent-hover: #4a80e8;
  --accent-danger: #d94040;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg-canvas);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  display: flex; align-items: center; justify-content: center;
  min-height: 100vh;
}
.card {
  background: var(--bg-primary);
  border: 1px solid var(--border-primary);
  border-radius: 12px;
  padding: 2.5rem;
  width: 100%; max-width: 480px;
  text-align: center;
}
h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
.subtitle { color: var(--text-secondary); margin-bottom: 2rem; font-size: 0.9rem; }
.file-input-wrapper {
  border: 2px dashed var(--border-primary);
  border-radius: 8px;
  padding: 2rem 1rem;
  margin-bottom: 1.5rem;
  cursor: pointer;
  transition: border-color 0.2s;
}
.file-input-wrapper:hover, .file-input-wrapper.dragover {
  border-color: var(--accent-primary);
}
.file-input-wrapper input { display: none; }
.file-label { color: var(--text-secondary); font-size: 0.9rem; }
.file-name { color: var(--text-primary); margin-top: 0.5rem; font-weight: 500; }
button {
  background: var(--accent-primary);
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0.75rem 2rem;
  font-size: 1rem;
  cursor: pointer;
  width: 100%;
  transition: background 0.2s;
}
button:hover { background: var(--accent-hover); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.status {
  margin-top: 1.5rem;
  font-size: 0.9rem;
  color: var(--text-secondary);
}
.error { color: var(--accent-danger); }
.spinner {
  display: inline-block;
  width: 18px; height: 18px;
  border: 2px solid var(--border-primary);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  vertical-align: middle;
  margin-right: 0.5rem;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>TT Memory Profiler</h1>
  <p class="subtitle">Upload a log file to generate an interactive report</p>
  <div class="file-input-wrapper" id="dropzone">
    <input type="file" id="file" accept=".log,.txt">
    <div class="file-label" id="fileLabel">Click or drag &amp; drop a .log file</div>
    <div class="file-name" id="fileName"></div>
  </div>
  <button id="btn" disabled>Create Report</button>
  <div class="status" id="status"></div>
</div>
<script>
const fileInput = document.getElementById('file');
const dropzone = document.getElementById('dropzone');
const fileLabel = document.getElementById('fileLabel');
const fileName = document.getElementById('fileName');
const btn = document.getElementById('btn');
const status = document.getElementById('status');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; onFileChange(); }
});
fileInput.addEventListener('change', onFileChange);

function onFileChange() {
  if (fileInput.files.length) {
    fileName.textContent = fileInput.files[0].name;
    fileLabel.textContent = 'Selected file:';
    btn.disabled = false;
    status.innerHTML = '';
  }
}

btn.addEventListener('click', async () => {
  if (!fileInput.files.length) return;
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Uploading...';

  const form = new FormData();
  form.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error('Upload failed: ' + res.statusText);
    const data = await res.json();
    const reqId = data.request_id;
    status.innerHTML = '<span class="spinner"></span> Processing log file...';
    pollStatus(reqId);
  } catch (e) {
    status.innerHTML = '<span class="error">Error: ' + e.message + '</span>';
    btn.disabled = false;
  }
});

function pollStatus(id) {
  const iv = setInterval(async () => {
    try {
      const res = await fetch('/status/' + id);
      const data = await res.json();
      if (data.state === 'done') {
        clearInterval(iv);
        window.location.href = data.report_url;
      } else if (data.state === 'error') {
        clearInterval(iv);
        status.innerHTML = '<span class="error">Error: ' + data.error + '</span>';
        btn.disabled = false;
      }
    } catch (e) {
      clearInterval(iv);
      status.innerHTML = '<span class="error">Polling error: ' + e.message + '</span>';
      btn.disabled = false;
    }
  }, 2000);
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._respond_html(200, UPLOAD_PAGE)

        elif path.startswith("/status/"):
            job_id = path[len("/status/"):]
            job = jobs.get(job_id)
            if not job:
                self._respond_json(404, {"error": "not found"})
            else:
                self._respond_json(200, {
                    "state": job["state"],
                    "report_url": job.get("report_url"),
                    "error": job.get("error"),
                })

        elif path.startswith("/reports/"):
            # Serve from ~/.ttmem/reports/
            rel = path[len("/reports/"):]
            # Prevent directory traversal
            if ".." in rel:
                self._respond_json(400, {"error": "invalid path"})
                return
            file_path = get_reports_dir() / rel
            if file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                if mime is None:
                    mime = "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                content = file_path.read_bytes()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._respond_json(404, {"error": "not found"})

        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/upload":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if "multipart/form-data" not in content_type:
                self._respond_json(400, {"error": "expected multipart/form-data"})
                return

            body = self.rfile.read(content_length)

            try:
                filename, file_data = parse_multipart(body, content_type)
            except ValueError as e:
                self._respond_json(400, {"error": str(e)})
                return

            job_id = str(uuid.uuid4())
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            upload_path = UPLOADS_DIR / f"{job_id}.log"
            upload_path.write_bytes(file_data)

            jobs[job_id] = {
                "state": "queued",
                "report_url": None,
                "error": None,
                "filename": filename,
            }

            thread = threading.Thread(
                target=process_job,
                args=(job_id, upload_path, filename),
                daemon=True,
            )
            thread.start()

            self._respond_json(200, {"request_id": job_id})
        else:
            self._respond_json(404, {"error": "not found"})

    def _respond_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_html(self, code, html):
        body = html.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Use simpler log format
        print(f"[ttmem-web] {args[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Web UI for TT Memory Profiler"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to listen on (default: 8001)"
    )
    parser.add_argument(
        "--bind", default="0.0.0.0", help="Address to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"TT Memory Profiler Web UI")
    print(f"Listening on http://{args.bind}:{args.port}")
    print(f"Reports directory: {get_reports_dir()}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
