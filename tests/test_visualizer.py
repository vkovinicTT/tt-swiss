# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HTML visualizer: report generation, program boundary lines,
per-program IR tabs, and backward compatibility with old JSON format."""

import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from memory_profiler.visualizer import MemoryVisualizer


@pytest.fixture
def vae_report_dir(parsed_vae):
    """Set up a report directory with proper file naming for the visualizer."""
    tmpdir = tempfile.mkdtemp(prefix="ttmem_viz_vae_")
    name = "vae"
    for key in ("memory", "operations", "registry", "ir"):
        src = parsed_vae["paths"][key]
        dst = Path(tmpdir) / f"{name}_{key.replace('registry', 'inputs_registry')}.json"
        shutil.copy2(src, dst)
    yield Path(tmpdir), name
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def multihost_report_dir(parsed_multihost):
    """Set up a report directory with proper file naming for the visualizer."""
    tmpdir = tempfile.mkdtemp(prefix="ttmem_viz_mh_")
    name = "multihost"
    for key in ("memory", "operations", "registry", "ir"):
        src = parsed_multihost["paths"][key]
        dst = Path(tmpdir) / f"{name}_{key.replace('registry', 'inputs_registry')}.json"
        shutil.copy2(src, dst)
    yield Path(tmpdir), name
    shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_plotly_layout(html: str):
    """Extract the memoryData layout from the HTML."""
    match = re.search(r"const memoryData = (.+?);\n", html)
    if match:
        return json.loads(match.group(1)).get("layout", {})
    return {}


# ---------------------------------------------------------------------------
# Vae report (single-program)
# ---------------------------------------------------------------------------

class TestVaeReport:
    def test_generates_html(self, vae_report_dir):
        run_dir, name = vae_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        assert report.exists()
        assert report.stat().st_size > 0

    def test_no_program_boundary_lines(self, vae_report_dir):
        run_dir, name = vae_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        html = report.read_text()
        layout = _extract_plotly_layout(html)
        # Single program: no boundary shapes
        assert "shapes" not in layout or len(layout["shapes"]) == 0

    def test_no_program_tabs(self, vae_report_dir):
        run_dir, name = vae_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        html = report.read_text()
        buttons = re.findall(r'<button class="ir-program-tab', html)
        assert len(buttons) == 0


# ---------------------------------------------------------------------------
# Multihost report (multi-program)
# ---------------------------------------------------------------------------

class TestMultihostReport:
    def test_generates_html(self, multihost_report_dir):
        run_dir, name = multihost_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        assert report.exists()
        assert report.stat().st_size > 0

    def test_has_boundary_lines(self, multihost_report_dir):
        run_dir, name = multihost_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        html = report.read_text()
        layout = _extract_plotly_layout(html)
        shapes = layout.get("shapes", [])
        assert len(shapes) >= 1
        # Check it's a green dashed line
        assert shapes[0]["line"]["dash"] == "dash"
        assert "76, 175, 80" in shapes[0]["line"]["color"]

    def test_has_program_tabs(self, multihost_report_dir):
        run_dir, name = multihost_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        html = report.read_text()
        buttons = re.findall(r'<button class="ir-program-tab', html)
        assert len(buttons) >= 2

    def test_boundary_annotation_text(self, multihost_report_dir):
        run_dir, name = multihost_report_dir
        viz = MemoryVisualizer(run_dir, script_name=name)
        report = viz.generate_report()
        html = report.read_text()
        layout = _extract_plotly_layout(html)
        annotations = layout.get("annotations", [])
        assert len(annotations) >= 1
        assert annotations[0]["text"] == "program_1"


# ---------------------------------------------------------------------------
# Backward compatibility: old JSON format (no 'programs' in metadata)
# ---------------------------------------------------------------------------

class TestOldFormatBackwardCompat:
    def test_loads_without_programs_metadata(self, vae_report_dir):
        """Simulate old format by removing programs from metadata."""
        run_dir, name = vae_report_dir
        mem_file = run_dir / f"{name}_memory.json"
        with open(mem_file) as f:
            data = json.load(f)
        # Remove programs key to simulate old format
        data["metadata"].pop("programs", None)
        with open(mem_file, "w") as f:
            json.dump(data, f)

        viz = MemoryVisualizer(run_dir, script_name=name)
        assert len(viz.programs) == 1
        assert viz.programs[0]["name"] == "main"
        report = viz.generate_report()
        assert report.exists()
