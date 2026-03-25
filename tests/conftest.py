# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from memory_profiler.parser import parse_log_file

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_LOGS = REPO_ROOT / "example_logs"


@pytest.fixture(scope="session")
def vae_log_path():
    return str(EXAMPLE_LOGS / "vae.log")


@pytest.fixture(scope="session")
def multihost_log_path():
    return str(EXAMPLE_LOGS / "multihost.log")


def _parse_log_to_dir(log_path, prefix):
    """Run parse_log_file into a temp directory, return paths dict."""
    tmpdir = tempfile.mkdtemp(prefix=f"ttmem_test_{prefix}_")
    mem = f"{tmpdir}/{prefix}_memory.json"
    ops = f"{tmpdir}/{prefix}_operations.json"
    reg = f"{tmpdir}/{prefix}_inputs_registry.json"
    ir = f"{tmpdir}/{prefix}_ir.json"
    parse_log_file(log_path, mem, ops, reg, ir)
    return {
        "dir": tmpdir,
        "memory": mem,
        "operations": ops,
        "registry": reg,
        "ir": ir,
    }


def _load_parsed(paths):
    """Load all JSON outputs into a single dict."""
    with open(paths["memory"]) as f:
        mem_json = json.load(f)
    with open(paths["operations"]) as f:
        ops_json = json.load(f)
    with open(paths["registry"]) as f:
        reg_json = json.load(f)
    with open(paths["ir"]) as f:
        ir_json = json.load(f)
    return {
        "mem": mem_json,
        "ops": ops_json,
        "reg": reg_json,
        "ir": ir_json,
        "paths": paths,
    }


@pytest.fixture(scope="session")
def parsed_vae(vae_log_path):
    paths = _parse_log_to_dir(vae_log_path, "vae")
    result = _load_parsed(paths)
    yield result
    shutil.rmtree(paths["dir"], ignore_errors=True)


@pytest.fixture(scope="session")
def parsed_multihost(multihost_log_path):
    paths = _parse_log_to_dir(multihost_log_path, "multihost")
    result = _load_parsed(paths)
    yield result
    shutil.rmtree(paths["dir"], ignore_errors=True)
