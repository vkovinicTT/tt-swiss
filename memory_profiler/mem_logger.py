# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight RSS memory logger for diagnosing memory spikes in the parsing pipeline.

Usage:
    from mem_logger import log_mem

    log_mem("before parsing")
    ...
    log_mem("after parsing")

Enable by setting environment variable: TTMEM_LOG_MEMORY=1
"""

import os
import resource
import sys

_ENABLED = os.environ.get("TTMEM_LOG_MEMORY", "0") == "1"


def get_rss_mb() -> float:
    """Return current RSS in MB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def log_mem(label: str) -> None:
    """Print current RSS with a label. No-op unless TTMEM_LOG_MEMORY=1."""
    if not _ENABLED:
        return
    rss = get_rss_mb()
    print(f"[MEM] {label}: RSS = {rss:.1f} MB", file=sys.stderr)
