# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multihost MPI log detection and filtering utilities.

MPI-based multihost logs prefix each line with [rank,device]<stdout>:
e.g., [1,0]<stdout>:                  Always |     INFO | Device DRAM memory state: ...

This module provides utilities to:
- Detect whether a log file contains multihost MPI prefixes
- Filter lines to a single host
- Strip the MPI prefix so downstream parsers work unchanged
"""

import re
from typing import List, Optional

# Pattern matching MPI host prefix: [rank,device]<stdout>:
# Captures the host identifier (e.g., "1,0") and allows optional whitespace after colon
MPI_PREFIX_PATTERN = re.compile(r"^\[(\d+,\d+)\]<stdout>:\s?")


def strip_mpi_prefix(line: str) -> str:
    """Strip the [rank,device]<stdout>: prefix from a line, if present."""
    return MPI_PREFIX_PATTERN.sub("", line)


def extract_host(line: str) -> Optional[str]:
    """Extract the host identifier (e.g., '1,0') from an MPI-prefixed line.

    Returns None if the line has no MPI prefix.
    """
    match = MPI_PREFIX_PATTERN.match(line)
    return match.group(1) if match else None


def line_matches_host(line: str, host: str) -> bool:
    """Check if a line belongs to the specified host (e.g., '1,0')."""
    return line.startswith(f"[{host}]<stdout>:")


def detect_multihost(log_path: str, scan_lines: int = 200) -> Optional[List[str]]:
    """
    Scan the first N lines of a log file for MPI host prefixes.

    Args:
        log_path: Path to the log file
        scan_lines: Number of lines to scan (default 200)

    Returns:
        Sorted list of unique host identifiers found (e.g., ['0,0', '1,0', '1,1']),
        or None if this is not a multihost log.
    """
    hosts = set()
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= scan_lines:
                    break
                host = extract_host(line)
                if host:
                    hosts.add(host)
    except (FileNotFoundError, OSError):
        return None

    return sorted(hosts) if hosts else None
