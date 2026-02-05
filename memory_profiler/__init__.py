# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Memory profiler package for analyzing memory usage in model execution on Tenstorrent hardware.

This package provides tools to:
- Parse runtime logs containing MLIR operations and memory statistics
- Extract synchronized operation and memory data
- Generate JSON outputs for analysis
- Create interactive HTML visualization reports
- Generate LLM-friendly text reports
"""

from .parser import parse_log_file
from .text_formatter import LLMTextFormatter

__all__ = ["parse_log_file", "LLMTextFormatter"]
