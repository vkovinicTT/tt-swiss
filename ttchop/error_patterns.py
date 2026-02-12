# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared error pattern definitions for Python-side matching.

Keep in sync with the JS ERROR_PATTERNS array in visualizer.py.
"""

import re
from typing import List, Optional, Tuple

ERROR_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "L1 Circular Buffer Overflow",
        re.compile(
            r"Statically allocated circular buffers on core range "
            r"\[\(x=\d+,y=\d+\) - \(x=\d+,y=\d+\)\] grow to \d+ B "
            r"which is beyond max L1 size of \d+ B"
        ),
    ),
]


def match_error_pattern(text: str) -> Optional[str]:
    """Return first pattern match found in text, or None."""
    if not text:
        return None
    for _name, pattern in ERROR_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None
