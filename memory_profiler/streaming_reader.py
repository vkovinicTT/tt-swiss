# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Memory-efficient streaming line reader with look-ahead buffer.

Instead of loading an entire file into memory with readlines(), this reader
streams lines from the file while maintaining a small look-ahead buffer
for parsers that need to peek at upcoming lines.
"""

from collections import deque
from typing import List, Optional


class BufferedLineReader:
    """
    Reads a file line-by-line, keeping only a sliding window in memory.

    Supports:
    - current_line: the line at the current position
    - peek(offset): look ahead by `offset` lines from current
    - peek_slice(start, count): get a list of lines for look-ahead parsing
    - advance(): move forward one line
    """

    def __init__(self, file_path: str, buffer_size: int = 10):
        self._file = open(file_path, "r", encoding="utf-8", errors="replace")
        self._buffer: deque = deque()
        self._buffer_size = buffer_size
        self._exhausted = False
        self._lines_consumed = 0  # total lines that have passed through
        self._fill()

    def _fill(self) -> None:
        """Fill the buffer up to buffer_size."""
        while len(self._buffer) < self._buffer_size and not self._exhausted:
            line = self._file.readline()
            if line:
                self._buffer.append(line)
            else:
                self._exhausted = True
                break

    @property
    def has_lines(self) -> bool:
        """True if there are lines remaining to process."""
        return len(self._buffer) > 0

    @property
    def current_line(self) -> Optional[str]:
        """The line at the current read position."""
        return self._buffer[0] if self._buffer else None

    def peek(self, offset: int) -> Optional[str]:
        """Look ahead by `offset` lines from the current position."""
        if offset < len(self._buffer):
            return self._buffer[offset]
        return None

    def peek_slice(self, start: int, count: int) -> List[str]:
        """
        Get a list of lines starting at `start` offset from current position.

        This is compatible with functions that expect (lines, start_idx) by
        returning a list where index 0 corresponds to `start` offset.
        """
        result = []
        for i in range(start, start + count):
            if i < len(self._buffer):
                result.append(self._buffer[i])
            else:
                break
        return result

    def advance(self) -> None:
        """Move to the next line."""
        if self._buffer:
            self._buffer.popleft()
            self._lines_consumed += 1
            self._fill()

    @property
    def total_consumed(self) -> int:
        """Total number of lines consumed (advanced past) so far."""
        return self._lines_consumed

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
