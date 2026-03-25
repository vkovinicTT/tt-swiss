# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for tile padding overhead data.

The primary parsing path (vae) produces output_layout_info with padded/unpadded
bytes, so unpadded_memory data is non-zero. The secondary path (multihost) lacks
MLIR op details, so tile padding data is flat zeros -- this is a known limitation.
"""

import pytest


class TestVaeTilePadding:
    def test_unpadded_memory_nonzero(self, parsed_vae):
        """Vae log should have meaningful tile padding overhead data."""
        nonzero_count = 0
        for op in parsed_vae["mem"]["operations"]:
            unpadded = op.get("unpadded_memory", {})
            for mt in ("DRAM", "L1"):
                if unpadded.get(mt, {}).get("padded_bytes", 0) > 0:
                    nonzero_count += 1
                    break
        assert nonzero_count > 0, "Expected some ops with non-zero tile padding data"

    def test_output_layout_info_present(self, parsed_vae):
        """Primary path ops should have output_layout_info."""
        with_layout = sum(
            1 for op in parsed_vae["ops"] if op.get("output_layout_info")
        )
        assert with_layout > 100, f"Expected many ops with layout info, got {with_layout}"


class TestMultihostTilePadding:
    def test_unpadded_memory_nonzero(self, parsed_multihost):
        """Multihost should have non-zero tile padding data (backfilled from IR)."""
        nonzero_count = 0
        for op in parsed_multihost["mem"]["operations"]:
            unpadded = op.get("unpadded_memory", {})
            for mt in ("DRAM", "L1"):
                if unpadded.get(mt, {}).get("padded_bytes", 0) > 0:
                    nonzero_count += 1
                    break
        assert nonzero_count > 0, "Tile padding data is all zeros"

    def test_output_layout_info_present(self, parsed_multihost):
        """Secondary path ops should have output_layout_info (backfilled from IR)."""
        with_layout = sum(
            1 for op in parsed_multihost["ops"] if op.get("output_layout_info")
        )
        assert with_layout > 0, f"Expected some ops with layout info, got {with_layout}"
