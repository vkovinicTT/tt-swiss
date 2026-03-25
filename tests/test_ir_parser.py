# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for IR parser: single-program and multi-program IR extraction."""

from memory_profiler.ir_parser import parse_ir_modules, parse_all_ir_modules


class TestVaeIR:
    def test_single_program_ir(self, vae_log_path):
        ir = parse_ir_modules(vae_log_path)
        assert ir["ttir"]["text"], "TTIR text should be non-empty"
        assert ir["ttnn"]["text"], "TTNN text should be non-empty"

    def test_loc_index_has_entries(self, vae_log_path):
        ir = parse_ir_modules(vae_log_path)
        assert len(ir["ttir"]["loc_index"]) > 0
        assert len(ir["ttnn"]["loc_index"]) > 0

    def test_ir_in_parsed_output(self, parsed_vae):
        """IR from parse_log_file should be single-program (flat) format."""
        ir = parsed_vae["ir"]
        # Single-program format: top-level ttir/ttnn keys
        assert "ttir" in ir
        assert "ttnn" in ir
        assert ir["ttir"]["text"]


class TestMultihostIR:
    def test_all_ir_modules_count(self, multihost_log_path):
        all_ir = parse_all_ir_modules(multihost_log_path)
        assert len(all_ir) == 4, f"Expected 4 IR module pairs, got {len(all_ir)}"

    def test_all_ir_modules_have_text(self, multihost_log_path):
        all_ir = parse_all_ir_modules(multihost_log_path)
        for i, pair in enumerate(all_ir):
            assert pair["ttir"]["text"], f"IR pair {i} missing TTIR text"
            assert pair["ttnn"]["text"], f"IR pair {i} missing TTNN text"

    def test_multi_program_ir_format(self, parsed_multihost):
        """Multi-program IR output uses 'programs' wrapper."""
        ir = parsed_multihost["ir"]
        assert "programs" in ir
        assert len(ir["programs"]) >= 2
