# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the main log parser: rank prefix stripping, program detection,
single-program backward compat (vae), and multi-program (multihost)."""

from memory_profiler.parser import strip_rank_prefix, validate_log_content, validate_outputs


# ---------------------------------------------------------------------------
# Unit tests: strip_rank_prefix
# ---------------------------------------------------------------------------

class TestStripRankPrefix:
    def test_stdout_prefix(self):
        line = "[1,0]<stdout>:                  Always |     INFO | foo"
        cleaned, rank = strip_rank_prefix(line)
        assert rank == "1,0"
        assert cleaned.strip().startswith("Always")

    def test_stderr_prefix(self):
        line = "[1,1]<stderr>: some error"
        cleaned, rank = strip_rank_prefix(line)
        assert rank == "1,1"
        assert cleaned == "some error"

    def test_no_prefix(self):
        line = "2026-03-23 13:23:02.098 | INFO | Something"
        cleaned, rank = strip_rank_prefix(line)
        assert rank is None
        assert cleaned == line

    def test_empty_line(self):
        cleaned, rank = strip_rank_prefix("")
        assert rank is None
        assert cleaned == ""


# ---------------------------------------------------------------------------
# Unit tests: validate_log_content
# ---------------------------------------------------------------------------

class TestValidateLogContent:
    def test_vae_valid(self, vae_log_path):
        assert validate_log_content(vae_log_path) is None

    def test_multihost_valid(self, multihost_log_path):
        # Multihost logs have memory states but no "Executing operation:" lines
        assert validate_log_content(multihost_log_path) is None


# ---------------------------------------------------------------------------
# Vae log: single-program backward compatibility
# ---------------------------------------------------------------------------

class TestVaeParsing:
    def test_total_operations(self, parsed_vae):
        ops = parsed_vae["ops"]
        assert len(ops) == 2495

    def test_memory_ops_aligned(self, parsed_vae):
        mem_ops = parsed_vae["mem"]["operations"]
        assert len(mem_ops) == len(parsed_vae["ops"])

    def test_output_alignment(self, parsed_vae):
        paths = parsed_vae["paths"]
        assert validate_outputs(paths["memory"], paths["operations"]) is True

    def test_single_program(self, parsed_vae):
        programs = parsed_vae["mem"]["metadata"]["programs"]
        assert len(programs) == 1
        assert programs[0]["name"] == "program_0"
        assert programs[0]["start_op_index"] == 0
        assert programs[0]["end_op_index"] == 2494

    def test_program_index_on_ops(self, parsed_vae):
        for op in parsed_vae["ops"]:
            assert op["program_index"] == 0

    def test_const_eval_detection(self, parsed_vae):
        const_eval_ops = [op for op in parsed_vae["ops"] if op.get("const_eval_graph")]
        assert len(const_eval_ops) == 615
        graphs = set(op["const_eval_graph"] for op in const_eval_ops)
        assert len(graphs) == 4

    def test_memory_types_complete(self, parsed_vae):
        required = {"DRAM", "L1", "L1_SMALL", "TRACE"}
        for op in parsed_vae["mem"]["operations"]:
            assert required.issubset(set(op["memory"].keys())), (
                f"Op {op['index']} missing types: {required - set(op['memory'].keys())}"
            )

    def test_weight_ops(self, parsed_vae):
        weight_ops = sum(1 for op in parsed_vae["ops"] if op.get("is_weight_op"))
        activation_ops = sum(1 for op in parsed_vae["ops"] if not op.get("is_weight_op"))
        assert weight_ops == 671
        assert activation_ops == 1824

    def test_no_deallocate_in_output(self, parsed_vae):
        for op in parsed_vae["ops"]:
            assert "deallocate" not in op.get("mlir_op", "").lower()


# ---------------------------------------------------------------------------
# Multihost log: multi-program, multi-host
# ---------------------------------------------------------------------------

class TestMultihostParsing:
    def test_total_operations(self, parsed_multihost):
        assert len(parsed_multihost["ops"]) == 8398

    def test_program_count(self, parsed_multihost):
        programs = parsed_multihost["mem"]["metadata"]["programs"]
        assert len(programs) == 2

    def test_program_names(self, parsed_multihost):
        programs = parsed_multihost["mem"]["metadata"]["programs"]
        assert programs[0]["name"] == "test_simple_sharded_addition"
        assert programs[1]["name"] == "program_1"

    def test_program_boundaries(self, parsed_multihost):
        programs = parsed_multihost["mem"]["metadata"]["programs"]
        # First program: small test
        assert programs[0]["start_op_index"] == 0
        assert programs[0]["end_op_index"] == 7
        # Second program: large llama run
        assert programs[1]["start_op_index"] == 8
        assert programs[1]["end_op_index"] == 8397

    def test_output_alignment(self, parsed_multihost):
        paths = parsed_multihost["paths"]
        assert validate_outputs(paths["memory"], paths["operations"]) is True

    def test_secondary_path_used(self, parsed_multihost):
        """Multihost logs use secondary path (no MLIR details), so result is None."""
        for op in parsed_multihost["ops"]:
            assert op["result"] is None

    def test_dram_present_in_all_ops(self, parsed_multihost):
        for op in parsed_multihost["mem"]["operations"]:
            assert "DRAM" in op["memory"], f"Op {op['index']} missing DRAM"

    def test_no_deallocate_in_output(self, parsed_multihost):
        for op in parsed_multihost["ops"]:
            assert op["mlir_op"] != "DeallocateOp"

    def test_rank_field_consistent(self, parsed_multihost):
        """Ops with rank field should all be from the target rank [1,0]."""
        ops_with_rank = [op for op in parsed_multihost["ops"] if op.get("rank")]
        assert len(ops_with_rank) > 0, "Some ops should have rank field"
        for op in ops_with_rank:
            assert op["rank"] == "1,0"
