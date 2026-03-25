# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for inputs registry parser."""

from memory_profiler.inputs_registry_parser import (
    parse_inputs_registry,
    parse_all_inputs_registries,
)


class TestVaeRegistry:
    def test_entry_count(self, vae_log_path):
        reg = parse_inputs_registry(vae_log_path)
        assert reg["metadata"]["total_entries"] == 3

    def test_weight_input_split(self, vae_log_path):
        reg = parse_inputs_registry(vae_log_path)
        assert reg["metadata"]["total_weights"] == 2
        assert reg["metadata"]["total_inputs"] == 1

    def test_from_parsed_output(self, parsed_vae):
        reg = parsed_vae["reg"]
        assert len(reg["entries"]) == 3


class TestMultihostRegistry:
    def test_all_registries(self, multihost_log_path):
        regs = parse_all_inputs_registries(multihost_log_path)
        assert len(regs) >= 1, "Should find at least one registry"

    def test_from_parsed_output(self, parsed_multihost):
        reg = parsed_multihost["reg"]
        assert "entries" in reg
