# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for QuarkConfig handling of absent optional top-level export keys."""

import torch

from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

GLOBAL_QUANT_CONFIG = {
    "weight": {"dtype": "fp8", "is_dynamic": True},
    "activation": {"dtype": "fp8", "is_dynamic": True},
}

LAYER = "model.layers.0.self_attn.q_proj"
# `_find_matched_config` keys this dict by the class object; its `cast(str, ...)`
# annotation is not enforced at runtime.
LINEAR_KEY = torch.nn.Linear


def test_global_only_config_resolves_to_global():
    """A checkpoint carrying only `global_quant_config` is legal per the
    documented layer -> layer_type -> global priority and must not raise."""
    config = QuarkConfig({"global_quant_config": GLOBAL_QUANT_CONFIG})

    assert config._find_matched_config(LAYER, LINEAR_KEY) == GLOBAL_QUANT_CONFIG


def test_layer_type_config_still_wins_over_global():
    """The default must not change precedence when the key is present."""
    per_type = {LINEAR_KEY: {"weight": {"dtype": "int4"}}}
    config = QuarkConfig(
        {
            "global_quant_config": GLOBAL_QUANT_CONFIG,
            "layer_type_quant_config": per_type,
        }
    )

    assert config._find_matched_config(LAYER, LINEAR_KEY) == per_type[LINEAR_KEY]


def test_config_with_neither_key_reports_no_match():
    """With no `layer_type_quant_config` and no `global_quant_config` there is
    nothing to match. `None` is this method's existing no-match result (see its
    `return None` when `matched_configs` is empty), asserted so the new default
    cannot quietly turn a crash into some other, undocumented outcome."""
    config = QuarkConfig({})

    assert config._find_matched_config(LAYER, LINEAR_KEY) is None
