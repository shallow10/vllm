# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for QuarkConfig handling of an absent `layer_type_quant_config`."""

import torch

from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

GLOBAL_QUANT_CONFIG = {
    "weight": {"dtype": "fp8", "is_dynamic": True},
    "activation": {"dtype": "fp8", "is_dynamic": True},
}

LAYER = "model.layers.0.self_attn.q_proj"


def test_global_only_config_resolves_to_global():
    """A checkpoint carrying only `global_quant_config` is legal per the
    documented layer -> layer_type -> global priority and must not raise."""
    config = QuarkConfig({"global_quant_config": GLOBAL_QUANT_CONFIG})

    assert config._find_matched_config(LAYER, torch.nn.Linear) == GLOBAL_QUANT_CONFIG


def test_layer_type_config_still_wins_over_global():
    """Supplying the default must not change precedence when the key exists."""
    per_type = {torch.nn.Linear: {"weight": {"dtype": "int4"}}}
    config = QuarkConfig(
        {
            "global_quant_config": GLOBAL_QUANT_CONFIG,
            "layer_type_quant_config": per_type,
        }
    )

    assert config._find_matched_config(LAYER, torch.nn.Linear) == per_type[
        torch.nn.Linear
    ]
