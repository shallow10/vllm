# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.cache import CacheConfig


def test_kv_sharing_fast_prefill_changes_compilation_hash():
    """`kv_sharing_fast_prefill` decides which module trees get compiled at all.

    It is the `enable_if=` predicate on three classes per architecture:
    `Gemma3nSelfDecoder` / `Gemma3nCrossDecoder` / `Gemma3nTextModel`, and the
    Gemma4 equivalents. The first two compile only when the flag is on, the
    third only when it is off -- so toggling it swaps which trees
    `@support_torch_compile` wraps, which is exactly what this hash keys.
    """
    disabled = CacheConfig()
    enabled = CacheConfig(kv_sharing_fast_prefill=True)

    assert disabled.compute_hash() != enabled.compute_hash()
