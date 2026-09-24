# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery hints raised by TimedTrace must name flags the CLI actually registers."""

import inspect

import pytest

from vllm.benchmarks.datasets import datasets as datasets_module
from vllm.benchmarks.datasets.datasets import TimedTrace


@pytest.fixture
def trace(monkeypatch):
    """A TimedTrace whose single entry lacks every expected label field."""
    monkeypatch.setattr(TimedTrace, "load_data", lambda self: None)
    instance = TimedTrace(dataset_path="unused.jsonl")
    instance.data = ['{"prompt": "hi"}']
    return instance


def test_input_length_hint_names_an_existing_flag(trace):
    """The message tells the user which flag to set to recover. The flag it used to
    name (--label-input-length) is not registered, so following the advice produced
    a second failure ('unrecognized arguments') instead of fixing the first."""
    with pytest.raises(ValueError) as excinfo:
        trace.sample(tokenizer=object(), num_requests=1)

    message = str(excinfo.value)
    assert "Input length field" in message
    assert "--timed-trace-label-input-length" in message


def test_no_hint_advises_an_unregistered_flag():
    """Guards the sibling messages (output length, timestamp) and any future one:
    the un-prefixed --label-* spellings are not accepted by the CLI, so no error
    may point a user at them."""
    source = inspect.getsource(datasets_module)

    assert "Use --label-" not in source
