# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base VideoBackend frame sampling when the container reports no duration."""

import pytest

from vllm.multimodal.video import VideoBackend, VideoSourceMetadata, VideoTargetMetadata

SOURCE_FPS = 30.0


def sample(total_frames, duration, num_frames=-1, fps=1, source_fps=SOURCE_FPS):
    source = VideoSourceMetadata(total_frames, source_fps, duration)
    target = VideoTargetMetadata(num_frames, fps, 300)
    return VideoBackend.compute_frames_index_to_sample(source, target)


@pytest.mark.parametrize("total_frames", [300, 3000, 30000])
@pytest.mark.parametrize("missing", [0, None], ids=["duration=0", "duration=None"])
def test_unknown_duration_matches_the_reported_duration(total_frames, missing):
    """An unreported duration must behave like the correct value derived from
    frames/fps -- neither collapsed to one frame nor widened to every frame."""
    known = total_frames / SOURCE_FPS

    assert sample(total_frames, missing) == sample(total_frames, known)
    assert len(sample(total_frames, missing)) == int(known)


@pytest.mark.parametrize("missing", [0, None])
def test_unknown_duration_does_not_collapse_or_explode(missing):
    assert len(sample(3000, missing)) == 100


def test_known_duration_is_untouched():
    assert len(sample(300, 10.0)) == 10
    assert len(sample(300, 60.0, fps=2)) == 120


def test_explicit_num_frames_still_bounds_the_sample():
    assert len(sample(3000, 100.0, num_frames=8, fps=-1)) == 8


def test_zero_source_fps_with_zero_duration_is_unchanged():
    """When neither duration nor source fps is usable nothing can be derived, so
    the pre-existing result stands: 1 frame. Asserted explicitly so this change
    cannot silently alter the degenerate path."""
    assert len(sample(300, 0, source_fps=0.0)) == 1


def test_zero_source_fps_with_none_duration_still_raises():
    """Out of scope for this fix: `None` duration with an unusable source fps still
    raises exactly as on main (zero source fps is handled separately elsewhere)."""
    with pytest.raises(TypeError):
        sample(300, None, source_fps=0.0)
