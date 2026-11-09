"""
Unit and integration tests for Uncensored Talking pipeline modules.
"""

import json
import subprocess
from pathlib import Path
import pytest

from src.audio import extract_audio
from src.selector import (
    Segment,
    ClipProposal,
    select_clips_heuristic,
)
from src.subtitles import (
    format_ass_timestamp,
    map_words_to_clip_timeline,
    chunk_words_for_subtitles,
    apply_hold_duration,
    generate_ass_subtitles,
)
from src.video import check_nvenc_support, render_clip, escape_ffmpeg_filter_path


def test_segment_validation():
    # Valid segment
    seg = Segment(start=1.234, end=5.678)
    assert seg.start == 1.23
    assert seg.end == 5.68

    # Invalid segment: end <= start
    with pytest.raises(ValueError):
        Segment(start=10.0, end=9.0)


def test_clip_proposal_total_duration():
    clip = ClipProposal(
        clip_id=1,
        title="test_clip",
        hook="Watch out!",
        rationale="Controversial topic",
        segments=[
            Segment(start=0.0, end=15.0),
            Segment(start=30.0, end=65.0),
        ],
    )
    assert clip.total_duration == 50.0
    out = clip.to_output_dict()
    assert out["clip_id"] == 1
    assert out["title"] == "test_clip"
    assert len(out["segments"]) == 2


def test_ass_timestamp_formatting():
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(1.25) == "0:00:01.25"
    assert format_ass_timestamp(65.43) == "0:01:05.43"
    assert format_ass_timestamp(3661.05) == "1:01:01.05"


def test_subtitle_chunking_and_hold():
    # 5 words with a speech pause
    words = [
        {"word": "THE", "start": 0.0, "end": 0.3},
        {"word": "GOVERNMENT", "start": 0.35, "end": 0.9},
        {"word": "IS", "start": 0.95, "end": 1.1},
        {"word": "LYING", "start": 1.15, "end": 1.6},
        {"word": "TOTALLY", "start": 3.0, "end": 3.5},  # 1.4s pause before TOTALLY
    ]
    clip = ClipProposal(
        clip_id=1,
        title="gov_lying",
        segments=[Segment(start=0.0, end=4.0)],
    )

    local_words = map_words_to_clip_timeline(words, clip.segments)
    assert len(local_words) == 5

    chunks = chunk_words_for_subtitles(local_words)
    # GOVERNMENT is 10 letters, so it gets flushed
    assert len(chunks) >= 2

    timed_chunks = apply_hold_duration(chunks, clip_total_duration=4.0)
    for c in timed_chunks:
        assert c["start"] < c["end"]
        # Max chunk words <= 3
        assert len(c["text"].split()) <= 3


def test_heuristic_selector_constraints():
    # Generate mock 120s transcript with word timestamps
    mock_words = []
    t = 0.0
    topics = [
        "hello", "welcome", "to", "the", "podcast", "today", "we", "discuss",
        "the", "secret", "matrix", "and", "why", "the", "media", "is", "lying",
        "to", "everyone", "about", "money", "and", "government", "control",
        "this", "is", "an", "insane", "scandal", "exposed", "completely",
    ]
    for i in range(150):
        w = topics[i % len(topics)]
        mock_words.append({
            "word": w,
            "start": round(t, 2),
            "end": round(t + 0.4, 2),
        })
        t += 0.55

    clips = select_clips_heuristic(mock_words, num_clips=2)
    assert len(clips) >= 1
    for cp in clips:
        # Check duration constraint 30s to 60s
        assert 30.0 <= cp.total_duration <= 60.0
        # Check first 3 seconds hook exists
        assert len(cp.hook) > 0


def test_escape_ffmpeg_path():
    path = Path("/home/user/my test:dir/clip.ass")
    escaped = escape_ffmpeg_filter_path(path)
    assert "\\:" in escaped
