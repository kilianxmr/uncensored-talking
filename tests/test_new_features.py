"""
Tests for enhanced features:
- Pillow-based gradient subtitle rendering
- Sentence boundary chunking & clean text
- OpenCV saliency focal point detection
- Proportional anti-stretch scaling & smoothed face tracking
- Real footage motion snippet extraction
"""

import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.selector import Segment, SceneCut, ClipProposal, generate_heuristic_scenes
from src.subtitles import (
    clean_subtitle_text,
    chunk_words_gradient,
    render_gradient_subtitle_png,
    generate_png_subtitles,
)
from src.broll_engine import find_visual_focal_point, extract_peak_motion_snippet
from src.video import get_video_dimensions, compute_smoothed_face_trajectory, render_clip


def test_clean_subtitle_text():
    raw = 'Wait, are you serious?! "No way..." - said the guest.'
    cleaned = clean_subtitle_text(raw)
    assert cleaned == "WAIT ARE YOU SERIOUS NO WAY SAID THE GUEST"
    assert not any(p in cleaned for p in [".", ",", "!", "?", '"', "'", "-"]), "Punctuation must be stripped"


def test_chunk_words_gradient_sentence_boundaries():
    local_words = [
        {"word": "THE", "raw_word": "The", "is_sentence_end": False, "start": 0.0, "end": 0.3},
        {"word": "MATRIX", "raw_word": "Matrix.", "is_sentence_end": True, "start": 0.35, "end": 0.8},
        {"word": "IS", "raw_word": "Is", "is_sentence_end": False, "start": 0.85, "end": 1.0},
        {"word": "REAL", "raw_word": "real!", "is_sentence_end": True, "start": 1.05, "end": 1.5},
        {"word": "UNBELIEVABLE", "raw_word": "Unbelievable", "is_sentence_end": False, "start": 1.6, "end": 2.2},
    ]
    chunks = chunk_words_gradient(local_words)
    # The Matrix ends sentence, so "THE MATRIX" must be a chunk
    assert chunks[0]["text"] == "THE MATRIX"
    assert chunks[1]["text"] == "IS REAL"
    # UNBELIEVABLE is >8 chars, so it must be a standalone 1-word chunk
    assert chunks[2]["text"] == "UNBELIEVABLE"


def test_render_gradient_subtitle_png(tmp_path: Path):
    out_png = tmp_path / "sub_test.png"
    render_gradient_subtitle_png(
        text="THE SECRET",
        output_png_path=out_png,
        width=1080,
        height=1920,
        font_size=88,
        stroke_width=12,
    )
    assert out_png.exists()
    img = Image.open(out_png)
    assert img.size == (1080, 1920)
    assert img.mode == "RGBA"

    # Check non-empty pixels in center
    arr = np.array(img)
    center_region = arr[900:1020, 480:600]
    # Alpha channel should have opaque content
    assert np.max(center_region[:, :, 3]) == 255


def test_find_visual_focal_point(tmp_path: Path):
    img_path = tmp_path / "focal_test.jpg"
    # Create image with high contrast dot in bottom-right
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[750:850, 1350:1450] = 255
    Image.fromarray(img).save(img_path)

    nx, ny = find_visual_focal_point(img_path)
    assert 0.15 <= nx <= 0.85
    assert 0.15 <= ny <= 0.85
    # Should point towards bottom-right quadrant
    assert nx > 0.5
    assert ny > 0.5


def test_extract_peak_motion_snippet(tmp_path: Path):
    # Create a 6s video where motion occurs only between second 2 and 4
    raw_video = tmp_path / "motion_source.mp4"
    # First 2s static, next 2s moving testsrc, last 2s static
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=2:r=30",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=2:r=30",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(raw_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_snippet = tmp_path / "peak_motion.mp4"
    extract_peak_motion_snippet(
        video_path=raw_video,
        output_path=out_snippet,
        target_duration=2.0,
        target_width=1080,
        target_height=1920,
    )
    assert out_snippet.exists()
    w, h = get_video_dimensions(out_snippet)
    assert w == 1080
    assert h == 1920


def test_anti_stretch_rendering_and_png_overlay(tmp_path: Path):
    # 1. 16:9 source video
    source_video = tmp_path / "source.mp4"
    cmd_src = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-c:a", "aac",
        str(source_video),
    ]
    subprocess.run(cmd_src, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    clip = ClipProposal(
        clip_id=99,
        title="anti_stretch_test",
        hook="Anti stretch and gradient test",
        rationale="Validation test",
        segments=[Segment(start=0.5, end=3.5)],
        scenes=[SceneCut(start=0.0, end=3.0, layout="full_speaker")],
    )

    timestamps_file = tmp_path / "words.json"
    timestamps_file.write_text(json.dumps([
        {"word": "NO", "start": 0.6, "end": 1.0},
        {"word": "STRETCH", "start": 1.1, "end": 1.8},
        {"word": "PERFECTION", "start": 2.0, "end": 2.8},
    ]))

    concat_subs, _ = generate_png_subtitles(clip, word_timestamps_file=timestamps_file, output_dir=tmp_path / "subs")
    assert concat_subs.exists()

    out_clip = render_clip(
        source_video_path=source_video,
        clip=clip,
        subtitles_source=concat_subs,
        output_dir=tmp_path / "out",
        force_encoder="libx264",
    )

    assert out_clip.exists()
    w, h = get_video_dimensions(out_clip)
    assert w == 1080
    assert h == 1920
