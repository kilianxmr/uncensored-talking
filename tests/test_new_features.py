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


def test_excon_font_loading():
    from src.subtitles import get_subtitle_font
    font = get_subtitle_font(size=63)
    name, style = font.getname()
    assert "Excon" in name or "Outfit" in name
    assert style in ("Black", "Bold")


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


def test_contextual_broll_queries():
    from src.selector import extract_contextual_broll_query

    # Category A: Named Persons
    q_trump = extract_contextual_broll_query("and President Trump said we must reform this")
    assert "Trump" in q_trump
    assert "rare oval office" in q_trump

    q_musk = extract_contextual_broll_query("Elon Musk thinks AI will replace work soon")
    assert "Elon Musk" in q_musk

    q_terry = extract_contextual_broll_query("Terry McAuliffe was the governor at that time")
    assert "Terry McAuliffe" in q_terry

    # Category A: Concrete Brands (Logos)
    q_pepsi = extract_contextual_broll_query("Pepsi and Coca-Cola spend millions lobbying")
    assert "Pepsi" in q_pepsi
    assert "logo" in q_pepsi

    # Category B: Direct Nouns
    q_lunch = extract_contextual_broll_query("school lunch is full of donuts and pastries for kids")
    assert "cafeteria lunch tray" in q_lunch

    q_pharma = extract_contextual_broll_query("pharmaceutical companies sell expensive pills")
    assert "pills blister pack macro" in q_pharma

    # Strict Topic Guard: Abstract or unrelated talk yields empty string (NO forced B-roll)
    q_general = extract_contextual_broll_query("they talked about the rules of the game")
    assert q_general == "", "Unrelated talk must return empty string to avoid wrong B-roll"


def test_validate_broll_frame(tmp_path: Path):
    from src.broll_engine import validate_broll_frame

    # 1. Black video
    black_video = tmp_path / "black.mp4"
    cmd_black = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(black_video),
    ]
    subprocess.run(cmd_black, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    is_valid, reason = validate_broll_frame(black_video)
    assert not is_valid
    assert "black" in reason.lower()

    # 2. Rich visual video with edges and color
    rich_video = tmp_path / "rich.mp4"
    cmd_rich = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(rich_video),
    ]
    subprocess.run(cmd_rich, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    is_valid, reason = validate_broll_frame(rich_video)
    assert is_valid
    assert reason == "OK"


def test_no_ambiguity_hook_rules():
    from src.selector import find_context_anchor_start, enforce_no_ambiguity_hook, ClipProposal, Segment

    words = [
        {"word": "Wait,", "start": 10.0, "end": 10.4},
        {"word": "so", "start": 10.45, "end": 10.6},
        {"word": "you're", "start": 10.65, "end": 10.8},
        {"word": "saying", "start": 10.85, "end": 11.2},
        {"word": "Terry", "start": 11.25, "end": 11.6},
        {"word": "is", "start": 11.65, "end": 11.8},
        {"word": "taking", "start": 11.85, "end": 12.2},
        {"word": "money?", "start": 12.25, "end": 12.7},
        {"word": "Yeah,", "start": 13.0, "end": 13.3},
        {"word": "it's", "start": 13.35, "end": 13.6},
        {"word": "based", "start": 13.65, "end": 14.0},
        {"word": "on", "start": 14.05, "end": 14.2},
        {"word": "a", "start": 14.25, "end": 14.3},
        {"word": "lie", "start": 14.35, "end": 14.8},
        {"word": "and", "start": 14.85, "end": 15.0},
        {"word": "70%", "start": 15.05, "end": 15.6},
        {"word": "is", "start": 15.65, "end": 15.8},
        {"word": "toxic", "start": 15.85, "end": 16.3},
    ]

    # Test 1: Ambiguous start at 13.0s ("Yeah, it's based on...")
    # Must anchor backwards to the host question at 10.0s ("Wait, so you're saying Terry...")
    anchored = find_context_anchor_start(words, target_start=13.0)
    assert anchored == 10.0

    # Test 2: enforce_no_ambiguity_hook adjusts proposal segments and total duration
    clip = ClipProposal(
        clip_id=1,
        title="ambiguous_hook_test",
        hook="Yeah it's based on a lie",
        rationale="Testing anchor rule",
        segments=[Segment(start=13.0, end=45.0)],
    )
    fixed_clip = enforce_no_ambiguity_hook(clip, words)
    assert fixed_clip.segments[0].start == 10.0
    assert 30.0 <= fixed_clip.total_duration <= 60.0
    assert "Wait" in fixed_clip.hook or "Terry" in fixed_clip.hook


def test_strict_topic_guard():
    from src.broll_engine import check_topic_guard
    from src.selector import SceneCut

    # 1. Aligned: school lunch tray with cafeteria words
    scene_ok = SceneCut(
        start=0.0,
        end=3.0,
        layout="splitscreen",
        search_query="american high school cafeteria lunch tray",
    )
    is_valid, reason = check_topic_guard(scene_ok, spoken_text="they give donuts and pastries for lunch")
    assert is_valid

    # 2. Misaligned: skyscraper construction workers with school lunch words
    scene_bad = SceneCut(
        start=0.0,
        end=3.0,
        layout="splitscreen",
        search_query="construction workers skyscraper high altitude",
    )
    is_valid, reason = check_topic_guard(scene_bad, spoken_text="school children eat processed food and pastries")
    assert not is_valid
    assert "no noun overlap" in reason.lower() or "rejected" in reason.lower()


def test_background_music_asset_and_pipeline_integration():
    from src.video import DEFAULT_BG_MUSIC_PATH

    # 1. Verify permanent pre-cut music asset exists in assets/audio/
    assert DEFAULT_BG_MUSIC_PATH.is_file(), f"Background music file not found at {DEFAULT_BG_MUSIC_PATH}"
    file_size = DEFAULT_BG_MUSIC_PATH.stat().st_size
    assert file_size > 500_000, f"Background music file unexpectedly small ({file_size} bytes)"

    # 2. Check duration with ffprobe: must be >= 120s (cut from 32s onwards of the original 3m05s track)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(DEFAULT_BG_MUSIC_PATH),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(res.stdout.strip())
    assert 140.0 <= duration <= 165.0, f"Expected cut duration ~153s from s32, got {duration}s"


