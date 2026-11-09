"""
Unit and integration tests for new pipeline features:
- Downloader module
- SceneCut and Visual Director
- B-Roll Engine with Ken-Burns motion
- Face Centering with OpenCV
- 3 Layout Modes (full_speaker, full_broll, splitscreen)
- Center-aligned subtitles (Alignment: 5, Gold/Yellow)
"""

import json
import subprocess
from pathlib import Path
import pytest

from src.downloader import sanitize_filename
from src.selector import Segment, SceneCut, ClipProposal, generate_heuristic_scenes
from src.broll_engine import apply_ken_burns_effect, generate_cinematic_fallback
from src.subtitles import generate_ass_subtitles, get_ass_header
from src.video import (
    get_video_dimensions,
    detect_speaker_face_x,
    render_scene_plate,
    render_clip,
)


def test_sanitize_filename():
    raw = 'Joe Rogan #2000 - Alien "Disclosure" & Conspiracy?!'
    clean = sanitize_filename(raw)
    assert '"' not in clean
    assert '?' not in clean
    assert ':' not in clean
    assert len(clean) > 5


def test_scenecut_model():
    sc = SceneCut(
        start=0.0,
        end=3.5,
        layout="splitscreen",
        search_query="military helicopter night flight",
    )
    assert sc.duration == 3.5
    assert sc.layout == "splitscreen"

    with pytest.raises(ValueError):
        SceneCut(start=5.0, end=4.0, layout="full_speaker")


def test_generate_heuristic_scenes():
    mock_words = [
        {"word": "government", "start": 0.5, "end": 1.2},
        {"word": "money", "start": 1.3, "end": 2.0},
        {"word": "secret", "start": 2.1, "end": 3.0},
        {"word": "conspiracy", "start": 3.1, "end": 4.5},
    ]
    scenes = generate_heuristic_scenes(clip_duration=35.0, clip_words=mock_words)
    assert len(scenes) >= 7
    # First scene must be the hook (<= 3.0s)
    assert scenes[0].start == 0.0
    assert scenes[0].end <= 3.2
    # Verify layouts include splitscreen, full_speaker, full_broll
    layouts = {s.layout for s in scenes}
    assert "splitscreen" in layouts
    assert "full_speaker" in layouts
    assert "full_broll" in layouts
    # Continuous timeline
    for i in range(len(scenes) - 1):
        assert scenes[i].end == scenes[i + 1].start


def test_center_subtitles_alignment_and_color(tmp_path: Path):
    clip = ClipProposal(
        clip_id=1,
        title="center_test",
        hook="Breaking truth",
        rationale="Testing center alignment",
        segments=[Segment(start=0.0, end=4.0)],
        scenes=[SceneCut(start=0.0, end=4.0, layout="splitscreen", search_query="gold vault")],
    )
    timestamps_file = tmp_path / "words.json"
    timestamps_file.write_text(json.dumps([
        {"word": "THIS", "start": 0.2, "end": 0.5},
        {"word": "CHANGES", "start": 0.6, "end": 1.1},
        {"word": "EVERYTHING", "start": 1.2, "end": 1.8},
    ]))

    ass_path = tmp_path / "test.ass"
    generate_ass_subtitles(clip, word_timestamps_file=timestamps_file, output_ass_path=ass_path)

    content = ass_path.read_text(encoding="utf-8")
    # Verify Alignment 5 (Exact Center)
    assert ",5,30,30,0,1" in content
    # Verify rich yellow/gold color
    assert "&H002BF7FF" in content


def test_ken_burns_effect_generation(tmp_path: Path):
    # Generate static 1920x1080 test image
    img_path = tmp_path / "test_image.jpg"
    cmd_img = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=1920x1080:d=1",
        "-frames:v", "1",
        str(img_path),
    ]
    subprocess.run(cmd_img, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_mp4 = tmp_path / "ken_burns.mp4"
    res_path = apply_ken_burns_effect(
        image_path=img_path,
        output_path=out_mp4,
        duration=2.5,
        width=1080,
        height=1920,
        fps=30,
    )
    assert res_path.exists()
    assert res_path.stat().st_size > 5000

    w, h = get_video_dimensions(res_path)
    assert w == 1080
    assert h == 1920


def test_dynamic_layouts_rendering(tmp_path: Path):
    # 1. Create a synthetic 16:9 source video (6 seconds, 1920x1080)
    source_video = tmp_path / "source.mp4"
    cmd_src = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-c:a", "aac",
        str(source_video),
    ]
    subprocess.run(cmd_src, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 2. Create synthetic B-roll video (3 seconds, 1080x1920)
    broll_video = tmp_path / "broll.mp4"
    cmd_broll = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=1080x1920:d=4:rate=30",
        "-c:v", "libx264", "-an",
        str(broll_video),
    ]
    subprocess.run(cmd_broll, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 3. Create ClipProposal with 3 scenes:
    # Scene 1: splitscreen (0.0s to 1.5s)
    # Scene 2: full_speaker (1.5s to 3.0s)
    # Scene 3: full_broll (3.0s to 4.5s)
    clip = ClipProposal(
        clip_id=42,
        title="layout_test",
        hook="Ultimate test hook",
        rationale="Testing all 3 layout modes",
        segments=[Segment(start=0.5, end=5.0)],
        scenes=[
            SceneCut(start=0.0, end=1.5, layout="splitscreen", search_query="military", broll_asset_path=str(broll_video)),
            SceneCut(start=1.5, end=3.0, layout="full_speaker", search_query=""),
            SceneCut(start=3.0, end=4.5, layout="full_broll", search_query="space", broll_asset_path=str(broll_video)),
        ],
    )

    # 4. Generate subtitles
    timestamps_file = tmp_path / "words.json"
    timestamps_file.write_text(json.dumps([
        {"word": "SCENE", "start": 0.6, "end": 1.0},
        {"word": "SWITCH", "start": 1.1, "end": 1.5},
        {"word": "TEST", "start": 2.0, "end": 2.8},
    ]))
    ass_path = tmp_path / "clip_42.ass"
    generate_ass_subtitles(clip, word_timestamps_file=timestamps_file, output_ass_path=ass_path)

    # 5. Render short
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_clip(
        source_video_path=source_video,
        clip=clip,
        subtitles_ass_path=ass_path,
        output_dir=out_dir,
        force_encoder="libx264",
    )

    assert rendered.exists()
    assert rendered.stat().st_size > 15000

    w, h = get_video_dimensions(rendered)
    assert w == 1080
    assert h == 1920
