"""
Unit and integration tests for professional postproduction mechanics:
1. Subtitle pop-in animation (Excon-Black, alignment 5, transform tag).
2. Audio-Tail Protection & Clean Outro (0.8s - 1.2s safety pause, 0.2s fade-out).
3. Voice-of-God Context Engine (evaluation, formula, TTS synthesis, audio sidechaining).
4. Cubic spline / smoothstep zoom & Dip-to-Black transitions.
"""

import json
import wave
from pathlib import Path

import pytest

from src.audio import generate_sub_boom, ensure_sfx_assets
from src.hook_voice import (
    evaluate_context_hook_need,
    generate_hook_formula,
    synthesize_voice_of_god,
    get_audio_duration,
)
from src.selector import (
    ClipProposal,
    Segment,
    SceneCut,
    apply_audio_tail_protection,
    enforce_no_ambiguity_hook,
    format_storyboard_display,
)
from src.subtitles import generate_ass_subtitles


def test_generate_sub_boom(tmp_path):
    output_wav = tmp_path / "sub_boom_test.wav"
    res = generate_sub_boom(output_wav, duration=1.0, freq=60.0, sample_rate=48000)

    assert res.exists()
    assert res.stat().st_size > 0

    with wave.open(str(res), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 48000
        assert wf.getnframes() == 48000


def test_ensure_sfx_assets():
    sfx = ensure_sfx_assets()
    assert "sub_boom" in sfx
    assert sfx["sub_boom"].exists()
    assert sfx["sub_boom"].stat().st_size > 0


def test_ass_pop_in_animation(tmp_path):
    word_ts_file = tmp_path / "words.json"
    dummy_words = [
        {"word": "ROBERT", "start": 1.0, "end": 1.4},
        {"word": "KENNEDY", "start": 1.4, "end": 1.9},
        {"word": "EXPOSED", "start": 2.0, "end": 2.6},
        {"word": "THE", "start": 2.6, "end": 2.8},
        {"word": "TRUTH", "start": 2.8, "end": 3.4},
    ]
    with open(word_ts_file, "w", encoding="utf-8") as f:
        json.dump(dummy_words, f)

    clip = ClipProposal(
        clip_id=42,
        title="test_pop",
        hook="ROBERT KENNEDY EXPOSED",
        rationale="Health conspiracy",
        segments=[Segment(start=1.0, end=3.5)],
    )

    out_ass = tmp_path / "test_clip_42.ass"
    res_ass = generate_ass_subtitles(
        clip=clip,
        word_timestamps_file=word_ts_file,
        output_ass_path=out_ass,
        pop_in_animation=True,
    )

    assert res_ass.exists()
    content = res_ass.read_text(encoding="utf-8")

    # Font and styling checks
    assert "Excon Black" in content
    assert "Alignment" in content
    # Pop-in transform tag check
    expected_tag = r"{\alpha&HFF&\fscx75\fscy75\t(0,120,\alpha&H00&\fscx100\fscy100)}"
    assert expected_tag in content
    assert "Dialogue:" in content


def test_audio_tail_protection():
    words = [
        {"word": "This", "start": 0.0, "end": 0.3},
        {"word": "is", "start": 0.3, "end": 0.5},
        {"word": "poison.", "start": 0.5, "end": 1.0},
        {"word": "Next", "start": 3.0, "end": 3.5},
    ]
    segments = [Segment(start=0.0, end=1.0)]

    # Segment ends at 1.0 (exact end of word 'poison')
    protected_segs = apply_audio_tail_protection(segments, words, min_tail=0.8, max_tail=1.2)

    # Must add 0.8s to 1.2s safety pause buffer
    assert protected_segs[-1].end >= 1.8
    assert protected_segs[-1].end <= 2.2


def test_voice_of_god_evaluation_and_formula():
    # Ambiguous pronoun start
    needs_hook, reason = evaluate_context_hook_need(
        story_summary="He talks about poison",
        story_hook="It is based on a lie",
        first_words=["it", "is", "based"],
    )
    assert needs_hook is True
    assert "pronoun" in reason.lower()

    # High-impact controversial topic
    needs_hook2, reason2 = evaluate_context_hook_need(
        story_summary="RFK explains the toxic food supply and poison in breakfast cereal",
        story_hook="They put toxic chemicals in cereal",
        first_words=["they", "put"],
    )
    assert needs_hook2 is True

    # Formula generation
    formula = generate_hook_formula(
        story_summary="RFK explains the toxic food supply and poison in breakfast cereal",
        story_hook="They put toxic chemicals in cereal",
    )
    assert "Robert F. Kennedy Jr." in formula
    assert "dark truth" in formula
    assert "breakfast" in formula.lower()


def test_synthesize_voice_of_god(tmp_path):
    out_audio = tmp_path / "vog_test.wav"
    text = "Robert F. Kennedy Jr. just exposed the dark truth about our food supply."
    path, dur = synthesize_voice_of_god(text, out_audio)

    assert path.exists()
    assert path.stat().st_size > 1000
    assert dur > 1.0


def test_format_storyboard_display_with_vog():
    clip = ClipProposal(
        clip_id=1,
        title="story_1",
        hook="THEY POISON YOUR FOOD",
        rationale="Food chemicals",
        needs_context_hook=True,
        voice_of_god_hook="Robert F. Kennedy Jr. just exposed the truth.",
        segments=[Segment(start=1.0, end=35.0)],
        scenes=[SceneCut(start=0.0, end=34.0, layout="full_speaker")],
    )
    display = format_storyboard_display(clip)
    assert "Voice-of-God Hook empfohlen: JA" in display
    assert "Hook-Text:" in display
    assert "Robert F. Kennedy Jr." in display


def test_ken_burns_smoothstep(tmp_path):
    from PIL import Image
    from src.broll_engine import apply_focal_ken_burns

    img_path = tmp_path / "test_image.png"
    Image.new("RGB", (640, 480), color=(180, 50, 50)).save(img_path)

    out_mp4 = tmp_path / "smooth_zoom.mp4"
    res = apply_focal_ken_burns(
        image_path=img_path,
        output_path=out_mp4,
        duration=1.0,
        target_width=1080,
        target_height=1920,
        fps=30,
    )
    assert res.exists()
    assert res.stat().st_size > 1000


def test_render_scene_plate_with_dip_to_black(tmp_path):
    import subprocess
    from src.video import render_scene_plate

    # Generate small 1280x720 dummy source mp4 with audio
    source_mp4 = tmp_path / "dummy_source.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=1280x720:d=4",
        "-f", "lavfi", "-i", "sine=f=440:d=4",
        "-c:v", "libx264", "-c:a", "aac",
        "-t", "4",
        str(source_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scene = SceneCut(start=0.0, end=2.0, layout="full_speaker")
    segments = [Segment(start=0.0, end=2.0)]
    out_plate = tmp_path / "plate_dip.mp4"

    res = render_scene_plate(
        source_video=source_mp4,
        scene=scene,
        clip_segments=segments,
        orig_w=1280,
        orig_h=720,
        output_plate=out_plate,
        is_first_scene=False,
        is_last_scene=False,
        dip_to_black=True,
    )
    assert res.exists()
    assert res.stat().st_size > 1000

