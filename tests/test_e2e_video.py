"""
End-to-end rendering and FFmpeg filter integration tests.
"""

import json
import subprocess
from pathlib import Path
import pytest

from src.audio import extract_audio
from src.selector import Segment, ClipProposal
from src.subtitles import generate_ass_subtitles
from src.video import render_clip


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """Generates a 6-second synthetic 1920x1080 MP4 with audio tone."""
    tmp_dir = tmp_path_factory.mktemp("media")
    video_path = tmp_dir / "synthetic_podcast.mp4"

    # FFmpeg command to generate test video with moving box and 440Hz beep tone
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return video_path


def test_audio_extraction_and_rendering_e2e(synthetic_video: Path, tmp_path: Path):
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test audio extraction
    audio_wav = temp_dir / "audio.wav"
    extracted_wav = extract_audio(synthetic_video, output_path=audio_wav)
    assert extracted_wav.exists()
    assert extracted_wav.stat().st_size > 1000

    # 2. Mock word timestamps
    word_timestamps_file = temp_dir / "word_timestamps.json"
    words = [
        {"word": "UNEXPECTED", "start": 0.5, "end": 1.2},
        {"word": "REVELATION", "start": 1.3, "end": 2.1},
        {"word": "CHANGES", "start": 2.2, "end": 2.6},
        {"word": "EVERYTHING", "start": 2.7, "end": 3.4},
    ]
    with open(word_timestamps_file, "w", encoding="utf-8") as f:
        json.dump(words, f)

    # 3. Create clip proposal (3.5s duration)
    clip = ClipProposal(
        clip_id=1,
        title="mind_blown",
        hook="Unexpected revelation changes everything!",
        rationale="Synthetic test clip",
        segments=[Segment(start=0.5, end=4.0)],
    )

    # 4. Generate ASS subtitles
    ass_path = temp_dir / "clip_1.ass"
    generate_ass_subtitles(clip, word_timestamps_file=word_timestamps_file, output_ass_path=ass_path)
    assert ass_path.exists()
    ass_content = ass_path.read_text(encoding="utf-8")
    assert "UNEXPECTED" in ass_content
    assert "PlayResX: 1080" in ass_content
    assert "PlayResY: 1920" in ass_content

    # 5. Render video clip with 9:16 crop and burned subtitles
    out_clip = render_clip(
        source_video_path=synthetic_video,
        clip=clip,
        subtitles_ass_path=ass_path,
        output_dir=output_dir,
        force_encoder="libx264",
    )

    assert out_clip.exists()
    assert out_clip.stat().st_size > 10000

    # 6. Probe rendered video with ffprobe to verify 1080x1920 resolution
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(out_clip),
    ]
    res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True, check=True)
    width, height = res.stdout.strip().split(",")
    assert width == "1080"
    assert height == "1920"
