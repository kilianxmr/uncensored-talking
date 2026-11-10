#!/usr/bin/env python3
"""
=============================================================================
Uncensored Talking (@uncensored_talking) - Automated YouTube & Short Clipper
=============================================================================
Pipeline:
  1. YouTube Download  -> yt-dlp direct stream MP4 download (src/downloader.py)
  2. Audio Extraction  -> 16kHz mono WAV (src/audio.py)
  3. Speech Recognition -> faster-whisper large-v3 / YouTube JSON3 (src/transcribe.py)
  4. Non-Linear Selector -> 3-5s shocking hook mandate, tight jump cuts,
                            2-5s scenes with 2-4 visual search queries (src/selector.py)
  5. Real-Footage Engine -> YouTube search, OpenCV peak-motion cut,
                            focal-point Ken-Burns (src/broll_engine.py)
  6. Gradient Subtitles  -> Pillow transparent PNG sequence with vertical
                            linear gradient (Yellow->Warm-Orange) & 12px outer stroke (src/subtitles.py)
  7. Video Engine        -> Anti-stretch proportional scaling, 30-frame smoothed
                            face tracking, 3 layout modes & subtitle burn (src/video.py)
=============================================================================
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline modules
from src.downloader import download_youtube_video
from src.audio import extract_audio
from src.transcribe import transcribe_audio
from src.selector import select_viral_clips
from src.broll_engine import source_broll_for_clip
from src.subtitles import generate_png_subtitles
from src.video import render_clip, check_nvenc_support


BANNER = r"""
╔═══════════════════════════════════════════════════════════════════╗
║                   UNCENSORED TALKING CLI                          ║
║                @uncensored_talking Automation                     ║
║   Non-Linear Hooks • Real Footage • Anti-Stretch • Gradient Subs  ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("uncensored_talking")
    root_logger.setLevel(level)
    root_logger.handlers = [handler]
    return root_logger


def resolve_video_source(url: str | None, local_path: str | None, overwrite: bool = False) -> Path:
    """Downloads from YouTube or locates an existing video in input/."""
    input_dir = Path("input").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    if url:
        print(f"\n>>> Sourcing video from YouTube URL: {url}")
        video_path = download_youtube_video(
            url=url,
            output_dir=input_dir,
            output_filename="raw_video.mp4",
            max_height=1080,
            overwrite=overwrite,
        )
        return video_path

    if local_path:
        video_path = Path(local_path).resolve()
        if not video_path.exists():
            print(f"Error: Specified input video does not exist: {video_path}")
            sys.exit(1)
        return video_path

    candidates = (
        list(input_dir.glob("*.mp4"))
        + list(input_dir.glob("*.mkv"))
        + list(input_dir.glob("*.mov"))
    )
    valid_candidates = [c for c in candidates if not c.name.startswith(".") and "broll" not in c.as_posix()]

    if not valid_candidates:
        print(f"Error: No video found and no --url or --input specified.")
        print("Usage: python main.py --url 'https://www.youtube.com/watch?v=...'")
        sys.exit(1)

    selected = valid_candidates[0]
    print(f"Auto-detected existing video: {selected.name}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uncensored Talking (@uncensored_talking) Automated Short-Form Clipper & B-Roll Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-u", "--url",
        type=str,
        default=None,
        help="YouTube video URL to download and process automatically.",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to an existing local video file (e.g. input/podcast.mp4).",
    )
    parser.add_argument(
        "-c", "--clips",
        type=int,
        default=3,
        help="Number of viral clips to extract and render.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=os.getenv("WHISPER_MODEL", "large-v3"),
        help="Whisper model size (e.g. large-v3, medium, small, base).",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=os.getenv("WHISPER_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu"],
        help="Compute device for Whisper transcription.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=os.getenv("WHISPER_COMPUTE_TYPE", "auto"),
        help="Whisper compute type: 'auto', 'float16', 'int8', 'float32'.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=os.getenv("VIDEO_ENCODER", "auto"),
        choices=["auto", "h264_nvenc", "libx264"],
        help="FFmpeg video encoder.",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Reuse existing temp/full_transcript.txt and temp/word_timestamps.json if present.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force heuristic clip selector and visual planner (skip LLM API call).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download, re-transcription, and re-rendering of clips.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    print(BANNER)

    start_total_time = time.time()
    input_video = resolve_video_source(args.url, args.input, overwrite=args.force)

    temp_dir = Path("temp").resolve()
    output_dir = Path("output").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardware check
    has_nvenc = check_nvenc_support()
    logger.info(f"Target video: {input_video.name}")
    logger.info(f"NVENC Acceleration: {'ACTIVE' if has_nvenc else 'CPU FALLBACK (libx264)'}")

    # =========================================================================
    # STEP 1: Audio Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> STEP 1: Extracting 16kHz Mono WAV Audio via FFmpeg")
    print("=" * 60)
    audio_wav = temp_dir / "audio.wav"
    extract_audio(
        video_path=input_video,
        output_path=audio_wav,
        overwrite=args.force or not audio_wav.exists(),
    )

    # =========================================================================
    # STEP 2: Speech Recognition
    # =========================================================================
    print("\n" + "=" * 60)
    print(f">>> STEP 2: Transcribing Audio (faster-whisper / YouTube Auto-Subs)")
    print("=" * 60)
    transcript_txt, word_timestamps_json = transcribe_audio(
        audio_path=audio_wav,
        output_dir=temp_dir,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        overwrite=args.force or (not args.skip_transcribe and not (temp_dir / "full_transcript.txt").exists()),
    )

    # =========================================================================
    # STEP 3: Non-Linear Hook & Visual Direction Planning
    # =========================================================================
    print("\n" + "=" * 60)
    print(f">>> STEP 3: Non-Linear Hook Curation & 2-5s Visual Scene Timeline")
    print("=" * 60)
    prefer_llm = not args.offline
    clip_proposals = select_viral_clips(
        transcript_file=transcript_txt,
        word_timestamps_file=word_timestamps_json,
        num_clips=args.clips,
        prefer_llm=prefer_llm,
        output_json_path=temp_dir / "selected_clips.json",
    )

    if not clip_proposals:
        logger.error("No valid clip proposals were generated. Aborting.")
        sys.exit(1)

    print(f"\nIdentified {len(clip_proposals)} explosive non-linear clips:")
    for cp in clip_proposals:
        print(f"  • [Clip {cp.clip_id}] '{cp.title}' ({cp.total_duration:.1f}s)")
        print(f"    Hook (first 3-5s): \"{cp.hook[:75]}...\"")
        print(f"    Segments: {len(cp.segments)} (Non-linear cuts)")
        print(f"    Scenes ({len(cp.scenes)}): " + ", ".join(f"{sc.layout} ({sc.duration:.1f}s)" for sc in cp.scenes[:4]) + "...")

    # =========================================================================
    # STEP 4: Real-Footage YouTube Sourcing & Peak Motion Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> STEP 4: Sourcing Real B-Roll Footage & Focal-Point Ken-Burns")
    print("=" * 60)
    for cp in clip_proposals:
        print(f"Sourcing real B-roll footage for Clip {cp.clip_id}: '{cp.title}'...")
        source_broll_for_clip(cp, temp_dir=temp_dir)

    # =========================================================================
    # STEP 5 & 6: Gradient Subtitles & Anti-Stretch Video Rendering
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> STEP 5 & 6: Rendering Shorts (Pillow Gradient Subs & Proportional Face Tracking)")
    print("=" * 60)

    rendered_clips: list[Path] = []
    for cp in clip_proposals:
        print(f"\nRendering Short {cp.clip_id}/{len(clip_proposals)}: '{cp.title}'")

        # 5. Generate Transparent PNG Subtitle Sequence with Vertical Gradient
        sub_dir = temp_dir / "subtitles" / f"clip_{cp.clip_id}"
        concat_subs_file, _ = generate_png_subtitles(
            clip=cp,
            word_timestamps_file=word_timestamps_json,
            output_dir=sub_dir,
        )

        # 6. Render video clip with proportional anti-stretch scaling & smoothed face tracking
        encoder_choice = None if args.encoder == "auto" else args.encoder
        out_clip = render_clip(
            source_video_path=input_video,
            clip=cp,
            subtitles_source=concat_subs_file,
            output_dir=output_dir,
            force_encoder=encoder_choice,
            overwrite=True,
        )
        rendered_clips.append(out_clip)

    total_elapsed = time.time() - start_total_time

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("                 PIPELINE COMPLETE! 🚀")
    print("=" * 60)
    print(f"Total processing time: {total_elapsed:.1f} seconds")
    print(f"Generated {len(rendered_clips)} dynamic layout shorts for @uncensored_talking:")
    for clip_path in rendered_clips:
        size_mb = clip_path.stat().st_size / (1024 * 1024)
        print(f"  -> {clip_path.relative_to(Path.cwd())} ({size_mb:.2f} MB)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
