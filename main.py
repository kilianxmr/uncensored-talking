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
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.downloader import download_youtube_video
from src.audio import extract_audio
from src.transcribe import transcribe_audio
from src.selector import (
    select_viral_clips,
    discover_stories,
    plan_story_script_and_broll,
    format_storyboard_display,
    DiscoveredStory,
)
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
        "--bg-music",
        type=str,
        default=os.getenv("BACKGROUND_MUSIC_PATH", "assets/audio/background_music.mp3"),
        help="Path to permanent background music track (default: assets/audio/background_music.mp3).",
    )
    parser.add_argument(
        "--bg-music-volume",
        type=float,
        default=float(os.getenv("BACKGROUND_MUSIC_VOLUME", "0.12")),
        help="Volume of background music relative to speech (default: 0.12).",
    )
    parser.add_argument(
        "--no-bg-music",
        action="store_true",
        help="Disable background music mixing in output shorts.",
    )
    parser.add_argument(
        "--story",
        type=int,
        default=None,
        help="Directly select a specific story number (skips interactive selection prompt).",
    )
    parser.add_argument(
        "--confirm-storyboard",
        action="store_true",
        help="Automatically confirm storyboard without interactive (y/anpassen/n) prompt.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all discovered stories or default top stories non-interactively.",
    )
    parser.add_argument(
        "--no-voice-of-god",
        action="store_true",
        help="Disable Voice-of-God contextual intro hook.",
    )
    parser.add_argument(
        "--voice-of-god-text",
        type=str,
        default=None,
        help="Explicit text for Voice-of-God contextual hook.",
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
    # PHASE 1: Story Discovery (Terminal-Ausgabe & Pause)
    # =========================================================================
    print("\n" + "=" * 64)
    print(">>> PHASE 1: STORY DISCOVERY (Analyse des gesamten Rohtranskripts)")
    print("=" * 64)
    prefer_llm = not args.offline
    discovered_stories = discover_stories(
        transcript_file=transcript_txt,
        word_timestamps_file=word_timestamps_json,
        prefer_llm=prefer_llm,
    )

    if not discovered_stories:
        logger.error("Keine Stories im Transkript gefunden. Beende.")
        sys.exit(1)

    with open(word_timestamps_json, "r", encoding="utf-8") as f:
        all_words = json.load(f)

    rendered_clips: list[Path] = []

    # Interactive Human-in-the-Loop Loop
    while True:
        print("\nGefundene Micro-Stories im Rohtranskript:")
        print("-" * 64)
        for st in discovered_stories:
            print(st.to_terminal_display())
            print()
        print("-" * 64)

        if args.story is not None:
            chosen_id = args.story
            print(f"[CLI-Flag] Wähle Story [{chosen_id}] direkt aus.")
        elif args.batch:
            print("[Batch-Modus] Verarbeite alle Stories nacheinander.")
            chosen_id = None
        else:
            try:
                choice = input("Wähle eine Story-Nummer zum Schneiden (oder 'q' zum Beenden): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nPipeline durch Benutzer abgebrochen.")
                break

            if choice.lower() in ("q", "quit", "exit"):
                print("Pipeline beendet. Bis zum nächsten Mal!")
                break

            if not choice.isdigit():
                print("Ungültige Eingabe. Bitte eine Zahl oder 'q' eingeben.")
                continue

            chosen_id = int(choice)

        selected_stories = (
            discovered_stories if (args.batch and args.story is None)
            else [s for s in discovered_stories if s.story_id == chosen_id]
        )

        if not selected_stories:
            print(f"Story mit Nummer {chosen_id} nicht gefunden. Bitte erneut wählen.")
            if args.story is not None:
                break
            continue

        for story in selected_stories:
            # =================================================================
            # PHASE 2: Radikaler Story-Schnitt & B-Roll-Planung
            # =================================================================
            print(f"\nPlane radikalen Schnitt (30-60s) und B-Roll für Story [{story.story_id}]...")
            feedback_text = None
            proposal = plan_story_script_and_broll(
                story=story,
                words=all_words,
                prefer_llm=prefer_llm,
                feedback=feedback_text,
            )

            # Storyboard Confirmation Loop
            confirmed = False
            while True:
                print(format_storyboard_display(proposal, story))

                if args.confirm_storyboard or args.batch:
                    print("[Auto-Confirm] Storyboard automatisch bestätigt.")
                    confirmed = True
                    break

                try:
                    conf_input = input("\nStoryboard bestätigen? (y / anpassen / n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    conf_input = "n"

                if conf_input in ("y", "yes", "ja"):
                    confirmed = True
                    break
                elif conf_input in ("n", "no", "nein"):
                    print("Storyboard verworfen. Zurück zur Story-Auswahl...")
                    confirmed = False
                    break
                elif conf_input in ("anpassen", "adjust", "edit", "a"):
                    try:
                        feedback_text = input("Was möchtest du anpassen? (z.B. anderer Hook, B-Roll-Änderung): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        feedback_text = None
                    print("\nPasse Storyboard basierend auf Feedback an...")
                    proposal = plan_story_script_and_broll(
                        story=story,
                        words=all_words,
                        prefer_llm=prefer_llm,
                        feedback=feedback_text,
                    )
                else:
                    print("Ungültige Eingabe. Bitte 'y', 'anpassen' oder 'n' eingeben.")

            if not confirmed or proposal is None:
                continue

            # Voice-of-God Hook Configuration Prompt
            enable_vog = not args.no_voice_of_god and (
                proposal.needs_context_hook
                or bool(proposal.voice_of_god_hook)
                or bool(args.voice_of_god_text)
            )
            vog_text = args.voice_of_god_text or proposal.voice_of_god_hook

            if enable_vog and not args.confirm_storyboard and not args.batch and not args.no_voice_of_god:
                print("\n" + "-" * 64)
                print(">>> VOICE-OF-GOD HOOK KONFIGURATION")
                print(f"Voice-of-God Hook empfohlen: {'JA' if proposal.needs_context_hook else 'NEIN'}")
                print(f'Hook-Text: "{vog_text}"')
                print("Optionen:")
                print("  [y] Hook bestätigen und rendern")
                print("  [e] Hook-Text manuell editieren")
                print("  [n] Hook überspringen (reines Podcast-Audio)")
                try:
                    vog_choice = input("Wähle Option (y / e / n) [Standard: y]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    vog_choice = "y"

                if vog_choice in ("n", "no", "nein"):
                    print("Voice-of-God Hook übersprungen.")
                    enable_vog = False
                    vog_text = None
                elif vog_choice in ("e", "edit", "anpassen"):
                    try:
                        custom_vog = input("Neuer Voice-of-God Text: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        custom_vog = vog_text
                    if custom_vog:
                        vog_text = custom_vog
                        enable_vog = True
                        print(f'Aktualisierter Voice-of-God Hook: "{vog_text}"')
                else:
                    print("Voice-of-God Hook bestätigt.")
                    enable_vog = True

            # =================================================================
            # PHASE 3: Asset-Sourcing & Render
            # =================================================================
            print("\n" + "=" * 64)
            print(f">>> PHASE 3: ASSET-SOURCING & RENDERING (Story {story.story_id})")
            print("=" * 64)
            print("Sourcing B-Roll Footage mit strikter Deduplizierung...")
            source_broll_for_clip(proposal, temp_dir=temp_dir, word_timestamps_file=word_timestamps_json)

            sub_dir = temp_dir / "subtitles" / f"story_{story.story_id}"
            concat_subs_file, _ = generate_png_subtitles(
                clip=proposal,
                word_timestamps_file=word_timestamps_json,
                output_dir=sub_dir,
            )

            encoder_choice = None if args.encoder == "auto" else args.encoder
            bg_music = None if args.no_bg_music else args.bg_music
            out_clip = render_clip(
                source_video_path=input_video,
                clip=proposal,
                subtitles_source=concat_subs_file,
                output_dir=output_dir,
                force_encoder=encoder_choice,
                overwrite=True,
                bg_music_path=bg_music,
                bg_music_volume=args.bg_music_volume,
                output_filename=f"story_{story.story_id}.mp4",
                enable_voice_of_god=enable_vog,
                voice_of_god_hook=vog_text,
            )
            rendered_clips.append(out_clip)
            size_mb = out_clip.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 64)
            print(f"  Short erfolgreich gerendert: {out_clip.relative_to(Path.cwd())} ({size_mb:.2f} MB)")
            print("=" * 64)

        if args.story is not None or args.batch:
            break

        try:
            another = input("\nMöchtest du eine weitere Story schneiden? (y / n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            another = "n"

        if another not in ("y", "yes", "ja"):
            print("Pipeline beendet. Bis zum nächsten Mal!")
            break

    total_elapsed = time.time() - start_total_time

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 64)
    print("                 PIPELINE RUN COMPLETE! 🚀")
    print("=" * 64)
    print(f"Total processing time: {total_elapsed:.1f} seconds")
    if rendered_clips:
        print(f"Gerenderte Shorts in 'output/':")
        for clip_path in rendered_clips:
            size_mb = clip_path.stat().st_size / (1024 * 1024)
            print(f"  -> {clip_path.relative_to(Path.cwd())} ({size_mb:.2f} MB)")
    else:
        print("Keine neuen Videos gerendert.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
