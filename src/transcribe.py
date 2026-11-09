"""
Speech transcription module for Uncensored Talking.
Uses faster-whisper (large-v3) with CUDA/float16 acceleration and automatic CPU fallback.
Generates full_transcript.txt and word_timestamps.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("uncensored_talking.transcribe")


def get_whisper_model(
    model_size: str = "large-v3",
    preferred_device: str = "auto",
    preferred_compute_type: str = "auto",
):
    """
    Initializes faster-whisper WhisperModel with CUDA acceleration if available,
    falling back to CPU if CUDA is unavailable or encounters an error.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as err:
        raise ImportError(
            "faster-whisper is not installed. Please install dependencies via: pip install -r requirements.txt"
        ) from err

    devices_to_try: list[tuple[str, str]] = []

    if preferred_device == "cuda":
        comp = "float16" if preferred_compute_type == "auto" else preferred_compute_type
        devices_to_try = [("cuda", comp), ("cpu", "int8")]
    elif preferred_device == "cpu":
        comp = "int8" if preferred_compute_type == "auto" else preferred_compute_type
        devices_to_try = [("cpu", comp)]
    else:  # auto
        devices_to_try = [("cuda", "float16"), ("cpu", "int8")]

    last_error: Exception | None = None
    for device, compute_type in devices_to_try:
        try:
            logger.info(
                f"Attempting to load Whisper model '{model_size}' on device='{device}' (compute_type='{compute_type}')..."
            )
            model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )
            logger.info(f"Successfully loaded Whisper model on {device} ({compute_type}).")
            return model, device, compute_type
        except Exception as exc:
            logger.warning(
                f"Failed to load Whisper model on {device} ({compute_type}): {exc}. Trying fallback..."
            )
            last_error = exc

    raise RuntimeError(
        f"Unable to initialize WhisperModel '{model_size}' on any device configuration: {last_error}"
    )


def transcribe_audio(
    audio_path: Path | str,
    output_dir: Path | str = Path("temp"),
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Transcribes the audio file using faster-whisper and generates:
    1) full_transcript.txt: Continuous text for LLM topic analysis.
    2) word_timestamps.json: Word-level timestamps list [{"word": str, "start": float, "end": float}].

    Args:
        audio_path: Path to the 16kHz mono WAV audio file.
        output_dir: Directory where transcripts will be stored (default: 'temp').
        model_size: Whisper model size (default: 'large-v3').
        device: 'cuda', 'cpu', or 'auto'.
        compute_type: 'float16', 'int8', 'float32', or 'auto'.
        language: Optional language code (e.g. 'en', 'de'). If None, automatically detected.
        overwrite: Overwrite existing transcript files if already present.

    Returns:
        tuple of (Path to full_transcript.txt, Path to word_timestamps.json).
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / "full_transcript.txt"
    word_timestamps_path = output_dir / "word_timestamps.json"

    if (
        not overwrite
        and transcript_path.exists()
        and word_timestamps_path.exists()
        and transcript_path.stat().st_size > 0
        and word_timestamps_path.stat().st_size > 0
    ):
        logger.info(f"Reusing existing transcript files in: {output_dir}")
        return transcript_path, word_timestamps_path

    # Check for downloaded YouTube auto-subtitles in json3 format
    json3_files = list(output_dir.glob("*subs*.json3")) + list(Path("input").glob("*.json3"))
    if json3_files:
        chosen_json3 = json3_files[0]
        logger.info(f"Found YouTube auto-subtitles at '{chosen_json3}'. Extracting word-level timestamps...")
        try:
            with open(chosen_json3, "r", encoding="utf-8") as f:
                data = json.load(f)

            events = data.get("events", [])
            words: list[dict[str, Any]] = []
            full_text_parts: list[str] = []

            for ev in events:
                start_ms = ev.get("tStartMs", 0)
                segs = ev.get("segs", [])
                for idx, s in enumerate(segs):
                    txt = s.get("utf8", "").strip()
                    if not txt or txt == "\n":
                        continue
                    # Remove >> speaker markers if any
                    txt = txt.replace(">>", "").strip()
                    if not txt:
                        continue
                    offset = s.get("tOffsetMs", 0)
                    w_start = round((start_ms + offset) / 1000.0, 3)
                    if idx + 1 < len(segs) and "tOffsetMs" in segs[idx + 1]:
                        w_end = round((start_ms + segs[idx + 1]["tOffsetMs"]) / 1000.0, 3)
                    else:
                        w_end = round(w_start + 0.35, 3)
                    if w_end <= w_start:
                        w_end = round(w_start + 0.25, 3)
                    words.append({"word": txt, "start": w_start, "end": w_end})
                    full_text_parts.append(txt)

            if words:
                full_transcript = " ".join(full_text_parts)
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(full_transcript + "\n")
                with open(word_timestamps_path, "w", encoding="utf-8") as f:
                    json.dump(words, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"Successfully parsed {len(words)} word timestamps from YouTube captions into {output_dir}."
                )
                return transcript_path, word_timestamps_path
        except Exception as exc:
            logger.warning(f"Error parsing YouTube json3 subtitles: {exc}. Falling back to faster-whisper.")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file for transcription not found: {audio_path}")

    logger.info(f"Starting faster-whisper transcription for: {audio_path.name}")
    model, used_device, used_compute = get_whisper_model(
        model_size=model_size,
        preferred_device=device,
        preferred_compute_type=compute_type,
    )

    # Transcribe with word timestamps enabled
    # vad_filter removes long silences and prevents hallucinations
    segments_gen, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(
        f"Detected language: '{info.language}' with probability {info.language_probability:.2f}. "
        f"Audio duration: {info.duration:.1f}s"
    )

    full_text_parts: list[str] = []
    word_timestamps: list[dict[str, Any]] = []

    for segment in segments_gen:
        seg_text = segment.text.strip()
        if seg_text:
            full_text_parts.append(seg_text)

        if segment.words:
            for w in segment.words:
                clean_word = w.word.strip()
                if not clean_word:
                    continue
                word_timestamps.append(
                    {
                        "word": clean_word,
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                    }
                )

    full_transcript = " ".join(full_text_parts)

    # 1. Save full_transcript.txt
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(full_transcript + "\n")

    # 2. Save word_timestamps.json
    with open(word_timestamps_path, "w", encoding="utf-8") as f:
        json.dump(word_timestamps, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Transcription complete: {len(word_timestamps)} words captured. "
        f"Saved full_transcript.txt and word_timestamps.json in {output_dir}."
    )

    return transcript_path, word_timestamps_path
