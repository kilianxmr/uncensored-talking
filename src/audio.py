"""
Audio extraction module for Uncensored Talking.
Extracts 16kHz mono WAV audio from source video files using FFmpeg.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("uncensored_talking.audio")


def extract_audio(
    video_path: Path | str,
    output_path: Path | str = Path("temp/audio.wav"),
    sample_rate: int = 16000,
    overwrite: bool = True,
) -> Path:
    """
    Extracts the audio track from a video file and converts it to a 16kHz mono WAV file.
    This format is optimal for Whisper speech recognition models.

    Args:
        video_path: Path to the input MP4/video file.
        output_path: Destination path for the WAV audio file.
        sample_rate: Target sample rate in Hz (default: 16000).
        overwrite: Whether to overwrite the destination if it already exists.

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If the source video file does not exist.
        RuntimeError: If FFmpeg fails during audio extraction.
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Source video file not found: {video_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        logger.info(f"Reusing existing audio dump at: {output_path}")
        return output_path

    logger.info(f"Extracting audio from {video_path.name} -> {output_path.name} (16kHz, mono)...")

    # Construct ffmpeg command:
    # -vn: disable video recording
    # -acodec pcm_s16le: uncompressed 16-bit PCM audio
    # -ar 16000: 16kHz sampling rate
    # -ac 1: mono channel
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.info(f"Audio extracted successfully to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.strip() if exc.stderr else "Unknown FFmpeg error"
        logger.error(f"FFmpeg audio extraction failed:\n{err_msg}")
        raise RuntimeError(f"FFmpeg audio extraction failed: {err_msg}") from exc
