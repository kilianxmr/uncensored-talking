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


def generate_sub_boom(
    output_path: Path | str = Path("assets/sfx/sub_boom.wav"),
    duration: float = 1.2,
    freq: float = 60.0,
    sample_rate: int = 48000,
    decay_rate: float = 3.5,
) -> Path:
    """
    Programmatically generates a 60Hz sub-bass boom impulse with exponential decay.
    Format: 48kHz, 16-bit Stereo PCM WAV.
    """
    import math
    import struct
    import wave

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = int(duration * sample_rate)
    max_amplitude = 30000.0  # Safe headroom below 32767

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(2)  # Stereo
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(total_samples):
            t = i / sample_rate
            # Pitch slide down slightly for cinematic thump (e.g. 65Hz down to 45Hz)
            instant_freq = freq * (1.0 - 0.25 * (t / duration))
            # Exponential decay envelope
            envelope = math.exp(-decay_rate * t)
            # Sine wave
            sample_val = int(max_amplitude * envelope * math.sin(2.0 * math.pi * instant_freq * t))
            sample_val = max(-32768, min(32767, sample_val))
            # Pack as 16-bit signed stereo
            frame = struct.pack("<hh", sample_val, sample_val)
            frames.extend(frame)

        wav_file.writeframes(frames)

    logger.info(f"Generated 60Hz Sub-Boom SFX impulse at: {output_path}")
    return output_path


def ensure_sfx_assets(sfx_dir: Path | str = Path("assets/sfx")) -> dict[str, Path]:
    """Ensures necessary SFX assets like sub_boom.wav exist; generates if missing."""
    sfx_dir = Path(sfx_dir).resolve()
    sfx_dir.mkdir(parents=True, exist_ok=True)

    sub_boom = sfx_dir / "sub_boom.wav"
    if not sub_boom.exists() or sub_boom.stat().st_size == 0:
        generate_sub_boom(sub_boom)

    return {
        "sub_boom": sub_boom,
    }
