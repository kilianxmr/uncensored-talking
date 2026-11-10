"""
YouTube downloader module for Uncensored Talking (@uncensored_talking).
Uses yt-dlp to download high-resolution videos (preferred 1080p MP4).
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("uncensored_talking.downloader")


def sanitize_filename(name: str) -> str:
    """Removes or replaces invalid characters for filenames."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = clean.replace(" ", "_").strip()
    return clean[:60] or "youtube_video"


def download_youtube_video(
    url: str,
    output_dir: Path | str = Path("input"),
    output_filename: str = "raw_video.mp4",
    max_height: int = 1080,
    overwrite: bool = False,
) -> Path:
    """
    Downloads a YouTube video at up to 1080p in MP4 format using yt-dlp.

    Args:
        url: The YouTube video URL.
        output_dir: Target directory (default: 'input').
        output_filename: Target filename (default: 'raw_video.mp4').
        max_height: Maximum video resolution height (default: 1080).
        overwrite: Whether to re-download if file already exists.

    Returns:
        Path to the downloaded video file.

    Raises:
        ImportError: If yt-dlp is not installed.
        RuntimeError: If download fails.
    """
    try:
        import yt_dlp
    except ImportError as err:
        raise ImportError("yt-dlp is not installed. Run: pip install yt-dlp") from err

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / output_filename

    if target_path.exists() and not overwrite and target_path.stat().st_size > 100000:
        logger.info(f"Reusing existing downloaded video: {target_path}")
        return target_path

    logger.info(f"Downloading YouTube video from: {url}")
    logger.info(f"Target path: {target_path} (max height: {max_height}p)")

    # yt-dlp options: prefer direct HTTP streams for blistering download speed
    ydl_opts = {
        "format": f"bestvideo[height<={max_height}][protocol^=http][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "outtmpl": str(target_path.with_suffix("")),  # yt-dlp appends .mp4
        "merge_output_format": "mp4",
        "nocheckcertificate": True,
        "js_runtimes": {"node": {"path": "/usr/local/bin/node"}},
        "writeautosub": True,
        "subtitleslangs": ["en", "de"],
        "subtitlesformat": "json3",
        "postprocessors": [
            {
                "key": "FFmpegVideoRemuxer",
                "preferedformat": "mp4",
            }
        ],
        "quiet": False,
        "no_warnings": False,
        "noprogress": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown Title") if info else "Unknown"
            duration = info.get("duration", 0) if info else 0
            logger.info(f"Successfully downloaded '{title}' ({duration}s) -> {target_path}")

        # In case yt-dlp saved with slight extension variance
        if not target_path.exists():
            matched = list(output_dir.glob(f"{target_path.stem}.*"))
            if matched:
                target_path = matched[0]

        # Extract and save timed json3 captions if available
        if info:
            auto_subs = info.get("automatic_captions", {}) or info.get("subtitles", {})
            for lang in ["en", "de"]:
                if lang in auto_subs:
                    for fmt in auto_subs[lang]:
                        if fmt.get("ext") == "json3" and fmt.get("url"):
                            try:
                                import ssl
                                import urllib.request

                                ctx = ssl.create_default_context()
                                ctx.check_hostname = False
                                ctx.verify_mode = ssl.CERT_NONE
                                req = urllib.request.Request(fmt["url"], headers={"User-Agent": "Mozilla/5.0"})
                                with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                                    sub_data = resp.read()
                                if sub_data:
                                    temp_sub = Path("temp") / f"yt_subs.{lang}.json3"
                                    temp_sub.parent.mkdir(parents=True, exist_ok=True)
                                    with open(temp_sub, "wb") as f:
                                        f.write(sub_data)
                                    logger.info(f"Saved direct YouTube {lang} json3 captions to {temp_sub}")
                                    break
                            except Exception as sub_err:
                                logger.debug(f"Direct caption fetch failed for {lang}: {sub_err}")
                    if (Path("temp") / f"yt_subs.{lang}.json3").exists():
                        break

        return target_path
    except Exception as exc:
        logger.error(f"yt-dlp download failed for URL '{url}': {exc}")
        raise RuntimeError(f"YouTube download failed: {exc}") from exc
