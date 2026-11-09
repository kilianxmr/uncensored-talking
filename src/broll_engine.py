"""
B-Roll asset sourcing and motion engine for Uncensored Talking (@uncensored_talking).
Fetches relevant vertical/horizontal video clips or high-res images from:
- Pexels API (PEXELS_API_KEY)
- Pixabay API (PIXABAY_API_KEY)
- Wikimedia Commons / CC0 public search
Applies smooth Ken-Burns effect (1.0x -> 1.15x zoom/pan) on static images via FFmpeg.
"""

import hashlib
import logging
import os
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .selector import ClipProposal, SceneCut

load_dotenv()
logger = logging.getLogger("uncensored_talking.broll")


def sanitize_query(query: str) -> str:
    """Cleans up search query string."""
    return query.strip().replace('"', "").replace("'", "")


def download_file(url: str, dest_path: Path, timeout: int = 15) -> Path:
    """Downloads a file from a URL to local disk."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UncensoredTalkingBot/2.0"
    }
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    return dest_path


def search_pexels(
    query: str,
    api_key: str,
) -> tuple[str | None, str]:
    """
    Searches Pexels for videos or high-resolution images.
    Returns: (url, 'video' | 'image' | None)
    """
    headers = {"Authorization": api_key}
    clean_q = urllib.parse.quote(sanitize_query(query))

    # 1. Try Pexels Video Search (portrait preferred)
    try:
        v_url = f"https://api.pexels.com/videos/search?query={clean_q}&per_page=5&orientation=portrait"
        res = requests.get(v_url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            videos = data.get("videos", [])
            for vid in videos:
                for file_info in vid.get("video_files", []):
                    # Prefer HD portrait MP4
                    if file_info.get("file_type") == "video/mp4" and file_info.get("width", 0) >= 720:
                        return file_info["link"], "video"
            # Fallback to any orientation
            if videos:
                first_files = videos[0].get("video_files", [])
                if first_files:
                    return first_files[0]["link"], "video"
    except Exception as exc:
        logger.debug(f"Pexels video search error: {exc}")

    # 2. Try Pexels Photo Search
    try:
        p_url = f"https://api.pexels.com/v1/search?query={clean_q}&per_page=5"
        res = requests.get(p_url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            photos = data.get("photos", [])
            if photos:
                # Use large or original
                src = photos[0].get("src", {})
                img_url = src.get("large2x") or src.get("large") or src.get("original")
                if img_url:
                    return img_url, "image"
    except Exception as exc:
        logger.debug(f"Pexels photo search error: {exc}")

    return None, ""


def search_pixabay(
    query: str,
    api_key: str,
) -> tuple[str | None, str]:
    """
    Searches Pixabay for videos or high-resolution photos.
    """
    clean_q = urllib.parse.quote(sanitize_query(query))

    # 1. Video search
    try:
        v_url = f"https://pixabay.com/api/videos/?key={api_key}&q={clean_q}&per_page=5"
        res = requests.get(v_url, timeout=8)
        if res.status_code == 200:
            hits = res.json().get("hits", [])
            for hit in hits:
                vids = hit.get("videos", {})
                # Try large or medium
                choice = vids.get("large") or vids.get("medium") or vids.get("small")
                if choice and choice.get("url"):
                    return choice["url"], "video"
    except Exception as exc:
        logger.debug(f"Pixabay video search error: {exc}")

    # 2. Image search
    try:
        p_url = f"https://pixabay.com/api/?key={api_key}&q={clean_q}&per_page=5&image_type=photo"
        res = requests.get(p_url, timeout=8)
        if res.status_code == 200:
            hits = res.json().get("hits", [])
            if hits:
                img_url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
                if img_url:
                    return img_url, "image"
    except Exception as exc:
        logger.debug(f"Pixabay image search error: {exc}")

    return None, ""


def search_wikimedia_public(query: str) -> tuple[str | None, str]:
    """
    Public CC0 / royalty-free image search using Wikimedia Commons API.
    """
    clean_q = urllib.parse.quote(sanitize_query(query))
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={clean_q}&gsrnamespace=6&gsrlimit=5&prop=imageinfo"
        f"&iiprop=url|mime&format=json"
    )
    headers = {"User-Agent": "UncensoredTalkingBot/2.0 (research automation)"}
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                infos = page.get("imageinfo", [])
                for info in infos:
                    mime = info.get("mime", "")
                    img_url = info.get("url")
                    if img_url and ("image/jpeg" in mime or "image/png" in mime or "image/webp" in mime):
                        return img_url, "image"
    except Exception as exc:
        logger.debug(f"Wikimedia search error: {exc}")

    return None, ""


def generate_cinematic_fallback(
    output_path: Path,
    duration: float,
    width: int,
    height: int,
    query_seed: str,
) -> Path:
    """
    Generates a cinematic motion visual texture using FFmpeg filters
    when no external API or internet access is available.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Generate abstract dynamic motion plate (gradient + moving geometric patterns)
    hue_val = (hash(query_seed) % 360)
    filter_expr = (
        f"color=c=black:s={width}x{height}:d={duration},"
        f"geq=r='128+100*sin(X/80+T*2)':g='80+60*cos(Y/60+T*1.5)':b='180+50*sin((X+Y)/100+T)',"
        f"boxblur=20:1,"
        f"hue=h={hue_val}:s=1.2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", filter_expr,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        # Fallback to simple solid color motion
        simple_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x111122:s={width}x{height}:d={duration}",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(simple_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_path


def apply_ken_burns_effect(
    image_path: Path,
    output_path: Path,
    duration: float,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> Path:
    """
    Applies a smooth Ken-Burns effect (continuous zoom 1.0x -> 1.15x) on a static image
    and renders it into an MP4 clip.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(duration * fps) + 5
    zoom_step = 0.15 / max(1, total_frames)

    # FFmpeg zoompan filter:
    # zoom starts at 1.0 and grows up to 1.15
    filter_str = (
        f"zoompan=z='min(zoom+{zoom_step:.6f},1.15)':"
        f"d={total_frames}:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", filter_str,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def process_broll_video(
    video_path: Path,
    output_path: Path,
    duration: float,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """
    Crops, centers, scales, and trims/loops a B-Roll video to the exact target dimensions and duration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Center crop and scale to width:height, loop if shorter than duration
    filter_str = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"setsar=1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "3",
        "-i", str(video_path),
        "-vf", filter_str,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def fetch_broll_asset(
    query: str,
    duration: float,
    target_width: int,
    target_height: int,
    output_path: Path,
) -> Path:
    """
    Orchestrates finding, downloading, and motion-processing of a B-Roll asset for a given query.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 5000:
        return output_path

    pexels_key = os.getenv("PEXELS_API_KEY")
    pixabay_key = os.getenv("PIXABAY_API_KEY")

    asset_url: str | None = None
    asset_type: str = ""

    # 1. Try Pexels
    if pexels_key:
        asset_url, asset_type = search_pexels(query, pexels_key)

    # 2. Try Pixabay
    if not asset_url and pixabay_key:
        asset_url, asset_type = search_pixabay(query, pixabay_key)

    # 3. Try Wikimedia Commons
    if not asset_url:
        asset_url, asset_type = search_wikimedia_public(query)

    # 4. If asset found, download and process
    if asset_url:
        try:
            h = hashlib.md5(asset_url.encode()).hexdigest()[:10]
            ext = ".mp4" if asset_type == "video" else ".jpg"
            raw_download = output_path.parent / f"raw_{h}{ext}"

            if not raw_download.exists():
                logger.info(f"Downloading B-Roll ({asset_type}) for query '{query}' from {asset_url[:50]}...")
                download_file(asset_url, raw_download)

            if asset_type == "video":
                return process_broll_video(
                    video_path=raw_download,
                    output_path=output_path,
                    duration=duration,
                    width=target_width,
                    height=target_height,
                )
            else:
                return apply_ken_burns_effect(
                    image_path=raw_download,
                    output_path=output_path,
                    duration=duration,
                    width=target_width,
                    height=target_height,
                )
        except Exception as exc:
            logger.warning(f"Error processing downloaded B-Roll for '{query}': {exc}. Using cinematic fallback.")

    # 5. Fallback generator
    logger.info(f"Generating cinematic motion plate for query '{query}' ({duration:.1f}s)...")
    return generate_cinematic_fallback(
        output_path=output_path,
        duration=duration,
        width=target_width,
        height=target_height,
        query_seed=query,
    )


def source_broll_for_clip(
    clip: ClipProposal,
    temp_dir: Path | str = Path("temp"),
) -> list[SceneCut]:
    """
    Iterates over all scenes in a clip. For every scene requiring B-Roll
    ('full_broll' or 'splitscreen'), fetches or generates the visual asset.
    """
    temp_dir = Path(temp_dir)
    broll_dir = temp_dir / "broll" / f"clip_{clip.clip_id}"
    broll_dir.mkdir(parents=True, exist_ok=True)

    for idx, scene in enumerate(clip.scenes, start=1):
        if scene.layout in ("full_broll", "splitscreen"):
            # Target resolution:
            # full_broll: 1080x1920
            # splitscreen: top half is 1080x960
            if scene.layout == "splitscreen":
                w, h = 1080, 960
            else:
                w, h = 1080, 1920

            clean_slug = sanitize_query(scene.search_query).replace(" ", "_")[:30] or "scene"
            dest_file = broll_dir / f"scene_{idx}_{clean_slug}.mp4"

            broll_path = fetch_broll_asset(
                query=scene.search_query or "cinematic dramatic lighting",
                duration=scene.duration,
                target_width=w,
                target_height=h,
                output_path=dest_file,
            )
            scene.broll_asset_path = str(broll_path.resolve())

    return clip.scenes
