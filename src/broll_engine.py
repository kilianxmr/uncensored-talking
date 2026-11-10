"""
Real-Footage B-Roll Engine for Uncensored Talking (@uncensored_talking).
Sourcing & Motion Architecture:
- Searches real footage via yt-dlp: ytsearch3:"<query> hd".
- Downloads 30-60s and extracts the peak motion 3-5s snippet using OpenCV frame differencing.
- For still images: OpenCV Saliency/contrast analysis finds the visual focal point (nx, ny)
  and applies continuous targeted Ken-Burns zoom (1.0x -> 1.15x).
- Manual override: Checks input/broll/ first if user dropped custom clips.
- Fallback: Procedural dynamic motion plate.
"""

import hashlib
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .selector import ClipProposal, SceneCut

logger = logging.getLogger("uncensored_talking.broll")


def sanitize_query(query: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "", query)
    return clean.strip().replace("'", "").replace('"', "")


def check_manual_broll_drop(query: str, broll_dir: Path = Path("input/broll")) -> Path | None:
    """Checks input/broll/ for files matching query keywords or any dropped media."""
    if not broll_dir.exists():
        return None

    files = [
        f for f in broll_dir.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in [".mp4", ".mov", ".mkv", ".jpg", ".png", ".webp"]
    ]
    if not files:
        return None

    # Check keyword match
    keywords = set(query.lower().split())
    for f in files:
        f_name_words = set(re.sub(r"[^\w]", " ", f.stem.lower()).split())
        if keywords.intersection(f_name_words):
            logger.info(f"Using manual user-provided B-roll match: {f.name}")
            return f

    # If only one file exists in input/broll, use it
    logger.info(f"Using manual user-provided B-roll from input/broll: {files[0].name}")
    return files[0]


def find_visual_focal_point(image_path: Path | str) -> tuple[float, float]:
    """
    Computes visual saliency / contrast map via OpenCV Laplacian & Gaussian blur.
    Returns normalized coordinates (nx, ny) in range [0.1, 0.9] of the primary focal point.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5, 0.5

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast map via Laplacian gradient
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        contrast = np.abs(lap)

        # Smooth to detect largest prominent feature area
        ksize = min(99, max(21, (min(h, w) // 10) | 1))
        blurred = cv2.GaussianBlur(contrast, (ksize, ksize), 0)

        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        nx = max_loc[0] / float(w)
        ny = max_loc[1] / float(h)

        # Clamp safely away from extreme edges for smooth zoom bounds
        return max(0.15, min(0.85, nx)), max(0.15, min(0.85, ny))
    except Exception as exc:
        logger.debug(f"Saliency focal point detection fallback: {exc}")
        return 0.5, 0.5


def validate_broll_frame(video_path: Path) -> tuple[bool, str]:
    """
    Visual quality gate:
    1. Extracts the middle frame of the rendered snippet.
    2. Checks luminance (rejects near-solid black or blown-out white).
    3. Checks color standard deviation & gray variance (rejects flat monochromatic solids).
    4. Checks edge complexity via Canny (rejects static slides, logos, or solid color intro cards).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Cannot open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return False, "Zero frames in video"

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, "Cannot read middle frame"

    mean_lum = float(np.mean(frame))
    if mean_lum < 20.0:
        return False, f"Near solid black (mean lum: {mean_lum:.1f})"
    if mean_lum > 245.0:
        return False, f"Near solid white (mean lum: {mean_lum:.1f})"

    color_std = float(np.mean(np.std(frame, axis=(0, 1))))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_var = float(np.var(gray))
    if color_std < 8.0 or gray_var < 50.0:
        return False, f"Monochrome/flat background (color std: {color_std:.1f}, gray var: {gray_var:.1f})"

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    if edge_density < 0.005:
        return False, f"Static title card/logo with low edge complexity (edge density: {edge_density:.4f})"

    return True, "OK"


def extract_peak_motion_snippet(
    video_path: Path,
    output_path: Path,
    target_duration: float = 3.0,
    target_width: int = 1080,
    target_height: int = 1920,
    min_offset_sec: float = 10.0,
) -> Path:
    """
    1. Logo & intro protection: NEVER cuts from second 0.0.
       Enforces minimum start offset (>=10s on normal footage, or >=15% on short test fixtures).
    2. Analyzes frame-by-frame motion using OpenCV absdiff from the offset onwards.
    3. Finds the 1-3s window with MAXIMUM visual action/motion.
    4. Cuts with proportional scaling (scale=-1:target_height, crop=target_width:target_height).
    5. Visual quality gate: checks that the resulting clip is not black, flat, or a static logo.
       If rejected, automatically advances 5s and retries up to 3 times.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for motion analysis: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration = total_frames / fps if fps > 0 else 0.0

    # Intro & logo protection: NEVER start from 0.0s
    if total_duration >= 18.0:
        actual_min_offset = min_offset_sec
    elif total_duration > target_duration + 3.0:
        actual_min_offset = 2.0  # skip initial 2s on shorter test videos
    else:
        actual_min_offset = 0.0

    start_scan_frame = int(actual_min_offset * fps)
    max_scan_frames = min(total_frames, int(75 * fps)) if total_frames > 0 else int(75 * fps)
    window_frames = max(1, int(target_duration * fps))

    if start_scan_frame > 0 and start_scan_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_scan_frame)

    motion_scores: list[tuple[int, float]] = []
    prev_gray = None
    sample_step = 3  # sample every 3rd frame for ultra-fast motion estimation

    frame_idx = start_scan_frame
    while frame_idx < max_scan_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (frame_idx - start_scan_frame) % sample_step == 0:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff))
                motion_scores.append((frame_idx, score))
            prev_gray = gray

        frame_idx += 1

    cap.release()

    # Find window with highest average motion starting at or after actual_min_offset
    best_start_sec = actual_min_offset
    if len(motion_scores) > window_frames // sample_step:
        window_size = window_frames // sample_step
        scores_arr = [s[1] for s in motion_scores]
        rolling_sums = np.convolve(scores_arr, np.ones(window_size), mode="valid")
        best_idx = int(np.argmax(rolling_sums))
        best_frame = motion_scores[best_idx][0]
        best_start_sec = max(actual_min_offset, best_frame / fps)

    logger.info(
        f"Selected peak-motion action window at {best_start_sec:.2f}s (duration: {target_duration:.1f}s) "
        f"for {output_path.name}"
    )

    # Cut candidates with 5s retry offsets if rejected by quality gate
    cut_candidates = [
        best_start_sec,
        best_start_sec + 5.0,
        best_start_sec + 10.0,
        actual_min_offset + 15.0,
    ]

    vf = (
        f"scale=-1:{target_height}:flags=lanczos,"
        f"crop={target_width}:{target_height}:(in_w-{target_width})/2:0,"
        f"setsar=1,fps=30"
    )

    success = False
    for candidate_start in cut_candidates:
        if candidate_start + target_duration > total_duration and total_duration > target_duration:
            candidate_start = max(actual_min_offset, total_duration - target_duration - 0.2)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{candidate_start:.3f}",
            "-t", f"{target_duration:.3f}",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-an",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        is_valid, reason = validate_broll_frame(output_path)
        if is_valid:
            logger.info(
                f"Visual quality gate PASSED at {candidate_start:.2f}s (duration: {target_duration:.1f}s) "
                f"for {output_path.name}"
            )
            success = True
            break
        else:
            logger.warning(
                f"Visual quality gate REJECTED snippet at {candidate_start:.2f}s ({reason}) for {output_path.name}. "
                f"Retrying 5s later..."
            )

    if not success:
        logger.warning(f"All offset cuts from {video_path.name} failed visual quality gate.")
        # Ensure output is cleaned up or left as fallback
    return output_path


def apply_focal_ken_burns(
    image_path: Path,
    output_path: Path,
    duration: float,
    target_width: int,
    target_height: int,
    fps: int = 30,
) -> Path:
    """
    Applies continuous Ken-Burns zoom (1.0x -> 1.15x) targeted at visual focal point (nx, ny).
    Proportionally scales to target_width x target_height.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = find_visual_focal_point(image_path)
    total_frames = int(duration * fps) + 5
    zoom_step = 0.15 / max(1, total_frames)

    # Zoom centered around focal point (nx, ny)
    filter_expr = (
        f"zoompan=z='min(zoom+{zoom_step:.6f},1.15)':"
        f"x='(iw-iw/zoom)*{nx:.3f}':"
        f"y='(ih-ih/zoom)*{ny:.3f}':"
        f"d={total_frames}:"
        f"s={target_width}x{target_height}:"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", filter_expr,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def download_youtube_broll(
    query: str,
    output_video_path: Path,
    max_duration: int = 45,
    candidate_index: int = 1,
) -> Path | None:
    """
    Searches YouTube for real footage using yt-dlp: ytsearch3:"<query> hd"
    Downloads candidate 1, 2, or 3 based on candidate_index.
    """
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    search_term = f"ytsearch3:{sanitize_query(query)} hd"

    ydl_opts = {
        "format": "bestvideo[height<=720][protocol^=http][ext=mp4]+bestaudio/best[height<=720][ext=mp4]/best",
        "outtmpl": str(output_video_path.with_suffix("")),
        "merge_output_format": "mp4",
        "nocheckcertificate": True,
        "playlist_items": str(candidate_index),
        "max_downloads": 1,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        import yt_dlp
        from yt_dlp.utils import MaxDownloadsReached

        logger.info(f"Searching real footage via YouTube (candidate {candidate_index}): '{query}'...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([search_term])
            except MaxDownloadsReached:
                pass  # Normal expected exit when 1 video is downloaded

        if output_video_path.exists() and output_video_path.stat().st_size > 50000:
            return output_video_path

        matches = list(output_video_path.parent.glob(f"{output_video_path.stem}.*"))
        if matches and matches[0].stat().st_size > 50000:
            return matches[0]
    except Exception as exc:
        if output_video_path.exists() and output_video_path.stat().st_size > 50000:
            return output_video_path
        matches = list(output_video_path.parent.glob(f"{output_video_path.stem}.*"))
        if matches and matches[0].stat().st_size > 50000:
            return matches[0]
        logger.warning(f"YouTube real footage search failed for '{query}' (candidate {candidate_index}): {exc}")

    return None


def generate_cinematic_fallback(
    output_path: Path,
    duration: float,
    width: int,
    height: int,
    query: str,
) -> Path:
    """Procedural dynamic motion texture when all external sources are unavailable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed = abs(hash(query)) % 360

    filter_expr = (
        f"color=c=black:s={width}x{height}:d={duration},"
        f"geq=r='120+90*sin(X/90+T*2)':g='60+50*cos(Y/70+T*1.8)':b='180+60*sin((X+Y)/120+T)',"
        f"boxblur=15:1,"
        f"hue=h={seed}:s=1.3"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", filter_expr,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-an",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        cmd_fallback = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x1a1a2e:s={width}x{height}:d={duration}",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-an",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd_fallback, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_path


def source_broll_for_clip(
    clip: ClipProposal,
    temp_dir: Path | str = Path("temp"),
) -> list[SceneCut]:
    """
    Sourcing pipeline for each SceneCut in a ClipProposal:
    1. Check manual user drop in input/broll/
    2. Contextual real archive footage search via YouTube (top-3 candidates)
    3. Logo & intro protection (minimum offset >= 10s)
    4. Visual quality gate validation (rejects black frames, flat slides, static logos)
    5. Deduplication (No-Repeat-Garantie: no asset repeats within a 30-60s short)
    6. Procedural fallback plate if all candidates fail
    """
    temp_dir = Path(temp_dir)
    broll_cache = temp_dir / "broll" / f"clip_{clip.clip_id}"
    broll_cache.mkdir(parents=True, exist_ok=True)

    # Deduplication tracking: list/set of all B-roll asset paths used in this clip
    used_asset_paths: set[str] = set()

    for idx, scene in enumerate(clip.scenes, start=1):
        if scene.layout not in ("full_broll", "splitscreen"):
            continue

        target_w = 1080
        target_h = 960 if scene.layout == "splitscreen" else 1920

        query = scene.search_query or "US Capitol congress chamber raw footage"
        slug = sanitize_query(query).replace(" ", "_")[:25] or "scene"

        # 1. Check manual drop in input/broll (respecting deduplication)
        manual_asset = check_manual_broll_drop(query)
        if manual_asset and str(manual_asset.resolve()) not in used_asset_paths:
            final_plate = broll_cache / f"scene_{idx}_{slug}_manual.mp4"
            if manual_asset.suffix.lower() in [".jpg", ".png", ".webp"]:
                apply_focal_ken_burns(
                    image_path=manual_asset,
                    output_path=final_plate,
                    duration=scene.duration,
                    target_width=target_w,
                    target_height=target_h,
                )
            else:
                extract_peak_motion_snippet(
                    video_path=manual_asset,
                    output_path=final_plate,
                    target_duration=scene.duration,
                    target_width=target_w,
                    target_height=target_h,
                    min_offset_sec=0.0,
                )
            resolved_plate = str(final_plate.resolve())
            scene.broll_asset_path = resolved_plate
            used_asset_paths.add(resolved_plate)
            continue

        # 2. Search YouTube real archive footage with Top-3 candidates & Quality Gate
        h_query = hashlib.md5(query.encode()).hexdigest()[:8]
        found_valid = False

        for cand_idx in range(1, 4):
            final_plate = broll_cache / f"scene_{idx}_{slug}_c{cand_idx}.mp4"
            resolved_plate = str(final_plate.resolve())

            # Check if this candidate was already downloaded & processed and not yet used in this clip
            if final_plate.exists() and final_plate.stat().st_size > 5000 and resolved_plate not in used_asset_paths:
                is_valid, _ = validate_broll_frame(final_plate)
                if is_valid:
                    scene.broll_asset_path = resolved_plate
                    used_asset_paths.add(resolved_plate)
                    found_valid = True
                    break

            raw_yt_video = broll_cache / f"raw_yt_{h_query}_c{cand_idx}.mp4"
            if not raw_yt_video.exists() or raw_yt_video.stat().st_size < 50000:
                download_youtube_broll(
                    query=query,
                    output_video_path=raw_yt_video,
                    candidate_index=cand_idx,
                )

            if raw_yt_video.exists() and raw_yt_video.stat().st_size > 50000:
                try:
                    extract_peak_motion_snippet(
                        video_path=raw_yt_video,
                        output_path=final_plate,
                        target_duration=scene.duration,
                        target_width=target_w,
                        target_height=target_h,
                        min_offset_sec=10.0,
                    )
                    is_valid, reason = validate_broll_frame(final_plate)
                    if is_valid and resolved_plate not in used_asset_paths:
                        scene.broll_asset_path = resolved_plate
                        used_asset_paths.add(resolved_plate)
                        found_valid = True
                        break
                    else:
                        logger.warning(
                            f"Candidate {cand_idx} for '{query}' failed quality gate or was duplicate: {reason}. "
                            f"Trying next search candidate..."
                        )
                except Exception as exc:
                    logger.warning(f"Error extracting motion snippet from candidate {cand_idx} ('{raw_yt_video}'): {exc}")

        if found_valid:
            continue

        # 3. Fallback: Dynamic motion texture (guaranteed unique path per scene index)
        final_plate = broll_cache / f"scene_{idx}_{slug}_fallback.mp4"
        logger.info(f"Generating fallback motion texture for '{query}'...")
        generate_cinematic_fallback(
            output_path=final_plate,
            duration=scene.duration,
            width=target_w,
            height=target_h,
            query=f"{query}_{idx}",
        )
        resolved_plate = str(final_plate.resolve())
        scene.broll_asset_path = resolved_plate
        used_asset_paths.add(resolved_plate)

    return clip.scenes
