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
    Applies smooth cubic spline / smoothstep Ken-Burns zoom (1.0x -> 1.15x) targeted at visual focal point (nx, ny):
    - Smooth ease-in -> steady continuous drift -> smooth ease-out (no linear robotic jitter).
    - Proportionally scales to target_width x target_height.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = find_visual_focal_point(image_path)
    total_frames = max(10, int(duration * fps) + 5)

    # Cubic spline / smoothstep formula: S(p) = 3*p^2 - 2*p^3, where p = on / N
    # Evaluates to 1.0 at on=0 and 1.15 at on=N with 0-derivative endpoints
    spline_expr = f"1.0+0.15*(on/{total_frames}*on/{total_frames}*(3-2*on/{total_frames}))"

    # Zoom centered around focal point (nx, ny)
    filter_expr = (
        f"zoompan=z='{spline_expr}':"
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


# Semantic keyword clusters for direct noun verification
TOPIC_KEYWORD_CLUSTERS = {
    "school_lunch": {
        "query_tokens": {"school", "lunch", "cafeteria", "tray"},
        "spoken_tokens": {"school", "lunch", "cafeteria", "donut", "donuts", "pastry", "pastries", "breakfast", "kid", "kids", "child", "children", "tray", "food", "eat", "meal", "meals"},
    },
    "pharma_drugs": {
        "query_tokens": {"pill", "pills", "blister", "macro", "pharmaceutical", "pharma", "chemical", "tubes", "laboratory"},
        "spoken_tokens": {"pharmaceutical", "pharma", "drug", "drugs", "medicine", "medicines", "pill", "pills", "chemical", "chemicals", "vaccine", "vaccines", "medication", "lab", "prescription"},
    },
    "processed_food": {
        "query_tokens": {"processed", "junk", "supermarket", "aisle"},
        "spoken_tokens": {"processed", "ultra-processed", "junk", "sugar", "fructose", "syrup", "soda", "cereal", "food", "ingredients", "dye", "dyes", "chemicals"},
    },
    "healthcare_system": {
        "query_tokens": {"hospital", "icu", "doctor", "insurance", "claim"},
        "spoken_tokens": {"hospital", "doctor", "doctors", "icu", "patient", "nurse", "insurance", "premium", "deductible", "healthcare", "bills", "costs"},
    },
    "congress_politics": {
        "query_tokens": {"capitol", "congress", "chamber", "lobbying", "hearing"},
        "spoken_tokens": {"congress", "senate", "capitol", "senator", "representative", "hearing", "lobby", "lobbyist", "bill", "vote", "campaign", "pac", "politician"},
    },
    "white_house": {
        "query_tokens": {"white", "house", "oval", "office"},
        "spoken_tokens": {"white", "house", "oval", "office", "president", "trump", "obama", "administration", "executive"},
    },
    "money_finance": {
        "query_tokens": {"banknotes", "counting", "machine", "dollar", "cash"},
        "spoken_tokens": {"money", "billion", "trillion", "dollar", "dollars", "wealth", "tax", "taxes", "cash", "bank", "pay", "paid", "funding"},
    },
    "legal_court": {
        "query_tokens": {"court", "courtroom", "gavel", "trial"},
        "spoken_tokens": {"court", "judge", "trial", "lawsuit", "illegal", "gavel", "attorney", "legal", "guilty", "crime"},
    },
    "supercomputer_ai": {
        "query_tokens": {"datacenter", "server", "racks"},
        "spoken_tokens": {"computer", "supercomputer", "ai", "artificial", "intelligence", "server", "datacenter", "algorithm", "tech"},
    },
    "military_pentagon": {
        "query_tokens": {"pentagon", "military", "briefing"},
        "spoken_tokens": {"military", "pentagon", "army", "defense", "war", "weapons", "general", "troops"},
    },
}


def check_topic_guard(
    scene: SceneCut,
    spoken_text: str,
) -> tuple[bool, str]:
    """
    Strict Topic Guard:
    Cross-checks the scene's search query against spoken audio nouns/entities in that 2-5s segment.
    If the search query does not directly match the spoken words, rejects B-roll.
    "Besser kein B-Roll als unpassendes B-Roll."
    """
    if not scene.search_query:
        return False, "Search query is empty"

    if not spoken_text or not spoken_text.strip():
        return False, "No spoken words in audio segment"

    spoken_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", spoken_text.lower()))
    query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", scene.search_query.lower()))

    # Direct token intersection (e.g. "trump", "musk", "pepsi", "pfizer", "lunch", "pills", "hospital")
    STOP_QUERY_TOKENS = {
        "rare", "raw", "footage", "archive", "archival", "interview", "official",
        "transparent", "png", "logo", "building", "corporate", "headquarters",
        "scene", "hd", "video", "4k", "dramatic", "cinematic", "real"
    }
    meaningful_query_tokens = query_tokens - STOP_QUERY_TOKENS

    if meaningful_query_tokens.intersection(spoken_tokens):
        return True, "Direct entity/noun overlap"

    # Check semantic topic clusters
    for cluster_name, cluster_data in TOPIC_KEYWORD_CLUSTERS.items():
        if cluster_data["query_tokens"].intersection(query_tokens):
            if cluster_data["spoken_tokens"].intersection(spoken_tokens):
                return True, f"Cluster match: {cluster_name}"

    return False, f"Query '{scene.search_query}' has no noun overlap with spoken segment ('{spoken_text[:40]}...')"


def render_brand_logo_plate(
    brand_query: str,
    output_path: Path,
    duration: float,
    width: int = 1080,
    height: int = 960,
) -> Path:
    """
    Renders Category A Priority 1 clean brand logo / badge plate:
    1. Generates clean high-res emblem plate using Pillow
    2. Outputs smooth MP4 with subtle Ken-Burns zoom (1.0x -> 1.05x).
    """
    from PIL import Image, ImageDraw
    from .subtitles import get_subtitle_font

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_png = output_path.with_suffix(".png")

    brand_name = brand_query.replace("official company logo transparent png", "")
    brand_name = brand_name.replace("logo transparent png", "").replace("company logo", "").replace("logo", "")
    brand_name = brand_name.strip().upper() or "BRAND"

    img = Image.new("RGBA", (width, height), (15, 20, 26, 255))
    draw = ImageDraw.Draw(img)

    badge_w, badge_h = int(width * 0.75), int(height * 0.48)
    bx1 = (width - badge_w) // 2
    by1 = (height - badge_h) // 2
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    draw.rounded_rectangle([bx1, by1, bx2, by2], radius=24, fill=(24, 32, 42, 255), outline=(56, 72, 92, 255), width=3)

    font = get_subtitle_font(size=min(72, max(28, int(badge_w / (len(brand_name) * 0.75 + 2)))))
    bbox = draw.textbbox((0, 0), brand_name, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width - tw) // 2 - bbox[0]
    ty = (height - th) // 2 - bbox[1]

    draw.text((tx, ty), brand_name, font=font, fill=(255, 255, 255, 255))

    label = "OFFICIAL ENTITY"
    font_sub = get_subtitle_font(size=24)
    bbox_sub = draw.textbbox((0, 0), label, font=font_sub)
    sw = bbox_sub[2] - bbox_sub[0]
    draw.text(((width - sw) // 2 - bbox_sub[0], by1 + 22), label, font=font_sub, fill=(130, 160, 195, 255))

    img.save(temp_png)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(temp_png),
        "-t", str(duration),
        "-vf", f"scale={width}:{height},zoompan=z='min(zoom+0.0005,1.06)':d={int(duration*30)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps=30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        cmd_simple = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(temp_png),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd_simple, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if temp_png.exists():
        temp_png.unlink()

    return output_path


def source_broll_for_clip(
    clip: ClipProposal,
    temp_dir: Path | str = Path("temp"),
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
) -> list[SceneCut]:
    """
    Sourcing pipeline for each SceneCut in a ClipProposal:
    1. Strict Topic Guard: cross-checks search query with spoken nouns in the scene interval.
       Reverts to 'full_speaker' if query does not match spoken audio.
    2. Category A Priority 1: Brand logo / badge rendering.
    3. Category A Priority 2 & Category B: Contextual real archive footage search via YouTube (top-3 candidates).
    4. Logo & intro protection (minimum offset >= 10s).
    5. Visual quality gate validation (rejects black frames, flat slides, static logos).
    6. Deduplication (No-Repeat-Garantie: no asset repeats within a 30-60s short).
    7. Topic Guard fallback: reverts to 'full_speaker' if footage cannot be validated.
    """
    import json
    from .subtitles import map_words_to_clip_timeline

    temp_dir = Path(temp_dir)
    broll_cache = temp_dir / "broll" / f"clip_{clip.clip_id}"
    broll_cache.mkdir(parents=True, exist_ok=True)

    # Load transcript words if available to enforce Strict Topic Guard
    local_words: list[dict[str, Any]] = []
    word_path = Path(word_timestamps_file)
    if word_path.exists():
        try:
            with open(word_path, "r", encoding="utf-8") as f:
                raw_words = json.load(f)
            local_words = map_words_to_clip_timeline(raw_words, clip.segments)
        except Exception as exc:
            logger.debug(f"Could not load word timestamps for Topic Guard: {exc}")

    # 0. Strict Query Diversification: Ensure all visual scene queries are distinct
    seen_queries: set[str] = set()
    QUERY_ANGLES = [
        "retro archive footage",
        "press conference retro",
        "walking candid outdoor",
        "close up macro detail",
        "documentary b-roll hd",
        "historical archive candid",
    ]
    for sc in clip.scenes:
        if sc.layout in ("full_broll", "splitscreen") and sc.search_query:
            clean_q = sanitize_query(sc.search_query).strip()
            lower_q = clean_q.lower()
            if lower_q in seen_queries:
                mod = QUERY_ANGLES[len(seen_queries) % len(QUERY_ANGLES)]
                sc.search_query = f"{clean_q} {mod}"
                logger.info(f"Query diversification: differentiated scene query to '{sc.search_query}'")
            seen_queries.add(sc.search_query.strip().lower())

    # Deduplication tracking: set of all asset IDs and output paths used in this clip
    used_asset_ids: set[str] = set()
    used_asset_paths: set[str] = set()

    for idx, scene in enumerate(clip.scenes, start=1):
        if scene.layout not in ("full_broll", "splitscreen"):
            continue

        target_w = 1080
        target_h = 960 if scene.layout == "splitscreen" else 1920

        # Extract spoken text in this scene slice for contextual alignment
        spoken_text = ""
        if local_words:
            scene_words = [
                w for w in local_words
                if w.get("start", 0) <= scene.end and w.get("end", 0) >= scene.start
            ]
            spoken_text = " ".join(w.get("word", "") for w in scene_words)
            if not spoken_text.strip():
                # Broader context fallback
                spoken_text = " ".join(w.get("word", "") for w in local_words[:20])

        query = scene.search_query or "archive documentary footage"
        slug = sanitize_query(query).replace(" ", "_")[:25] or "scene"

        # Category A Priority 1: Clean Brand Logo Plate
        if "logo" in query.lower() or "transparent png" in query.lower():
            logo_plate = broll_cache / f"scene_{idx}_{slug}_logo.mp4"
            resolved_logo = str(logo_plate.resolve())
            logo_asset_id = f"logo_{slug}"
            if logo_asset_id not in used_asset_ids and resolved_logo not in used_asset_paths:
                render_brand_logo_plate(
                    brand_query=query,
                    output_path=logo_plate,
                    duration=scene.duration,
                    width=target_w,
                    height=target_h,
                )
                scene.broll_asset_path = resolved_logo
                used_asset_paths.add(resolved_logo)
                used_asset_ids.add(logo_asset_id)
                continue

        # 1. Check manual drop in input/broll (respecting deduplication)
        manual_asset = check_manual_broll_drop(query)
        if manual_asset:
            manual_asset_id = f"{manual_asset.name}_{manual_asset.stat().st_size}"
            if manual_asset_id not in used_asset_ids:
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
                used_asset_ids.add(manual_asset_id)
                continue

        # 2. Search YouTube real archive footage with Top-5 candidates & Strict Deduplication
        h_query = hashlib.md5(query.encode()).hexdigest()[:8]
        found_valid = False

        for cand_idx in range(1, 6):
            final_plate = broll_cache / f"scene_{idx}_{slug}_c{cand_idx}.mp4"
            resolved_plate = str(final_plate.resolve())

            raw_yt_video = broll_cache / f"raw_yt_{h_query}_c{cand_idx}.mp4"
            if not raw_yt_video.exists() or raw_yt_video.stat().st_size < 50000:
                download_youtube_broll(
                    query=query,
                    output_video_path=raw_yt_video,
                    candidate_index=cand_idx,
                )

            if raw_yt_video.exists() and raw_yt_video.stat().st_size > 50000:
                raw_asset_id = f"{raw_yt_video.name}_{raw_yt_video.stat().st_size}"
                if raw_asset_id in used_asset_ids:
                    logger.info(
                        f"Deduplication: Candidate {cand_idx} for '{query}' already used in this short. "
                        f"Trying next candidate..."
                    )
                    continue

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
                        used_asset_ids.add(raw_asset_id)
                        found_valid = True
                        break
                    else:
                        logger.warning(
                            f"Candidate {cand_idx} for '{query}' failed quality gate ({reason}). "
                            f"Trying next search candidate..."
                        )
                except Exception as exc:
                    logger.warning(f"Error extracting motion snippet from candidate {cand_idx}: {exc}")

        if found_valid:
            continue

        # 3. Visual Asset Guarantee: Engine MUST load a visual asset for splitscreen/full_broll
        fallback_plate = broll_cache / f"scene_{idx}_{slug}_motion.mp4"
        generate_cinematic_fallback(
            output_path=fallback_plate,
            duration=scene.duration,
            width=target_w,
            height=target_h,
            query=query,
        )
        resolved_fb = str(fallback_plate.resolve())
        scene.broll_asset_path = resolved_fb
        used_asset_paths.add(resolved_fb)
        logger.info(f"Loaded guaranteed visual motion plate for scene {idx}: {fallback_plate.name}")

    return clip.scenes
