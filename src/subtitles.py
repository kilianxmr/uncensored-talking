"""
Pillow-based Subtitle Generator for Uncensored Talking (@uncensored_talking).
Renders subtitle chunks as transparent 1080x1920 PNG overlays:
- Font: The Bold Font / Impact / Anton (ALL CAPS, no punctuation).
- Vertical linear gradient: Yellow (#FFDD00, top) -> Warm-Orange (#FF7700, bottom).
- Outline: Bold opaque black outer stroke (10-14px).
- Placement: Exactly screen-centered (X: 540, Y: 960).
- Chunking: 2 words standard, 1 word for >8 chars, max 3 words for short words.
  Never cross sentence boundaries.
- Hold duration: 0.5s - 0.8s during pauses, immediate switch on next word.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .selector import ClipProposal, Segment

logger = logging.getLogger("uncensored_talking.subtitles")

# Search paths for bold typography (Excon-Black/Bold preferred, Outfit-Black, Arial Black fallback)
FONT_CANDIDATES = [
    Path("assets/fonts/Excon-Black.otf"),
    Path("assets/fonts/Excon-Black.ttf"),
    Path("assets/fonts/Excon-Bold.otf"),
    Path("assets/fonts/Excon-Bold.ttf"),
    Path("assets/fonts/Outfit-Black.ttf"),
    Path("assets/fonts/Satoshi-Black.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"),
    Path("assets/fonts/TheBoldFont.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf"),
]


def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds into ASS timestamp format: H:MM:SS.cs."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis -= 100
        if secs >= 60:
            minutes += 1
            secs -= 60
            if minutes >= 60:
                hours += 1
                minutes -= 60
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def get_subtitle_font(size: int = 63) -> ImageFont.FreeTypeFont:
    """Finds and loads the best available bold typography font (Excon-Black preferred)."""
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                continue

    # Fallback default
    try:
        return ImageFont.load_default()
    except Exception:
        raise RuntimeError("No suitable font could be loaded for subtitle rendering.")


def clean_subtitle_text(text: str) -> str:
    """
    Strips all punctuation marks (. , ! ? " ' : ; -) and forces uppercase.
    Per specification: Nur Großbuchstaben, keine Satzzeichen (. , ! ?).
    """
    cleaned = re.sub(r"[.,!?:;\"'()\[\]\-_/\\><]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    return cleaned


def is_sentence_terminator(word_raw: str) -> bool:
    """Checks if a word raw text contained sentence-ending punctuation."""
    return any(p in word_raw for p in [".", "!", "?", "\n"])


def map_words_to_clip_timeline(
    words: list[dict[str, Any]],
    segments: list[Segment],
) -> list[dict[str, Any]]:
    """
    Extracts words falling within the clip's segments and maps their timestamps
    from source video timeline to local clip timeline starting at 0.0s.
    Preserves raw word punctuation to detect sentence boundaries.
    """
    local_words: list[dict[str, Any]] = []
    cumulative_offset = 0.0

    for seg in segments:
        seg_duration = seg.end - seg.start
        for w in words:
            if w["start"] >= seg.start and w["end"] <= seg.end + 0.15:
                local_start = cumulative_offset + max(0.0, w["start"] - seg.start)
                local_end = cumulative_offset + min(seg_duration, w["end"] - seg.start)
                if local_end > local_start:
                    raw_w = w["word"].strip()
                    cleaned_w = clean_subtitle_text(raw_w)
                    if cleaned_w:
                        local_words.append(
                            {
                                "word": cleaned_w,
                                "raw_word": raw_w,
                                "is_sentence_end": is_sentence_terminator(raw_w),
                                "start": round(local_start, 3),
                                "end": round(local_end, 3),
                            }
                        )
        cumulative_offset += seg_duration

    local_words.sort(key=lambda x: x["start"])
    return local_words


def chunk_words_gradient(local_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Groups words according to exact chunking specifications:
    - Standard: 2 words.
    - Long words (>8 chars): 1 word.
    - Short words (all <= 4 chars): max 3 words.
    - NEVER cross sentence boundaries (ends immediately on sentence end).
    - Speech pauses (> 0.25s) trigger chunk split.
    """
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush():
        nonlocal current
        if not current:
            return
        text = " ".join(w["word"] for w in current)
        chunks.append({
            "text": text,
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "ends_sentence": current[-1]["is_sentence_end"],
        })
        current = []

    for w in local_words:
        word_len = len(w["word"])
        is_long = word_len > 8

        if current:
            last = current[-1]
            gap = w["start"] - last["end"]

            # If previous word ended a sentence, flush immediately
            if last["is_sentence_end"]:
                flush()
            elif gap > 0.25 or is_long or len(current) >= 3:
                flush()
            elif len(current) == 2 and not all(len(x["word"]) <= 4 for x in current):
                # Standard is 2 words unless all words are short
                flush()

        current.append(w)

        # Long words stay alone
        if is_long:
            flush()

    flush()
    return chunks


# Alias for backward compatibility
chunk_words_for_subtitles = chunk_words_gradient


def apply_hold_duration(
    chunks: list[dict[str, Any]],
    clip_total_duration: float,
    min_hold: float = 0.5,
    max_hold: float = 0.8,
) -> list[dict[str, Any]]:
    """
    Hold logic: During pauses, chunk remains visible for 0.5s to 0.8s.
    Switches immediately when subsequent word arrives.
    """
    timed_chunks: list[dict[str, Any]] = []
    num_chunks = len(chunks)

    for i, c in enumerate(chunks):
        c_start = c["start"]
        c_end = c["end"]

        if i < num_chunks - 1:
            next_start = chunks[i + 1]["start"]
            gap = next_start - c_end

            if gap <= 0:
                effective_end = min(c_end, next_start)
            elif gap <= max_hold:
                effective_end = next_start
            else:
                effective_end = c_end + max_hold
        else:
            effective_end = min(clip_total_duration, c_end + max_hold)

        if effective_end <= c_start:
            effective_end = c_start + 0.3

        timed_chunks.append({
            "text": c["text"],
            "start": c_start,
            "end": round(effective_end, 3),
        })

    return timed_chunks


def render_gradient_subtitle_png(
    text: str,
    output_png_path: Path,
    width: int = 1080,
    height: int = 1920,
    font_size: int = 63,
    stroke_width: int = 8,
    top_color: tuple[int, int, int] = (255, 221, 0),  # #FFDD00 Yellow
    bot_color: tuple[int, int, int] = (255, 119, 0),  # #FF7700 Warm-Orange
) -> Path:
    """
    Renders a single subtitle chunk as a transparent 1080x1920 PNG with:
    - Font: Excon-Black (UPPERCASE, no punctuation).
    - Compact sizing (reduced ~10%: 63px) and clean 7-8px black outer contour.
    - Vertical linear gradient (Yellow -> Warm-Orange).
    - Exact center alignment (X: 540, Y: 960).
    """
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    font = get_subtitle_font(font_size)

    # 1. Calculate text bounding box
    dummy_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_dummy = ImageDraw.Draw(dummy_img)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Center placement
    tx = (width - tw) // 2 - bbox[0]
    ty = (height - th) // 2 - bbox[1]

    # 2. Black stroke silhouette (thick outward outline)
    stroke_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    stroke_draw = ImageDraw.Draw(stroke_img)
    stroke_draw.text(
        (tx, ty),
        text,
        font=font,
        fill=(0, 0, 0, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )

    # 3. Text mask (un-stroked interior)
    mask_img = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.text((tx, ty), text, font=font, fill=255)

    # 4. Vertical linear gradient
    text_top_y = ty + bbox[1]
    text_bot_y = text_top_y + max(1, th)

    top_c = np.array(top_color, dtype=float)
    bot_c = np.array(bot_color, dtype=float)

    grad_arr = np.zeros((height, width, 4), dtype=np.uint8)
    y_coords = np.arange(height)
    t_vals = np.clip((y_coords - text_top_y) / max(1.0, float(th)), 0.0, 1.0)

    for c in range(3):
        grad_arr[:, :, c] = (top_c[c] + t_vals[:, None] * (bot_c[c] - top_c[c])).astype(np.uint8)
    grad_arr[:, :, 3] = 255

    grad_img = Image.fromarray(grad_arr, "RGBA")

    # Composite inner gradient over expanded stroke
    inner_text = Image.composite(grad_img, Image.new("RGBA", (width, height), (0, 0, 0, 0)), mask_img)
    final_img = Image.alpha_composite(stroke_img, inner_text)

    final_img.save(str(output_png_path), "PNG")
    return output_png_path


def generate_png_subtitles(
    clip: ClipProposal,
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
    output_dir: Path | str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """
    Generates all PNG subtitle overlays and an FFmpeg concat instruction file.

    Returns:
        tuple of (Path to subtitles_concat.txt, list of timed chunks)
    """
    word_timestamps_file = Path(word_timestamps_file)
    if not word_timestamps_file.exists():
        raise FileNotFoundError(f"Word timestamps file not found: {word_timestamps_file}")

    if output_dir is None:
        output_dir = Path(f"temp/subtitles/clip_{clip.clip_id}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(word_timestamps_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    # 1. Map to clip timeline & chunk
    local_words = map_words_to_clip_timeline(words, clip.segments)
    raw_chunks = chunk_words_gradient(local_words)
    timed_chunks = apply_hold_duration(
        raw_chunks,
        clip_total_duration=clip.total_duration,
        min_hold=0.5,
        max_hold=0.8,
    )

    # Blank transparent slide for gaps
    blank_png = output_dir / "blank.png"
    if not blank_png.exists():
        Image.new("RGBA", (1080, 1920), (0, 0, 0, 0)).save(blank_png, "PNG")

    concat_entries: list[str] = []
    current_time = 0.0

    for idx, c in enumerate(timed_chunks, start=1):
        c_start = c["start"]
        c_end = c["end"]
        duration = max(0.04, c_end - c_start)

        # Gap before this chunk?
        if c_start > current_time + 0.02:
            gap_dur = c_start - current_time
            concat_entries.append(f"file '{blank_png.resolve().as_posix()}'\nduration {gap_dur:.3f}")

        # Render subtitle chunk PNG
        png_path = output_dir / f"sub_{idx:03d}.png"
        render_gradient_subtitle_png(c["text"], png_path)

        concat_entries.append(f"file '{png_path.resolve().as_posix()}'\nduration {duration:.3f}")
        current_time = c_end

    # Remaining clip tail gap
    if current_time < clip.total_duration:
        tail_dur = clip.total_duration - current_time
        concat_entries.append(f"file '{blank_png.resolve().as_posix()}'\nduration {tail_dur:.3f}")

    # Concat file requirement: last file listed twice
    if concat_entries:
        last_file_line = concat_entries[-1].split("\n")[0]
        concat_entries.append(last_file_line)

    concat_file_path = output_dir / "subtitles_concat.txt"
    with open(concat_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_entries) + "\n")

    logger.info(
        f"Generated {len(timed_chunks)} PNG gradient subtitles and concat list for clip {clip.clip_id}: {concat_file_path}"
    )
    return concat_file_path, timed_chunks


def generate_ass_subtitles(
    clip: ClipProposal,
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
    output_ass_path: Path | str | None = None,
    **kwargs,
) -> Path:
    """Wrapper ensuring backward compatibility: writes ASS file and generates PNGs."""
    concat_txt, timed_chunks = generate_png_subtitles(clip, word_timestamps_file=word_timestamps_file)

    if output_ass_path is None:
        output_ass_path = Path(f"temp/clip_{clip.clip_id}.ass")
    output_ass_path = Path(output_ass_path)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,The Bold Font,84,&H002BF7FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,4.0,2.0,5,30,30,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for c in timed_chunks:
        start_str = format_ass_timestamp(c["start"])
        end_str = format_ass_timestamp(c["end"])
        lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{c['text']}")

    output_ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_ass_path
