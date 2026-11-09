"""
ASS (Advanced SubStation Alpha) subtitle generator for Uncensored Talking (@uncensored_talking).
Screen-center alignment (Alignment: 5) for viral shorts & splitscreen dividing lines.
Styling: Rich yellow/gold with thick black outline & shadow, 1-3 word chunking,
and 0.5-0.8s hold duration on speech pauses.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .selector import ClipProposal, Segment

logger = logging.getLogger("uncensored_talking.subtitles")


def format_ass_timestamp(seconds: float) -> str:
    """
    Formats seconds into ASS timestamp format: H:MM:SS.cs (centiseconds).
    Example: 65.432 -> '0:01:05.43'
    """
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


def get_ass_header(
    font_name: str = "Montserrat-Black",
    fallback_font: str = "Arial Black",
    font_size: int = 80,
    primary_color: str = "&H002BF7FF",  # Rich yellow/gold in ASS BGR format
    outline_color: str = "&H00000000",  # Pure black
    outline_width: float = 4.0,
    shadow_depth: float = 2.0,
    alignment: int = 5,  # 5 = Exact Center of screen (sits right on splitscreen dividing line)
) -> str:
    """
    Generates the ASS script header and styles tailored for 1080x1920 vertical shorts.
    """
    return f"""[Script Info]
; Uncensored Talking Subtitle Script (@uncensored_talking)
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,-1,0,0,0,100,100,1,0,1,{outline_width},{shadow_depth},{alignment},30,30,0,1
Style: Fallback,{fallback_font},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,-1,0,0,0,100,100,1,0,1,{outline_width},{shadow_depth},{alignment},30,30,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def map_words_to_clip_timeline(
    words: list[dict[str, Any]],
    segments: list[Segment],
) -> list[dict[str, Any]]:
    """
    Extracts words falling within the clip's segments and maps their timestamps
    from source video timeline to local clip timeline starting at 0.0s.
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
                    local_words.append(
                        {
                            "word": w["word"].strip(),
                            "start": local_start,
                            "end": local_end,
                        }
                    )
        cumulative_offset += seg_duration

    local_words.sort(key=lambda x: x["start"])
    return local_words


def chunk_words_for_subtitles(
    local_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Groups words into punchy 1 to 3 word chunks according to rules:
    - Maximum 3 words per chunk.
    - Long words (>= 8 chars) get their own 1-word chunk.
    - Pauses (> 0.25s) trigger a chunk split.
    """
    chunks: list[dict[str, Any]] = []
    current_chunk: list[dict[str, Any]] = []

    def flush_chunk():
        nonlocal current_chunk
        if not current_chunk:
            return
        text = " ".join(w["word"] for w in current_chunk).upper()
        start = current_chunk[0]["start"]
        end = current_chunk[-1]["end"]
        chunks.append({"text": text, "start": start, "end": end})
        current_chunk = []

    for w in local_words:
        word_text = w["word"].strip()
        if not word_text:
            continue

        if current_chunk:
            last_word = current_chunk[-1]
            gap = w["start"] - last_word["end"]

            is_long = len(word_text) >= 8
            prev_has_long = any(len(x["word"]) >= 8 for x in current_chunk)

            if gap > 0.25 or len(current_chunk) >= 3 or is_long or prev_has_long:
                flush_chunk()

        current_chunk.append(w)

        if len(word_text) >= 10:
            flush_chunk()

    flush_chunk()
    return chunks


def apply_hold_duration(
    chunks: list[dict[str, Any]],
    clip_total_duration: float,
    min_hold: float = 0.5,
    max_hold: float = 0.8,
) -> list[dict[str, Any]]:
    """
    Applies timing specification:
    - Chunks are shown as a whole block.
    - Hold duration: During speech pauses, chunk remains visible for 0.5s to 0.8s.
    - If the next chunk comes earlier, switch immediately without a gap.
    """
    timed_chunks: list[dict[str, Any]] = []
    num_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        c_start = chunk["start"]
        c_end = chunk["end"]

        if i < num_chunks - 1:
            next_start = chunks[i + 1]["start"]
            gap = next_start - c_end

            if gap <= 0:
                effective_end = min(c_end, next_start)
            elif gap <= max_hold:
                # Next chunk comes within 0.8s: switch immediately without gap
                effective_end = next_start
            else:
                # Silence > 0.8s: hold for max_hold then disappear
                effective_end = c_end + max_hold
        else:
            effective_end = min(clip_total_duration, c_end + max_hold)

        if effective_end <= c_start:
            effective_end = c_start + 0.3

        timed_chunks.append(
            {
                "text": chunk["text"],
                "start": c_start,
                "end": effective_end,
            }
        )

    return timed_chunks


def generate_ass_subtitles(
    clip: ClipProposal,
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
    output_ass_path: Path | str | None = None,
    font_name: str = "Montserrat-Black",
    fallback_font: str = "Arial Black",
    primary_color: str = "&H002BF7FF",  # Rich yellow/gold
    alignment: int = 5,  # Center of screen (splitscreen dividing line)
) -> Path:
    """
    Generates an ASS subtitle file for a given clip proposal.
    Aligned in the exact center (Alignment 5) for viral shorts & splitscreen layouts.
    """
    word_timestamps_file = Path(word_timestamps_file)
    if not word_timestamps_file.exists():
        raise FileNotFoundError(f"Word timestamps file not found: {word_timestamps_file}")

    if output_ass_path is None:
        output_ass_path = Path(f"temp/clip_{clip.clip_id}.ass")
    output_ass_path = Path(output_ass_path)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    with open(word_timestamps_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    local_words = map_words_to_clip_timeline(words, clip.segments)
    raw_chunks = chunk_words_for_subtitles(local_words)
    timed_chunks = apply_hold_duration(
        raw_chunks,
        clip_total_duration=clip.total_duration,
        min_hold=0.5,
        max_hold=0.8,
    )

    header = get_ass_header(
        font_name=font_name,
        fallback_font=fallback_font,
        font_size=82,
        primary_color=primary_color,
        outline_width=4.0,
        shadow_depth=2.0,
        alignment=alignment,
    )

    lines = [header]
    for chunk in timed_chunks:
        start_str = format_ass_timestamp(chunk["start"])
        end_str = format_ass_timestamp(chunk["end"])
        text = chunk["text"]
        lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(
        f"Generated ASS subtitles for clip {clip.clip_id} ({len(timed_chunks)} dialogue lines, Alignment={alignment}): {output_ass_path}"
    )
    return output_ass_path
