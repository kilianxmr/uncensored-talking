"""
Topic selection and visual director module for Uncensored Talking (@uncensored_talking).
Analyzes transcripts for controversies, taboos, and emotional peaks.
Extracts 30-60s clips where the first 3 seconds serve as the viral hook.
Creates a visual direction plan (SceneCuts every 2-5s) with layouts:
'full_speaker', 'full_broll', and 'splitscreen'.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logger = logging.getLogger("uncensored_talking.selector")

LayoutType = Literal["full_speaker", "full_broll", "splitscreen"]


class Segment(BaseModel):
    start: float = Field(..., description="Segment start timestamp in seconds")
    end: float = Field(..., description="Segment end timestamp in seconds")

    @field_validator("end")
    @classmethod
    def check_duration(cls, v: float, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"Segment end ({v}) must be greater than start ({start})")
        return round(v, 2)

    @field_validator("start")
    @classmethod
    def round_start(cls, v: float):
        return round(v, 2)


class SceneCut(BaseModel):
    start: float = Field(..., description="Local clip start time in seconds (0.0 <= start < end)")
    end: float = Field(..., description="Local clip end time in seconds")
    layout: LayoutType = Field(
        default="full_speaker",
        description="Layout mode: 'full_speaker' (1080x1920), 'full_broll' (1080x1920), or 'splitscreen' (top B-Roll, bottom speaker)",
    )
    search_query: str = Field(
        default="",
        description="Abstract, visual English search terms for B-roll sourcing (e.g. 'helicopter military extraction')",
    )
    broll_asset_path: str | None = Field(
        default=None,
        description="Local path to downloaded/processed B-Roll asset (video or Ken-Burns image)",
    )

    @field_validator("end")
    @classmethod
    def check_scene_duration(cls, v: float, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"Scene end ({v}) must be greater than start ({start})")
        return round(v, 2)

    @property
    def duration(self) -> float:
        return self.end - self.start


class ClipProposal(BaseModel):
    clip_id: int
    title: str = Field(..., description="Short slug / title for the viral clip")
    hook: str = Field(default="", description="The explosive hook statement used in the first 3s")
    rationale: str = Field(default="", description="Why this clip is controversial / viral")
    segments: list[Segment] = Field(
        ...,
        description="List of video segments to cut from original source (total 30-60s)",
    )
    scenes: list[SceneCut] = Field(
        default_factory=list,
        description="Timeline of visual direction scene cuts every 2-5s with layout modes and search queries",
    )

    @property
    def total_duration(self) -> float:
        return sum(s.end - s.start for s in self.segments)

    def to_output_dict(self) -> dict[str, Any]:
        """Returns the format expected by the specification."""
        return {
            "clip_id": self.clip_id,
            "title": self.title,
            "hook": self.hook,
            "rationale": self.rationale,
            "total_duration": round(self.total_duration, 2),
            "segments": [{"start": s.start, "end": s.end} for s in self.segments],
            "scenes": [
                {
                    "start": sc.start,
                    "end": sc.end,
                    "layout": sc.layout,
                    "search_query": sc.search_query,
                    "broll_asset_path": sc.broll_asset_path,
                }
                for sc in self.scenes
            ],
        }


UNCENSORED_SYSTEM_PROMPT = """You are the Lead Visual Director and Social Media Strategist for the viral channel "@uncensored_talking".
Your mission:
1. Identify the TOP most viral, controversial, taboo-breaking debate moments (duration 30-60 seconds).
2. The FIRST 3 SECONDS MUST be an irresistible, pattern-interrupting hook.
3. VISUAL REGIE & B-ROLL TIMELINE: For each clip, create a dynamic visual timeline with scene cuts every 2 to 5 seconds.
   - layout must be one of:
     * "full_speaker": 1080x1920 speaker full frame (face centered).
     * "full_broll": 1080x1920 high impact B-roll footage/image.
     * "splitscreen": Top half B-roll (1080x960), bottom half speaker (1080x960).
   - search_query: Abstract, cinematic English search terms matching the subtopic/metaphor (e.g. "helicopter military extraction", "dry aged meat cellar", "laboratory microbes zoom", "money vault gold bars", "deep space galaxy", "police interrogation shadows"). DO NOT use literal names of podcast hosts; use visceral, sensory visual queries.
   - Timeline starts at 0.0s and ends at total clip duration without gaps.

OUTPUT FORMAT:
Return ONLY a valid JSON list of clip objects matching this schema:
[
  {
    "clip_id": 1,
    "title": "slug_format_title",
    "hook": "Transcript excerpt of the opening 3 seconds",
    "rationale": "Why this debate triggers outrage or curiosity",
    "segments": [
      {"start": 12.5, "end": 48.0}
    ],
    "scenes": [
      {
        "start": 0.0,
        "end": 3.0,
        "layout": "splitscreen",
        "search_query": "secret government documents burning"
      },
      {
        "start": 3.0,
        "end": 6.5,
        "layout": "full_speaker",
        "search_query": ""
      },
      {
        "start": 6.5,
        "end": 10.0,
        "layout": "full_broll",
        "search_query": "matrix code computer green screen"
      }
    ]
  }
]
"""


def _load_transcript_data(
    transcript_file: Path | str,
    word_timestamps_file: Path | str,
) -> tuple[str, list[dict[str, Any]]]:
    """Loads plain transcript and word timestamps."""
    transcript_file = Path(transcript_file)
    word_timestamps_file = Path(word_timestamps_file)

    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
    if not word_timestamps_file.exists():
        raise FileNotFoundError(f"Word timestamps file not found: {word_timestamps_file}")

    with open(transcript_file, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    with open(word_timestamps_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    return full_text, words


def generate_heuristic_scenes(
    clip_duration: float,
    clip_words: list[dict[str, Any]],
) -> list[SceneCut]:
    """
    Creates a dynamic visual director plan for a clip (scene cuts every 2-5 seconds).
    Alternates between 'splitscreen', 'full_speaker', and 'full_broll'.
    Derives abstract visual English search queries from keywords in each time slice.
    """
    scenes: list[SceneCut] = []
    current_time = 0.0

    # Abstract visual B-roll topics mapped from keywords
    TOPIC_BROLL_MAP = [
        (["secret", "cia", "fbi", "government", "staat", "geheim"], "secret government vault files"),
        (["money", "billion", "rich", "dollar", "wealth", "geld"], "money cash stack luxury vault"),
        (["lie", "truth", "fake", "media", "wahrheit", "lüge"], "television studio dark broadcasting"),
        (["matrix", "ai", "robot", "tech", "code"], "cyberpunk digital matrix computer screen"),
        (["space", "alien", "ufo", "universe", "mars"], "deep space nebula galaxy stars"),
        (["food", "meat", "health", "poison", "toxic"], "laboratory microscope cells chemical"),
        (["war", "army", "military", "weapon", "krieg"], "military drone surveillance tactical"),
        (["death", "danger", "destroy", "fear", "angst"], "dark storm lightning dramatic clouds"),
        (["mind", "brain", "think", "insane", "crazy"], "human brain neural network glowing"),
    ]

    # Pre-select pool of high-impact queries
    FALLBACK_QUERIES = [
        "cinematic smoke dark spotlight",
        "money vault gold bullion cinematic",
        "military helicopter night flight",
        "cyberpunk neon city aerial",
        "laboratory scientific research microscope",
        "ancient mystery temple dust beams",
        "police siren lights dark street",
    ]

    # Sequence of layouts to cycle through for maximum visual pacing
    # Opening 3 seconds: splitscreen (Hook + visual reinforcement)
    layout_cycle: list[LayoutType] = ["splitscreen", "full_speaker", "full_broll", "splitscreen", "full_speaker", "full_broll"]
    cycle_idx = 0

    while current_time < clip_duration - 0.5:
        # Determine duration of this scene cut (between 2.5s and 4.0s)
        if current_time == 0.0:
            scene_dur = min(3.0, clip_duration)
        else:
            remaining = clip_duration - current_time
            if remaining <= 4.5:
                scene_dur = remaining
            else:
                scene_dur = min(3.2, remaining)

        scene_end = round(min(clip_duration, current_time + scene_dur), 2)
        layout = layout_cycle[cycle_idx % len(layout_cycle)]
        cycle_idx += 1

        # Determine visual query for B-roll scenes
        search_query = ""
        if layout in ("full_broll", "splitscreen"):
            # Inspect words in this time slice
            slice_words = [
                w["word"].lower()
                for w in clip_words
                if current_time <= w.get("start", 0) <= scene_end
            ]
            slice_text = " ".join(slice_words)

            matched_query = ""
            for keywords, query in TOPIC_BROLL_MAP:
                if any(kw in slice_text for kw in keywords):
                    matched_query = query
                    break

            if not matched_query:
                matched_query = FALLBACK_QUERIES[cycle_idx % len(FALLBACK_QUERIES)]

            search_query = matched_query

        scenes.append(
            SceneCut(
                start=round(current_time, 2),
                end=scene_end,
                layout=layout,
                search_query=search_query,
            )
        )
        current_time = scene_end

    return scenes


def select_clips_with_llm(
    full_text: str,
    words: list[dict[str, Any]],
    num_clips: int = 3,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> list[ClipProposal]:
    """
    Calls OpenAI / OpenAI-compatible API to select viral clips and build B-roll direction timelines.
    """
    try:
        from openai import OpenAI
    except ImportError as err:
        raise ImportError("openai package required for LLM selector. Run pip install openai.") from err

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OPENAI_API_KEY provided or found in environment.")

    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    model_name = model or os.getenv("LLM_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Format transcript with periodic timestamps for LLM reference
    formatted_chunks: list[str] = []
    chunk_words: list[str] = []
    chunk_start: float | None = None
    last_end: float = 0.0

    for w in words:
        if chunk_start is None:
            chunk_start = w["start"]
        chunk_words.append(w["word"])
        last_end = w["end"]

        if last_end - chunk_start >= 12.0:
            formatted_chunks.append(
                f"[{chunk_start:.1f}s - {last_end:.1f}s] {' '.join(chunk_words)}"
            )
            chunk_words = []
            chunk_start = None

    if chunk_words and chunk_start is not None:
        formatted_chunks.append(
            f"[{chunk_start:.1f}s - {last_end:.1f}s] {' '.join(chunk_words)}"
        )

    context_transcript = "\n".join(formatted_chunks)
    if len(context_transcript) > 100000:
        context_transcript = context_transcript[:100000] + "\n...[Transcript truncated]"

    user_prompt = f"""Analyze the podcast transcript and extract the TOP {num_clips} most viral, controversial clips for @uncensored_talking.

Remember:
- Total clip length must be between 30 and 60 seconds.
- The first 3 seconds MUST be the ultimate hook.
- Create a visual director plan ('scenes') with scene cuts every 2-5 seconds.
- Use layouts: 'full_speaker', 'full_broll', 'splitscreen'.
- Provide abstract visual English search queries for B-roll (e.g. 'helicopter military extraction', 'laboratory microbes zoom').

TIMESTAMPED TRANSCRIPT:
{context_transcript}
"""

    logger.info(f"Sending transcript to LLM ({model_name}) for viral clip curation & B-roll direction...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": UNCENSORED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    logger.debug(f"LLM Raw Response: {content}")

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for key in ["clips", "proposals", "results", "data"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                if "segments" in data:
                    data = [data]

        proposals: list[ClipProposal] = []
        for idx, item in enumerate(data, start=1):
            if "clip_id" not in item:
                item["clip_id"] = idx
            if "title" not in item:
                item["title"] = f"clip_{idx}"

            # Convert scenes if present
            raw_scenes = item.get("scenes", [])
            item["scenes"] = []
            prop = ClipProposal(**item)

            # Validate or populate scenes
            dur = prop.total_duration
            if raw_scenes:
                validated_scenes: list[SceneCut] = []
                for sc in raw_scenes:
                    try:
                        validated_scenes.append(SceneCut(**sc))
                    except Exception:
                        pass
                prop.scenes = validated_scenes

            # If scenes are missing or invalid, generate them via heuristic planner
            if not prop.scenes:
                clip_words = [
                    w for w in words
                    if any(s.start <= w["start"] <= s.end for s in prop.segments)
                ]
                prop.scenes = generate_heuristic_scenes(dur, clip_words)

            proposals.append(prop)

        logger.info(f"LLM successfully generated {len(proposals)} viral clip proposals with visual direction.")
        return proposals
    except Exception as exc:
        logger.error(f"Failed to parse LLM response: {exc}. Response was:\n{content}")
        raise


def select_clips_heuristic(
    words: list[dict[str, Any]],
    num_clips: int = 3,
    target_duration: float = 45.0,
) -> list[ClipProposal]:
    """
    Advanced rule-based heuristic for offline use or when LLM API is unavailable.
    Scans for controversy/taboo keywords, emotional peaks, and generates
    the complete 2-5s visual scene direction plan with dynamic layout modes.
    """
    logger.info("Running heuristic viral clip selector & visual director (offline mode)...")

    if not words:
        raise ValueError("Word timestamps list is empty.")

    total_audio_duration = words[-1]["end"]
    if total_audio_duration < 30.0:
        dur = total_audio_duration
        scenes = generate_heuristic_scenes(dur, words)
        return [
            ClipProposal(
                clip_id=1,
                title="full_take",
                hook=" ".join(w["word"] for w in words[:10]),
                rationale="Short audio snippet taken in full.",
                segments=[Segment(start=words[0]["start"], end=dur)],
                scenes=scenes,
            )
        ]

    # Keyword bank for @uncensored_talking
    CONTROVERSY_KEYWORDS = {
        "secret", "lie", "lying", "truth", "conspiracy", "matrix", "censored", "illegal",
        "insane", "crazy", "corrupt", "government", "money", "fake", "destroy", "kill",
        "scandal", "exposed", "dangerous", "never", "always", "shocking", "forbidden",
        "alien", "ai", "cia", "fbi", "control", "power", "war", "death", "god", "hell",
        "fear", "manipulated", "silenced", "trap", "elite", "propaganda", "hoax", "hate",
        "lüge", "wahrheit", "zensiert", "skandal", "geheim", "krass", "verrückt", "elite",
        "verschwörung", "staat", "medien", "kontrolle", "gefährlich", "geld", "macht",
        "verboten", "manipuliert", "vernichtet", "schockiert", "angst", "hölle", "zerstört",
    }

    word_scores: list[float] = []
    for w in words:
        clean = re.sub(r"[^\w\s]", "", w["word"].lower())
        score = 1.0
        if clean in CONTROVERSY_KEYWORDS:
            score += 6.0
        if "?" in w["word"] or "!" in w["word"]:
            score += 2.5
        if w["word"].isupper() and len(w["word"]) > 2:
            score += 2.0
        word_scores.append(score)

    candidate_windows: list[dict[str, Any]] = []
    n = len(words)

    for i in range(n):
        start_t = words[i]["start"]
        for j in range(i + 10, n):
            end_t = words[j]["end"]
            dur = end_t - start_t
            if dur > 58.0:
                break
            if 35.0 <= dur <= 55.0:
                score = sum(word_scores[i : j + 1]) / dur
                candidate_windows.append(
                    {
                        "start_idx": i,
                        "end_idx": j,
                        "start": start_t,
                        "end": end_t,
                        "duration": dur,
                        "score": score,
                    }
                )

    candidate_windows.sort(key=lambda x: x["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    for cand in candidate_windows:
        overlap = False
        for sel in selected:
            if not (cand["end"] < sel["start"] or cand["start"] > sel["end"]):
                overlap = True
                break
        if not overlap:
            selected.append(cand)
            if len(selected) >= num_clips:
                break

    if not selected:
        step = max(30.0, total_audio_duration / max(1, num_clips))
        for k in range(min(num_clips, int(total_audio_duration // 30))):
            s = k * step
            e = min(s + 45.0, total_audio_duration)
            if e - s >= 30.0:
                selected.append({
                    "start": s, "end": e, "duration": e - s, "score": 1.0,
                    "start_idx": 0, "end_idx": len(words)-1
                })

    proposals: list[ClipProposal] = []
    for idx, sel in enumerate(selected, start=1):
        clip_words = [w for w in words if sel["start"] <= w["start"] <= sel["end"]]
        hook_words = [w["word"] for w in clip_words if w["start"] - sel["start"] <= 3.2]
        hook_text = " ".join(hook_words) or "Watch this revelation!"
        slug = f"uncensored_clip_{idx}_{int(sel['start'])}s"

        # Generate 2-5s scene timeline with layout modes and search queries
        scenes = generate_heuristic_scenes(sel["duration"], clip_words)

        proposal = ClipProposal(
            clip_id=idx,
            title=slug,
            hook=hook_text,
            rationale=f"High controversy score ({sel['score']:.2f}) featuring intense discussion.",
            segments=[Segment(start=sel["start"], end=sel["end"])],
            scenes=scenes,
        )
        proposals.append(proposal)

    logger.info(f"Heuristic selector created {len(proposals)} viral clip proposals with visual scene directions.")
    return proposals


def select_viral_clips(
    transcript_file: Path | str = Path("temp/full_transcript.txt"),
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
    num_clips: int = 3,
    prefer_llm: bool = True,
    output_json_path: Path | str = Path("temp/selected_clips.json"),
) -> list[ClipProposal]:
    """
    Main entry point for clip selection & visual direction planning.
    Saves the result to output_json_path.
    """
    full_text, words = _load_transcript_data(transcript_file, word_timestamps_file)

    proposals: list[ClipProposal] = []
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if prefer_llm and has_api_key:
        try:
            proposals = select_clips_with_llm(
                full_text=full_text,
                words=words,
                num_clips=num_clips,
            )
        except Exception as exc:
            logger.warning(
                f"LLM clip selection failed ({exc}). Falling back to heuristic selector..."
            )
            proposals = select_clips_heuristic(words, num_clips=num_clips)
    else:
        if prefer_llm and not has_api_key:
            logger.info("No OPENAI_API_KEY found in environment. Using heuristic selector.")
        proposals = select_clips_heuristic(words, num_clips=num_clips)

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = [p.to_output_dict() for p in proposals]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(proposals)} selected clip definitions with visual timelines to: {output_json_path}")
    return proposals
