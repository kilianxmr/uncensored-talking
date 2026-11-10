"""
Topic selection, non-linear hook curation, and visual direction module for Uncensored Talking (@uncensored_talking).
Analyzes transcripts for controversies, taboos, and emotional peaks.
Enforces non-linear hooks:
- First 3-5s: The most controversial / shocking quote of the topic (even if it occurred at minute 15).
- Followed by hard cut to reaction/question (e.g. Joe Rogan: "Wait, are you serious?!").
- Followed by tight narrative payoff with pauses & filler removed.
- Total clip duration strictly 30 to 60 seconds.
- Scene cuts every 2-5s with 2-4 abstract visual English search queries.
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
        description="Layout mode: 'full_speaker' (1080x1920), 'full_broll' (1080x1920), or 'splitscreen' (top B-Roll 1080x960, bottom speaker 1080x960)",
    )
    search_query: str = Field(
        default="",
        description="2-4 abstract, visual English search terms for B-roll sourcing (e.g. 'military helicopter extraction night')",
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
    hook: str = Field(default="", description="The explosive non-linear hook statement used in the first 3-5s")
    rationale: str = Field(default="", description="Why this clip is controversial / viral")
    segments: list[Segment] = Field(
        ...,
        description="Non-linear sequence of segments (hook + reaction + payoff) to cut from original source (total 30-60s)",
    )
    scenes: list[SceneCut] = Field(
        default_factory=list,
        description="Timeline of visual direction scene cuts every 2-5s with layout modes and 2-4 search queries",
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


UNCENSORED_SYSTEM_PROMPT = """You are the Lead Visual Director and Executive Editor for the viral channel "@uncensored_talking".
Your mission:
Create explosive, highly shareable 30-60 second vertical shorts from podcast/interview transcripts using NON-LINEAR EDITING.

CRITICAL EDITING PRINCIPLES:
1. NON-LINEAR HOOK MANDATE:
   - The first 3 to 5 seconds of the clip MUST be the single most controversial, shocking, or taboo sentence of the discussion (even if it was spoken at minute 15 or 45).
   - Immediately hard-cut from this hook to the reaction or provocative question (e.g. Joe Rogan: "Wait, are you serious?!"), then continue the core explanation.
2. TIGHT PACING & JUMP CUTS:
   - Remove long thinking pauses, filler words, and small talk. Use multiple tight segments if needed.
3. DURATION:
   - Total clip duration MUST be strictly between 30 and 60 seconds (sum of all segments).
4. STRICT CONTEXTUAL B-ROLL DIRECTIVES (No generic stock, real archives):
   - Break the clip down into scenes every 2 to 5 seconds.
   - Layouts: 'splitscreen', 'full_speaker', 'full_broll'.
   - search_query: 1-2 focused, visual English search queries directly targeting the REAL subjects being discussed:
     * When real personalities are mentioned (e.g. "Trump", "Elon Musk", "Joe Rogan", "Bobby Kennedy / RFK", "Obamacare"):
       MANDATORY: search real, rare archive or inside material (e.g. "Donald Trump rare oval office footage", "Elon Musk 1990s retro interview archive", "Robert F Kennedy Jr speech archive raw", "White house press briefing historic raw footage").
     * When institutions or topics are mentioned: search real footage (e.g. "American school lunch cafeteria children raw footage", "pharmaceutical lab factory inspection archive", "corporate lobbying congress hearing archive raw").
     * NEVER use generic stock prompts (STRICTLY FORBIDDEN: 'steak', 'meat', 'gold vault', 'matrix code', 'smoke').
     * Avoid literal interpretations of idioms or metaphors (e.g. 'having beef' is NOT meat; 'cash cow' is NOT cows).

OUTPUT SCHEMA (Return ONLY valid JSON):
[
  {
    "clip_id": 1,
    "title": "slug_title",
    "hook": "Transcript excerpt of the opening 3-5 seconds",
    "rationale": "Why this shocks the audience",
    "segments": [
      {"start": 915.2, "end": 919.5},
      {"start": 120.0, "end": 124.2},
      {"start": 920.0, "end": 952.0}
    ],
    "scenes": [
      {
        "start": 0.0,
        "end": 3.5,
        "layout": "splitscreen",
        "search_query": "Donald Trump rare oval office footage"
      },
      {
        "start": 3.5,
        "end": 7.0,
        "layout": "full_speaker",
        "search_query": ""
      },
      {
        "start": 7.0,
        "end": 11.0,
        "layout": "full_broll",
        "search_query": "corporate lobbying congress hearing archive raw"
      }
    ]
  }
]
"""


def _load_transcript_data(
    transcript_file: Path | str,
    word_timestamps_file: Path | str,
) -> tuple[str, list[dict[str, Any]]]:
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


def extract_contextual_broll_query(
    slice_text: str,
    clip_words: list[dict[str, Any]] | None = None,
) -> str:
    """
    Generates 1-2 focused, visual English search queries directly targeting REAL subjects:
    - Real personalities (Trump, Elon Musk, Joe Rogan, RFK Jr, Obama) -> Rare archive footage.
    - Real institutions & public interest topics (White House, Congress, School Lunch, Food Chemistry Lab).
    - NEVER uses generic stock prompts ('steak', 'gold vault', 'meat', 'smoke').
    - Avoids literal interpretations of idioms/metaphors.
    """
    text_lower = slice_text.lower()

    # 1. Real Personalities & Political Figures (Highest priority)
    if any(k in text_lower for k in ["trump", "donald"]):
        return "Donald Trump rare oval office footage"
    if any(k in text_lower for k in ["elon", "musk"]):
        return "Elon Musk early archive raw footage"
    if any(k in text_lower for k in ["rogan", "joe rogan"]):
        return "Joe Rogan studio rare interview archive"
    if any(k in text_lower for k in ["kennedy", "bobby", "rfk"]):
        return "Robert F Kennedy Jr speech archive raw"
    if any(k in text_lower for k in ["obamacare", "obama"]):
        return "Barack Obama healthcare signing historic raw"
    if any(k in text_lower for k in ["susie", "wiles", "brooke", "rollins"]):
        return "White House administration briefing historic footage"

    # 2. Real Institutions & Contextual Subjects
    if any(k in text_lower for k in ["white house", "oval office", "president"]):
        return "White house oval office rare archive footage"
    if any(k in text_lower for k in ["congress", "capitol", "senate", "constituen"]):
        return "US Capitol congress chamber raw footage"
    if any(k in text_lower for k in ["school", "lunch", "breakfast", "donut", "pastr", "child", "kid"]):
        return "American school lunch cafeteria children raw footage"
    if any(k in text_lower for k in ["ultra-process", "processed food", "coca", "oreo", "mondelez", "sweetener", "dye", "chemic"]):
        return "food chemistry lab factory inspection archive"
    if any(k in text_lower for k in ["lobby", "special interest", "fda", "usda", "regulat"]):
        return "corporate lobbying congress hearing archive raw"
    if any(k in text_lower for k in ["healthcare", "insurance", "hospital", "premium", "doctor", "medicine"]):
        return "hospital healthcare system archival footage"
    if any(k in text_lower for k in ["wall street", "stock", "econom", "billion", "wealth", "inflation"]):
        return "Wall Street stock exchange trading floor historic footage"
    if any(k in text_lower for k in ["ai", "artificial intelligence", "robot", "supercomputer", "algorithm"]):
        return "supercomputer datacenter artificial intelligence raw footage"
    if any(k in text_lower for k in ["military", "pentagon", "defense", "army"]):
        return "Pentagon military briefing archival footage"
    if any(k in text_lower for k in ["classified", "intelligence", "cia", "fbi"]):
        return "FBI intelligence headquarters rare archival footage"

    # 3. Broader clip context fallback
    if clip_words:
        broad_text = " ".join(w["word"].lower() for w in clip_words)
        if any(k in broad_text for k in ["food", "school", "diet", "fda", "usda"]):
            return "American school lunch cafeteria children raw footage"
        if any(k in broad_text for k in ["healthcare", "insurance", "obamacare"]):
            return "hospital healthcare system archival footage"
        if any(k in broad_text for k in ["trump", "white house", "president", "policy"]):
            return "White house oval office rare archive footage"

    return "US Capitol congress chamber raw footage"


def generate_heuristic_scenes(
    clip_duration: float,
    clip_words: list[dict[str, Any]],
) -> list[SceneCut]:
    """
    Creates visual director plan with scene cuts every 2-5 seconds.
    Generates strictly contextual, real archive search queries per B-roll scene.
    """
    scenes: list[SceneCut] = []
    current_time = 0.0

    layout_cycle: list[LayoutType] = [
        "splitscreen",
        "full_speaker",
        "full_broll",
        "splitscreen",
        "full_speaker",
        "full_broll",
    ]
    cycle_idx = 0

    while current_time < clip_duration - 0.5:
        if current_time == 0.0:
            scene_dur = min(3.5, clip_duration)
        else:
            rem = clip_duration - current_time
            if rem <= 4.5:
                scene_dur = rem
            else:
                scene_dur = min(3.2, rem)

        scene_end = round(min(clip_duration, current_time + scene_dur), 2)
        layout = layout_cycle[cycle_idx % len(layout_cycle)]
        cycle_idx += 1

        search_query = ""
        if layout in ("full_broll", "splitscreen"):
            slice_text = " ".join(
                w["word"]
                for w in clip_words
                if current_time <= w.get("start", 0) <= scene_end
            )
            search_query = extract_contextual_broll_query(slice_text, clip_words)

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
    """Calls LLM with strict non-linear hook mandate and visual scene direction."""
    try:
        from openai import OpenAI
    except ImportError as err:
        raise ImportError("openai package required for LLM selector.") from err

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OPENAI_API_KEY provided or found in environment.")

    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    model_name = model or os.getenv("LLM_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key, base_url=base_url)

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

    user_prompt = f"""Analyze the podcast transcript and extract the TOP {num_clips} most viral, controversial clips for @uncensored_talking using NON-LINEAR EDITING.

MANDATORY EDITING RULES:
1. NON-LINEAR HOOK: The first segment (3-5s) MUST be the explosive shocker quote of the topic (taken from anywhere in the video).
2. Hard-cut to the reaction/question, then give the narrative explanation.
3. Total duration strictly 30-60s.
4. Provide scenes every 2-5s with 2-4 cinematic visual English keywords.

TRANSCRIPT:
{context_transcript}
"""

    logger.info(f"Querying LLM ({model_name}) for non-linear hook cuts and visual direction...")
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

        raw_scenes = item.get("scenes", [])
        item["scenes"] = []
        prop = ClipProposal(**item)

        if raw_scenes:
            prop.scenes = [SceneCut(**sc) for sc in raw_scenes if "start" in sc and "end" in sc]

        if not prop.scenes:
            clip_words = [
                w for w in words
                if any(s.start <= w["start"] <= s.end for s in prop.segments)
            ]
            prop.scenes = generate_heuristic_scenes(prop.total_duration, clip_words)

        proposals.append(prop)

    logger.info(f"LLM generated {len(proposals)} non-linear clip proposals.")
    return proposals


def select_clips_heuristic(
    words: list[dict[str, Any]],
    num_clips: int = 3,
    target_duration: float = 40.0,
) -> list[ClipProposal]:
    """
    Advanced heuristic with non-linear hook splicing:
    1. Finds peak controversy/taboo quotes across the entire duration.
    2. Takes the top punchy 3.5s quote as the non-linear hook.
    3. Finds the supporting context/discussion window (30-45s).
    4. Splices Hook -> Reaction/Body into a cohesive high-retention 35-50s clip.
    """
    logger.info("Running heuristic non-linear clip selector (offline mode)...")

    if not words:
        raise ValueError("Word timestamps list is empty.")

    total_duration = words[-1]["end"]
    if total_duration < 30.0:
        return [
            ClipProposal(
                clip_id=1,
                title="full_take",
                hook=" ".join(w["word"] for w in words[:8]),
                rationale="Short audio snippet taken in full.",
                segments=[Segment(start=words[0]["start"], end=total_duration)],
                scenes=generate_heuristic_scenes(total_duration, words),
            )
        ]

    CONTROVERSY_KEYWORDS = {
        "secret", "lie", "lying", "truth", "conspiracy", "matrix", "censored", "illegal",
        "insane", "crazy", "corrupt", "government", "money", "fake", "destroy", "kill",
        "scandal", "exposed", "dangerous", "never", "always", "shocking", "forbidden",
        "alien", "ai", "cia", "fbi", "control", "power", "war", "death", "god", "hell",
        "fear", "manipulated", "silenced", "trap", "elite", "propaganda", "hoax", "hate",
        "meat", "poison", "toxic", "pharma", "fda", "billion", "hospital", "cancer",
    }

    # Score words
    word_scores: list[float] = []
    for w in words:
        cl = re.sub(r"[^\w\s]", "", w["word"].lower())
        score = 1.0
        if cl in CONTROVERSY_KEYWORDS:
            score += 7.0
        if "?" in w["word"] or "!" in w["word"]:
            score += 2.5
        word_scores.append(score)

    # 1. Identify top 3-5s explosive hook candidates across the ENTIRE audio
    hook_candidates: list[dict[str, Any]] = []
    n = len(words)
    for i in range(n):
        s_t = words[i]["start"]
        for j in range(i + 3, min(i + 15, n)):
            e_t = words[j]["end"]
            dur = e_t - s_t
            if 3.0 <= dur <= 4.5:
                score = sum(word_scores[i : j + 1]) / dur
                hook_candidates.append({
                    "start": s_t,
                    "end": e_t,
                    "score": score,
                    "words": words[i : j + 1],
                })
            elif dur > 4.5:
                break

    hook_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 2. Identify top 30-45s discussion segments
    body_candidates: list[dict[str, Any]] = []
    for i in range(0, n, 5):
        s_t = words[i]["start"]
        for j in range(i + 20, n, 5):
            e_t = words[j]["end"]
            dur = e_t - s_t
            if 30.0 <= dur <= 45.0:
                score = sum(word_scores[i : j + 1]) / dur
                body_candidates.append({
                    "start": s_t,
                    "end": e_t,
                    "score": score,
                    "words": words[i : j + 1],
                })
            elif dur > 45.0:
                break

    body_candidates.sort(key=lambda x: x["score"], reverse=True)

    proposals: list[ClipProposal] = []
    used_times: list[tuple[float, float]] = []

    for idx in range(1, num_clips + 1):
        # Pick non-overlapping body
        body = None
        for b in body_candidates:
            if not any(not (b["end"] < u[0] or b["start"] > u[1]) for u in used_times):
                body = b
                break
        if not body:
            body = body_candidates[(idx - 1) % len(body_candidates)]

        used_times.append((body["start"], body["end"]))

        # Pick punchiest hook (ideally non-linear from outside or peak of the body)
        hook_choice = hook_candidates[(idx - 1) % len(hook_candidates)]
        hook_text = " ".join(w["word"] for w in hook_choice["words"])

        # Non-linear structure:
        # Segment 1: Hook quote (3-4s)
        # Segment 2: Context & explanation body (30-40s)
        segments: list[Segment] = []
        if abs(hook_choice["start"] - body["start"]) > 10.0:
            # True non-linear splice
            segments = [
                Segment(start=hook_choice["start"], end=hook_choice["end"]),
                Segment(start=body["start"], end=min(body["end"], body["start"] + 36.0)),
            ]
        else:
            # Peak starts right at the body
            segments = [Segment(start=body["start"], end=body["end"])]

        # Visual scenes
        clip_words = [
            w for w in words
            if any(s.start <= w["start"] <= s.end for s in segments)
        ]
        total_dur = sum(s.end - s.start for s in segments)
        scenes = generate_heuristic_scenes(total_dur, clip_words)

        proposals.append(
            ClipProposal(
                clip_id=idx,
                title=f"uncensored_clip_{idx}_{int(segments[0].start)}s",
                hook=hook_text,
                rationale=f"High controversy score ({body['score']:.2f}) with explosive non-linear hook.",
                segments=segments,
                scenes=scenes,
            )
        )

    logger.info(f"Heuristic selector created {len(proposals)} non-linear hook clip proposals.")
    return proposals


def select_viral_clips(
    transcript_file: Path | str = Path("temp/full_transcript.txt"),
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
    num_clips: int = 3,
    prefer_llm: bool = True,
    output_json_path: Path | str = Path("temp/selected_clips.json"),
) -> list[ClipProposal]:
    full_text, words = _load_transcript_data(transcript_file, word_timestamps_file)
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if prefer_llm and has_api_key:
        try:
            proposals = select_clips_with_llm(full_text, words, num_clips=num_clips)
        except Exception as exc:
            logger.warning(f"LLM selection failed: {exc}. Using heuristic non-linear selector.")
            proposals = select_clips_heuristic(words, num_clips=num_clips)
    else:
        proposals = select_clips_heuristic(words, num_clips=num_clips)

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump([p.to_output_dict() for p in proposals], f, indent=2, ensure_ascii=False)

    return proposals
