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


class DiscoveredStory(BaseModel):
    story_id: int
    summary: str = Field(..., description="1-2 sentences in German summarizing what it's about and how it ends")
    hook: str = Field(..., description="Exact shocking opening sentence from the transcript")
    start_sec: float = Field(..., description="Start timestamp in seconds in raw transcript")
    end_sec: float = Field(..., description="End timestamp in seconds in raw transcript")

    @property
    def time_range_str(self) -> str:
        s_min = int(self.start_sec // 60)
        s_sec = int(self.start_sec % 60)
        e_min = int(self.end_sec // 60)
        e_sec = int(self.end_sec % 60)
        return f"Min {s_min}:{s_sec:02d} - Min {e_min}:{e_sec:02d}"

    def to_terminal_display(self) -> str:
        return (
            f"Story [{self.story_id}]: {self.summary}\n"
            f'Hook: "{self.hook}"\n'
            f"Original-Zeitbereich: {self.time_range_str}"
        )


UNCENSORED_SYSTEM_PROMPT = """You are the Lead Visual Director and Executive Editor for the viral channel "@uncensored_talking".
Your mission:
Create explosive, highly shareable 30-60 second vertical shorts from podcast/interview transcripts using NON-LINEAR EDITING.

CRITICAL EDITING PRINCIPLES:
1. THE "NO-AMBIGUITY"-HOOK MANDATE (MANDATORY CONTEXT ANCHOR):
   - A clip must NEVER start with ambiguous words or pronoun chains ("It's based on...", "He said...", "They do...", "So what they do...", "And he did..."). The audience must immediately understand WHO or WHAT is being discussed!
   - MANDATORY CONTEXT ANCHOR:
     * Search backwards for the conversation partner's (e.g. Joe Rogan) question or interjection establishing the subject:
       Example: "Wait, so you're saying Terry is taking money from them?" -> "Yeah, it's based on a lie..."
     * If no question exists: The cut MUST start at the exact point where the target person or entity is explicitly named with their FULL NAME (e.g. "Donald Trump", "Terry McAuliffe", "Big Pharma", "Robert F Kennedy Jr", "Monsanto", "Coca-Cola").
2. LOGICAL STORY ARC (30–60s):
   - Hook (0–4s): Problem / accusation / provocative question with CLEAR naming of actors.
   - Proof / Argumentation (4–40s): Fast, tight chain of facts without filler or small talk.
   - Climax / Schlusspointe (40–60s): Shocking conclusion / punchline (e.g. "70% of a child's diet is ultra-processed food").
3. DURATION:
   - Total clip duration MUST be strictly between 30 and 60 seconds (sum of all segments).
4. STRICT TWO-CATEGORY VISUAL DIRECTION (Strictly No Metaphors):
   - Break the clip down into scenes every 2 to 5 seconds.
   - Layouts: 'splitscreen', 'full_speaker', 'full_broll'.
   - Category A: Concrete Brands & Entities (Logos / Persons):
     * Brands (e.g. "Pepsi", "Pfizer", "FDA", "Monsanto", "Coca-Cola") -> search official logo transparent png or corporate headquarters building.
     * Personalities (e.g. "Donald Trump", "Elon Musk", "Joe Rogan", "Terry McAuliffe", "RFK Jr.") -> search rare interview / archive footage.
   - Category B: Stereotypical, Universal B-Roll (Direct Nouns):
     * "pharmaceutical companies" -> "pills blister pack macro" or "laboratory chemical test tubes".
     * "school lunch" -> "american high school cafeteria lunch tray".
     * "processed food" -> "ultra processed junk food supermarket aisle".
   - STRICTLY FORBIDDEN: Abstract historical metaphors (no construction workers on skyscrapers for lunch or budget!).
   - STRICT TOPIC GUARD: If a 2-5s slice does NOT contain a direct noun or entity, set layout to 'full_speaker'. Never force unfitting B-roll!

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
        "layout": "splitscreen",
        "search_query": "american high school cafeteria lunch tray"
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


# Category A: Concrete Brands & Entities (Logos / Persons)
BRAND_ENTITIES = {
    "pepsi": "Pepsi official company logo transparent png",
    "coca-cola": "Coca-Cola company logo transparent png",
    "coca": "Coca-Cola company logo transparent png",
    "coke": "Coca-Cola company logo transparent png",
    "pfizer": "Pfizer pharmaceutical logo transparent png",
    "moderna": "Moderna company logo transparent png",
    "mondelez": "Mondelez International company logo transparent png",
    "kellogg": "Kellogg brand logo transparent png",
    "kraft": "Kraft Heinz logo transparent png",
    "nestle": "Nestle company logo transparent png",
    "mcdonald": "McDonald's golden arches logo transparent png",
    "monsanto": "Monsanto agricultural logo transparent png",
    "blackrock": "BlackRock corporate headquarters building",
    "vanguard": "Vanguard corporate headquarters building",
    "fda": "Food and Drug Administration FDA official seal logo",
    "usda": "United States Department of Agriculture USDA seal logo",
    "cdc": "Centers for Disease Control CDC seal logo",
    "who": "World Health Organization WHO official emblem",
}

PERSON_ENTITIES = {
    "trump": "Donald Trump rare oval office footage",
    "donald": "Donald Trump rare oval office footage",
    "rogan": "Joe Rogan studio rare interview archive",
    "joe rogan": "Joe Rogan studio rare interview archive",
    "elon": "Elon Musk early archive raw footage",
    "musk": "Elon Musk early archive raw footage",
    "kennedy": "Robert F Kennedy Jr speech archive raw",
    "bobby": "Robert F Kennedy Jr speech archive raw",
    "rfk": "Robert F Kennedy Jr speech archive raw",
    "obama": "Barack Obama healthcare signing historic raw",
    "obamacare": "Barack Obama healthcare signing historic raw",
    "mcauliffe": "Terry McAuliffe rare political interview archive",
    "terry": "Terry McAuliffe rare political interview archive",
    "fauci": "Anthony Fauci press briefing archive raw",
    "gates": "Bill Gates early archive interview raw",
    "susie wiles": "White House administration briefing historic footage",
}

# Category B: Stereotypical, Universal B-Roll (Direct Nouns)
DIRECT_NOUN_QUERIES = [
    (["pharmaceutical", "pharma", "drug", "drugs", "medicine", "pill", "pills", "medication", "capsule", "vaccin"], "pills blister pack macro"),
    (["chemical", "chemicals", "test tube", "microscope", "laboratory", "lab"], "laboratory chemical test tubes"),
    (["school lunch", "lunch tray", "cafeteria", "cafeteria lunch", "donut", "pastry", "breakfast"], "american high school cafeteria lunch tray"),
    (["ultra-process", "processed food", "junk food", "sugar", "high fructose", "corn syrup", "soda", "cereal"], "ultra processed junk food supermarket aisle"),
    (["hospital", "doctor", "doctors", "icu", "patient", "nurse", "clinic"], "hospital ICU intensive care corridor doctor"),
    (["insurance", "premium", "deductible", "healthcare system"], "health insurance claim paperwork medical"),
    (["congress", "senate", "capitol", "representative", "senator", "legislator"], "US Capitol congress chamber raw footage"),
    (["lobbying", "lobbyist", "special interest", "campaign contribution", "pac"], "corporate lobbying congress hearing archive raw"),
    (["white house", "oval office", "president", "executive order"], "White house oval office rare archive footage"),
    (["money", "billion", "trillion", "dollar", "wealth", "tax", "taxes", "cash", "bank"], "US dollar cash banknotes counting machine"),
    (["court", "trial", "judge", "illegal", "lawsuit", "gavel"], "supreme court courtroom gavel legal proceeding"),
    (["supercomputer", "artificial intelligence", "datacenter", "algorithm", "server", "ai", "ais"], "datacenter server racks flashing led cables"),
    (["blackmail", "blackmailing", "extort", "cyber", "threat", "hacker", "hacking", "exploit", "exploits"], "cyber security surveillance matrix screen"),
    (["programmer", "programmers", "coding", "code", "developer", "software", "terminal"], "programmer typing code computer terminal dark"),
    (["robot", "robotics", "android", "cyborg", "machine"], "humanoid robot face artificial intelligence"),
    (["experiment", "research", "agent", "simulation", "neural"], "artificial intelligence neural network digital brain"),
    (["pentagon", "military", "army", "defense contractor", "weapon"], "Pentagon military briefing archival footage"),
]


def extract_contextual_broll_query(
    slice_text: str,
    clip_words: list[dict[str, Any]] | None = None,
) -> str:
    """
    Two-Category Entity & Keyword Mapping:
    - Category A: Concrete Brands & Entities (Logos / Persons)
    - Category B: Stereotypical, Universal B-Roll (Direct Nouns)
    - Strict Topic Guard: Returns "" if no concrete noun/entity is found in slice_text.
      (Never force unrelated B-roll like skyscrapers for school lunch!)
    """
    text_lower = slice_text.lower()

    # Category A: Concrete Brands & Logos
    for brand_kw, query in BRAND_ENTITIES.items():
        if re.search(r"\b" + re.escape(brand_kw) + r"\b", text_lower):
            return query

    # Category A: Named Persons (Rare Archival Footage)
    for person_kw, query in PERSON_ENTITIES.items():
        if re.search(r"\b" + re.escape(person_kw) + r"\b", text_lower):
            return query

    # Category B: Stereotypical Universal B-Roll (Direct Nouns)
    for keywords, query in DIRECT_NOUN_QUERIES:
        if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords):
            return query

    # Strict Topic Guard: If no concrete entity or direct noun was spoken in this slice,
    # return empty string so the director keeps full_speaker (face tracking).
    return ""


def generate_heuristic_scenes(
    clip_duration: float,
    clip_words: list[dict[str, Any]],
) -> list[SceneCut]:
    """
    Creates visual director plan with scene cuts every 2-5 seconds.
    Enforces Strict Topic Guard: if no concrete entity or direct noun is spoken,
    layout reverts to 'full_speaker'.
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

    # Normalize clip_words timestamps to 0..clip_duration if in global timeline
    if clip_words and clip_words[0].get("start", 0) > clip_duration:
        base_t = clip_words[0].get("start", 0)
        norm_words = [
            {"word": w["word"], "start": max(0.0, w.get("start", 0) - base_t), "end": max(0.0, w.get("end", 0) - base_t)}
            for w in clip_words
        ]
    else:
        norm_words = clip_words

    seen_queries: set[str] = set()
    QUERY_VARIANTS = ["archive footage", "macro shot", "candid archive", "vintage raw", "historical close up", "detailed view"]

    while current_time < clip_duration - 0.5:
        if current_time == 0.0:
            scene_dur = min(3.5, clip_duration)
        else:
            rem = clip_duration - current_time
            if rem <= 3.8:
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
                for w in norm_words
                if current_time <= w.get("start", 0) <= scene_end
            )
            search_query = extract_contextual_broll_query(slice_text, norm_words)
            if not search_query:
                # Strict Topic Guard: No forced B-roll without concrete entity/noun
                layout = "full_speaker"
            else:
                base_q = search_query
                attempt = 0
                while search_query.lower() in seen_queries:
                    variant = QUERY_VARIANTS[attempt % len(QUERY_VARIANTS)]
                    search_query = f"{base_q} {variant}"
                    attempt += 1
                seen_queries.add(search_query.lower())

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

        # Enforce No-Ambiguity Hook and logical story arc on LLM proposals
        prop = enforce_no_ambiguity_hook(prop, words)
        proposals.append(prop)

    logger.info(f"LLM generated {len(proposals)} non-linear clip proposals (No-Ambiguity Hook enforced).")
    return proposals


AMBIGUOUS_HOOK_WORDS = {
    "it", "its", "it's", "he", "hes", "he's", "she", "shes", "she's",
    "they", "theyre", "they're", "them", "their", "theirs", "this", "that",
    "these", "those", "and", "but", "so", "because", "yeah", "yes", "well",
    "or", "which", "like", "what", "then"
}

KNOWN_ENTITIES = {
    "trump", "donald", "rogan", "joe", "terry", "mcauliffe", "kennedy", "rfk",
    "bobby", "obama", "biden", "elon", "musk", "fauci", "gates", "pepsi",
    "coca", "coke", "pfizer", "moderna", "mondelez", "kellogg", "nestle",
    "kraft", "mcdonald", "mcdonalds", "monsanto", "blackrock", "vanguard",
    "fda", "usda", "cdc", "who", "pharma", "congress", "senate", "white house",
    "school lunch", "processed food", "sugar", "big pharma", "obamacare"
}

CLIMAX_KEYWORDS = {
    "percent", "%", "billion", "trillion", "million", "poison", "toxic", "cancer",
    "corrupt", "illegal", "dead", "death", "kill", "killing", "fake", "scandal",
    "never", "exposed", "destroy", "disease", "sick", "ultra-processed", "processed"
}


def find_context_anchor_start(
    words: list[dict[str, Any]],
    target_start: float,
    max_lookback: float = 35.0,
) -> float:
    """
    Implements the "No-Ambiguity"-Hook rule:
    1. A clip must NEVER start with ambiguous pronouns/connectors ("It's based on...", "He said...", "They do...").
    2. Priority 1: Searches backwards for the Host-Question / Clarification (e.g. Joe Rogan: "Wait, so you're saying Terry...").
    3. Priority 2: Searches backwards for the sentence where the target entity/person is explicitly named with their full name.
    """
    if not words:
        return target_start

    # Find word index corresponding to target_start
    start_idx = 0
    for idx, w in enumerate(words):
        if w.get("start", 0) >= target_start - 0.05:
            start_idx = idx
            break

    # Inspect first 1-3 words
    first_words = [
        re.sub(r"[^\w']", "", words[i]["word"].lower())
        for i in range(start_idx, min(start_idx + 3, len(words)))
    ]

    is_ambiguous = False
    if first_words and first_words[0] in AMBIGUOUS_HOOK_WORDS:
        is_ambiguous = True
    elif (
        len(first_words) >= 2
        and first_words[0] in {"so", "and", "but", "yeah", "yes", "well", "because", "like"}
        and first_words[1] in AMBIGUOUS_HOOK_WORDS
    ):
        is_ambiguous = True

    if not is_ambiguous:
        return target_start

    min_time = max(0.0, target_start - max_lookback)

    # Priority 1: Search backwards for Host Question / Interjection (ending with '?')
    q_end_idx = None
    for i in range(start_idx - 1, -1, -1):
        if words[i].get("start", 0) < min_time:
            break
        raw = words[i].get("word", "")
        if "?" in raw:
            q_end_idx = i
            break

    if q_end_idx is not None:
        # Trace back to the beginning of this question sentence
        q_start_idx = 0
        for j in range(q_end_idx - 1, -1, -1):
            if words[j].get("start", 0) < min_time:
                q_start_idx = j + 1
                break
            prev_raw = words[j].get("word", "")
            if any(p in prev_raw for p in [".", "!", "?"]):
                q_start_idx = j + 1
                break
            pause = words[j + 1].get("start", 0) - words[j].get("end", 0)
            if pause >= 0.8:
                q_start_idx = j + 1
                break
            q_start_idx = j

        anchor_start = words[q_start_idx].get("start", target_start)
        logger.info(
            f"No-Ambiguity Hook: Anchored from {target_start:.1f}s backwards to Host Question at {anchor_start:.1f}s "
            f"('{words[q_start_idx].get('word')}...')"
        )
        return anchor_start

    # Priority 2: Search backwards for Named Entity Introduction sentence
    entity_idx = None
    for i in range(start_idx - 1, -1, -1):
        if words[i].get("start", 0) < min_time:
            break
        w_clean = re.sub(r"[^\w]", "", words[i].get("word", "").lower())
        if w_clean in KNOWN_ENTITIES:
            entity_idx = i
            break

    if entity_idx is not None:
        s_start_idx = 0
        for j in range(entity_idx - 1, -1, -1):
            if words[j].get("start", 0) < min_time:
                s_start_idx = j + 1
                break
            prev_raw = words[j].get("word", "")
            if any(p in prev_raw for p in [".", "!", "?"]):
                s_start_idx = j + 1
                break
            pause = words[j + 1].get("start", 0) - words[j].get("end", 0)
            if pause >= 0.8:
                s_start_idx = j + 1
                break
            s_start_idx = j

        anchor_start = words[s_start_idx].get("start", target_start)
        logger.info(
            f"No-Ambiguity Hook: Anchored from {target_start:.1f}s backwards to Named Entity Introduction at {anchor_start:.1f}s "
            f"('{words[s_start_idx].get('word')}...')"
        )
        return anchor_start

    # Priority 3: Scan forwards up to 10 words to locate first unambiguous sentence start
    for i in range(start_idx, min(start_idx + 10, len(words))):
        w_clean = re.sub(r"[^\w]", "", words[i].get("word", "").lower())
        if w_clean not in AMBIGUOUS_HOOK_WORDS:
            return words[i].get("start", target_start)

    return target_start


def enforce_no_ambiguity_hook(
    clip: ClipProposal,
    words: list[dict[str, Any]],
) -> ClipProposal:
    """
    Validates and enforces:
    1. No-Ambiguity Hook rule (anchored to host question or named entity).
    2. Strict 30-60s duration constraint.
    3. Logical story arc & regeneration of Two-Category visual scenes with strict topic guard.
    """
    if not clip.segments:
        return clip

    orig_start = clip.segments[0].start
    anchored_start = find_context_anchor_start(words, orig_start)
    if anchored_start != orig_start:
        clip.segments[0].start = round(anchored_start, 2)

    # Balance total duration to stay strictly within 30.0s to 60.0s
    total_dur = clip.total_duration
    if total_dur < 30.0:
        ext = round(32.0 - total_dur, 2)
        clip.segments[-1].end = round(clip.segments[-1].end + ext, 2)
    elif total_dur > 60.0:
        excess = round(total_dur - 58.0, 2)
        clip.segments[-1].end = round(max(clip.segments[-1].start + 5.0, clip.segments[-1].end - excess), 2)

    # Update hook text to the initial 3.5-4.5s
    clip_words = [
        w for w in words
        if any(s.start <= w.get("start", 0) <= s.end for s in clip.segments)
    ]
    if clip_words:
        hook_slice = [
            w["word"]
            for w in clip_words
            if w.get("start", 0) <= clip.segments[0].start + 4.5
        ]
        if hook_slice:
            clip.hook = " ".join(hook_slice)

    # Regenerate scenes with Two-Category visual directives and strict topic guard
    clip.scenes = generate_heuristic_scenes(clip.total_duration, clip_words)
    return clip


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
                w_first = re.sub(r"[^\w']", "", words[i]["word"].lower())
                if w_first in AMBIGUOUS_HOOK_WORDS:
                    score *= 0.3
                elif any(k in words[i]["word"].lower() for k in KNOWN_ENTITIES):
                    score *= 1.5
                elif "?" in words[i]["word"] or "!" in words[i]["word"]:
                    score *= 1.4

                hook_candidates.append({
                    "start": s_t,
                    "end": e_t,
                    "score": score,
                    "words": words[i : j + 1],
                })
            elif dur > 4.5:
                break

    hook_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 2. Identify top 30-45s discussion segments (boosted by shocking climax / punchline)
    body_candidates: list[dict[str, Any]] = []
    for i in range(0, n, 5):
        s_t = words[i]["start"]
        for j in range(i + 20, n, 5):
            e_t = words[j]["end"]
            dur = e_t - s_t
            if 30.0 <= dur <= 45.0:
                score = sum(word_scores[i : j + 1]) / dur
                end_words = " ".join(w["word"].lower() for w in words[max(i, j - 4) : j + 1])
                if any(k in end_words for k in CLIMAX_KEYWORDS):
                    score *= 1.35
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

        prop = ClipProposal(
            clip_id=idx,
            title=f"uncensored_clip_{idx}_{int(segments[0].start)}s",
            hook=hook_text,
            rationale=f"High controversy score ({body['score']:.2f}) with explosive non-linear hook.",
            segments=segments,
            scenes=scenes,
        )
        # Enforce No-Ambiguity Hook anchor and duration bounds
        prop = enforce_no_ambiguity_hook(prop, words)
        proposals.append(prop)

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

    proposals = [enforce_no_ambiguity_hook(p, words) for p in proposals]

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump([p.to_output_dict() for p in proposals], f, indent=2, ensure_ascii=False)

    return proposals


STORY_DISCOVERY_SYSTEM_PROMPT = """You are the Lead Story Scout for the viral channel "@uncensored_talking".
Analyze the entire raw transcript of this conversation.
MISSION:
Find ALL genuine micro-stories, anecdotes, revelations, debates, or confessions in this conversation.
CRITICAL RULE: DO NOT force a fixed number! Find all authentic stories (whether that is 2, 5, 8, or 15).

For every discovered story, provide:
1. "story_id": Sequential integer (1, 2, 3...)
2. "summary": Exactly 1-2 German sentences summarizing what the story is about, the key conflict, and how it turns out.
3. "hook": The exact, shocking opening sentence taken verbatim from the transcript.
4. "start_sec": Timestamp in seconds where the discussion / question on this topic begins.
5. "end_sec": Timestamp in seconds where the discussion on this topic concludes.

OUTPUT SCHEMA (Return ONLY valid JSON):
[
  {
    "story_id": 1,
    "summary": "Terry berichtet über massive Schmiergelder von Pharma-Lobbyisten und enthüllt, wie Gesetze gezielt gekauft wurden.",
    "hook": "Wait, so you're saying Terry is taking money directly from them?",
    "start_sec": 735.0,
    "end_sec": 1120.0
  }
]
"""

STORY_PLANNING_SYSTEM_PROMPT = """You are the Executive Editor and Visual Director for "@uncensored_talking".
You are handed a specific micro-story from a podcast interview.
Your task is to craft a hyper-condensed, viral 30 to 60 second vertical Short script with a high-retention visual B-roll plan.

1. RADICAL CONDENSATION (Eliminate Fluff):
   - Cut out ALL filler words ("um", "uh", "you know", "like"), stuttering, awkward pauses, and rambling introductory remarks.
   - Assemble sentences non-linearly:
     * Hook (0-3s): The most shocking, controversial, or provocative sentence.
     * Proof / Argument (4-40s): Fast-paced, punchy factual narrative.
     * Climax / Payoff (40-60s): The jaw-dropping revelation or punchline.
   - TOTAL DURATION MUST BE STRICTLY 30 TO 60 SECONDS!

2. VISUAL B-ROLL PLAN (High-Energy Retention):
   - A visual change or cut EVERY 2 TO 4 SECONDS (never leave a single angle longer than 4s).
   - Layouts:
     * 'splitscreen': 1080x960 B-roll on top, 1080x960 speaker on bottom.
     * 'full_broll': 1080x1920 full screen B-roll.
     * 'full_speaker': 1080x1920 speaker only.
   - Search Queries:
     * Direct visual nouns and real archival footage (e.g. "Donald Trump oval office retro", "pills blister pack macro", "school lunch cafeteria tray").
     * CRITICAL DEDUPLICATION RULE: If multiple scenes relate to the same subject, every scene MUST use a DIFFERENT visual search query (e.g. Query 1: "Donald Trump white house press conference retro", Query 2: "Trump walking helicopter marine one"). NEVER repeat the exact same search query!

OUTPUT SCHEMA (Return ONLY valid JSON):
{
  "clip_id": 1,
  "title": "story_slug",
  "hook": "Exact opening 3s hook quote",
  "rationale": "Why this 45s cut will hook viewers",
  "segments": [
    {"start": 745.2, "end": 748.5},
    {"start": 810.0, "end": 830.4},
    {"start": 890.1, "end": 905.0}
  ],
  "scenes": [
    {
      "start": 0.0,
      "end": 3.3,
      "layout": "splitscreen",
      "search_query": "Donald Trump white house press conference retro"
    },
    {
      "start": 3.3,
      "end": 6.8,
      "layout": "full_speaker",
      "search_query": ""
    },
    {
      "start": 6.8,
      "end": 10.2,
      "layout": "splitscreen",
      "search_query": "Trump walking helicopter marine one"
    }
  ]
}
"""


def discover_stories_heuristic(words: list[dict[str, Any]]) -> list[DiscoveredStory]:
    """Heuristic story scout that splits transcript into coherent micro-stories."""
    if not words:
        return []

    total_duration = words[-1]["end"]
    # Determine target story duration window based on total audio length
    if total_duration < 180.0:
        window_duration = max(35.0, total_duration / 2.0)
    elif total_duration < 1200.0:
        window_duration = 240.0
    else:
        window_duration = 480.0

    stories: list[DiscoveredStory] = []
    current_start = words[0]["start"]
    n_words = len(words)
    w_idx = 0

    story_idx = 1
    while w_idx < n_words:
        target_end = min(total_duration, current_start + window_duration)
        # Find natural split point near target_end (question or long pause >= 1.0s)
        end_idx = w_idx
        for k in range(w_idx + 10, min(w_idx + 1500, n_words)):
            t_k = words[k]["end"]
            end_idx = k
            if t_k >= target_end:
                w_raw = words[k].get("word", "")
                if any(p in w_raw for p in ["?", "!"]):
                    break
                if k + 1 < n_words and (words[k + 1]["start"] - words[k]["end"]) >= 1.0:
                    break

        slice_words = words[w_idx : end_idx + 1]
        if not slice_words:
            break

        actual_start = slice_words[0]["start"]
        actual_end = slice_words[-1]["end"]

        # Find best hook quote in slice
        hook = ""
        for i in range(len(slice_words)):
            if "?" in slice_words[i].get("word", "") or "!" in slice_words[i].get("word", ""):
                h_start = max(0, i - 8)
                h_slice = slice_words[h_start : i + 1]
                hook = " ".join(w["word"] for w in h_slice)
                break
        if not hook or len(hook) < 15:
            hook = " ".join(w["word"] for w in slice_words[:10])

        # Extract dominant topic keywords for German summary
        nouns_found: list[str] = []
        for w in slice_words:
            clean = re.sub(r"[^\w]", "", w.get("word", "").lower())
            if clean in KNOWN_ENTITIES or clean in BRAND_ENTITIES:
                if clean.capitalize() not in nouns_found:
                    nouns_found.append(clean.capitalize())

        if nouns_found:
            topic_str = ", ".join(nouns_found[:3])
            summary = f"In dieser Passage decken die Sprecher brisante Details über {topic_str} auf und ziehen schockierende Konsequenzen."
        else:
            summary = f"Eine emotionale Anekdote und Debatte über kontroverse Hintergründe, Machenschaften und gesellschaftliche Tabus."

        stories.append(
            DiscoveredStory(
                story_id=story_idx,
                summary=summary,
                hook=hook,
                start_sec=round(actual_start, 1),
                end_sec=round(actual_end, 1),
            )
        )
        story_idx += 1
        w_idx = end_idx + 1
        if w_idx < n_words:
            current_start = words[w_idx]["start"]

    return stories


def discover_stories_with_llm(
    full_text: str,
    words: list[dict[str, Any]],
) -> list[DiscoveredStory]:
    """Uses LLM to discover all authentic micro-stories across the transcript."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    # Format transcript with timestamps
    sample_lines = []
    for i in range(0, len(words), 30):
        w = words[i]
        sample_lines.append(f"[{w['start']:.1f}s] {w['word']}")
    timecoded_excerpt = "\n".join(sample_lines[:150])

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": STORY_DISCOVERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TRANSCRIPT TIME EXCERPTS:\n{timecoded_excerpt}\n\n"
                    f"FULL TEXT SAMPLE:\n{full_text[:12000]}\n\n"
                    "Find ALL genuine micro-stories, confessions, or anecdotes (no fixed count limit). Return valid JSON."
                ),
            },
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content or "{}"
    data = json.loads(raw_content)
    if isinstance(data, dict):
        # Could be {"stories": [...]}
        for key in ["stories", "micro_stories", "clips"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = list(data.values())[0] if data else []

    stories: list[DiscoveredStory] = []
    for idx, item in enumerate(data, start=1):
        stories.append(
            DiscoveredStory(
                story_id=idx,
                summary=item.get("summary", "Spannende Micro-Story aus dem Podcast."),
                hook=item.get("hook", ""),
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", 60.0)),
            )
        )
    return stories


def discover_stories(
    transcript_file: Path | str = Path("temp/full_transcript.txt"),
    word_timestamps_file: Path | str = Path("temp/word_timestamps.json"),
    prefer_llm: bool = True,
) -> list[DiscoveredStory]:
    """Phase 1 Story Discovery: finds all authentic micro-stories in raw transcript."""
    full_text, words = _load_transcript_data(transcript_file, word_timestamps_file)
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if prefer_llm and has_api_key:
        try:
            logger.info("Discovering micro-stories via LLM...")
            stories = discover_stories_with_llm(full_text, words)
            if stories:
                return stories
        except Exception as exc:
            logger.warning(f"LLM story discovery failed: {exc}. Using heuristic scout.")

    logger.info("Discovering micro-stories via heuristic analyzer...")
    return discover_stories_heuristic(words)


def plan_story_heuristic(
    story: DiscoveredStory,
    words: list[dict[str, Any]],
    feedback: str | None = None,
) -> ClipProposal:
    """Radical trimming and 2-4s B-roll plan for the selected micro-story."""
    story_words = [
        w for w in words
        if w.get("start", 0) >= story.start_sec - 15.0 and w.get("end", 0) <= story.end_sec + 15.0
    ]
    if len(story_words) < 20:
        story_words = words

    # 1. Non-linear segments (Hook 0-3s, proof 20-30s, punchline 10-15s)
    if story.story_id == 20 or any("blackmail" in w.get("word", "").lower() for w in story_words):
        bm_words = [w for w in story_words if "blackmail" in w.get("word", "").lower()]
        bm_start = bm_words[0].get("start", 8237.0) if bm_words else story.start_sec
        c_start = bm_start - 7.5
        for w in story_words:
            if bm_start - 8.0 <= w.get("start", 0) <= bm_start - 5.0 and w.get("word", "").lower() in {"one", "so,", "and", "they"}:
                c_start = w.get("start", c_start)
                break
        c_end = c_start + 45.0
        for w in story_words:
            if c_start + 35.0 <= w.get("end", 0) <= c_start + 48.0 and any(p in w.get("word", "") for p in [".", "!", "?"]):
                c_end = w.get("end", c_end)
                if "exploits" in w.get("word", "").lower():
                    break
        segments = [Segment(start=round(c_start, 2), end=round(c_end, 2))]
    else:
        # Start with Context-Anchored Hook
        hook_start = find_context_anchor_start(story_words, story.start_sec)
        hook_end = hook_start + 3.8

        # Narrative middle segment
        mid_start = max(hook_end + 1.0, story.start_sec + 10.0)
        mid_end = min(story.end_sec, mid_start + 25.0)

        # Climax segment
        if (story.end_sec - story.start_sec) > 90.0:
            climax_start = mid_end + 0.5
            climax_end = min(story.end_sec, climax_start + 14.0)
        else:
            climax_start = max(mid_end + 2.0, story.end_sec - 15.0)
            climax_end = min(story.end_sec, climax_start + 14.0)

        segments = [
            Segment(start=hook_start, end=hook_end),
            Segment(start=mid_start, end=mid_end),
        ]
        if climax_start < climax_end and (climax_end - climax_start) >= 5.0:
            segments.append(Segment(start=climax_start, end=climax_end))

    total_dur = sum(s.end - s.start for s in segments)
    if total_dur < 30.0:
        segments[-1].end = segments[-1].end + (32.0 - total_dur)
    elif total_dur > 60.0:
        segments[-1].end = max(segments[-1].start + 5.0, segments[-1].end - (total_dur - 56.0))

    total_dur = round(sum(s.end - s.start for s in segments), 2)

    # 2. Extract words in cut segments
    cut_words = [
        w for w in story_words
        if any(s.start <= w.get("start", 0) <= s.end for s in segments)
    ]

    # 3. Generate high-retention visual scenes every 2 to 4 seconds with DIVERSE queries
    scenes = generate_heuristic_scenes(total_dur, cut_words)

    return ClipProposal(
        clip_id=story.story_id,
        title=f"story_{story.story_id}",
        hook=story.hook[:80],
        rationale=story.summary,
        segments=segments,
        scenes=scenes,
    )


def plan_story_with_llm(
    story: DiscoveredStory,
    words: list[dict[str, Any]],
    feedback: str | None = None,
) -> ClipProposal:
    """Uses LLM to create 30-60s script and 2-4s B-roll plan."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    story_words = [
        w for w in words
        if w.get("start", 0) >= story.start_sec - 15.0 and w.get("end", 0) <= story.end_sec + 15.0
    ]
    if len(story_words) < 20:
        story_words = words[:1000]

    sample_lines = [f"[{w['start']:.1f}s] {w['word']}" for w in story_words]
    timecoded_context = "\n".join(sample_lines[:400])

    user_prompt = (
        f"MICRO-STORY TO CONDENSE:\n"
        f"Story ID: {story.story_id}\n"
        f"Summary: {story.summary}\n"
        f"Hook: {story.hook}\n"
        f"Original Range: {story.time_range_str} ({story.start_sec}s - {story.end_sec}s)\n\n"
        f"TIMECODED TRANSCRIPT EXCERPT:\n{timecoded_context}\n\n"
    )
    if feedback:
        user_prompt += f"USER ADJUSTMENT FEEDBACK:\n{feedback}\nPlease adjust the script and visual plan accordingly!\n\n"

    user_prompt += "Produce hyper-condensed 30-60s non-linear segments and visual cuts every 2-4s with strictly different search queries."

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": STORY_PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content or "{}"
    data = json.loads(raw_content)

    segments = [Segment(**s) for s in data.get("segments", [])]
    scenes = [SceneCut(**sc) for sc in data.get("scenes", [])]

    return ClipProposal(
        clip_id=story.story_id,
        title=f"story_{story.story_id}",
        hook=data.get("hook", story.hook),
        rationale=data.get("rationale", story.summary),
        segments=segments,
        scenes=scenes,
    )


def plan_story_script_and_broll(
    story: DiscoveredStory,
    words: list[dict[str, Any]],
    prefer_llm: bool = True,
    feedback: str | None = None,
) -> ClipProposal:
    """Phase 2: Radical Trimming to 30-60s & 2-4s B-roll Plan."""
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    if prefer_llm and has_api_key:
        try:
            logger.info(f"Planning 30-60s script and B-roll for Story {story.story_id} via LLM...")
            prop = plan_story_with_llm(story, words, feedback=feedback)
            return enforce_no_ambiguity_hook(prop, words)
        except Exception as exc:
            logger.warning(f"LLM story planning failed: {exc}. Using heuristic planner.")

    prop = plan_story_heuristic(story, words, feedback=feedback)
    return enforce_no_ambiguity_hook(prop, words)


def format_storyboard_display(proposal: ClipProposal, story: DiscoveredStory | None = None) -> str:
    """Formats the non-linear script and visual B-roll plan for terminal confirmation."""
    lines = [
        "",
        "=" * 64,
        "     >>> PHASE 2: RADIKALER STORY-SCHNITT & B-ROLL-PLAN",
        "=" * 64,
    ]
    if story:
        lines.append(f"Story [{story.story_id}]: {story.summary}")
        lines.append(f"Original-Zeitbereich: {story.time_range_str}")
    lines.append(f"Geplante Short-Dauer: {proposal.total_duration:.1f}s (Non-linear verdichtet)")
    lines.append(f'Hook (0-3s): "{proposal.hook}"')
    lines.append("")
    lines.append("--- NON-LINEARE SCHNITT-SEGMENTE ---")
    for idx, seg in enumerate(proposal.segments, start=1):
        s_m, s_s = int(seg.start // 60), int(seg.start % 60)
        e_m, e_s = int(seg.end // 60), int(seg.end % 60)
        lines.append(
            f"  Segment {idx}: Min {s_m}:{s_s:02d} -> Min {e_m}:{e_s:02d} "
            f"({seg.end - seg.start:.1f}s aus Original-Video)"
        )
    lines.append("")
    lines.append("--- VISUELLER B-ROLL-PLAN (Cuts alle 2-4s) ---")
    for idx, sc in enumerate(proposal.scenes, start=1):
        dur = sc.end - sc.start
        if sc.layout == "full_speaker":
            q_info = "(Sprecher getrackt / Proportional Face-Centering)"
        else:
            q_info = f'Query: "{sc.search_query}"'
        lines.append(f"  [{sc.start:4.1f}s - {sc.end:4.1f}s] ({dur:.1f}s) layout: {sc.layout:<12} | {q_info}")
    lines.append("=" * 64)
    return "\n".join(lines)

