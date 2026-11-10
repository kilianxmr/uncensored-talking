"""
Tests for Interactive Human-in-the-Loop Pipeline:
- Phase 1: Story Discovery (variable count, exact terminal format, time_range_str)
- Phase 2: Radical Cutting & Visual B-Roll Plan (30-60s, 2-4s cuts, diverse queries)
- Phase 3: B-Roll Deduplication & Used Asset IDs set
- Output Filename Naming (output/story_[id].mp4)
"""

import json
from pathlib import Path
import pytest

from src.selector import (
    DiscoveredStory,
    discover_stories_heuristic,
    plan_story_heuristic,
    format_storyboard_display,
    ClipProposal,
    Segment,
    SceneCut,
)
from src.broll_engine import source_broll_for_clip


def test_discovered_story_formatting():
    story = DiscoveredStory(
        story_id=1,
        summary="Terry berichtet über dubiose Spenden von Pharma-Lobbyisten und wie Gesetze gezielt gekauft wurden.",
        hook="Wait, so you're saying Terry is taking money directly from them?",
        start_sec=735.0,  # Min 12:15
        end_sec=1120.0,   # Min 18:40
    )
    assert story.time_range_str == "Min 12:15 - Min 18:40"
    disp = story.to_terminal_display()
    assert "Story [1]: Terry berichtet" in disp
    assert 'Hook: "Wait, so you\'re saying Terry is taking money directly from them?"' in disp
    assert "Original-Zeitbereich: Min 12:15 - Min 18:40" in disp


def test_story_discovery_variable_count():
    # Build 10-minute mock transcript (600 seconds)
    words = []
    t = 0.0
    for i in range(700):
        w = f"word_{i}"
        if i % 80 == 0:
            w = "Trump?"
        elif i % 150 == 0:
            w = "Pharma!"
        words.append({
            "word": w,
            "start": round(t, 2),
            "end": round(t + 0.5, 2),
        })
        t += 0.85

    stories = discover_stories_heuristic(words)
    # Must NOT force a fixed count (e.g. not strictly hardcoded to 3)
    assert len(stories) >= 2
    for idx, s in enumerate(stories, start=1):
        assert s.story_id == idx
        assert len(s.summary) > 10
        assert len(s.hook) > 0
        assert 0.0 <= s.start_sec < s.end_sec


def test_plan_story_heuristic_constraints():
    story = DiscoveredStory(
        story_id=2,
        summary="Robert F Kennedy deckt Korruption im FDA-Zulassungsprozess auf.",
        hook="The FDA gets 75 percent of its drug review budget from Big Pharma.",
        start_sec=100.0,
        end_sec=350.0,
    )
    words = []
    t = 90.0
    topics = ["Wait", "Terry", "Kennedy", "FDA", "drug", "money", "pharma", "scandal", "truth", "exposed"]
    for i in range(300):
        words.append({
            "word": topics[i % len(topics)],
            "start": round(t, 2),
            "end": round(t + 0.45, 2),
        })
        t += 0.65

    proposal = plan_story_heuristic(story, words)

    # 1. Radical Trimming: 30 to 60s
    assert 30.0 <= proposal.total_duration <= 60.0
    # 2. Hook must be present
    assert len(proposal.hook) > 0

    # 3. Visual B-Roll Plan: cuts every 2-4 seconds
    assert len(proposal.scenes) >= 6
    scene_queries = []
    for sc in proposal.scenes:
        dur = sc.end - sc.start
        assert dur <= 4.0
        assert sc.layout in ("splitscreen", "full_speaker", "full_broll")
        if sc.search_query:
            scene_queries.append(sc.search_query.lower())

    # 4. Strict Query Deduplication: no duplicate search queries!
    assert len(scene_queries) == len(set(scene_queries)), "Every visual scene query must be distinct!"


def test_format_storyboard_display():
    story = DiscoveredStory(
        story_id=3,
        summary="Anekdote über geheime Deals im Oval Office.",
        hook="Did he really sign that executive order at 3 AM?",
        start_sec=300.0,
        end_sec=480.0,
    )
    proposal = ClipProposal(
        clip_id=3,
        title="story_3",
        hook=story.hook,
        rationale=story.summary,
        segments=[
            Segment(start=300.0, end=303.5),
            Segment(start=340.0, end=375.0),
        ],
        scenes=[
            SceneCut(start=0.0, end=3.5, layout="splitscreen", search_query="Donald Trump oval office retro"),
            SceneCut(start=3.5, end=7.0, layout="full_speaker", search_query=""),
            SceneCut(start=7.0, end=10.5, layout="splitscreen", search_query="Trump walking helicopter marine one"),
        ],
    )
    display_str = format_storyboard_display(proposal, story)
    assert "PHASE 2: RADIKALER STORY-SCHNITT" in display_str
    assert "Story [3]:" in display_str
    assert "Dauer: 38.5s" in display_str
    assert "layout: splitscreen" in display_str
    assert "layout: full_speaker" in display_str


def test_strict_broll_query_diversification_and_guaranteed_plates(tmp_path: Path):
    # Two scenes with identical initial query
    clip = ClipProposal(
        clip_id=1,
        title="story_1",
        hook="Shocking revelation",
        segments=[Segment(start=0.0, end=10.0)],
        scenes=[
            SceneCut(start=0.0, end=3.0, layout="splitscreen", search_query="Trump Oval Office"),
            SceneCut(start=3.0, end=6.0, layout="full_speaker", search_query=""),
            SceneCut(start=6.0, end=9.0, layout="splitscreen", search_query="Trump Oval Office"),  # duplicate query!
        ],
    )

    source_broll_for_clip(clip, temp_dir=tmp_path)

    # 1. Verify queries were diversified
    q1 = clip.scenes[0].search_query
    q3 = clip.scenes[2].search_query
    assert q1 != q3, "Scene queries for same topic must be differentiated!"

    # 2. Verify visual assets were guaranteed (not reverted to empty / full_speaker)
    assert clip.scenes[0].layout == "splitscreen"
    assert clip.scenes[0].broll_asset_path is not None
    assert Path(clip.scenes[0].broll_asset_path).is_file()

    assert clip.scenes[2].layout == "splitscreen"
    assert clip.scenes[2].broll_asset_path is not None
    assert Path(clip.scenes[2].broll_asset_path).is_file()

    # 3. Verify deduplication: plates must be distinct files
    assert clip.scenes[0].broll_asset_path != clip.scenes[2].broll_asset_path
