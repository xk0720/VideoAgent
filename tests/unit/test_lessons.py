"""LessonBook persistence + retrieval tests."""
from __future__ import annotations

import json
from pathlib import Path

from longvideoagent.memory.lessons import LessonBook


def test_add_and_reload(tmp_path: Path):
    p = tmp_path / "lessons.jsonl"
    book = LessonBook(p)
    book.add_simple(trigger="low_reward", scope="editor",
                    lesson_text="Avoid query X when memory is small",
                    context={"memory_size": 4})
    book.add_simple(trigger="disagreement", scope="validator",
                    lesson_text="Judge and metric mean diverged")
    assert len(book) == 2

    # New LessonBook over the same file must re-read the lessons.
    book2 = LessonBook(p)
    assert len(book2) == 2
    assert book2.all()[0].trigger == "low_reward"


def test_filter_by_scope(tmp_path: Path):
    book = LessonBook(tmp_path / "lessons.jsonl")
    book.add_simple(trigger="a", scope="editor", lesson_text="x")
    book.add_simple(trigger="b", scope="director", lesson_text="y")
    assert len(book.filter(scope="editor")) == 1
    assert len(book.filter(scope="director")) == 1


def test_retrieve_relevant_keyword_ranking(tmp_path: Path):
    book = LessonBook(tmp_path / "lessons.jsonl")
    book.add_simple(trigger="x", scope="screenwriter",
                    lesson_text="prefer specific shot descriptions over generic phrases",
                    context={"user_prompt_keywords": ["chase", "action"]})
    book.add_simple(trigger="x", scope="screenwriter",
                    lesson_text="match cinematography hints to music energy",
                    context={"user_prompt_keywords": ["calm", "romantic"]})
    hits = book.retrieve_relevant("screenwriter", keywords=["chase"], limit=5)
    # First hit should be the chase-related lesson.
    assert "shot descriptions" in hits[0].lesson


def test_malformed_lines_are_skipped(tmp_path: Path):
    p = tmp_path / "lessons.jsonl"
    p.write_text("{not json}\n" + json.dumps({
        "lesson_id": "abc", "created_at": 1.0, "trigger": "x", "scope": "global",
        "context": {}, "lesson": "valid", "evidence": {},
    }) + "\n")
    book = LessonBook(p)
    assert len(book) == 1
    assert book.all()[0].lesson == "valid"
