from maestro.memory.lesson_library import LessonLibrary
from maestro.types import PhysFailureMode


def test_add_and_retrieve(tmp_path):
    lib = LessonLibrary(tmp_path / "lessons.jsonl")
    lib.add("a ball is thrown and bounces", "add gravity arc constraint",
            PhysFailureMode.GRAVITY_INERTIA)
    hits = lib.retrieve("ball thrown into the air", top_k=3)
    assert hits
    assert "gravity" in hits[0].fix


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "lessons.jsonl"
    lib = LessonLibrary(p)
    lib.add("water pours from a cup", "enforce fluid continuity", PhysFailureMode.FLUID)
    reloaded = LessonLibrary(p)
    assert len(reloaded) == 1
    assert reloaded.lessons[0].failure_mode == PhysFailureMode.FLUID


def test_irrelevant_query_returns_nothing(tmp_path):
    lib = LessonLibrary(tmp_path / "l.jsonl")
    lib.add("water pours from a cup", "fluid continuity", PhysFailureMode.FLUID)
    assert lib.retrieve("zzzzz qqqqq", threshold=0.5) == []
