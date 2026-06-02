"""Memory subsystem. AssetMemory lives in types; this package adds the
cross-task LessonLibrary (C4) and a tiny persistence helper."""
from .lesson_library import LessonLibrary

__all__ = ["LessonLibrary"]
