"""Hierarchical narrative memory.

Layers (design doc §1.2):
    shot ⇢ event ⇢ story ⇢ character
Stored across three backends:
    • SQLite (stdlib)  — relational metadata
    • FAISS (faiss-cpu) — CLIP embedding ANN index   [optional; v0.1 falls back to numpy]
    • .npz files       — heavy ShotFeatures arrays
"""
from .schema import StoreSchemas
from .store import MemoryStore
from .builder import build_memory_from_shots
from .retriever import MemoryRetriever
from .lessons import Lesson, LessonBook

__all__ = [
    "StoreSchemas", "MemoryStore", "MemoryRetriever", "build_memory_from_shots",
    "Lesson", "LessonBook",
]
