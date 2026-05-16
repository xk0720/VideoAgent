"""SQL schemas for the narrative memory store.

Open-source dependency: ``sqlite3`` from the Python standard library.

Rationale (design doc §13 step 4): keep metadata in SQLite (small, queryable),
push CLIP embeddings to FAISS (or numpy), and dump heavy per-shot ndarrays
to ``.npz`` so memory.store can lazy-load.
"""
from __future__ import annotations


SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS shots (
        shot_id          TEXT PRIMARY KEY,
        source_video     TEXT NOT NULL,
        start_time       REAL NOT NULL,
        end_time         REAL NOT NULL,
        caption          TEXT DEFAULT '',
        shot_scale       TEXT DEFAULT 'medium',
        shot_movement    TEXT DEFAULT 'static',
        shot_angle       TEXT DEFAULT 'eye_level',
        framing          TEXT DEFAULT 'single',
        avg_flow_mag     REAL DEFAULT 0.0,
        dialogue         TEXT,
        feature_path     TEXT          -- relative path to .npz
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_shots_source ON shots(source_video);
    """,
    """
    CREATE TABLE IF NOT EXISTS shot_characters (
        shot_id          TEXT NOT NULL,
        character_id     TEXT NOT NULL,
        PRIMARY KEY (shot_id, character_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id    TEXT PRIMARY KEY,
        summary     TEXT,
        start_time  REAL,
        end_time    REAL,
        shot_ids    TEXT     -- JSON array string
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS stories (
        story_id   TEXT PRIMARY KEY,
        title      TEXT,
        summary    TEXT,
        arc_role   TEXT,
        event_ids  TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS characters (
        character_id        TEXT PRIMARY KEY,
        name                TEXT,
        appearance_shot_ids TEXT,
        profile_summary     TEXT,
        face_emb_path       TEXT,
        voice_emb_path      TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS music (
        audio_path   TEXT PRIMARY KEY,
        duration     REAL,
        bpm          REAL,
        beats_path   TEXT,
        sections     TEXT   -- JSON array
    );
    """,
]


class StoreSchemas:
    """Tiny wrapper so callers can ``StoreSchemas.apply(conn)``."""

    @staticmethod
    def apply(conn) -> None:  # ``conn`` is sqlite3.Connection
        cur = conn.cursor()
        for stmt in SCHEMA_SQL:
            cur.executescript(stmt)
        conn.commit()
