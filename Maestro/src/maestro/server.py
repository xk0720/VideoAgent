"""FastAPI deployment shim — `upload to server and run`.

Borrowed (cite): UniVA's `univa_server.py` (FastAPI on :8000 with /health).
Maestro adds /tools (registry manifest) and runs jobs in-process with a simple
threadpool so we don't drag in Celery/Redis for a v0.2.2 deploy.

Run:
    uvicorn maestro.server:app --host 0.0.0.0 --port 8000
or via the CLI helper:
    python -m maestro.cli serve --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                → liveness probe (UniVA-compatible)
    GET  /tools                  → tool manifest from default_registry
    POST /generate {prompt, ...} → enqueue a generation job, returns job_id
    GET  /jobs/{job_id}          → poll job status / result

Note: we intentionally do NOT use `from __future__ import annotations` in this
file — pydantic v2 + FastAPI need real (non-stringified) class refs to bind a
locally-defined request model to the request body without a ForwardRef dance.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

from .config import load_config
from .pipeline.run import run_maestro
from .tools.base import default_registry

# Optional dep probe (pyflakes-clean; no actual import needed just to detect).
_HAS_FASTAPI = find_spec("fastapi") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Job tracker — process-local, in-memory. Good enough for single-replica v0.2.2;
# v0.3 would back this with Redis + worker queue. The interface is stable so the
# upgrade is local to this module.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    job_id: str
    state: str = "queued"          # queued | running | done | error
    prompt: str = ""
    output_path: Optional[str] = None
    report: Optional[dict] = None
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)   # CPU pipeline; bump for real backends

    def submit(self, prompt: str, output_root: Path, **kwargs) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, prompt=prompt)
        with self._lock:
            self._jobs[job_id] = job
        fut = self._pool.submit(self._run, job_id, prompt, output_root, kwargs)
        with self._lock:
            self._futures[job_id] = fut
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, prompt: str, output_root: Path, kwargs: dict) -> None:
        job = self._jobs[job_id]
        job.state = "running"
        job.started_at = time.time()
        try:
            out_path = output_root / f"{job_id}.mp4"
            result = run_maestro(
                user_prompt=prompt,
                output_path=out_path,
                config=load_config(),
                cache_dir=output_root / job_id / "cache",
                trajectory_path=output_root / f"{job_id}.trajectory.jsonl",
                lesson_path=output_root / "lessons.jsonl",
                **kwargs,
            )
            job.output_path = str(result["output_path"])
            job.report = result["report"]
            job.state = "done"
        except Exception as e:                              # pragma: no cover
            job.error = repr(e)
            job.state = "error"
        finally:
            job.finished_at = time.time()


_STORE: Optional[JobStore] = None


def store() -> JobStore:
    global _STORE
    if _STORE is None:
        _STORE = JobStore()
    return _STORE


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI wiring — only instantiated if fastapi is installed (gated optional).
# ─────────────────────────────────────────────────────────────────────────────
def create_app(output_root: str | Path = "outputs"):
    """Construct the FastAPI app. Lazy so importing this module on a box without
    fastapi installed doesn't crash (e.g. CI running plain pytest)."""
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "fastapi is not installed. `pip install fastapi uvicorn` or use the CLI "
            "(`python -m maestro.cli run-once`) for non-server inference."
        )
    # Import inside the function so pydantic BaseModel resolves to the real
    # pydantic class at class-definition time. `Body(...)` is required because
    # `from __future__ import annotations` (top of this file) stringifies the
    # endpoint's parameter annotation, defeating FastAPI's body-vs-query
    # inference and otherwise causing a 422 "field required in query".
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel

    out_root = Path(output_root); out_root.mkdir(parents=True, exist_ok=True)

    class GenerateRequest(BaseModel):
        prompt: str
        source_videos: list[str] = []
        images: list[str] = []
        music: Optional[str] = None

    app = FastAPI(title="Maestro", version="0.2.2")

    @app.get("/health")
    def health() -> dict:
        # UniVA-compatible shape: short, JSON, no auth needed.
        return {
            "status": "ok",
            "service": "maestro",
            "version": "0.2.2",
            "n_tools": len(default_registry().names()),
        }

    @app.get("/tools")
    def tools() -> list[dict]:
        return [asdict(s) for s in default_registry().list_specs()]

    @app.post("/generate")
    def generate(req: GenerateRequest = Body(...)) -> dict:
        job = store().submit(
            prompt=req.prompt,
            output_root=out_root,
            source_videos=[Path(p) for p in req.source_videos] or None,
            images=[Path(p) for p in req.images] or None,
            music=Path(req.music) if req.music else None,
        )
        return {"job_id": job.job_id, "state": job.state}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return asdict(job)

    return app


# Lazily expose a module-level `app` so `uvicorn maestro.server:app` works
# when fastapi is available. Falls back to a plain object so `import` is safe.
if _HAS_FASTAPI:
    try:
        app = create_app()
    except Exception:                                        # pragma: no cover
        app = None                                           # type: ignore
else:
    app = None                                               # type: ignore
