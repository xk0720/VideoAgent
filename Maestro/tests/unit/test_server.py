"""Server smoke test.

The FastAPI server is an OPTIONAL piece (only needed if you `python -m
maestro.cli serve`). The tests below skip cleanly when fastapi is not on the
box — but when it IS installed, they verify the UniVA-compatible /health
shape, the /tools manifest, and that /generate enqueues a job that eventually
finishes.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from maestro.server import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(output_root=tmp_path)
    return TestClient(app)


def test_health_returns_ok(client):
    """UniVA's /health contract: 200 + JSON with status==ok."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "maestro"
    assert body["n_tools"] >= 7   # at least the new tool batch


def test_tools_endpoint_lists_categories(client):
    r = client.get("/tools")
    assert r.status_code == 200
    specs = r.json()
    cats = {s["category"] for s in specs}
    for required in ("analysis", "generation", "editing", "tracking",
                     "metric"):
        assert required in cats, cats


def test_generate_enqueues_and_completes(client):
    r = client.post("/generate", json={"prompt": "a ball is thrown"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert r.json()["state"] in {"queued", "running", "done"}

    # Poll up to 15s for the mock pipeline to finish.
    final = None
    for _ in range(60):
        r2 = client.get(f"/jobs/{job_id}")
        assert r2.status_code == 200
        final = r2.json()
        if final["state"] in {"done", "error"}:
            break
        time.sleep(0.25)
    assert final is not None
    assert final["state"] == "done", final
    assert final["report"]["n_shots"] >= 1


def test_jobs_404_for_unknown(client):
    r = client.get("/jobs/zzzznotreal")
    assert r.status_code == 404
