# tests/test_diarize_router.py
"""Tests for POST /api/diarize/{video_id} endpoint."""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from tests.conftest import stub_gpu_models
    stub_gpu_models(monkeypatch)
    from api.src.main import app
    with TestClient(app) as c:
        yield c


def test_diarize_route_registered(client):
    """The diarize endpoint should be registered in OpenAPI schema."""
    schema = client.get("/openapi.json").json()
    paths = list(schema["paths"].keys())
    assert any("/api/diarize" in p for p in paths)


def test_diarize_unknown_video_returns_404(client):
    resp = client.post("/api/diarize/nonexistent-id")
    assert resp.status_code == 404


def test_diarize_returns_cached(client, tmp_path, monkeypatch):
    """When cached JSON without a skipped flag exists, skipped defaults to False.

    This covers the case where real diarization ran and the cache was written
    with skipped=False (or pre-fix caches missing the key entirely).
    """
    from api.src.core import config

    # Point diarizations_dir to tmp
    monkeypatch.setattr(type(config.settings), "diarizations_dir",
                        property(lambda self: tmp_path))

    # Create cached result without a skipped key (pre-fix legacy format)
    cached = {"speakers": ["SPEAKER_00"], "segments": [{"start_s": 0, "end_s": 3, "speaker": "SPEAKER_00"}]}
    (tmp_path / "test_video.json").write_text(json.dumps(cached))

    # Patch resolve_title to return our test title
    with patch("api.src.routers.diarize.resolve_title", return_value="test_video"):
        resp = client.post("/api/diarize/test-id")

    assert resp.status_code == 200
    data = resp.json()
    # Legacy cache without "skipped" key → defaults to False (real diarization ran)
    assert data["skipped"] is False
    assert data["speakers"] == ["SPEAKER_00"]


def test_diarize_returns_cached_skipped(client, tmp_path, monkeypatch):
    """When cached JSON has skipped=True (pyannote unavailable), return skipped=True."""
    from api.src.core import config

    monkeypatch.setattr(type(config.settings), "diarizations_dir",
                        property(lambda self: tmp_path))

    cached = {"speakers": [], "segments": [], "skipped": True}
    (tmp_path / "test_video.json").write_text(json.dumps(cached))

    with patch("api.src.routers.diarize.resolve_title", return_value="test_video"):
        resp = client.post("/api/diarize/test-id")

    assert resp.status_code == 200
    data = resp.json()
    assert data["skipped"] is True
    assert data["speakers"] == []


def test_diarize_response_schema(client, tmp_path, monkeypatch):
    """Response should match DiarizeResponse schema."""
    from api.src.core import config

    monkeypatch.setattr(type(config.settings), "diarizations_dir",
                        property(lambda self: tmp_path))

    cached = {"speakers": [], "segments": [], "skipped": True}
    (tmp_path / "test_video.json").write_text(json.dumps(cached))

    with patch("api.src.routers.diarize.resolve_title", return_value="test_video"):
        resp = client.post("/api/diarize/test-id")

    data = resp.json()
    assert "video_id" in data
    assert "speakers" in data
    assert "segments" in data
    assert "skipped" in data
