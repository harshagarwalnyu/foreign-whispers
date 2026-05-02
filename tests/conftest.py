"""Shared test fixtures."""

import importlib
from unittest.mock import MagicMock


def stub_gpu_models(monkeypatch):
    """Stub whisper and TTS model loading when GPU packages aren't installed."""
    if importlib.util.find_spec("whisper"):
        monkeypatch.setattr("whisper.load_model", lambda *a, **kw: MagicMock())
    if importlib.util.find_spec("TTS"):
        monkeypatch.setattr("TTS.api.TTS", lambda *a, **kw: MagicMock())
