# tests/test_voice_resolution.py
"""Tests for foreign_whispers.voice_resolution.resolve_speaker_wav (TTS Task 2)."""

import pytest
from foreign_whispers.voice_resolution import resolve_speaker_wav


def test_speaker_specific_wav(tmp_path):
    """Speaker-specific WAV is selected when available."""
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "SPEAKER_00.wav").write_bytes(b"fake")
    result = resolve_speaker_wav(tmp_path, "es", speaker_id="SPEAKER_00")
    assert result == "es/SPEAKER_00.wav"


def test_fallback_to_language_default(tmp_path):
    """Falls back to language default when speaker-specific WAV is missing."""
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "default.wav").write_bytes(b"fake")
    result = resolve_speaker_wav(tmp_path, "es", speaker_id="SPEAKER_99")
    assert result == "es/default.wav"


def test_fallback_to_global_default(tmp_path):
    """Falls back to global default when language directory is empty/missing."""
    (tmp_path / "default.wav").write_bytes(b"fake")
    result = resolve_speaker_wav(tmp_path, "fr")
    assert result == "default.wav"


def test_default_without_speaker_id(tmp_path):
    """When speaker_id is None, goes straight to language default."""
    (tmp_path / "es").mkdir()
    (tmp_path / "es" / "default.wav").write_bytes(b"fake")
    result = resolve_speaker_wav(tmp_path, "es")
    assert result == "es/default.wav"


def test_unknown_language_falls_to_global(tmp_path):
    """Unknown language with no directory falls back to global default."""
    (tmp_path / "default.wav").write_bytes(b"fake")
    result = resolve_speaker_wav(tmp_path, "zz", speaker_id="SPEAKER_00")
    assert result == "default.wav"


def test_no_wav_raises_file_not_found(tmp_path):
    """Raises FileNotFoundError when no suitable WAV exists at any level."""
    with pytest.raises(FileNotFoundError):
        resolve_speaker_wav(tmp_path, "es", speaker_id="SPEAKER_00")
