# tests/test_assign_speakers.py
"""Tests for foreign_whispers.diarization.assign_speakers (Diarization Task 1)."""

from foreign_whispers.diarization import assign_speakers


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def _diar(start_s, end_s, speaker):
    return {"start_s": start_s, "end_s": end_s, "speaker": speaker}


def test_single_speaker_across_multiple_segments():
    segments = [_seg(0, 3, "Hello"), _seg(3, 6, "World")]
    diarization = [_diar(0, 6, "SPEAKER_00")]
    result = assign_speakers(segments, diarization)
    assert all(s["speaker"] == "SPEAKER_00" for s in result)


def test_multiple_speakers_distinct_ranges():
    segments = [_seg(0, 3, "Hello"), _seg(5, 8, "World")]
    diarization = [_diar(0, 4, "SPEAKER_00"), _diar(4, 9, "SPEAKER_01")]
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_empty_diarization_defaults_to_speaker_00():
    segments = [_seg(0, 3, "Hello"), _seg(3, 6, "World")]
    result = assign_speakers(segments, [])
    assert all(s["speaker"] == "SPEAKER_00" for s in result)


def test_does_not_mutate_input():
    segments = [_seg(0, 3, "Hello")]
    diarization = [_diar(0, 3, "SPEAKER_01")]
    original_seg_text = segments[0]["text"]
    result = assign_speakers(segments, diarization)
    # Original must be unchanged
    assert "speaker" not in segments[0]
    assert segments[0]["text"] == original_seg_text
    # Result must have speaker
    assert result[0]["speaker"] == "SPEAKER_01"


def test_max_overlap_wins():
    """When a segment overlaps two diarization intervals, the one with more overlap wins."""
    segments = [_seg(2, 5, "Test")]
    # 2-4 overlap with SPEAKER_00 = 2s, 4-5 overlap with SPEAKER_01 = 1s
    diarization = [_diar(0, 4, "SPEAKER_00"), _diar(4, 7, "SPEAKER_01")]
    result = assign_speakers(segments, diarization)
    assert result[0]["speaker"] == "SPEAKER_00"
