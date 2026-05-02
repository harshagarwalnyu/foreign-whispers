"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import copy
import logging

logger = logging.getLogger(__name__)


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign speaker labels to transcription segments based on diarization overlap.

    For each transcription segment, finds the diarization interval with maximum
    temporal overlap and assigns that speaker label. If diarization is empty,
    all segments default to "SPEAKER_00".

    Does not mutate input lists — returns a deep copy.

    Args:
        segments: List of ``{start: float, end: float, text: str, ...}`` dicts.
        diarization: List of ``{start_s: float, end_s: float, speaker: str}`` dicts.

    Returns:
        Deep copy of *segments* with a ``speaker`` key added to each dict.
    """
    result = copy.deepcopy(segments)

    if not diarization:
        for seg in result:
            seg["speaker"] = "SPEAKER_00"
        return result

    for seg in result:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for diar in diarization:
            overlap = max(0.0, min(seg_end, diar["end_s"]) - max(seg_start, diar["start_s"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker"]

        seg["speaker"] = best_speaker

    return result


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []
