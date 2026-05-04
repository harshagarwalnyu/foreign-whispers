"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import difflib as _difflib
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
)

_MAX_DURATION_ERROR_S: float = 3.0
_DRIFT_BUDGET_FRACTION: float = 0.05
_CV_NATURALNESS_SCALE: float = 2.0


def _semantic_fidelity(metrics: list) -> float | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as _np
        pairs = [(m.source_text, m.translated_text) for m in metrics if m.source_text and m.translated_text]
        if not pairs:
            return None
        src, tgt = zip(*pairs)
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        mat = vec.fit_transform(list(src) + list(tgt))
        n = len(src)
        sims = [float(cosine_similarity(mat[i], mat[n + i])[0, 0]) for i in range(n)]
        return round(float(_np.mean(sims)), 3)
    except Exception:
        return None


def _intelligibility(metrics: list, tts_wav_path: str | None, whisper_url: str | None) -> float | None:
    if not tts_wav_path or not whisper_url:
        return None
    import pathlib
    import requests
    wav = pathlib.Path(tts_wav_path)
    if not wav.exists():
        return None
    try:
        with open(wav, 'rb') as f:
            resp = requests.post(
                f"{whisper_url}/v1/audio/transcriptions",
                files={'file': (wav.name, f, 'audio/wav')},
                data={'model': 'whisper-1', 'language': 'es'},
                timeout=120,
            )
        resp.raise_for_status()
        stt_text = resp.json().get('text', '')
        ref_words = ' '.join(m.translated_text for m in metrics if m.translated_text).lower().split()
        hyp_words = stt_text.lower().split()
        if not ref_words:
            return None
        sm = _difflib.SequenceMatcher(None, ref_words, hyp_words)
        correct = sum(i2 - i1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag == 'equal')
        return round(max(0.0, 1.0 - (len(ref_words) - correct) / len(ref_words)), 3)
    except Exception:
        return None


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for a in aligned if a.action == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned_segments: list[AlignedSegment],
    align_report: dict,
    *,
    tts_wav_path: str | None = None,
    whisper_url: str | None = None,
) -> dict:
    """Return a scorecard dict for alignment quality and performance.

    intelligibility: WER-based score from TTS->STT round-trip. Pass tts_wav_path + whisper_url to activate.
    semantic_fidelity: Char n-gram TF-IDF cosine similarity between source_text and translated_text per segment.
    """
    n = max(len(metrics), 1)

    timing_accuracy = max(0.0, 1.0 - align_report.get("mean_abs_duration_error_s", 0.0) / _MAX_DURATION_ERROR_S)
    stretch_quality = max(0.0, 1.0 - align_report.get("pct_severe_stretch", 0.0) / 100.0)
    gap_efficiency = max(0.0, 1.0 - align_report.get("n_gap_shifts", 0) / n)
    retry_rate = max(0.0, 1.0 - align_report.get("n_translation_retries", 0) / n)

    total_video_duration_s = (
        metrics[-1].source_end - metrics[0].source_start if metrics else 1.0
    )
    drift_limit = max(total_video_duration_s * _DRIFT_BUDGET_FRACTION, 1.0)
    drift_score = max(
        0.0, 1.0 - abs(align_report.get("total_cumulative_drift_s", 0.0)) / drift_limit
    )

    speaking_rates = [
        m.tgt_char_count / m.source_duration_s for m in metrics if m.source_duration_s > 0
    ]
    if len(speaking_rates) < 2:
        naturalness = 1.0
    else:
        mu = _stats.mean(speaking_rates)
        cv = _stats.stdev(speaking_rates) / mu
        naturalness = max(0.0, 1.0 - cv / _CV_NATURALNESS_SCALE)

    overall = (
        timing_accuracy * 0.3
        + stretch_quality * 0.2
        + gap_efficiency * 0.15
        + retry_rate * 0.15
        + drift_score * 0.1
        + naturalness * 0.1
    )

    if overall >= 0.85:
        grade = "A"
    elif overall >= 0.70:
        grade = "B"
    elif overall >= 0.55:
        grade = "C"
    elif overall >= 0.40:
        grade = "D"
    else:
        grade = "F"

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "stretch_quality": round(stretch_quality, 3),
        "gap_efficiency": round(gap_efficiency, 3),
        "retry_rate": round(retry_rate, 3),
        "drift_score": round(drift_score, 3),
        "naturalness": round(naturalness, 3),
        "intelligibility": _intelligibility(metrics, tts_wav_path, whisper_url),
        "semantic_fidelity": _semantic_fidelity(metrics),
        "overall_score": round(overall, 3),
        "grade": grade,
        "report": align_report,
    }

