"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
)

_MAX_DURATION_ERROR_S: float = 3.0
_DRIFT_BUDGET_FRACTION: float = 0.05
_CV_NATURALNESS_SCALE: float = 2.0


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
) -> dict:
    """Return a scorecard dict for alignment quality and performance.

    New stub dimensions (return None until implemented):
        intelligibility: None (stub) — TTS->STT round-trip WER; requires GPU speech services.
        semantic_fidelity: None (stub) — embedding cosine similarity EN<->back-translated EN; requires LLM.
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
        "intelligibility": None,
        "semantic_fidelity": None,
        "overall_score": round(overall, 3),
        "grade": grade,
        "report": align_report,
    }

