"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum

_DP_GAP_SHIFT_BOUNDARY: float = 1.55
_STRETCH_QUALITY_WEIGHT: float = 5.0


def _silence_after(end_s: float, silence_regions: list[dict]) -> float:
    """Return duration of the first silence region starting at or after end_s."""
    for r in silence_regions:
        if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
            return r["end_s"] - r["start_s"]
    return 0.0


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


def _estimate_tts_duration(text: str) -> float:
    """Estimate TTS output duration in seconds for target-language text.

    Weighted blend of two complementary heuristics:
    - Syllable rate (0.6 weight): Romance languages ~4.5 syllables/second.
      More linguistically accurate but degrades on long utterances.
    - Character rate (0.4 weight): ~15 chars/second for Spanish TTS.
      More stable for long texts where the syllable counter drifts.
    """
    syllable_estimate = _count_syllables(text) / 4.5
    char_estimate = max(1, len(text.strip())) / 15.0
    return 0.6 * syllable_estimate + 0.4 * char_estimate


class DurationPredictor:
    """Predicts TTS duration for target-language text.

    Two strategies:
    - 'syllable': syllables / syllable_rate  (default, no calibration needed)
    - 'regression': linear model on (syllable_count, char_count) with stubbed
      coefficients — replace COEF_* with values fitted to real TTS ground truth.

    The regression strategy is intentionally stubbed; it requires collecting
    actual TTS WAV durations (raw_duration_s from .align.json files) and fitting
    coefficients with e.g. scipy.stats.linregress. See alignment_integration
    notebook Task 1 for the calibration workflow.
    """

    # Regression coefficients — calibrated via linregress(syllable_counts, actual_durations)
    # on 2187 ground-truth segments from pipeline_data/api/tts_audio/chatterbox/*.align.json
    COEF_SYLLABLE: float = 0.1684         # seconds per syllable
    COEF_CHAR:     float = 0.0            # seconds per character (stub — add after calibration)
    INTERCEPT:     float = 0.6956

    def __init__(self, strategy: str = 'syllable', syllable_rate: float = 4.5) -> None:
        if strategy not in ('syllable', 'regression'):
            raise ValueError(f"Unknown strategy {strategy!r}; choose 'syllable' or 'regression'")
        self.strategy = strategy
        self.syllable_rate = syllable_rate

    def predict(self, text: str) -> float:
        """Return predicted TTS duration in seconds for *text*."""
        n_syl = _count_syllables(text)
        if self.strategy == 'syllable':
            return n_syl / self.syllable_rate
        # regression: weighted combination of syllable count and char count
        return max(0.1, self.COEF_SYLLABLE * n_syl + self.COEF_CHAR * len(text) + self.INTERCEPT)


_DEFAULT_PREDICTOR = DurationPredictor(strategy='syllable')


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        syllables = _count_syllables(self.translated_text)
        self.predicted_tts_s = syllables / 4.5
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
    predictor: 'DurationPredictor | None' = None,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.
        predictor: Optional DurationPredictor instance.  Defaults to the
            syllable-rate heuristic.  Pass DurationPredictor(strategy='regression')
            to use the regression model (requires calibrated coefficients).

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        m = SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        )
        if predictor is not None:
            m.predicted_tts_s = predictor.predict(m.translated_text)
            m.predicted_stretch = (
                m.predicted_tts_s / m.source_duration_s
                if m.source_duration_s > 0 else 1.0
            )
            m.overflow_s = max(0.0, m.predicted_tts_s - m.source_duration_s)
        metrics.append(m)
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end, silence_regions))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def global_align_dp(
    metrics: list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch: float = 1.4,
) -> list[AlignedSegment]:
    """Greedy alignment with DP-based trade-off for GAP_SHIFT vs MILD_STRETCH.

    Improves upon global_align by comparing the global drift impact of
    GAP_SHIFT against the local audio quality impact of MILD_STRETCH.

    When GAP_SHIFT is suggested but would cause excessive drift (where drift
    cost > stretch cost), it overrides to MILD_STRETCH.
    """
    n = len(metrics)
    aligned, cumulative_drift = [], 0.0

    for i, m in enumerate(metrics):
        action = decide_action(m, available_gap_s=_silence_after(m.source_end, silence_regions))
        gap_shift = 0.0
        stretch = 1.0

        if action == AlignAction.GAP_SHIFT and 1.4 <= m.predicted_stretch <= _DP_GAP_SHIFT_BOUNDARY:
            gap_shift_cost = m.overflow_s * (n - i)
            mild_stretch_cost = (m.predicted_stretch - 1.0) * m.source_duration_s * _STRETCH_QUALITY_WEIGHT

            if gap_shift_cost > mild_stretch_cost:
                action = AlignAction.MILD_STRETCH
                stretch = min(m.predicted_stretch, max_stretch)
            else:
                gap_shift = m.overflow_s
        elif action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)

        sched_start = m.source_start + cumulative_drift
        sched_end = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index=m.index,
            original_start=m.source_start,
            original_end=m.source_end,
            scheduled_start=sched_start,
            scheduled_end=sched_end,
            text=m.translated_text,
            action=action,
            gap_shift_s=gap_shift,
            stretch_factor=stretch,
        ))

        cumulative_drift += gap_shift

    return aligned
