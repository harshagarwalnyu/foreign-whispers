import pytest
from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    compute_segment_metrics,
    decide_action,
    global_align,
    global_align_dp,
)


def _make_metrics(src_dur: float, tgt_chars: int) -> SegmentMetrics:
    text = "ba" * tgt_chars  # each "ba" = 1 vowel cluster = 1 syllable
    return SegmentMetrics(
        index=0,
        source_start=0.0,
        source_end=src_dur,
        source_duration_s=src_dur,
        source_text="x" * 10,
        translated_text=text,
        src_char_count=10,
        tgt_char_count=len(text),
    )


def test_syllable_count_simple():
    # "hola mundo" → ho-la-mun-do = 4 syllables
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("hola mundo") == 4


def test_syllable_count_accents():
    # "cómo están" → có-mo-es-tán = 4 syllables
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("cómo están") == 4


def test_syllable_count_empty_string():
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("") == 1  # floor prevents zero-division in predicted_tts_s


def test_syllable_count_punctuation_only():
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("...") == 1  # no vowels → floor returns 1


def test_syllable_count_consonants_only():
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("grr") == 1  # no vowels → floor returns 1


def test_segment_metrics_predicted_tts_syllable_based():
    # "hola mundo" = 4 syllables → 4/4.5 ≈ 0.889s
    m = SegmentMetrics(
        index=0, source_start=0.0, source_end=2.0, source_duration_s=2.0,
        source_text="hello world", translated_text="hola mundo",
        src_char_count=11, tgt_char_count=10,
    )
    assert m.predicted_tts_s == pytest.approx(4 / 4.5, rel=0.01)


def test_segment_metrics_predicted_tts():
    # "ba" * 30 → 30 syllables → 30/4.5 ≈ 6.667s
    m = _make_metrics(src_dur=3.0, tgt_chars=30)
    assert m.predicted_tts_s == pytest.approx(30 / 4.5, rel=0.01)


def test_segment_metrics_predicted_stretch():
    # "ba" * 30 → 30/4.5 ≈ 6.667s vs 2.0s → stretch ≈ 3.33
    m = _make_metrics(src_dur=2.0, tgt_chars=30)
    assert m.predicted_stretch == pytest.approx((30 / 4.5) / 2.0, rel=0.01)


def test_segment_metrics_overflow():
    # "ba" * 60 → 60/4.5 ≈ 13.33s predicted, 2.0s budget → overflow ≈ 11.33s
    m = _make_metrics(src_dur=2.0, tgt_chars=60)
    assert m.overflow_s == pytest.approx((60 / 4.5) - 2.0, rel=0.01)


def test_decide_action_accept():
    # stretch <= 1.1  → N/4.5 / 3.0 <= 1.1  → N <= 14.85  → tgt_chars=14
    assert decide_action(_make_metrics(3.0, 14)) == AlignAction.ACCEPT


def test_decide_action_mild_stretch():
    # 1.1 < s <= 1.4  → 14.85 < N <= 18.9   → tgt_chars=17
    assert decide_action(_make_metrics(3.0, 17)) == AlignAction.MILD_STRETCH


def test_decide_action_gap_shift():
    # 1.4 < s <= 1.8  → 18.9  < N <= 24.3   → tgt_chars=22 (with gap)
    m = _make_metrics(3.0, 22)
    assert decide_action(m, available_gap_s=2.0) == AlignAction.GAP_SHIFT


def test_decide_action_request_shorter():
    # 1.8 < s <= 2.5  → 24.3  < N <= 33.75  → tgt_chars=27
    assert decide_action(_make_metrics(3.0, 27)) == AlignAction.REQUEST_SHORTER


def test_decide_action_fail():
    # s > 2.5  → N > 33.75  → tgt_chars=35
    assert decide_action(_make_metrics(3.0, 35)) == AlignAction.FAIL


def test_compute_segment_metrics_length():
    en = {"segments": [
        {"start": 0.0, "end": 3.0, "text": " Hello world"},
        {"start": 3.0, "end": 6.0, "text": " How are you"},
    ]}
    es = {"segments": [
        {"start": 0.0, "end": 3.0, "text": " Hola mundo"},
        {"start": 3.0, "end": 6.0, "text": " Como estas"},
    ]}
    metrics = compute_segment_metrics(en, es)
    assert len(metrics) == 2
    assert metrics[0].index == 0
    assert metrics[1].index == 1


def test_compute_segment_metrics_text_stripped():
    en = {"segments": [{"start": 0.0, "end": 2.0, "text": "  hi  "}]}
    es = {"segments": [{"start": 0.0, "end": 2.0, "text": "  hola  "}]}
    m = compute_segment_metrics(en, es)[0]
    assert m.source_text == "hi"
    assert m.translated_text == "hola"


def test_global_align_accept_no_drift():
    en = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hello"}]}
    es = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hola"}]}
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    assert aligned[0].scheduled_start == pytest.approx(0.0)
    assert aligned[0].action == AlignAction.ACCEPT


def test_global_align_gap_shift_accumulates_drift():
    en = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "x"},
        {"start": 2.0, "end": 4.0, "text": "x"},
    ]}
    es = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "ba" * 7},   # 7 syl/4.5 ≈ 1.56s in 1.0s → stretch 1.56 → GAP_SHIFT
        {"start": 2.0, "end": 4.0, "text": "ba" * 4},   # 4 syl/4.5 ≈ 0.89s in 2.0s → ACCEPT
    ]}
    silence = [{"start_s": 1.0, "end_s": 3.0, "label": "silence"}]
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=silence)
    assert aligned[0].action == AlignAction.GAP_SHIFT
    assert aligned[1].scheduled_start > aligned[1].original_start

def test_global_align_dp_accept_identical_to_greedy():
    # Single ACCEPT segment (src_dur=3.0, tgt_chars=14 → stretch ≈ 14/4.5 / 3.0 ≈ 1.037 ≤ 1.1)
    m = _make_metrics(3.0, 14)
    m.index = 0
    metrics = [m]
    aligned_greedy = global_align(metrics, silence_regions=[])
    aligned_dp = global_align_dp(metrics, silence_regions=[])
    
    assert aligned_greedy[0].action == AlignAction.ACCEPT
    assert aligned_dp[0].action == AlignAction.ACCEPT
    assert aligned_dp[0].scheduled_start == 0.0
    assert aligned_dp[0].scheduled_start == aligned_greedy[0].scheduled_start

def test_global_align_dp_overrides_gap_shift_to_mild_stretch_when_drift_high():
    metrics = []
    silence = []
    for i in range(10):
        # src_dur=3.0, tgt_chars=20 → stretch = 20/4.5 / 3.0 = 1.481 (between 1.4 and 1.55)
        m = _make_metrics(3.0, 20)
        m.index = i
        m.source_start = i * 3.0
        m.source_end = (i + 1) * 3.0
        metrics.append(m)
        silence.append({'start_s': (i + 1) * 3.0, 'end_s': (i + 1) * 3.0 + 2.0, 'label': 'silence'})
    
    aligned = global_align_dp(metrics, silence_regions=silence)
    
    # Segment 0: gap_shift_cost > mild_stretch_cost → MILD_STRETCH
    assert aligned[0].action == AlignAction.MILD_STRETCH
    assert aligned[0].stretch_factor == pytest.approx(1.4)

def test_global_align_dp_keeps_gap_shift_when_last_segment():
    # Single borderline GAP_SHIFT segment (src_dur=3.0, tgt_chars=20, stretch ≈ 1.481)
    m = _make_metrics(3.0, 20)
    m.index = 0
    metrics = [m]
    silence = [{'start_s': 3.0, 'end_s': 5.0, 'label': 'silence'}]
    
    aligned = global_align_dp(metrics, silence_regions=silence)
    
    # gap_shift_cost < mild_stretch_cost → keep GAP_SHIFT
    assert aligned[0].action == AlignAction.GAP_SHIFT
    assert aligned[0].gap_shift_s == pytest.approx((20 / 4.5) - 3.0)


def test_dp_overrides_early_gap_to_stretch():
    # i=0, n=10, stretch~1.444, src_dur=2.0:
    # predicted = 13*1/4.5 = 2.889. overflow = 0.889
    # gap_shift_cost = 0.889 * 10 = 8.89
    # mild_stretch_cost = (1.444-1.0) * 2.0 * 5.0 = 4.44
    # 8.89 > 4.44 => override to MILD_STRETCH
    from foreign_whispers.alignment import global_align_dp, AlignAction, compute_segment_metrics
    # Build 10 segments: first problematic, rest easy
    segs_en = [{"start": float(i*3), "end": float(i*3+2), "text": "hello"} for i in range(10)]
    segs_es_first = {"start": 0.0, "end": 2.0, "text": "ba" * 13}  # stretch ~1.44 in DP zone
    segs_es_rest = [{"start": float(i*3), "end": float(i*3+2), "text": "ba"} for i in range(1, 10)]
    en = {"segments": segs_en}
    es = {"segments": [segs_es_first] + segs_es_rest}
    metrics = compute_segment_metrics(en, es)
    silence = [{"start_s": 2.0, "end_s": 5.0, "label": "silence"}]
    aligned = global_align_dp(metrics, silence)
    # First segment should be MILD_STRETCH (overridden from GAP_SHIFT)
    assert aligned[0].action == AlignAction.MILD_STRETCH


def test_dp_keeps_gap_for_late_segment():
    # i=9, n=10, stretch~1.44 in DP zone:
    # overflow ~0.889, gap_shift_cost = 0.889 * (10-9) = 0.889
    # mild_stretch_cost = 0.444 * 2.0 * 5.0 = 4.44
    # 0.889 < 4.44 => keep GAP_SHIFT
    from foreign_whispers.alignment import global_align_dp, AlignAction, compute_segment_metrics
    segs_en = [{"start": float(i*3), "end": float(i*3+2), "text": "hello"} for i in range(10)]
    segs_es_last = {"start": 27.0, "end": 29.0, "text": "ba" * 13}
    segs_es_rest = [{"start": float(i*3), "end": float(i*3+2), "text": "ba"} for i in range(9)]
    en = {"segments": segs_en}
    es = {"segments": segs_es_rest + [segs_es_last]}
    metrics = compute_segment_metrics(en, es)
    silence = [{"start_s": 29.0, "end_s": 32.0, "label": "silence"}]
    aligned = global_align_dp(metrics, silence)
    assert aligned[9].action == AlignAction.GAP_SHIFT


def test_dp_above_boundary_always_gap_shifts():
    # stretch > 1.55 boundary: DP override block not entered, GAP_SHIFT kept
    # "ba"*14 -> 14/4.5=3.111s in 2.0s -> stretch=1.556 > 1.55
    from foreign_whispers.alignment import global_align_dp, AlignAction, compute_segment_metrics
    en = {"segments": [{"start": 0.0, "end": 2.0, "text": "hello"}]}
    es = {"segments": [{"start": 0.0, "end": 2.0, "text": "ba" * 14}]}
    metrics = compute_segment_metrics(en, es)
    silence = [{"start_s": 2.0, "end_s": 5.0, "label": "silence"}]
    aligned = global_align_dp(metrics, silence)
    assert aligned[0].action == AlignAction.GAP_SHIFT


def test_dp_empty_input():
    from foreign_whispers.alignment import global_align_dp
    assert global_align_dp([], []) == []
