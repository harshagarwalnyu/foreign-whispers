"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    budget_chars = int(target_duration_s * 15.0)
    baseline_len = len(baseline_es)

    # If baseline already fits, nothing to do
    if baseline_len <= budget_chars:
        return []

    candidates: list[TranslationCandidate] = []

    # ── Strategy 0: Strip argostranslate quote/blockquote prefixes ─────
    # argostranslate sometimes copies markdown blockquote markers ("> ") or
    # typographic quote chars from captions verbatim into the translation.
    # These are transcription artifacts, not speech — strip them.
    _stripped = re.sub(r'^[\s>»]+', '', baseline_es).strip()
    if len(_stripped) < baseline_len:
        candidates.append(TranslationCandidate(
            text=_stripped,
            char_count=len(_stripped),
            brevity_rationale="stripped quote/blockquote prefix artifact",
        ))
        # Use stripped text as baseline for further shortening strategies
        baseline_es = _stripped

    # ── Strategy 1: Rule-based Spanish shortening ──────────────────────
    # Common filler/verbose phrases → shorter equivalents
    _CONTRACTIONS: list[tuple[str, str, str]] = [
        ("en este momento", "ahora", "temporal filler → ahora"),
        ("en la actualidad", "ahora", "temporal filler → ahora"),
        ("con el fin de", "para", "verbose purpose → para"),
        ("con el objetivo de", "para", "verbose purpose → para"),
        ("a pesar de que", "aunque", "concessive → aunque"),
        ("a pesar de", "pese a", "concessive shortened"),
        ("debido a que", "porque", "causal → porque"),
        ("con respecto a", "sobre", "prepositional → sobre"),
        ("en relación con", "sobre", "prepositional → sobre"),
        ("por lo tanto", "así", "connector → así"),
        ("sin embargo", "pero", "connector → pero"),
        ("no obstante", "pero", "connector → pero"),
        ("es necesario", "hay que", "modal → hay que"),
        ("tiene que", "debe", "modal → debe"),
        ("se encuentra", "está", "locative → está"),
        ("se encuentran", "están", "locative → están"),
        ("llevar a cabo", "hacer", "verbal → hacer"),
        ("dar lugar a", "causar", "verbal → causar"),
        ("poner en marcha", "iniciar", "verbal → iniciar"),
        ("tener en cuenta", "considerar", "verbal → considerar"),
        ("a lo largo de", "durante", "temporal → durante"),
        ("de acuerdo con", "según", "prepositional → según"),
        ("por medio de", "mediante", "prepositional → mediante"),
        ("una gran cantidad de", "muchos", "quantifier → muchos"),
        ("gran cantidad de", "muchos", "quantifier → muchos"),
        ("la mayoría de", "casi todos", "quantifier → casi todos"),
        ("el hecho de que", "que", "nominalizer stripped"),
        # conversational / reported-speech patterns
        ("cuando le dijiste", "al saberlo", "reported speech → al saberlo"),
        ("cuando te dije", "al decirte", "reported speech → al decirte"),
        ("cuando se lo dijiste", "al enterarse", "reported speech → al enterarse"),
        ("cuando lo supiste", "al saberlo", "reported speech → al saberlo"),
        ("cómo reaccionó", "su reacción", "nominalized → su reacción"),
        ("qué fue lo que", "qué", "relative clause stripped"),
        ("lo que pasó fue", "fue que", "cleft stripped"),
    ]

    # Apply contractions cumulatively (case-insensitive)
    shortened = baseline_es
    rationales: list[str] = []
    for pattern, replacement, rationale in _CONTRACTIONS:
        lower = shortened.lower()
        idx = lower.find(pattern)
        if idx != -1:
            # Preserve surrounding text, replace matched span
            shortened = shortened[:idx] + replacement + shortened[idx + len(pattern):]
            rationales.append(rationale)

    if len(shortened) < baseline_len:
        candidates.append(TranslationCandidate(
            text=shortened,
            char_count=len(shortened),
            brevity_rationale="; ".join(rationales),
        ))

    # ── Strategy 2: Strip filler words ─────────────────────────────────
    _FILLERS = [
        "realmente", "básicamente", "simplemente", "prácticamente",
        "verdaderamente", "absolutamente", "completamente", "totalmente",
        "generalmente", "normalmente", "aproximadamente", "exactamente",
        "especialmente", "particularmente", "específicamente",
        "probablemente", "posiblemente", "ciertamente", "obviamente",
        "claramente", "efectivamente", "definitivamente",
        "muy", "bastante", "algo", "un poco",
    ]

    filler_stripped = baseline_es
    filler_rationale: list[str] = []
    for filler in _FILLERS:
        pattern_re = re.compile(r'\b' + re.escape(filler) + r'\b\s*', re.IGNORECASE)
        if pattern_re.search(filler_stripped):
            filler_stripped = pattern_re.sub("", filler_stripped)
            filler_rationale.append(f"removed '{filler}'")

    # Clean double spaces
    filler_stripped = re.sub(r'\s{2,}', ' ', filler_stripped).strip()

    if len(filler_stripped) < baseline_len and filler_stripped != shortened:
        candidates.append(TranslationCandidate(
            text=filler_stripped,
            char_count=len(filler_stripped),
            brevity_rationale="; ".join(filler_rationale),
        ))

    # ── Strategy 3: Combined (contractions + filler removal) ───────────
    combined = shortened
    for filler in _FILLERS:
        pattern_re = re.compile(r'\b' + re.escape(filler) + r'\b\s*', re.IGNORECASE)
        combined = pattern_re.sub("", combined)
    combined = re.sub(r'\s{2,}', ' ', combined).strip()

    if len(combined) < baseline_len and combined not in (shortened, filler_stripped):
        candidates.append(TranslationCandidate(
            text=combined,
            char_count=len(combined),
            brevity_rationale="contractions + filler removal",
        ))

    # Sort shortest first
    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: %.1fs budget, %d chars baseline → %d candidates",
        target_duration_s,
        baseline_len,
        len(candidates),
    )
    return candidates
