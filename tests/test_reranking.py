# tests/test_reranking.py
"""Tests for foreign_whispers.reranking.get_shorter_translations."""

from foreign_whispers.reranking import get_shorter_translations, TranslationCandidate


def test_returns_empty_when_baseline_fits():
    """If baseline is within budget, no candidates needed."""
    result = get_shorter_translations(
        source_text="Hello world",
        baseline_es="Hola mundo",
        target_duration_s=5.0,  # 75 chars budget, "Hola mundo" is 10
    )
    assert result == []


def test_contraction_shortening():
    """Verbose phrases should be shortened via contraction rules."""
    long_text = "En este momento estamos llevando a cabo la investigación"
    result = get_shorter_translations(
        source_text="We are conducting the research right now",
        baseline_es=long_text,
        target_duration_s=2.0,  # 30 chars budget
    )
    assert len(result) > 0
    assert all(isinstance(c, TranslationCandidate) for c in result)
    # All candidates should be shorter than baseline
    assert all(c.char_count < len(long_text) for c in result)


def test_filler_removal():
    """Filler adverbs should be stripped."""
    text = "Realmente es completamente absolutamente increíble"
    result = get_shorter_translations(
        source_text="It is really incredible",
        baseline_es=text,
        target_duration_s=1.5,
    )
    assert len(result) > 0
    assert all(c.char_count < len(text) for c in result)


def test_sorted_shortest_first():
    """Candidates should be sorted shortest first."""
    text = "En este momento realmente estamos llevando a cabo la investigación completamente"
    result = get_shorter_translations(
        source_text="test",
        baseline_es=text,
        target_duration_s=1.0,
    )
    if len(result) > 1:
        for i in range(len(result) - 1):
            assert result[i].char_count <= result[i + 1].char_count


def test_brevity_rationale_populated():
    """Each candidate should have a non-empty rationale."""
    text = "Sin embargo en este momento es realmente importante"
    result = get_shorter_translations(
        source_text="test",
        baseline_es=text,
        target_duration_s=1.0,
    )
    assert len(result) > 0
    assert all(c.brevity_rationale for c in result)
