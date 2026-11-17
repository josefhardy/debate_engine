from pathlib import Path

import pytest

from debate.data import best_distractor, load_quality
from debate.protocols import parse_judgement
from debate.quotes import verify_quotes
from debate.retrieval import StoryIndex, chunk_text

FIXTURE = Path(__file__).parent / "fixtures" / "mini_quality.jsonl"


# ---- data -------------------------------------------------------------------
def test_best_distractor_majority_vote():
    q = {"gold_label": 2, "options": ["a", "b", "c", "d"],
         "validation": [{"untimed_best_distractor": 3}, {"untimed_best_distractor": 3},
                        {"untimed_best_distractor": 1}]}
    assert best_distractor(q) == 3


def test_best_distractor_tie_and_fallback():
    tie = {"gold_label": 1, "options": list("abcd"),
           "validation": [{"untimed_best_distractor": 4}, {"untimed_best_distractor": 2}]}
    assert best_distractor(tie) == 2
    assert best_distractor({"gold_label": 1, "options": list("abcd"), "validation": []}) == 2


def test_load_quality_hard_only_and_two_options():
    items = load_quality(FIXTURE, hard_only=True)
    assert len(items) == 3  # the easy question is excluded
    rescue = next(i for i in items if i.question_id == "q1")
    assert rescue.option(rescue.correct) == "Mara"
    assert rescue.option(rescue.incorrect) == "The captain"  # most-voted distractor


def test_load_quality_randomises_answer_position_reproducibly():
    a = load_quality(FIXTURE, hard_only=False, seed=1)
    b = load_quality(FIXTURE, hard_only=False, seed=1)
    assert [i.correct for i in a] == [i.correct for i in b]


# ---- retrieval ----------------------------------------------------------------
def test_chunks_overlap_and_cover_text():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, passage_words=100, overlap_words=20)
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert chunks[-1].text.split()[-1] == "w249"


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", passage_words=10, overlap_words=10)


def test_search_finds_relevant_passage():
    story = "Cats sleep a lot. " * 50 + "The treasure was buried under the old oak tree. " + "Dogs bark. " * 50
    index = StoryIndex(story, passage_words=30, overlap_words=5)
    hits = index.search("where was the treasure buried", k=1)
    assert "treasure was buried" in hits[0].text


# ---- quotes -------------------------------------------------------------------
STORY = "Mara rowed out alone and rescued the two sailors, though she broke her wrist doing it."


def test_exact_quote_is_verified():
    shown, checks = verify_quotes("See <quote>rescued the two sailors</quote>.", STORY)
    assert checks[0].verified
    assert "<v_quote>rescued the two sailors</v_quote>" in shown


def test_fabricated_quote_is_flagged():
    shown, checks = verify_quotes("<quote>the captain saved everyone</quote>", STORY)
    assert not checks[0].verified
    assert "<u_quote>" in shown


def test_quote_tolerates_punctuation_and_case():
    _, checks = verify_quotes("<quote>Rescued the two sailors though she broke</quote>", STORY)
    assert checks[0].verified


# ---- judge parsing --------------------------------------------------------------
@pytest.mark.parametrize("text,answer,conf", [
    ("Reasoning...\nAnswer: B\nConfidence: 80", "B", 0.8),
    ("**Answer:** (A)\n**Confidence:** 65", "A", 0.65),
    ("Answer: A ... on reflection\nAnswer: B\nConfidence: 150", "B", 1.0),
    ("I cannot decide.", None, None),
])
def test_parse_judgement(text, answer, conf):
    assert parse_judgement(text) == (answer, conf)
