"""Load QuALITY questions as two-option debate items.

QuALITY questions have four options. Following the debate literature (e.g. Khan et al.,
2024), each question is reduced to two options: the correct answer and the *best
distractor* -- the wrong option that human validators found most tempting. That makes
the incorrect side as persuasive as possible, which is what makes debate a real test.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DebateItem:
    question_id: str
    article_id: str
    title: str
    story: str
    question: str
    option_a: str
    option_b: str
    correct: str  # "A" or "B"
    difficult: bool

    @property
    def incorrect(self) -> str:
        return "B" if self.correct == "A" else "A"

    def option(self, letter: str) -> str:
        return self.option_a if letter == "A" else self.option_b

    def to_dict(self, include_story: bool = False) -> dict:
        d = asdict(self)
        if not include_story:
            d.pop("story")
        return d


def best_distractor(question: dict) -> int:
    """Return the 1-indexed option most often chosen as 'best distractor' by validators.

    Ties go to the lowest option number. Falls back to the first wrong option if no
    validator data is available.
    """
    gold = question["gold_label"]
    votes = [
        v.get("untimed_best_distractor")
        for v in question.get("validation", [])
        if v.get("untimed_best_distractor") not in (None, gold)
    ]
    if votes:
        counts = Counter(votes)
        top = max(counts.values())
        return min(opt for opt, c in counts.items() if c == top)
    return next(i for i in range(1, len(question["options"]) + 1) if i != gold)


def clean_story(text: str) -> str:
    """Tidy up QuALITY's HTML-stripped text: collapse odd line breaks into paragraphs."""
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)


def load_quality(
    path: str | Path,
    n_questions: int | None = None,
    hard_only: bool = True,
    seed: int = 0,
) -> list[DebateItem]:
    """Load QuALITY items, shuffle with a fixed seed, and randomise answer order.

    The correct answer is placed in position A or B at random (seeded), so no protocol
    can win just by favouring one position.
    """
    rng = random.Random(seed)
    items: list[DebateItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            story = clean_story(record["article"])
            for q in record["questions"]:
                if hard_only and not q.get("difficult"):
                    continue
                gold = q["options"][q["gold_label"] - 1]
                wrong = q["options"][best_distractor(q) - 1]
                correct_first = rng.random() < 0.5
                items.append(
                    DebateItem(
                        question_id=q.get("question_unique_id", f"{record['set_unique_id']}_{len(items)}"),
                        article_id=record["article_id"],
                        title=record.get("title", ""),
                        story=story,
                        question=q["question"],
                        option_a=gold if correct_first else wrong,
                        option_b=wrong if correct_first else gold,
                        correct="A" if correct_first else "B",
                        difficult=bool(q.get("difficult")),
                    )
                )
    rng.shuffle(items)
    return items[:n_questions] if n_questions else items
