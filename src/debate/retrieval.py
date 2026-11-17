"""Evidence retrieval: split a story into passages and search them with BM25.

Debaters don't receive the whole story. Instead they search it for passages relevant to
the question and their argument, like a lawyer looking through case files. This keeps
prompts small (important on free API tiers) and makes evidence-finding part of the task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Passage:
    index: int
    text: str


def chunk_text(text: str, passage_words: int = 120, overlap_words: int = 30) -> list[Passage]:
    """Split text into overlapping word windows so no evidence falls between two chunks."""
    if overlap_words >= passage_words:
        raise ValueError("overlap_words must be smaller than passage_words")
    words = text.split()
    step = passage_words - overlap_words
    passages = []
    for i, start in enumerate(range(0, max(len(words) - overlap_words, 1), step)):
        passages.append(Passage(i, " ".join(words[start : start + passage_words])))
    return passages


class StoryIndex:
    """A BM25 keyword index over one story's passages."""

    def __init__(self, story: str, passage_words: int = 120, overlap_words: int = 30):
        self.passages = chunk_text(story, passage_words, overlap_words)
        self._bm25 = BM25Okapi([tokenize(p.text) for p in self.passages])

    def search(self, query: str, k: int = 4) -> list[Passage]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # Return in story order so the debater reads them in context.
        return [self.passages[i] for i in sorted(ranked)]
