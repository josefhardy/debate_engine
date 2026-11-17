"""Quote verification: the judge can't read the story, but it can trust checked quotes.

Debaters cite evidence inside <quote>...</quote> tags. Before the judge sees an argument,
every quote is checked against the story:

* verified quotes become   <v_quote>...</v_quote>
* unverified quotes become <u_quote>...</u_quote>

The judge is told only verified quotes are guaranteed to be real. This is the mechanism
that makes lying harder than telling the truth, and is the same convention used in
published debate experiments (Khan et al., 2024).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

QUOTE_PATTERN = re.compile(r"<quote>(.*?)</quote>", re.DOTALL)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


@dataclass
class QuoteCheck:
    quote: str
    verified: bool
    score: float


def check_quote(quote: str, story_normalized: str, min_score: float = 95) -> QuoteCheck:
    q = normalize(quote)
    if not q:
        return QuoteCheck(quote, False, 0.0)
    if q in story_normalized:
        return QuoteCheck(quote, True, 100.0)
    score = fuzz.partial_ratio(q, story_normalized)
    return QuoteCheck(quote, score >= min_score, score)


def verify_quotes(argument: str, story: str, min_score: float = 95) -> tuple[str, list[QuoteCheck]]:
    """Return the argument with quotes re-tagged as verified/unverified, plus the checks."""
    story_norm = normalize(story)
    checks: list[QuoteCheck] = []

    def retag(match: re.Match) -> str:
        check = check_quote(match.group(1), story_norm, min_score)
        checks.append(check)
        tag = "v_quote" if check.verified else "u_quote"
        return f"<{tag}>{match.group(1).strip()}</{tag}>"

    return QUOTE_PATTERN.sub(retag, argument), checks
