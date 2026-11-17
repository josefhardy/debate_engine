"""The four experimental conditions.

* naive        -- the judge answers alone, without the story (lower bound)
* consultancy  -- one AI argues for an assigned answer (correct half the time)
* debate       -- two AIs argue for opposite answers over several rounds
* expert       -- the judge reads the full story itself (upper bound)

The research question: does debate get the judge closer to the expert than consultancy does?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import prompts
from .data import DebateItem
from .llm import LLM
from .quotes import verify_quotes
from .retrieval import StoryIndex

# Tolerates markdown and brackets, e.g. "**Answer:** (B)".
_ANSWER = re.compile(r"answer[\s*_:\-(\[]{1,8}([AB])\b", re.IGNORECASE)
_CONFIDENCE = re.compile(r"confidence[\s*_:\-]{1,8}(\d{1,3})", re.IGNORECASE)


def parse_judgement(text: str) -> tuple[str | None, float | None]:
    """Extract the judge's final answer and confidence (0-1). Uses the LAST match."""
    answers = _ANSWER.findall(text)
    confidences = _CONFIDENCE.findall(text)
    answer = answers[-1].upper() if answers else None
    confidence = None
    if confidences:
        confidence = min(max(int(confidences[-1]), 0), 100) / 100
    return answer, confidence


@dataclass
class Turn:
    speaker: str  # e.g. "debater_A", "consultant", "judge"
    round: int
    raw: str
    shown_to_judge: str
    quotes_total: int = 0
    quotes_verified: int = 0


@dataclass
class Result:
    protocol: str
    question_id: str
    correct: str
    judge_answer: str | None
    judge_correct: bool
    confidence: float | None
    difficult: bool
    assigned: str | None = None  # consultancy: "correct" / "incorrect"
    transcript: list[Turn] = field(default_factory=list)


class Runner:
    def __init__(self, llm: LLM, config: dict):
        self.llm = llm
        self.cfg = config

    # ---- helpers -------------------------------------------------------------
    def _gen(self, model_key: str, system: str, user: str) -> str:
        gen = self.cfg["generation"]
        return self.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=self.cfg["models"][model_key],
            temperature=gen["temperature"],
            max_tokens=gen["max_tokens"],
        )

    def _judge(self, item: DebateItem, system: str, body: str) -> tuple[str, str | None, float | None]:
        user = prompts.JUDGE_USER.format(
            question=item.question, option_a=item.option_a, option_b=item.option_b, body=body
        )
        text = self._gen("judge", system, user)
        answer, confidence = parse_judgement(text)
        return text, answer, confidence

    def _argue(self, item: DebateItem, index: StoryIndex, system: str, letter: str, speaker: str,
               transcript: list[Turn], round_no: int, rounds: int) -> Turn:
        r = self.cfg["retrieval"]
        query = f"{item.question} {item.option(letter)}"
        opponent_turns = [t for t in transcript if t.speaker != speaker]
        if opponent_turns:
            query += " " + opponent_turns[-1].raw  # also search for evidence against the opponent
        passages = index.search(query, k=r["top_k"])
        passage_text = "\n\n".join(f"[Passage {p.index}] {p.text}" for p in passages)
        user = prompts.ARGUE_USER.format(
            question=item.question, option_a=item.option_a, option_b=item.option_b,
            passages=passage_text, transcript=format_transcript(transcript) or "(none yet)",
            round=round_no, rounds=rounds,
        )
        raw = self._gen("debater", system, user)
        shown, checks = verify_quotes(raw, item.story, self.cfg["quotes"]["min_match_score"])
        return Turn(speaker, round_no, raw, shown, len(checks), sum(c.verified for c in checks))

    def _index(self, item: DebateItem) -> StoryIndex:
        r = self.cfg["retrieval"]
        return StoryIndex(item.story, r["passage_words"], r["passage_overlap_words"])

    def _result(self, protocol: str, item: DebateItem, judge_text: str, answer: str | None,
                confidence: float | None, transcript: list[Turn], assigned: str | None = None) -> Result:
        transcript.append(Turn("judge", 0, judge_text, judge_text))
        return Result(protocol, item.question_id, item.correct, answer, answer == item.correct,
                      confidence, item.difficult, assigned, transcript)

    # ---- protocols -----------------------------------------------------------
    def naive(self, item: DebateItem) -> Result:
        text, answer, conf = self._judge(item, prompts.JUDGE_NAIVE_SYSTEM, "")
        return self._result("naive", item, text, answer, conf, [])

    def expert(self, item: DebateItem) -> Result:
        body = f"\nStory:\n{item.story}"
        text, answer, conf = self._judge(item, prompts.JUDGE_EXPERT_SYSTEM, body)
        return self._result("expert", item, text, answer, conf, [])

    def consultancy(self, item: DebateItem, assigned: str) -> Result:
        """assigned: 'correct' or 'incorrect' -- which answer the consultant must defend."""
        letter = item.correct if assigned == "correct" else item.incorrect
        other = "B" if letter == "A" else "A"
        d = self.cfg["debate"]
        system = prompts.CONSULTANT_SYSTEM.format(
            letter=letter, answer=item.option(letter), other_letter=other,
            other_answer=item.option(other), quote_rules=prompts.QUOTE_RULES, words=d["words_per_turn"],
        )
        index, transcript = self._index(item), []
        for r in range(1, d["rounds"] + 1):
            transcript.append(
                self._argue(item, index, system, letter, "consultant", transcript, r, d["rounds"])
            )
        body = "\nConsultant's arguments:\n" + format_transcript(transcript)
        text, answer, conf = self._judge(item, prompts.JUDGE_CONSULTANCY_SYSTEM, body)
        return self._result("consultancy", item, text, answer, conf, transcript, assigned)

    def debate(self, item: DebateItem) -> Result:
        d = self.cfg["debate"]
        systems = {
            letter: prompts.DEBATER_SYSTEM.format(
                letter=letter, answer=item.option(letter),
                other_letter="B" if letter == "A" else "A",
                other_answer=item.option("B" if letter == "A" else "A"),
                quote_rules=prompts.QUOTE_RULES, words=d["words_per_turn"],
            )
            for letter in ("A", "B")
        }
        index, transcript = self._index(item), []
        for r in range(1, d["rounds"] + 1):
            # Simultaneous turns: both debaters see only previous rounds, so neither side
            # gets the advantage of always replying last.
            previous = list(transcript)
            for letter in ("A", "B"):
                transcript.append(self._argue(
                    item, index, systems[letter], letter, f"debater_{letter}", previous, r, d["rounds"]
                ))
        body = "\nDebate transcript:\n" + format_transcript(transcript)
        text, answer, conf = self._judge(item, prompts.JUDGE_DEBATE_SYSTEM, body)
        return self._result("debate", item, text, answer, conf, transcript)


def format_transcript(transcript: list[Turn]) -> str:
    """Render arguments with quotes already tagged as verified/unverified."""
    lines = []
    for t in transcript:
        if t.speaker == "judge":
            continue
        name = {"debater_A": "Debater A", "debater_B": "Debater B"}.get(t.speaker, "Consultant")
        lines.append(f"--- Round {t.round}, {name} ---\n{t.shown_to_judge}")
    return "\n\n".join(lines)
