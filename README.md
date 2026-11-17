# AI Debate Engine

**Can two AIs arguing against each other help a weaker AI find the truth?**

As AI systems become more capable, humans will need to supervise models that know more than
they do. One proposed answer is *debate* ([Irving et al., 2018](https://arxiv.org/abs/1805.00899)):
let two strong models argue opposite sides and have a weaker judge decide. If lying is harder
to defend than the truth, the judge should end up right more often than it would alone.

This project tests that idea on hard reading-comprehension questions. The debaters can search
a story for evidence, while the judge never sees it and has to rely on the arguments and on
automatically verified quotes.

> **Status:** pipeline complete and tested; experiments in progress. Results will be added
> below once the real runs are finished.

## Experimental design

Each question comes from the **hard subset of [QuALITY](https://github.com/nyu-mll/quality)**:
long stories where people skimming under time pressure answered wrongly most of the time.
Each question is reduced to two options: the correct answer and the *best distractor* (the wrong
option human validators found most tempting).

| Protocol | What the judge gets | Role |
|---|---|---|
| **Naive** | Question and two options only | Lower bound |
| **Consultancy** | One AI argues for an assigned answer, correct 50% of the time | Baseline for "one persuasive AI" |
| **Debate** | Two AIs argue opposite answers over several rounds | The method being tested |
| **Expert** | The full story | Upper bound |

The judge is a **smaller model** than the debaters, mimicking a human supervising a more
capable AI.

### Key mechanisms

- **Retrieval-grounded arguments.** Debaters don't read the whole story. They search it with
  BM25 for passages relevant to the question, their answer and their opponent's last argument
  (`src/debate/retrieval.py`).
- **Quote verification.** Debaters cite evidence in `<quote>` tags. Every quote is checked
  against the story before the judge sees it, and is shown as verified (`<v_quote>`) or
  unverified (`<u_quote>`) (`src/debate/quotes.py`). This is what should make lying costly.
- **Fair setup.** The correct answer is randomly placed in position A or B, debaters argue
  simultaneously so neither always gets the last word, and consultancy runs both honest and
  dishonest assignments for every question.
- **Reproducible and free-tier friendly.** Every model response is cached on disk, requests are
  rate-limited with backoff, and interrupted runs resume where they stopped.

### What gets measured

- Judge accuracy per protocol, with 95% bootstrap confidence intervals
- Accuracy when the consultant is honest vs dishonest
- How often honest vs dishonest arguers' quotes are verified
- Judge confidence, to check calibration

## Results

*Coming soon.*

## Setup (macOS / Linux)

```bash
git clone https://github.com/josefhardy/debate_engine.git
cd debate_engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/download_quality.sh
```

Get a free API key from an OpenAI-compatible provider (e.g. Groq, Google AI Studio or
OpenRouter), set it, and check `configs/default.yaml` uses model names your provider currently
offers:

```bash
export GROQ_API_KEY=your_key_here
```

## Running

```bash
# 1. Check everything works offline with a fake model (no key needed)
python -m debate.run --dry-run --n 5

# 2. A small pilot run
python -m debate.run --n 10 --run-name pilot
python -m debate.analyze results/pilot

# 3. The full experiment
python -m debate.run --run-name main
python -m debate.analyze results/main
```

`analyze` writes `summary.md` and `accuracy.png` into the run folder. Run only some protocols
with e.g. `--protocols naive,debate`.

## Tests

```bash
pytest
```

The tests use a scripted fake model, so they run in seconds without an API key.

## Project structure

```
src/debate/
  data.py        load QuALITY, pick best distractor, randomise answer order
  retrieval.py   passage chunking and BM25 search
  quotes.py      quote extraction and verification
  llm.py         OpenAI-compatible client with caching, rate limiting and retries
  prompts.py     debater, consultant and judge prompts
  protocols.py   naive, consultancy, debate and expert conditions
  run.py         experiment runner (resumable)
  analyze.py     accuracy, confidence intervals, quote stats, plot
```

## References

- Irving, Christiano & Amodei (2018). [AI safety via debate](https://arxiv.org/abs/1805.00899).
- Pang et al. (2022). [QuALITY: Question Answering with Long Input Texts, Yes!](https://arxiv.org/abs/2112.08608)
- Michael et al. (2023). [Debate Helps Supervise Unreliable Experts](https://arxiv.org/abs/2311.08702).
- Khan et al. (2024). [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782).

## Credits

Built by Josef Hardy with AI-assisted coding. QuALITY is by Pang et al.; see the
[dataset repository](https://github.com/nyu-mll/quality) for licensing.
