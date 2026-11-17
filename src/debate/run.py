"""Run experiments.

Examples:
    # Check everything works offline, with a fake model (no API key needed):
    python -m debate.run --dry-run --n 5

    # Real run on 10 questions first, then scale up:
    python -m debate.run --n 10 --run-name pilot
    python -m debate.run --run-name main

Results are appended to results/<run-name>/results.jsonl. Re-running the same command
resumes where it stopped, and cached model responses are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from .data import load_quality
from .llm import FakeLLM, OpenAICompatibleLLM
from .protocols import Runner

ALL_PROTOCOLS = ["naive", "consultancy", "debate", "expert"]


def fake_responder(messages, model):
    """Offline stand-in: quotes the first evidence passage and answers pseudo-randomly."""
    user = messages[-1]["content"]
    if "Write your argument" in user:
        match = re.search(r"\[Passage \d+\] ((?:\S+ ){8}\S+)", user)
        quote = match.group(1) if match else "no evidence"
        return f"My answer is right because the story says <quote>{quote}</quote>."
    seed = int(hashlib.md5(user.encode()).hexdigest(), 16)
    return f"Weighing it up.\nAnswer: {'AB'[seed % 2]}\nConfidence: {50 + seed % 50}"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--protocols", default=",".join(ALL_PROTOCOLS))
    parser.add_argument("--n", type=int, help="number of questions (overrides config)")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--dry-run", action="store_true", help="use a fake model, no API calls")
    parser.add_argument("--data", help="path to QuALITY file (overrides config)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    unknown = set(protocols) - set(ALL_PROTOCOLS)
    if unknown:
        parser.error(f"unknown protocols: {unknown}")

    d = cfg["data"]
    items = load_quality(args.data or d["path"], args.n or d["n_questions"], d["hard_only"], d["seed"])
    print(f"Loaded {len(items)} questions ({'hard only' if d['hard_only'] else 'all'})")

    if args.dry_run:
        llm = FakeLLM(fake_responder)
        args.run_name = f"{args.run_name}-dryrun"
    else:
        p = cfg["provider"]
        llm = OpenAICompatibleLLM(p["base_url"], p["api_key_env"], cfg["cache_dir"],
                                  p["min_seconds_between_requests"], p["max_retries"])
    runner = Runner(llm, cfg)

    out_dir = Path(cfg["output_dir"]) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    out_file = out_dir / "results.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["protocol"], r["question_id"], r.get("assigned")))
        print(f"Resuming: {len(done)} results already saved")

    jobs = []
    for item in items:
        for proto in protocols:
            if proto == "consultancy":
                # Run both assignments so honest and dishonest consultants are balanced.
                jobs += [(item, proto, "correct"), (item, proto, "incorrect")]
            else:
                jobs.append((item, proto, None))
    random.Random(cfg["data"]["seed"]).shuffle(jobs)  # spread protocols evenly over time

    with out_file.open("a") as f:
        for i, (item, proto, assigned) in enumerate(jobs, 1):
            if (proto, item.question_id, assigned) in done:
                continue
            fn = getattr(runner, proto)
            result = fn(item, assigned) if assigned else fn(item)
            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()
            mark = "correct" if result.judge_correct else "wrong"
            label = f"{proto}({assigned})" if assigned else proto
            print(f"[{i}/{len(jobs)}] {label:<24} {item.question_id[:28]:<28} -> {mark}")

    print(f"\nSaved to {out_file}. Summarise with: python -m debate.analyze {out_dir}")


if __name__ == "__main__":
    main()
