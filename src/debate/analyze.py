"""Summarise a run: judge accuracy per protocol with 95% bootstrap confidence intervals,
plus how often honest vs dishonest arguers' quotes were verified.

    python -m debate.analyze results/main
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["naive", "consultancy", "debate", "expert"]


def load_results(run_dir: str | Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (Path(run_dir) / "results.jsonl").read_text().splitlines() if l]
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for a mean, by resampling with replacement."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for proto in [p for p in ORDER if p in set(df["protocol"])]:
        sub = df[df["protocol"] == proto]
        acc = sub["judge_correct"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(acc)
        rows.append({
            "protocol": proto,
            "n": len(sub),
            "accuracy": acc.mean(),
            "ci_low": lo,
            "ci_high": hi,
            "unparsed": int(sub["judge_answer"].isna().sum()),
            "mean_confidence": sub["confidence"].mean(),
        })
        if proto == "consultancy":
            for side in ("correct", "incorrect"):
                s = sub[sub["assigned"] == side]["judge_correct"].astype(float).to_numpy()
                lo, hi = bootstrap_ci(s)
                rows.append({"protocol": f"  consultant argues {side}", "n": len(s),
                             "accuracy": s.mean() if len(s) else float("nan"),
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def quote_table(df: pd.DataFrame) -> pd.DataFrame:
    """Are dishonest arguers forced into unverifiable quotes more often than honest ones?"""
    counts = {"honest": [0, 0], "dishonest": [0, 0]}  # [verified, total]
    for _, r in df[df["protocol"].isin(["debate", "consultancy"])].iterrows():
        for t in r["transcript"]:
            if t["speaker"] == "judge" or not t["quotes_total"]:
                continue
            if t["speaker"] == "consultant":
                honest = r["assigned"] == "correct"
            else:
                honest = t["speaker"].endswith(r["correct"])
            key = "honest" if honest else "dishonest"
            counts[key][0] += t["quotes_verified"]
            counts[key][1] += t["quotes_total"]
    return pd.DataFrame([
        {"side": k, "quotes": tot, "verified_rate": (v / tot) if tot else float("nan")}
        for k, (v, tot) in counts.items()
    ])


def plot_accuracy(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = table[table["protocol"].isin(ORDER)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(main))
    err = [main["accuracy"] - main["ci_low"], main["ci_high"] - main["accuracy"]]
    ax.bar(x, main["accuracy"], color="#3b6ea5", width=0.6)
    ax.errorbar(x, main["accuracy"], yerr=err, fmt="none", ecolor="#222", capsize=4)
    ax.axhline(0.5, color="#999", linestyle="--", linewidth=1)
    ax.text(len(main) - 0.5, 0.51, "chance", color="#777", ha="right", fontsize=8)
    ax.set_xticks(x, main["protocol"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Judge accuracy")
    ax.set_title("Judge accuracy by protocol (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)

    df = load_results(run_dir)
    acc = accuracy_table(df)
    quotes = quote_table(df)
    plot_accuracy(acc, run_dir / "accuracy.png")

    fmt = lambda t: t.to_markdown(index=False, floatfmt=".3f")
    summary = (f"# Results: {run_dir.name}\n\n## Judge accuracy\n\n{fmt(acc)}\n\n"
               f"## Quote verification by side\n\n{fmt(quotes)}\n\n![accuracy](accuracy.png)\n")
    (run_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
