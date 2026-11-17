"""End-to-end tests with a scripted fake model (no API key needed)."""

from pathlib import Path

import yaml

from debate import analyze, run
from debate.data import load_quality
from debate.llm import FakeLLM
from debate.protocols import Runner

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "mini_quality.jsonl"


def config(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["data"]["path"] = str(FIXTURE)
    cfg["output_dir"] = str(tmp_path)
    cfg["debate"]["rounds"] = 2
    return cfg


def honest_judge_llm():
    """Debaters quote real text if honest, fabricate if not; judge believes verified quotes."""
    def responder(messages, model):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "Write your argument" in user:
            if '"Mara"' in system.split("defend")[-1][:40] or "answer is A" in system:
                return "<quote>Mara rowed out alone and rescued the two sailors</quote>"
            return "<quote>the captain bravely saved the sailors</quote>"
        # Judge: pick whichever option a verified quote supports, else A.
        if "<v_quote>Mara rowed out" in user:
            mara_letter = "A" if "A: Mara" in user else "B"
            return f"Answer: {mara_letter}\nConfidence: 90"
        return "Answer: A\nConfidence: 55"
    return FakeLLM(responder)


def test_debate_transcript_structure(tmp_path):
    cfg = config(tmp_path)
    item = next(i for i in load_quality(FIXTURE) if i.question_id == "q1")
    result = Runner(honest_judge_llm(), cfg).debate(item)
    speakers = [t.speaker for t in result.transcript]
    assert speakers == ["debater_A", "debater_B"] * 2 + ["judge"]
    assert result.judge_answer in ("A", "B")


def test_consultancy_assignment(tmp_path):
    cfg = config(tmp_path)
    item = next(i for i in load_quality(FIXTURE) if i.question_id == "q1")
    llm = honest_judge_llm()
    runner = Runner(llm, cfg)
    honest = runner.consultancy(item, "correct")
    dishonest = runner.consultancy(item, "incorrect")
    assert honest.assigned == "correct" and dishonest.assigned == "incorrect"
    first_system = llm.calls[0][1][0]["content"]
    assert item.option(item.correct) in first_system


def test_debaters_see_retrieved_passages_not_whole_story(tmp_path):
    cfg = config(tmp_path)
    cfg["retrieval"].update(passage_words=20, passage_overlap_words=5, top_k=2)
    item = load_quality(FIXTURE)[0]
    llm = honest_judge_llm()
    Runner(llm, cfg).debate(item)
    debater_prompt = llm.calls[0][1][-1]["content"]
    assert debater_prompt.count("[Passage") == 2
    assert len(debater_prompt) < len(item.story)


def test_dry_run_and_analysis_end_to_end(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config(tmp_path)))
    run.main(["--config", str(cfg_path), "--dry-run", "--n", "3", "--run-name", "t"])
    run_dir = tmp_path / "t-dryrun"
    df = analyze.load_results(run_dir)
    # 3 questions x (naive + 2 consultancy + debate + expert)
    assert len(df) == 15
    analyze.main([str(run_dir)])
    assert (run_dir / "summary.md").exists() and (run_dir / "accuracy.png").exists()

    # Re-running resumes instead of duplicating work.
    run.main(["--config", str(cfg_path), "--dry-run", "--n", "3", "--run-name", "t"])
    assert len(analyze.load_results(run_dir)) == 15
