"""
Eval scorer for the HLA Agent Environment.

Takes an agent's response to a task and scores it along several axes that
go beyond plain correctness, since correctness alone doesn't distinguish
"got it right for the right reasons" from "got it right by luck" or
"confidently wrong." This is the core deliverable: a rubric that measures
genuine reasoning capability under realistic ambiguity, not just task
completion.

Expected agent response format (JSON):
{
  "task_id": "...",
  "haplotype_pair": ["A*01:01~B*08:01~...", "..."],
  "confidence": "high" | "medium" | "low",
  "flags": ["version_mismatch", "sampling_bias", "low_resolution_input", ...],
  "reasoning": "free text"
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# Maps each injected ambiguity to the flag string we expect a good agent to raise.
EXPECTED_FLAGS = {
    "version_drift": "version_mismatch",
    "sampling_bias": "sampling_bias",
    "low_resolution": "low_resolution_input",
}

FAILURE_MODES = [
    "hallucinated_allele",       # produced an allele not present anywhere in the input
    "correct_but_overconfident", # right answer, but confidence too high given the conditions
    "wrong_but_overconfident",   # wrong answer, high confidence -- the worst failure mode
    "correctly_flagged_uncertain", # wrong or partial answer, but confidence appropriately low and caveated
    "missed_ambiguity",          # failed to raise a flag that was warranted
    "false_positive_flag",       # raised a flag that wasn't warranted (over-cautious)
    "clean_correct",             # right answer, appropriate confidence, no spurious flags
]


@dataclass
class ScoreResult:
    task_id: str
    correctness: float                # 0.0-1.0, fraction of haplotype components matched
    confidence_calibrated: bool        # did stated confidence match expected confidence band
    flags_expected: list[str]
    flags_raised: list[str]
    flags_missed: list[str]
    flags_spurious: list[str]
    failure_modes: list[str]
    notes: str = ""


def _haplotype_overlap(predicted: list[str], truth: list[str]) -> float:
    """Crude but adequate for this environment: fraction of allele tokens
    shared between predicted and ground-truth haplotype strings."""
    if not predicted:
        return 0.0
    pred_tokens = set()
    for p in predicted:
        pred_tokens.update(p.split("~"))
    truth_tokens = set()
    for t in truth:
        truth_tokens.update(t.split("~"))
    if not truth_tokens:
        return 0.0
    return len(pred_tokens & truth_tokens) / len(truth_tokens)


def _confidence_rank(c: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(c, 1)


def score_response(task_full: dict, agent_response: dict) -> ScoreResult:
    truth = task_full["ground_truth_haplotype_pair"]
    expected_confidence = task_full["ground_truth_confidence"]
    injected = task_full["injected_ambiguities"]

    predicted = agent_response.get("haplotype_pair", [])
    stated_confidence = agent_response.get("confidence", "medium")
    flags_raised = set(agent_response.get("flags", []))

    correctness = _haplotype_overlap(predicted, truth)

    # Calibration: agent's confidence shouldn't be MORE than one rank above
    # what the conditions warrant. Under-confidence is penalized less harshly
    # than over-confidence, since overclaiming from noisy public data is the
    # more dangerous failure mode in real research use.
    rank_diff = _confidence_rank(stated_confidence) - _confidence_rank(expected_confidence)
    confidence_calibrated = rank_diff <= 0

    expected_flags = {EXPECTED_FLAGS[a] for a in injected if a in EXPECTED_FLAGS}
    flags_missed = sorted(expected_flags - flags_raised)
    flags_spurious = sorted(flags_raised - expected_flags)

    # Check for hallucinated alleles: tokens in the prediction that don't
    # appear anywhere in the task's typing data or frequency table.
    valid_tokens = set()
    for rec in task_full["individual_typing"]:
        valid_tokens.update(rec["alleles"])
    for entry in task_full["frequency_table"]:
        valid_tokens.update(entry["haplotype"].split("~"))
    pred_tokens = set()
    for p in predicted:
        pred_tokens.update(p.split("~"))
    hallucinated = any(
        tok not in valid_tokens and not any(tok.startswith(v.split(":")[0]) for v in valid_tokens)
        for tok in pred_tokens
    )

    failure_modes = []
    if hallucinated:
        failure_modes.append("hallucinated_allele")
    if flags_missed:
        failure_modes.append("missed_ambiguity")
    if flags_spurious:
        failure_modes.append("false_positive_flag")

    if correctness >= 0.8 and not confidence_calibrated and rank_diff > 0:
        failure_modes.append("correct_but_overconfident")
    elif correctness < 0.5 and not confidence_calibrated and rank_diff > 0:
        failure_modes.append("wrong_but_overconfident")
    elif correctness < 0.5 and confidence_calibrated:
        failure_modes.append("correctly_flagged_uncertain")
    elif correctness >= 0.8 and confidence_calibrated and not flags_missed and not flags_spurious:
        failure_modes.append("clean_correct")

    return ScoreResult(
        task_id=task_full["task_id"],
        correctness=round(correctness, 3),
        confidence_calibrated=confidence_calibrated,
        flags_expected=sorted(expected_flags),
        flags_raised=sorted(flags_raised),
        flags_missed=flags_missed,
        flags_spurious=flags_spurious,
        failure_modes=failure_modes,
    )


def score_run_directory(tasks_dir: str = "runs/tasks", responses_dir: str = "runs/responses") -> list[ScoreResult]:
    """Scores every agent response found in responses_dir against the
    matching full task definition in tasks_dir. Response files are
    expected to be named <task_id>.response.json."""
    tasks_path = Path(tasks_dir)
    responses_path = Path(responses_dir)
    results = []

    for response_file in sorted(responses_path.glob("*.response.json")):
        task_id = response_file.stem.replace(".response", "")
        full_task_file = tasks_path / f"{task_id}.full.json"
        if not full_task_file.exists():
            continue
        task_full = json.loads(full_task_file.read_text())
        agent_response = json.loads(response_file.read_text())
        results.append(score_response(task_full, agent_response))

    return results


def summarize(results: list[ScoreResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    avg_correctness = sum(r.correctness for r in results) / n
    calibration_rate = sum(r.confidence_calibrated for r in results) / n
    failure_counts: dict[str, int] = {}
    for r in results:
        for fm in r.failure_modes:
            failure_counts[fm] = failure_counts.get(fm, 0) + 1
    return {
        "n_tasks": n,
        "avg_correctness": round(avg_correctness, 3),
        "calibration_rate": round(calibration_rate, 3),
        "failure_mode_counts": failure_counts,
    }


if __name__ == "__main__":
    results = score_run_directory()
    summary = summarize(results)
    Path("runs/scores.json").write_text(
        json.dumps({"results": [asdict(r) for r in results], "summary": summary}, indent=2)
    )
    print(json.dumps(summary, indent=2))
