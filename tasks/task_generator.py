"""
Task generator for the HLA Agent Environment.

Generates research tasks for an agent to solve: given partial genotype
typing data for an individual and a population haplotype frequency
reference, impute the most likely haplotype pair and report a
confidence-qualified answer.

Each generated task can have one or more "ambiguity injections" turned on,
simulating real-world conditions an agent must notice and adapt to rather
than silently ignore:

  - VERSION_DRIFT: the allele nomenclature in the reference table is from
    an older IMGT/HLA release than the genotype data; some allele names
    have been renamed/retired/split since.
  - SAMPLING_BIAS: the population frequency table was built from a
    non-representative cohort, with a caveat documented only in metadata.
  - LOW_RESOLUTION: part of the input typing data has been truncated to a
    lower resolution than is needed for a confident call.

This module produces a list of Task objects with everything the eval
scorer needs to score correctness AND whether the agent noticed/adapted
to the injected issue.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class Ambiguity(str, Enum):
    VERSION_DRIFT = "version_drift"
    SAMPLING_BIAS = "sampling_bias"
    LOW_RESOLUTION = "low_resolution"
    NONE = "none"


# A small, self-contained mock allele/frequency universe so the environment
# runs deterministically without needing live IMGT/HLA downloads. Swap
# `load_real_reference_data()` in to point at actual IPD-IMGT/HLA +
# NMDP/1000 Genomes frequency tables later — the task/eval logic doesn't
# need to change.
LOCI = ["A", "B", "C", "DRB1", "DQB1"]

MOCK_ALLELES = {
    "A": ["A*01:01", "A*02:01", "A*03:01", "A*24:02", "A*11:01"],
    "B": ["B*07:02", "B*08:01", "B*15:01", "B*44:02", "B*51:01"],
    "C": ["C*07:01", "C*07:02", "C*04:01", "C*03:04", "C*06:02"],
    "DRB1": ["DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01", "DRB1*15:01"],
    "DQB1": ["DQB1*02:01", "DQB1*03:01", "DQB1*05:01", "DQB1*06:02", "DQB1*02:02"],
}

# Renamed/retired alleles to simulate a real IMGT/HLA release boundary.
# Maps OLD (release N) name -> NEW (release N+1) name.
VERSION_DRIFT_RENAMES = {
    "A": {"A*02:01": "A*02:01:01"},
    "B": {"B*15:01": "B*15:01:01"},
    "DRB1": {"DRB1*03:01": "DRB1*03:01:01"},
}


@dataclass
class TypingRecord:
    locus: str
    alleles: list[str]               # 1 or 2 alleles depending on resolution
    resolution: str                  # "high" or "low"


@dataclass
class FrequencyEntry:
    haplotype: str                   # e.g. "A*01:01~B*08:01~DRB1*03:01"
    frequency: float
    reference_release: str           # e.g. "IMGT/HLA 3.45.0"
    cohort_note: Optional[str] = None  # populated when SAMPLING_BIAS is injected


@dataclass
class Task:
    task_id: str
    individual_typing: list[TypingRecord]
    frequency_table: list[FrequencyEntry]
    data_release: str                 # release the genotype data was typed against
    reference_release: str            # release the frequency table was built against
    injected_ambiguities: list[Ambiguity]
    ground_truth_haplotype_pair: list[str]
    ground_truth_confidence: str      # "high" / "medium" / "low" -- what a well-calibrated answer should say
    notes_for_scorer: dict = field(default_factory=dict)

    def to_agent_payload(self) -> dict:
        """What the agent actually sees. Ground truth and scorer notes are withheld."""
        return {
            "task_id": self.task_id,
            "individual_typing": [asdict(t) for t in self.individual_typing],
            "frequency_table": [asdict(f) for f in self.frequency_table],
            "data_release": self.data_release,
            "instructions": (
                "Given the individual's typing data and the population frequency "
                "table, determine the most likely haplotype pair. State your "
                "confidence (high/medium/low) and explicitly flag any data quality "
                "issues, version mismatches, or sampling caveats that affect your answer."
            ),
        }


def _make_typing(rng: random.Random, low_res_loci: set[str]) -> list[TypingRecord]:
    records = []
    for locus in LOCI:
        alleles = rng.sample(MOCK_ALLELES[locus], 2)
        if locus in low_res_loci:
            # Low resolution: truncate to 2-digit family, ambiguous between
            # the two real high-res alleles -- a realistic typing limitation.
            family = alleles[0].split(":")[0]
            records.append(TypingRecord(locus=locus, alleles=[family], resolution="low"))
        else:
            records.append(TypingRecord(locus=locus, alleles=alleles, resolution="high"))
    return records


def _make_frequency_table(
    rng: random.Random,
    typing: list[TypingRecord],
    inject_drift: bool,
    inject_bias: bool,
) -> tuple[list[FrequencyEntry], str, str]:
    data_release = "IMGT/HLA 3.49.0"
    reference_release = "IMGT/HLA 3.48.0" if inject_drift else data_release

    entries = []
    cohort_note = None
    if inject_bias:
        cohort_note = (
            "Frequencies derived from a single-registry donor cohort "
            "(n=4,200, predominantly one regional ancestry group); "
            "may not generalize to the queried individual's population."
        )

    # Build a small set of plausible haplotypes from the (possibly stale)
    # reference release naming.
    rename_map = {}
    if inject_drift:
        for locus, m in VERSION_DRIFT_RENAMES.items():
            rename_map.update({v: k for k, v in m.items()})  # new -> old

    for _ in range(4):
        combo = []
        for rec in typing:
            allele = rng.choice(rec.alleles) if rec.resolution == "high" else rec.alleles[0] + ":01"
            if inject_drift and allele in rename_map:
                allele = rename_map[allele]  # express in the OLDER nomenclature
            combo.append(allele)
        haplotype = "~".join(combo)
        entries.append(
            FrequencyEntry(
                haplotype=haplotype,
                frequency=round(rng.uniform(0.0001, 0.05), 5),
                reference_release=reference_release,
                cohort_note=cohort_note,
            )
        )
    return entries, data_release, reference_release


def generate_task(
    task_id: str,
    ambiguities: list[Ambiguity],
    seed: Optional[int] = None,
) -> Task:
    rng = random.Random(seed if seed is not None else hash(task_id) & 0xFFFFFFFF)

    low_res_loci = {"DQB1"} if Ambiguity.LOW_RESOLUTION in ambiguities else set()
    typing = _make_typing(rng, low_res_loci)

    inject_drift = Ambiguity.VERSION_DRIFT in ambiguities
    inject_bias = Ambiguity.SAMPLING_BIAS in ambiguities
    freq_table, data_release, reference_release = _make_frequency_table(
        rng, typing, inject_drift, inject_bias
    )

    # Ground truth: the "true" haplotype pair is just two of the high-res
    # alleles per locus, deterministic given the seed -- known exactly
    # because we generated it, which is the whole point of using synthetic
    # data for eval scoring before moving to real public datasets.
    pair = []
    for rec in typing:
        if rec.resolution == "high":
            pair.append(rec.alleles[0])
        else:
            pair.append(rec.alleles[0] + ":01")
    ground_truth_pair = ["~".join(pair), "~".join(pair)]  # homozygous-style simplification

    if low_res_loci or inject_bias:
        expected_confidence = "low" if (low_res_loci and inject_bias) else "medium"
    else:
        expected_confidence = "high"

    return Task(
        task_id=task_id,
        individual_typing=typing,
        frequency_table=freq_table,
        data_release=data_release,
        reference_release=reference_release,
        injected_ambiguities=ambiguities,
        ground_truth_haplotype_pair=ground_truth_pair,
        ground_truth_confidence=expected_confidence,
        notes_for_scorer={
            "low_res_loci": list(low_res_loci),
            "version_drift_injected": inject_drift,
            "sampling_bias_injected": inject_bias,
        },
    )


def generate_task_suite(out_dir: str = "runs/tasks", n_per_condition: int = 5) -> list[Task]:
    """Generates a balanced suite covering each ambiguity condition plus
    a clean control condition, and writes both the full task (for the
    scorer) and the agent-facing payload (for the agent) to disk."""
    conditions = [
        [],
        [Ambiguity.VERSION_DRIFT],
        [Ambiguity.SAMPLING_BIAS],
        [Ambiguity.LOW_RESOLUTION],
        [Ambiguity.VERSION_DRIFT, Ambiguity.SAMPLING_BIAS],
    ]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cond in conditions:
        cond_name = "_".join(a.value for a in cond) or "control"
        for i in range(n_per_condition):
            task_id = f"{cond_name}_{i:03d}"
            task = generate_task(task_id, cond, seed=hash(task_id) & 0xFFFFFFFF)
            tasks.append(task)

            full_path = out / f"{task_id}.full.json"
            agent_path = out / f"{task_id}.agent.json"
            full_path.write_text(json.dumps(asdict(task), indent=2, default=str))
            agent_path.write_text(json.dumps(task.to_agent_payload(), indent=2, default=str))

    return tasks


if __name__ == "__main__":
    tasks = generate_task_suite()
    print(f"Generated {len(tasks)} tasks across 5 conditions -> runs/tasks/")
