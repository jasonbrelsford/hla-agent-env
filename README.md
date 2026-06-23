# hla-agent-env

An evaluation environment for testing LLM agent reasoning on HLA haplotype imputation tasks under realistic ambiguity conditions.

## Structure

```
tasks/        Task generator (synthetic typing + frequency data with injected ambiguities)
env/          Agent harness (sends tasks to Claude, collects responses)
eval/         Scorer (correctness, calibration, failure mode classification)
runs/         Generated artifacts (gitignored)
```

## Quick Start

```bash
# Generate tasks
python tasks/task_generator.py

# Run the agent (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=...
python env/run_agent.py

# Score results
python eval/scorer.py
```

## Ambiguity Conditions

Tasks are generated across five conditions to test whether the agent notices and adapts to data quality issues:

- **control** — clean data, no tricks
- **version_drift** — frequency table uses older IMGT/HLA nomenclature than the typing data
- **sampling_bias** — frequency table built from a non-representative cohort (noted in metadata)
- **low_resolution** — some typing loci truncated to low resolution
- **version_drift + sampling_bias** — combined

## Scoring Axes

- Correctness (haplotype overlap)
- Confidence calibration (over-confidence penalized more than under-confidence)
- Flag detection (did the agent notice the injected issue?)
- Failure mode classification (hallucinated allele, overconfident, missed ambiguity, etc.)
