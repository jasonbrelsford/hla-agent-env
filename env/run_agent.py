"""
Harness that sends each generated task to an LLM agent (via the Anthropic
API) and writes its response in the format the scorer expects.

This is intentionally simple -- a single-turn call, no tool use yet -- so
the environment and eval pipeline can be validated end-to-end before
investing in a full multi-turn tool-using agent loop (which is the obvious
next iteration: let the agent query a mock IMGT lookup tool instead of
being handed the frequency table directly).

Usage:
    export ANTHROPIC_API_KEY=...
    python env/run_agent.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a research agent working on HLA haplotype imputation.
You will be given an individual's typing data and a population haplotype
frequency table. Determine the most likely haplotype pair.

You MUST respond with ONLY a JSON object, no other text, in this exact format:
{
  "task_id": "<copy from input>",
  "haplotype_pair": ["<haplotype string>", "<haplotype string>"],
  "confidence": "high" | "medium" | "low",
  "flags": ["version_mismatch" and/or "sampling_bias" and/or "low_resolution_input", ...],
  "reasoning": "<brief explanation>"
}

Only include a flag if the input data actually shows evidence of that issue
(e.g. a reference_release that differs from data_release, a cohort_note in
the frequency table, or a typing record with resolution "low"). Do not flag
things that aren't present in the input. Your confidence should reflect the
real reliability of your answer given any such issues -- do not default to
"high" out of habit.
"""


def run_task(client: anthropic.Anthropic, agent_payload: dict) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(agent_payload, indent=2)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"task_id": agent_payload["task_id"], "error": "no_json_found", "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"task_id": agent_payload["task_id"], "error": "json_decode_failed", "raw": text}


def main(tasks_dir: str = "runs/tasks", responses_dir: str = "runs/responses"):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out_dir = Path(responses_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_files = sorted(Path(tasks_dir).glob("*.agent.json"))
    print(f"Running {len(agent_files)} tasks against {MODEL}...")

    for agent_file in agent_files:
        task_id = agent_file.stem.replace(".agent", "")
        payload = json.loads(agent_file.read_text())
        result = run_task(client, payload)
        out_path = out_dir / f"{task_id}.response.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  {task_id}: confidence={result.get('confidence', 'ERROR')} flags={result.get('flags', [])}")


if __name__ == "__main__":
    main()
