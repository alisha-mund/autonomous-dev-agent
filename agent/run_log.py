"""One JSON file per run in runs/, so any run (especially a failed one) can be traced afterwards.

The records are also the input for evaluation: success rate, attempts, tokens and cost per issue.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import tools
from agent.llm import estimate_cost_usd, get_model_name

RUNS_DIR = Path(os.getenv("RUNS_DIR", "./runs"))
MAX_DIFF_CHARS = 20_000


def _outcome(state: dict[str, Any], dry_run: bool) -> str:
    if state.get("pr_url") and not state.get("error"):
        return "pr_opened"
    if state.get("error"):
        return "gave_up" if any(e.get("step") == "give_up" for e in state.get("logs", [])) else "error"
    return "passed_dry_run" if dry_run else "unknown"


def _final_diff(state: dict[str, Any]) -> str:
    """Code changes vs the base branch (committed or not), so a failed attempt can be inspected."""
    repo, base = state.get("repo_path"), state.get("default_branch")
    if not repo or not base:
        return ""
    result = tools.run_cmd(["git", "diff", f"origin/{base}"], cwd=Path(repo))
    diff = result.stdout if result.returncode == 0 else ""
    return diff if len(diff) <= MAX_DIFF_CHARS else diff[:MAX_DIFF_CHARS] + "\n...[trimmed]"


def _model_name() -> str:
    try:
        return get_model_name()
    except RuntimeError:  # config missing: the record should still be written
        return "unknown"


def build_record(state: dict[str, Any], started: datetime, finished: datetime, dry_run: bool) -> dict[str, Any]:
    logs = state.get("logs", [])
    llm_calls = [e for e in logs if "prompt_tokens" in e]
    tokens_in = sum(e["prompt_tokens"] for e in llm_calls)
    tokens_out = sum(e["completion_tokens"] for e in llm_calls)
    return {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ") + f"-issue{state.get('issue_number')}",
        "repo": state.get("repo_full_name"),
        "issue": state.get("issue_number"),
        "issue_title": state.get("issue_title"),
        "model": _model_name(),
        "dry_run": dry_run,
        "outcome": _outcome(state, dry_run),
        "attempts": state.get("attempts", 0),
        "pr_url": state.get("pr_url"),
        "error": state.get("error"),
        "started_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "llm_calls": len(llm_calls),
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "estimated_cost_usd": round(estimate_cost_usd(tokens_in, tokens_out), 5),
        "steps": logs,
        "final_diff": _final_diff(state),
    }


def write_record(record: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def now() -> datetime:
    return datetime.now(timezone.utc)