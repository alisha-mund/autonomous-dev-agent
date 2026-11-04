"""Graph nodes. Each node takes the current state and returns only the fields it changes.

setup     (no LLM)  fetch issue, clean workspace, record baseline test failures, create branch
planner   (LLM)     issue + source code -> Plan
coder     (LLM)     plan + current files (+ test feedback on retries) -> full new file contents
reviewer  (no LLM)  run the test suite and compare against the baseline
give_up   (no LLM)  record a failed run after MAX_ATTEMPTS (no PR)
"""

import functools
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent import tools
from agent.llm import structured_call
from agent.state import MAX_ATTEMPTS, AgentState, CodeChange, Plan

MAX_SOURCE_CHARS = 40_000  # cap on source code sent to the planner (cost + context guard)


# ---------- logging + error handling shared by all nodes ----------

def _entry(step: str, **data: Any) -> dict[str, Any]:
    return {"step": step, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **data}


def traced(step: str) -> Callable:
    """Wrap a node so it always leaves a log entry, and a crash becomes `error` in state
    (which the graph routes to END) instead of an unhandled exception."""

    def decorator(fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict:
            start = time.perf_counter()
            try:
                update = fn(state)
            except Exception as exc:  # noqa: BLE001 - we want every failure in the trace
                return {
                    "error": f"{step}: {exc}",
                    "logs": [_entry(step, status="error", error=str(exc),
                                    seconds=round(time.perf_counter() - start, 2))],
                }
            for entry in update.get("logs", []):
                entry.setdefault("seconds", round(time.perf_counter() - start, 2))
            return update

        return wrapper

    return decorator


def _format_files(repo: Path, paths: list[str]) -> str:
    return "\n\n".join(f"### {p}\n```python\n{tools.read_file(repo, p)}\n```" for p in paths)


# ---------- setup (deterministic) ----------

@traced("setup")
def setup(state: AgentState) -> dict:
    repo_name, number = state["repo_full_name"], state["issue_number"]
    branch = f"agent/issue-{number}"

    default_branch = tools.get_default_branch(repo_name)
    issue = tools.fetch_issue(repo_name, number)

    existing_pr = tools.find_open_pr(repo_name, branch)
    if existing_pr:
        raise tools.ToolError(f"An open PR already exists for this issue: {existing_pr}")

    repo = tools.prepare_workspace(repo_name, default_branch)
    baseline = tools.run_tests(repo)
    if not baseline.ran_ok:
        raise tools.ToolError(f"Tests don't run on {default_branch} (exit {baseline.returncode}).")
    tools.create_branch(repo, branch, default_branch)

    return {
        "issue_title": issue.title,
        "issue_body": issue.body,
        "issue_url": issue.url,
        "repo_path": str(repo),
        "default_branch": default_branch,
        "branch_name": branch,
        "baseline_failures": sorted(baseline.failed),
        "attempts": 0,
        "logs": [_entry("setup", status="ok", issue=issue.title, branch=branch,
                        baseline_failed=len(baseline.failed), baseline_total=baseline.total)],
    }


# ---------- planner (LLM) ----------

PLANNER_SYSTEM = """You are the planning agent in an automated bug-fixing pipeline.
Given a GitHub issue and a repository's source code, find the root cause and write a short,
concrete fix plan for a separate coding agent.

Rules:
- files_to_change: existing SOURCE files that must be edited. Never test files. Usually one file.
- test_files: the test file(s) for the module(s) in files_to_change, picked from the file list.
  Usually exactly one, named after the module (e.g. tests/test_<module>.py). Do not include
  tests for unrelated modules. You cannot see test contents; choose by file and module name.
- steps: concrete edits (which function, what to change). No large code blocks.
- Fix only what the issue describes. Do not refactor unrelated code."""


@traced("planner")
def planner(state: AgentState) -> dict:
    repo = Path(state["repo_path"])
    all_files = tools.list_files(repo)
    source_files = [f for f in all_files if f.endswith(".py") and not tools.is_test_file(f)]

    source_code = _format_files(repo, source_files)
    if len(source_code) > MAX_SOURCE_CHARS:
        raise tools.ToolError(
            f"Source code is {len(source_code)} chars (limit {MAX_SOURCE_CHARS}). "
            "This repo needs a file-selection step before planning."
        )

    user = (
        f"# Issue #{state['issue_number']}: {state['issue_title']}\n{state['issue_body']}\n\n"
        f"# Files in the repository\n" + "\n".join(all_files) + "\n\n"
        f"# Source code (test files are hidden)\n{source_code}"
    )
    plan, usage = structured_call(PLANNER_SYSTEM, user, Plan)

    # Schema-valid is not the same as correct: check the plan against the real repo.
    plan.files_to_change = [f for f in plan.files_to_change if f in source_files]
    plan.test_files = [f for f in plan.test_files if f in all_files and tools.is_test_file(f)]
    if not plan.files_to_change:
        raise tools.ToolError("Planner chose no valid source files to change.")
    if not plan.test_files:
        raise tools.ToolError("Planner chose no valid test files to verify the fix.")

    # The planner guesses test files by name; keep only those that really import the changed code.
    covering = tools.tests_covering(repo, plan.test_files, plan.files_to_change)
    dropped = [t for t in plan.test_files if t not in covering]
    if covering:
        plan.test_files = covering
    return {
        "plan": plan,
        "logs": [_entry("planner", status="ok", plan=plan.model_dump(), dropped_test_files=dropped,
                        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)],
    }


# ---------- coder (LLM) ----------

CODER_SYSTEM = """You are the coding agent in an automated bug-fixing pipeline.
You receive an issue, a fix plan, and the current content of the files you may change.
Return the COMPLETE new content of every file you change.

Rules:
- Only change files listed under "Files you may change".
- Return each file in full, first line to last. Never return diffs, snippets or "...".
- Keep unrelated code, imports, docstrings and formatting exactly as they are.
- Make the minimal change that fixes the issue. Do not add dependencies.
- If test failure output is included, your previous attempt failed. The test output is the
  source of truth about the expected behaviour: fix what it reports."""


@traced("coder")
def coder(state: AgentState) -> dict:
    repo = Path(state["repo_path"])
    plan: Plan = state["plan"]
    attempt = state.get("attempts", 0) + 1

    user = (
        f"# Issue #{state['issue_number']}: {state['issue_title']}\n{state['issue_body']}\n\n"
        f"# Plan\nRoot cause: {plan.root_cause}\n"
        + "\n".join(f"- {s}" for s in plan.steps) + "\n\n"
        f"# Files you may change (current content)\n{_format_files(repo, plan.files_to_change)}"
    )
    if attempt > 1 and state.get("test_output"):
        user += (
            f"\n\n# Your previous attempt (shown above) failed the tests (attempt {attempt - 1})\n"
            f"{state['test_output']}"
        )

    change, usage = structured_call(CODER_SYSTEM, user, CodeChange)

    written, ignored = [], []
    for f in change.files:
        if f.path in plan.files_to_change:
            tools.write_file(repo, f.path, f.content)
            written.append(f.path)
        else:
            ignored.append(f.path)
    if not written:
        raise tools.ToolError(f"Coder returned no files from the plan (got {ignored or 'none'}).")

    return {
        "code_change": change,
        "attempts": attempt,
        "logs": [_entry("coder", status="ok", attempt=attempt, files_written=written,
                        files_ignored=ignored, explanation=change.explanation,
                        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)],
    }


# ---------- reviewer (deterministic) ----------

@traced("reviewer")
def reviewer(state: AgentState) -> dict:
    repo = Path(state["repo_path"])
    run = tools.run_tests(repo)
    verdict = tools.evaluate_against_baseline(
        set(state["baseline_failures"]), run, state["plan"].test_files
    )
    return {
        "tests_passed": verdict.passed,
        "test_output": verdict.feedback,
        "logs": [_entry("reviewer", status="passed" if verdict.passed else "failed",
                        attempt=state.get("attempts", 0), summary=verdict.summary,
                        failed_total=len(run.failed), total=run.total)],
    }

# ---------- give_up (deterministic) ----------

@traced("give_up")
def give_up(state: AgentState) -> dict:
    attempts = state.get("attempts", 0)
    return {
        "error": f"Tests still failing after {attempts} attempt(s) (max {MAX_ATTEMPTS}); no PR opened.",
        "logs": [_entry("give_up", status="failed", attempts=attempts,
                        last_feedback=state.get("test_output", "")[:2000])],
    }

if __name__ == "__main__":
    # Smoke test: one pass through setup -> planner -> coder -> reviewer, no graph, no push.
    #   python -m agent.nodes alisha-mund/textutils 2
    import json
    import sys

    state: dict = {"repo_full_name": sys.argv[1], "issue_number": int(sys.argv[2]), "logs": []}
    for node in (setup, planner, coder, reviewer):
        update = node(state)
        state.update({k: v for k, v in update.items() if k != "logs"})
        state["logs"] += update["logs"]
        print(json.dumps(update["logs"][-1], indent=2, default=str))
        if state.get("error"):
            print("STOPPED:", state["error"])
            break

    if not state.get("error"):
        print("\n--- diff of the coder's change ---")
        print(tools.run_cmd(["git", "diff"], cwd=Path(state["repo_path"])).stdout)