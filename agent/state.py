"""Graph state and the structured outputs the agents produce.

- AgentState (TypedDict) is the shared state LangGraph passes between nodes.
- Plan / FileChange / CodeChange (Pydantic) are what the LLM must return;
  they're validated because model output is untrusted input.
"""

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


# ---------- LLM output schemas ----------

class Plan(BaseModel):
    """Planner output: what's wrong and where to fix it."""

    root_cause: str = Field(description="One or two sentences explaining why the bug happens.")
    files_to_change: list[str] = Field(
        description="Repo-relative paths of the SOURCE files to modify (not test files)."
    )
    test_files: list[str] = Field(
        description="Repo-relative paths of the test files that cover this behaviour."
    )
    steps: list[str] = Field(description="Concrete, ordered changes the coder should make.")


class FileChange(BaseModel):
    path: str = Field(description="Repo-relative path of the file. Must be one of the planned files.")
    content: str = Field(description="The COMPLETE new content of the file, not a diff or a snippet.")


class CodeChange(BaseModel):
    """Coder output: full replacement contents for each changed file."""

    explanation: str = Field(description="Short description of what was changed and why.")
    files: list[FileChange]


# ---------- Graph state ----------

class AgentState(TypedDict, total=False):
    # Input
    repo_full_name: str          # e.g. "alisha-mund/textutils"
    issue_number: int
    dry_run: bool                # True = run everything except commit/push/PR
    issue_title: str
    issue_body: str
    issue_url: str

    # Workspace
    repo_path: str               # local clone the agents work in
    default_branch: str          # e.g. "main" (read from GitHub, not assumed)
    branch_name: str             # e.g. "agent/issue-2"
    baseline_failures: list[str] # test IDs already failing on main before any change

    # Agent outputs
    plan: Plan
    code_change: CodeChange      # latest attempt only
    tests_passed: bool
    test_output: str             # trimmed pytest output, fed back to the coder on failure
    attempts: int                # coder attempts so far (capped at MAX_ATTEMPTS)

    # Result
    pr_url: str
    error: str

    # Append-only trace of every step (the reducer adds instead of overwriting)
    logs: Annotated[list[dict[str, Any]], operator.add]


MAX_ATTEMPTS = 3