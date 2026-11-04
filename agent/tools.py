"""Deterministic tools the agents use. No LLM calls in this file.

Part A: files and tests (local, no network).
Part B: git and GitHub.
"""

import base64
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from github import Auth, Github

load_dotenv()

MAX_OUTPUT_CHARS = 4000  # pytest output sent back to the coder is trimmed to save tokens
TEST_TIMEOUT_SECONDS = 120


class ToolError(RuntimeError):
    """Raised when a tool is asked to do something unsafe or impossible."""


# ---------- helpers ----------

def run_cmd(
    args: list[str], cwd: Path, timeout: int = 120, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a command without a shell (args as a list, so nothing gets shell-interpreted)."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _safe_path(repo_path: Path, rel_path: str) -> Path:
    """Resolve a repo-relative path and refuse anything that escapes the repo.

    Paths come from the LLM, so treat them as untrusted (e.g. '../../.env').
    """
    repo = repo_path.resolve()
    target = (repo / rel_path).resolve()
    if not target.is_relative_to(repo):
        raise ToolError(f"Path escapes the repository: {rel_path!r}")
    if ".git" in target.relative_to(repo).parts:
        raise ToolError(f"Refusing to touch git internals: {rel_path!r}")
    return target


def is_test_file(rel_path: str) -> bool:
    p = Path(rel_path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


# ---------- files ----------

IGNORED_PARTS = {"__pycache__", ".venv", "venv", "node_modules"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".exe"}


def list_files(repo_path: Path) -> list[str]:
    """Tracked files as repo-relative POSIX paths, minus build junk the LLM shouldn't see."""
    result = run_cmd(["git", "ls-files"], cwd=repo_path)
    if result.returncode != 0:
        raise ToolError(f"git ls-files failed: {result.stderr.strip()}")
    files = [line for line in result.stdout.splitlines() if line.strip()]
    return sorted(
        f for f in files
        if not IGNORED_PARTS.intersection(Path(f).parts) and Path(f).suffix not in IGNORED_SUFFIXES
    )


def read_file(repo_path: Path, rel_path: str) -> str:
    target = _safe_path(repo_path, rel_path)
    if not target.is_file():
        raise ToolError(f"File not found: {rel_path!r}")
    return target.read_text(encoding="utf-8")


def write_file(repo_path: Path, rel_path: str, content: str) -> None:
    """Overwrite a source file with full new content. Test files are off-limits."""
    if is_test_file(rel_path):
        raise ToolError(f"The coder may not modify test files: {rel_path!r}")
    target = _safe_path(repo_path, rel_path)
    if not target.is_file():
        raise ToolError(f"Can only modify existing files, not create new ones: {rel_path!r}")
    if not content.endswith("\n"):
        content += "\n"
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


# ---------- tests ----------

@dataclass
class TestRun:
    __test__ = False  # stop pytest from trying to collect this class

    returncode: int
    # failing test ID -> its failure details, e.g.
    # "tests.test_numbers::test_percentage[1-4-25.0]" -> "assert 0 == 25.0 ..."
    failures: dict[str, str] = field(default_factory=dict)
    total: int = 0
    output: str = ""  # raw pytest output (trimmed), used when pytest itself breaks

    @property
    def failed(self) -> set[str]:
        return set(self.failures)

    @property
    def ran_ok(self) -> bool:
        """0 = all passed, 1 = some failed. Anything else means pytest itself had a problem."""
        return self.returncode in (0, 1)


def _parse_junit(xml_path: Path) -> tuple[dict[str, str], int]:
    root = ET.parse(xml_path).getroot()
    failures: dict[str, str] = {}
    total = 0
    for case in root.iter("testcase"):
        total += 1
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
            failures[test_id] = (problem.text or problem.get("message", "")).strip()
    return failures, total


def _trim(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the end of the output: that's where the tracebacks and summary are."""
    return text if len(text) <= limit else "...[trimmed]...\n" + text[-limit:]


def run_tests(repo_path: Path, test_paths: list[str] | None = None) -> TestRun:
    """Run pytest in the repo and return which test IDs failed.

    Uses JUnit XML (stdlib parser) instead of scraping pytest's text output,
    which changes between versions.
    """
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "report.xml"
        args = [
            sys.executable, "-m", "pytest",
            *(test_paths or []),
            "-q", "--tb=long",
            "-p", "no:cacheprovider",
            f"--junitxml={xml_path}",
        ]
        try:
            proc = run_cmd(args, cwd=repo_path, timeout=TEST_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return TestRun(returncode=-1, output=f"pytest timed out after {TEST_TIMEOUT_SECONDS}s")

        failures, total = _parse_junit(xml_path) if xml_path.exists() else ({}, 0)
        output = _trim((proc.stdout + "\n" + proc.stderr).strip())
        return TestRun(returncode=proc.returncode, failures=failures, total=total, output=output)


def _test_module(rel_path: str) -> str:
    """'tests/test_numbers.py' -> 'tests.test_numbers' (matches JUnit classname)."""
    return Path(rel_path).with_suffix("").as_posix().replace("/", ".")


@dataclass
class Verdict:
    passed: bool
    summary: str   # one line, for logs and the PR body
    feedback: str  # failure details for the coder: ONLY tests relevant to this issue


def evaluate_against_baseline(
    baseline_failed: set[str], after: TestRun, target_test_files: list[str]
) -> Verdict:
    """Decide if a change fixed the issue without breaking anything.

    Pass = (1) pytest ran, (2) no test in the issue's test files fails,
           (3) no test that passed on main fails now (no regressions).
    Tests that were already failing for OTHER issues are allowed to keep failing,
    and they are left out of the coder's feedback.
    """
    if not after.ran_ok:
        return Verdict(False, f"pytest did not run cleanly (exit code {after.returncode}).", after.output)

    target_modules = tuple(_test_module(p) for p in target_test_files)
    regressions = sorted(after.failed - baseline_failed)
    still_failing = sorted(
        t for t in after.failed if t.split("::")[0] in target_modules and t not in regressions
    )

    if not still_failing and not regressions:
        return Verdict(True, "Target tests pass and no regressions.", "")

    parts, details = [], []
    if still_failing:
        parts.append(f"{len(still_failing)} target test(s) still failing")
        details += [f"FAILED {t}\n{after.failures[t]}" for t in still_failing]
    if regressions:
        parts.append(f"{len(regressions)} previously passing test(s) now fail")
        details += [f"REGRESSION {t}\n{after.failures[t]}" for t in regressions]
    return Verdict(False, "; ".join(parts), _trim("\n\n".join(details)))


# =====================================================================
# Part B: git and GitHub
# =====================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace"))
AGENT_GIT_NAME = os.getenv("AGENT_GIT_NAME", "autonomous-dev-agent")
AGENT_GIT_EMAIL = os.getenv("AGENT_GIT_EMAIL", "autonomous-dev-agent@users.noreply.github.com")


def _github_token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ToolError("Missing GITHUB_TOKEN in .env")
    return token


def _basic_auth() -> str:
    return base64.b64encode(f"x-access-token:{_github_token()}".encode()).decode()


def _redact(text: str) -> str:
    """Never let the token (raw or base64) end up in an error message or a log."""
    token = os.getenv("GITHUB_TOKEN")
    if token:
        text = text.replace(token, "***").replace(_basic_auth(), "***")
    return text


def _git(repo_path: Path, *args: str, network: bool = False) -> str:
    """Run a git command and return stdout. Raises ToolError (token redacted) on failure.

    For network commands the token is passed as a one-off HTTP header scoped to
    github.com, so it is never written into .git/config or the remote URL.
    """
    cmd = ["git"]
    if network:
        cmd += [
            "-c", "credential.helper=",  # don't let a credential manager pop up a login window
            "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {_basic_auth()}",
        ]
    cmd += list(args)
    # GIT_TERMINAL_PROMPT=0: fail with an error instead of hanging on a password prompt
    result = run_cmd(cmd, cwd=repo_path, timeout=180, extra_env={"GIT_TERMINAL_PROMPT": "0"})
    if result.returncode != 0:
        raise ToolError(_redact(f"git {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}"))
    return result.stdout


def _clone_url(repo_full_name: str) -> str:
    return f"https://github.com/{repo_full_name}.git"


def _normalize_url(url: str) -> str:
    return url.strip().lower().removesuffix("/").removesuffix(".git")


def prepare_workspace(repo_full_name: str, default_branch: str) -> Path:
    """Clone the repo (first run) or reset it to a clean copy of the remote default branch.

    Every run starts from exactly what's on GitHub, so leftovers from a failed run
    can never leak into the next one.
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    dest = (WORKSPACE_DIR / repo_full_name.split("/")[-1]).resolve()
    url = _clone_url(repo_full_name)

    if not (dest / ".git").exists():
        _git(WORKSPACE_DIR, "clone", url, str(dest), network=True)
    else:
        origin = _git(dest, "remote", "get-url", "origin").strip()
        if _normalize_url(origin) != _normalize_url(url):
            raise ToolError(f"{dest} is a clone of {origin}, not {url}. Remove it and retry.")
        _git(dest, "fetch", "origin", "--prune", network=True)

    _git(dest, "checkout", "-f", "-B", default_branch, f"origin/{default_branch}")
    _git(dest, "clean", "-fdx")
    return dest


def create_branch(repo_path: Path, branch: str, base: str) -> None:
    """Create (or reset) a local branch from the remote base branch."""
    _git(repo_path, "checkout", "-f", "-B", branch, f"origin/{base}")


def commit_files(repo_path: Path, files: list[str], message: str) -> str:
    """Stage exactly these files (never `git add -A`) and commit. Returns the commit SHA."""
    _git(repo_path, "add", "--", *files)
    if not _git(repo_path, "diff", "--cached", "--name-only").strip():
        raise ToolError("Nothing to commit: the changes are identical to the base branch.")
    _git(
        repo_path,
        "-c", f"user.name={AGENT_GIT_NAME}",
        "-c", f"user.email={AGENT_GIT_EMAIL}",
        "commit", "-m", message,
    )
    return _git(repo_path, "rev-parse", "HEAD").strip()


def push_branch(repo_path: Path, branch: str, default_branch: str) -> None:
    """Push the agent's branch. Force is allowed only because agent/* branches belong to the agent."""
    if branch == default_branch or not branch.startswith("agent/"):
        raise ToolError(f"Refusing to push to {branch!r}: the agent only pushes agent/* branches.")
    _git(repo_path, "push", "--force", "-u", "origin", branch, network=True)


# ---------- GitHub API ----------

@dataclass
class IssueInfo:
    number: int
    title: str
    body: str
    url: str


@lru_cache(maxsize=1)
def _github() -> Github:
    return Github(auth=Auth.Token(_github_token()), timeout=30)


def get_default_branch(repo_full_name: str) -> str:
    return _github().get_repo(repo_full_name).default_branch


def fetch_issue(repo_full_name: str, number: int) -> IssueInfo:
    issue = _github().get_repo(repo_full_name).get_issue(number=number)
    if issue.pull_request is not None:
        raise ToolError(f"#{number} is a pull request, not an issue.")
    if issue.state != "open":
        raise ToolError(f"Issue #{number} is {issue.state}; nothing to fix.")
    return IssueInfo(number=number, title=issue.title, body=issue.body or "", url=issue.html_url)


def find_open_pr(repo_full_name: str, branch: str) -> str | None:
    """Return the URL of an open PR from this branch, if one exists."""
    owner = repo_full_name.split("/")[0]
    for pr in _github().get_repo(repo_full_name).get_pulls(state="open", head=f"{owner}:{branch}"):
        return pr.html_url
    return None


def open_pull_request(repo_full_name: str, head: str, base: str, title: str, body: str) -> str:
    pr = _github().get_repo(repo_full_name).create_pull(title=title, body=body, head=head, base=base)
    return pr.html_url


def _smoke_local(repo: Path) -> None:
    files = list_files(repo)
    print("files:", files)
    print("read_file ok:", read_file(repo, "textutils/numbers.py").splitlines()[0])

    for bad in ["tests/test_numbers.py", "../.env", ".git/config"]:
        try:
            write_file(repo, bad, "x")
            print(f"UNSAFE: write to {bad} was allowed")
        except ToolError as e:
            print("blocked:", e)

    baseline = run_tests(repo)
    print(f"baseline: {len(baseline.failed)} failed / {baseline.total} total (exit {baseline.returncode})")

    verdict = evaluate_against_baseline(baseline.failed, baseline, ["tests/test_numbers.py"])
    print(f"evaluate unchanged code for issue #2 -> passed={verdict.passed}: {verdict.summary}")
    print("--- feedback the coder would get ---")
    print(verdict.feedback)


def _smoke_github(repo_full_name: str, issue_number: int) -> None:
    default_branch = get_default_branch(repo_full_name)
    print("default branch:", default_branch)

    issue = fetch_issue(repo_full_name, issue_number)
    print(f"issue #{issue.number}: {issue.title}")

    repo = prepare_workspace(repo_full_name, default_branch)
    print("workspace:", repo)
    print("HEAD:", _git(repo, "log", "-1", "--oneline").strip())
    print("clean:", not _git(repo, "status", "--porcelain").strip())

    # Check push permission without leaving anything behind:
    # push a temporary branch pointing at the current commit, then delete it.
    check_branch = "agent/permission-check"
    _git(repo, "push", "origin", f"HEAD:refs/heads/{check_branch}", network=True)
    _git(repo, "push", "origin", "--delete", check_branch, network=True)
    print("push permission: ok (temporary branch created and deleted)")

    print("open PR for agent/issue-%d:" % issue_number, find_open_pr(repo_full_name, f"agent/issue-{issue_number}"))


if __name__ == "__main__":
    # Smoke tests:
    #   python -m agent.tools local workspace/textutils
    #   python -m agent.tools github alisha-mund/textutils 2
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode == "local":
        _smoke_local(Path(sys.argv[2] if len(sys.argv) > 2 else "workspace/textutils"))
    elif mode == "github":
        _smoke_github(sys.argv[2], int(sys.argv[3]))
    else:
        sys.exit("usage: python -m agent.tools [local <repo_path> | github <owner/repo> <issue>]")