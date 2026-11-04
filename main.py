"""CLI entry point.

    python main.py --repo alisha-mund/textutils --issue 2
    python main.py --repo alisha-mund/textutils --issue 1 2 3
    python main.py --repo alisha-mund/textutils --issue 2 --dry-run
"""

import argparse
import sys

from agent.graph import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix GitHub issues with a planner/coder/reviewer agent pipeline.")
    parser.add_argument("--repo", required=True, help="Target repository, e.g. alisha-mund/textutils")
    parser.add_argument("--issue", required=True, type=int, nargs="+", help="Issue number(s) to fix")
    parser.add_argument("--dry-run", action="store_true", help="Run everything except commit/push/PR")
    args = parser.parse_args()

    results = []
    for number in args.issue:
        print(f"\n=== Issue #{number} ===")
        state = run(args.repo, number, dry_run=args.dry_run)
        attempts = state.get("attempts", 0)
        if state.get("error"):
            outcome = f"FAILED after {attempts} attempt(s): {state['error']}"
        elif args.dry_run:
            outcome = f"tests pass after {attempts} attempt(s) (dry run, nothing pushed)"
        else:
            outcome = f"PR opened after {attempts} attempt(s): {state.get('pr_url')}"
        results.append((number, not state.get("error"), outcome))

    print("\n=== Summary ===")
    for number, ok, outcome in results:
        print(f"#{number}: {outcome}")

    # Non-zero exit code if any issue failed: lets CI (e.g. GitHub Actions) detect failures.
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())