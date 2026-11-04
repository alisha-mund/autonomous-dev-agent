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

    records = []
    for number in args.issue:
        print(f"\n=== Issue #{number} ===")
        records.append(run(args.repo, number, dry_run=args.dry_run))

    print("\n=== Summary ===")
    for r in records:
        detail = r["pr_url"] if r["outcome"] == "pr_opened" else (r["error"] or "")
        print(f"#{r['issue']}: {r['outcome']} | attempts {r['attempts']} | "
              f"{r['prompt_tokens'] + r['completion_tokens']:,} tokens, ~${r['estimated_cost_usd']:.4f} | {detail}")
        print(f"     log: {r['log_file']}")
    total = sum(r["estimated_cost_usd"] for r in records)
    print(f"Total estimated cost: ~${total:.4f}")

    # Non-zero exit code if any issue failed: lets CI (e.g. GitHub Actions) detect failures.
    return 0 if all(not r["error"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())