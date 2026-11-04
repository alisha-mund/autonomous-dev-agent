"""LangGraph wiring: who runs after whom, and the retry loop.

    START -> setup -> planner -> coder -> reviewer --passed--> pr_creator -> END
                                  ^          |
                                  |          +--failed, attempts < MAX--> coder
                                  |          +--failed, attempts = MAX--> give_up -> END
    (any node that sets `error` goes straight to END; with dry_run, "passed" goes to END)
"""

import sys
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent import run_log
from agent.nodes import coder, give_up, planner, pr_creator, reviewer, setup
from agent.state import MAX_ATTEMPTS, AgentState

# Safety net on top of MAX_ATTEMPTS: LangGraph aborts if the graph ever takes more steps than this.
# Longest legit path: setup, planner, 3 x (coder, reviewer), give_up = 9 steps.
RECURSION_LIMIT = 15


def _next_unless_error(next_node: str):
    def route(state: AgentState) -> str:
        return END if state.get("error") else next_node
    route.__name__ = f"to_{next_node}_unless_error"
    return route


def make_review_router(dry_run: bool = False):
    def route_after_review(state: AgentState) -> str:
        if state.get("error"):
            return END
        if state.get("tests_passed"):
            return END if dry_run else "pr_creator"
        if state.get("attempts", 0) < MAX_ATTEMPTS:
            return "coder"
        return "give_up"
    return route_after_review


def build_graph(dry_run: bool = False):
    g = StateGraph(AgentState)
    g.add_node("setup", setup)
    g.add_node("planner", planner)
    g.add_node("coder", coder)
    g.add_node("reviewer", reviewer)
    g.add_node("give_up", give_up)
    g.add_node("pr_creator", pr_creator)

    g.add_edge(START, "setup")
    g.add_conditional_edges("setup", _next_unless_error("planner"), ["planner", END])
    g.add_conditional_edges("planner", _next_unless_error("coder"), ["coder", END])
    g.add_conditional_edges("coder", _next_unless_error("reviewer"), ["reviewer", END])
    g.add_conditional_edges("reviewer", make_review_router(dry_run), ["coder", "give_up", "pr_creator", END])
    g.add_edge("give_up", END)
    g.add_edge("pr_creator", END)
    return g.compile()


def _summarize(node: str, update: dict[str, Any]) -> str:
    entry = (update.get("logs") or [{}])[-1]
    if entry.get("status") == "error":
        return f"[{node}] ERROR: {entry.get('error')}"
    if node == "setup":
        return f"[setup] {entry.get('issue')} | baseline: {entry.get('baseline_failed')} failing"
    if node == "planner":
        plan = entry.get("plan", {})
        return f"[planner] change {plan.get('files_to_change')} | verify with {plan.get('test_files')}"
    if node == "coder":
        return f"[coder] attempt {entry.get('attempt')}: {entry.get('explanation')}"
    if node == "reviewer":
        return f"[reviewer] {entry.get('status').upper()}: {entry.get('summary')}"
    if node == "pr_creator":
        return f"[pr_creator] commit {entry.get('commit')} on {entry.get('branch')} -> {entry.get('pr_url')}"
    return f"[{node}] {entry.get('status')}"


def run(repo_full_name: str, issue_number: int, dry_run: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Run the pipeline for one issue, write a JSON run record, and return the record."""
    graph = build_graph(dry_run)
    initial: AgentState = {
        "repo_full_name": repo_full_name, "issue_number": issue_number, "dry_run": dry_run, "logs": [],
    }
    final: dict[str, Any] = dict(initial)
    started = run_log.now()

    try:
        for mode, data in graph.stream(initial, {"recursion_limit": RECURSION_LIMIT},
                                       stream_mode=["updates", "values"]):
            if mode == "updates" and verbose:
                for node, update in data.items():
                    print(_summarize(node, update))
            elif mode == "values":
                final = data
    except Exception as exc:  # e.g. recursion limit hit: still write a record of what happened
        final = {**final, "error": f"graph: {exc}"}

    record = run_log.build_record(final, started, run_log.now(), dry_run)
    record["log_file"] = str(run_log.write_record(record))
    return record


if __name__ == "__main__":
    #   python -m agent.graph --mermaid                    print the graph diagram (for the README)
    #   python -m agent.graph alisha-mund/textutils 2      dry run on one issue (no push, no PR)
    #   (use main.py for real runs)
    if sys.argv[1:] == ["--mermaid"]:
        print(build_graph().get_graph().draw_mermaid())
        sys.exit()

    rec = run(sys.argv[1], int(sys.argv[2]), dry_run=True)
    print(f"\nRESULT: {rec['outcome']} after {rec['attempts']} attempt(s) | "
          f"{rec['prompt_tokens'] + rec['completion_tokens']:,} tokens, ~${rec['estimated_cost_usd']:.4f}")
    if rec["error"]:
        print("ERROR:", rec["error"])
    print("log:", rec["log_file"])