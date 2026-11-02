# Autonomous Dev Agent
Multi-agent pipeline that takes a GitHub issue, writes a fix, runs the tests and opens a pull request.

Planner agent → Coder agent → Reviewer agent (runs pytest) → retry on failure → open PR

Stack: Python, LangGraph, Azure OpenAI, PyGithub, pytest, Docker
Status: in active development, first working version this week.
