#!/usr/bin/env python3
"""OODA Loop — Autonomous coding orchestrator.

Inner loop: Codebuff + local model iterates on a task until tests pass.
Outer loop: Grok reviews the diff, generates new tasks, repeats until done.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("ooda")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    description: str
    context: str = ""
    status: str = "pending"  # pending | running | done | failed
    attempts: int = 0


@dataclass
class Review:
    approved: bool
    summary: str
    issues: list[str] = field(default_factory=list)
    new_tasks: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path = ROOT / "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve env vars for API keys
    import os
    for key in ("xai_api_key", "openrouter_api_key"):
        val = cfg.get(key, "")
        if isinstance(val, str) and val.startswith("$"):
            cfg[key] = os.environ.get(val.lstrip("$"), "")
    return cfg


def load_constraints(constraints_dir: Path = ROOT / "constraints") -> str:
    """Concatenate all constraint files into a single system prompt."""
    parts = []
    for md in sorted(constraints_dir.glob("*.md")):
        parts.append(f"# {md.stem.upper()}\n\n{md.read_text()}")
    return "\n\n---\n\n".join(parts)


def load_tasks(path: Path = ROOT / "tasks.yaml") -> list[Task]:
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    return [Task(**t) for t in raw if t.get("status", "pending") == "pending"]


def save_tasks(tasks: list[Task], path: Path = ROOT / "tasks.yaml") -> None:
    data = [
        {"id": t.id, "description": t.description, "context": t.context, "status": t.status}
        for t in tasks
    ]
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_diff(project_dir: str) -> str:
    r = subprocess.run(
        ["git", "diff", "--cached", "HEAD"],
        cwd=project_dir, capture_output=True, text=True,
    )
    diff = r.stdout.strip()
    if not diff:
        r = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=project_dir, capture_output=True, text=True,
        )
        diff = r.stdout.strip()
    return diff or "(no changes)"


def git_stage_commit(project_dir: str, message: str) -> bool:
    subprocess.run(["git", "add", "-A"], cwd=project_dir)
    r = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=project_dir, capture_output=True, text=True,
    )
    return r.returncode == 0


def git_push(project_dir: str) -> bool:
    r = subprocess.run(["git", "push"], cwd=project_dir, capture_output=True, text=True)
    return r.returncode == 0

# ---------------------------------------------------------------------------
# Coding agent (inner loop)
# ---------------------------------------------------------------------------

def run_coding_agent(prompt: str, cfg: dict) -> str:
    """Run the configured coding agent with the given prompt.

    Supports: codebuff, aider, ollama-direct, or any custom command.
    """
    agent = cfg.get("coding_agent", {})
    backend = agent.get("backend", "codebuff")
    project_dir = cfg["project_dir"]

    if backend == "codebuff":
        cmd = ["codebuff", "--message", prompt]
        r = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True, timeout=600)
        return r.stdout

    elif backend == "aider":
        cmd = ["aider", "--message", prompt, "--yes-always", "--no-git"]
        r = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True, timeout=600)
        return r.stdout

    elif backend == "ollama":
        # Direct Ollama API call — returns a code patch as text
        model = agent.get("model", "qwen2.5-coder:14b")
        endpoint = agent.get("endpoint", "http://localhost:11434")
        resp = httpx.post(
            f"{endpoint}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        return resp.json().get("response", "")

    elif backend == "custom":
        # User-defined command template
        cmd_template = agent["command"]
        cmd = cmd_template.replace("{prompt}", prompt).replace("{project_dir}", project_dir)
        r = subprocess.run(cmd, shell=True, cwd=project_dir, capture_output=True, text=True, timeout=600)
        return r.stdout

    else:
        raise ValueError(f"Unknown coding agent backend: {backend}")

# ---------------------------------------------------------------------------
# Test gate
# ---------------------------------------------------------------------------

def run_tests(cfg: dict) -> tuple[bool, str]:
    """Run the project's test suite. Returns (passed, output)."""
    test_cmd = cfg.get("test_command", "echo 'no tests configured'")
    r = subprocess.run(
        test_cmd, shell=True, cwd=cfg["project_dir"],
        capture_output=True, text=True, timeout=300,
    )
    passed = r.returncode == 0
    output = (r.stdout + "\n" + r.stderr).strip()
    return passed, output

# ---------------------------------------------------------------------------
# Grok review (outer loop)
# ---------------------------------------------------------------------------

def grok_review(diff: str, task: Task, constraints: str, cfg: dict) -> Review:
    """Call Grok (xAI API) to review the diff and generate next tasks."""
    api_key = cfg.get("xai_api_key", "")
    base_url = cfg.get("review_api_base", "https://api.x.ai/v1")
    model = cfg.get("review_model", "grok-3-mini")

    if not api_key:
        LOG.warning("No xAI API key — skipping review, auto-approving")
        return Review(approved=True, summary="No reviewer configured")

    system_prompt = f"""You are a senior code reviewer. Review this diff against the task spec.
Apply these constraints:

{constraints}

Respond with ONLY valid JSON:
{{
  "approved": true/false,
  "summary": "brief review",
  "issues": ["issue1", "issue2"],
  "new_tasks": [
    {{"id": "task-N", "description": "what to do next", "context": "why"}}
  ]
}}

If the implementation is complete and correct, set approved=true and new_tasks=[].
If there are issues, set approved=false and describe what needs fixing in new_tasks."""

    user_prompt = f"""## Task
ID: {task.id}
Description: {task.description}
Context: {task.context}

## Diff
```
{diff[:12000]}
```"""

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    # Parse JSON from response (handle markdown code blocks)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]

    data = json.loads(content)
    return Review(
        approved=data.get("approved", False),
        summary=data.get("summary", ""),
        issues=data.get("issues", []),
        new_tasks=data.get("new_tasks", []),
    )

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    constraints = load_constraints(ROOT / "constraints")
    tasks = load_tasks()
    project_dir = cfg["project_dir"]
    max_attempts = cfg.get("max_attempts_per_task", 3)
    max_cycles = cfg.get("max_cycles", 20)
    cycle = 0

    LOG.info(f"OODA Loop starting — {len(tasks)} tasks, project: {project_dir}")

    while tasks and cycle < max_cycles:
        cycle += 1
        task = tasks.pop(0)
        task.status = "running"
        task.attempts += 1

        LOG.info(f"[cycle {cycle}] Task {task.id}: {task.description}")

        # --- INNER LOOP: code + test until pass or bail ---
        prompt = f"""## CONSTRAINTS
{constraints}

## TASK
{task.description}

## CONTEXT
{task.context}

## INSTRUCTIONS
Implement this task. Follow all constraints. Write tests if none exist.
Make minimal, focused changes. Do not refactor unrelated code."""

        LOG.info("  OBSERVE + ORIENT — loading constraints + codebase context")
        LOG.info("  DECIDE + ACT — running coding agent")

        agent_output = run_coding_agent(prompt, cfg)
        LOG.info(f"  Agent output: {len(agent_output)} chars")

        # Gate check
        passed, test_output = run_tests(cfg)
        if not passed:
            LOG.warning(f"  Tests FAILED (attempt {task.attempts}/{max_attempts})")
            if task.attempts < max_attempts:
                task.context += f"\n\nPrevious attempt failed tests:\n{test_output[:2000]}"
                task.status = "pending"
                tasks.insert(0, task)  # retry
                continue
            else:
                task.status = "failed"
                LOG.error(f"  Task {task.id} FAILED after {max_attempts} attempts")
                save_tasks(tasks)
                continue

        LOG.info("  Tests PASSED — staging for review")

        # --- OUTER LOOP: Grok review ---
        diff = git_diff(project_dir)
        review = grok_review(diff, task, constraints, cfg)

        LOG.info(f"  Review: {'APPROVED' if review.approved else 'NEEDS WORK'} — {review.summary}")
        for issue in review.issues:
            LOG.info(f"    - {issue}")

        if review.approved:
            task.status = "done"
            git_stage_commit(project_dir, f"ooda: {task.id} — {task.description[:60]}")
            LOG.info(f"  Committed: {task.id}")
        else:
            # Add review-generated fix tasks
            for nt in review.new_tasks:
                tasks.append(Task(
                    id=nt.get("id", f"fix-{cycle}"),
                    description=nt.get("description", "Fix review issues"),
                    context=nt.get("context", review.summary),
                ))
            LOG.info(f"  Added {len(review.new_tasks)} new tasks from review")

        save_tasks(tasks)

    # --- Done — push if configured ---
    remaining = [t for t in tasks if t.status == "pending"]
    if not remaining and cfg.get("auto_push", False):
        if git_push(project_dir):
            LOG.info("Pushed to remote")
        else:
            LOG.warning("Push failed")

    LOG.info(f"OODA Loop complete — {cycle} cycles, {len(remaining)} tasks remaining")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
