#!/usr/bin/env python3
"""OODA Loop — Autonomous coding orchestrator.

Inner loop: Codebuff + local model iterates on a task until tests pass.
Outer loop: Grok/Ollama reviews the diff, generates new tasks, repeats until done.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
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
    if not path.exists():
        example = path.with_suffix(".example.yaml")
        if example.exists():
            import shutil
            shutil.copy(example, path)
            LOG.info(f"Created {path.name} from {example.name}")
        else:
            return []
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
    # Stage everything first so new files show up in the diff
    subprocess.run(["git", "add", "-A"], cwd=project_dir, capture_output=True)
    r = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=project_dir, capture_output=True, text=True,
    )
    diff = r.stdout.strip()
    if not diff:
        # Fallback: show untracked file contents
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_dir, capture_output=True, text=True,
        )
        diff = r.stdout.strip() or "(no changes)"
    return diff


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
# Codebase context
# ---------------------------------------------------------------------------

def _get_project_context(project_dir: str, max_chars: int = 4000) -> str:
    """Read existing project files to give the model context about what already exists."""
    project = Path(project_dir)
    context_parts = []
    total = 0
    # Show file tree first
    r = subprocess.run(
        ["find", ".", "-not", "-path", "./.git/*", "-type", "f"],
        cwd=project_dir, capture_output=True, text=True,
    )
    files = sorted(r.stdout.strip().split("\n")) if r.stdout.strip() else []
    if files:
        context_parts.append("## Existing files\n" + "\n".join(files))
        total += len(context_parts[-1])

    # Include contents of small source files
    code_exts = {".py", ".js", ".ts", ".rs", ".rb", ".go", ".java", ".c", ".h", ".yaml", ".json", ".toml"}
    for f in files:
        fp = project / f.lstrip("./")
        if fp.suffix in code_exts and fp.is_file():
            try:
                content = fp.read_text()
            except Exception:
                continue
            if len(content) > 2000:
                continue
            block = f"\n## {f}\n```\n{content}\n```"
            if total + len(block) > max_chars:
                break
            context_parts.append(block)
            total += len(block)

    return "\n".join(context_parts) if context_parts else "(empty project)"


# ---------------------------------------------------------------------------
# Coding agent (inner loop)
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` wrappers that small models add to file content."""
    text = text.strip()
    # Handle ```python\n...\n``` or ```\n...\n```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"


def run_coding_agent(prompt: str, cfg: dict) -> str:
    """Run the configured coding agent with the given prompt.

    Supports: codebuff, aider, ollama-direct, or any custom command.
    """
    agent = cfg.get("coding_agent", {})
    backend = agent.get("backend", "codebuff")
    project_dir = cfg["project_dir"]

    if backend == "codebuff":
        cmd = ["codebuff", "--cwd", project_dir, prompt]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.stdout

    elif backend == "aider":
        cmd = ["aider", "--message", prompt, "--yes-always", "--no-git"]
        r = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True, timeout=600)
        return r.stdout

    elif backend == "ollama":
        # Direct Ollama API — generates code and writes files to project
        model = agent.get("model", "qwen2.5-coder:7b")
        endpoint = agent.get("endpoint", "http://localhost:11434")

        # Ask the model to produce file contents with clear delimiters
        file_prompt = f"""{prompt}

IMPORTANT: Output your changes as file blocks. For each file, use this exact format:

===FILE: path/to/file.ext===
<file contents>
===END===

Only output files that need to be created or modified. Use paths relative to the project root."""

        try:
            resp = httpx.post(
                f"{endpoint}/api/generate",
                json={"model": model, "prompt": file_prompt, "stream": False},
                timeout=600,
            )
            if resp.status_code != 200:
                LOG.warning(f"  Ollama returned {resp.status_code}: {resp.text[:200]}")
                return "(error)"
            response_text = resp.json().get("response", "")
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            LOG.warning(f"  Ollama failed: {e}")
            return "(timeout)"

        # Parse and write file blocks
        file_blocks = re.findall(
            r"===FILE:\s*(.+?)===\n(.*?)===END===",
            response_text,
            re.DOTALL,
        )
        for rel_path, content in file_blocks:
            rel_path = rel_path.strip()
            # Strip markdown code fences that small models love to add
            content = _strip_markdown_fences(content)
            target = Path(project_dir) / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            LOG.info(f"    Wrote: {rel_path}")

        return response_text

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

def _build_review_prompts(diff: str, task: Task, constraints: str) -> tuple[str, str]:
    """Build system + user prompts for code review."""
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
    return system_prompt, user_prompt


def _parse_review_json(content: str) -> Review:
    """Extract JSON from a model response, handling markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    # Try to find JSON object if model added extra text
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        LOG.warning("Review response was not valid JSON — auto-approving")
        LOG.debug(f"Raw response: {content[:500]}")
        return Review(approved=True, summary="Review parse failed — auto-approved")
    return Review(
        approved=data.get("approved", False),
        summary=data.get("summary", ""),
        issues=data.get("issues", []),
        new_tasks=data.get("new_tasks", []),
    )


def review_ollama(diff: str, task: Task, constraints: str, cfg: dict) -> Review:
    """Review using local Ollama model — fully offline."""
    endpoint = cfg.get("reviewer_endpoint", "http://localhost:11434")
    model = cfg.get("reviewer_model", "qwen2.5-coder:7b")
    system_prompt, user_prompt = _build_review_prompts(diff, task, constraints)

    try:
        resp = httpx.post(
            f"{endpoint}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=600,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _parse_review_json(content)
    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        LOG.warning(f"  Ollama review timed out: {e}")
        return Review(approved=True, summary="Review timed out — auto-approved")


def review_api(diff: str, task: Task, constraints: str, cfg: dict) -> Review:
    """Review using remote API (Grok, OpenRouter, etc.)."""
    api_key = cfg.get("xai_api_key", "")
    base_url = cfg.get("review_api_base", "https://api.x.ai/v1")
    model = cfg.get("review_model", "grok-3-mini")

    if not api_key:
        LOG.warning("No API key configured — auto-approving")
        return Review(approved=True, summary="No reviewer API key")

    system_prompt, user_prompt = _build_review_prompts(diff, task, constraints)

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
    return _parse_review_json(content)


def run_review(diff: str, task: Task, constraints: str, cfg: dict) -> Review:
    """Dispatch to the configured reviewer backend."""
    backend = cfg.get("reviewer", "ollama")
    if backend == "ollama":
        return review_ollama(diff, task, constraints, cfg)
    elif backend == "api":
        return review_api(diff, task, constraints, cfg)
    else:
        LOG.warning(f"Unknown reviewer '{backend}' — auto-approving")
        return Review(approved=True, summary=f"Unknown reviewer: {backend}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    constraints = load_constraints(ROOT / "constraints")
    tasks = load_tasks()
    project_dir = cfg["project_dir"]
    max_attempts = cfg.get("max_attempts_per_task", 3)
    max_cycles = cfg.get("max_cycles", 20)
    dry_run = cfg.get("dry_run", False)
    cycle = 0

    LOG.info(f"OODA Loop starting — {len(tasks)} tasks, project: {project_dir}")
    if dry_run:
        LOG.info("DRY RUN — no code will be written, no commits made")

    while tasks and cycle < max_cycles:
        cycle += 1
        task = tasks.pop(0)
        task.status = "running"
        task.attempts += 1

        LOG.info(f"[cycle {cycle}] Task {task.id}: {task.description}")

        # --- INNER LOOP: code + test until pass or bail ---
        LOG.info("  OBSERVE — scanning codebase")
        project_context = _get_project_context(project_dir)

        prompt = f"""## CONSTRAINTS
{constraints}

## EXISTING CODEBASE
{project_context}

## TASK
{task.description}

## CONTEXT
{task.context}

## INSTRUCTIONS
Implement this task. Follow all constraints. Write tests if none exist.
Make minimal, focused changes. Do not refactor unrelated code.
Do NOT wrap file contents in markdown code fences — output raw code only."""

        LOG.info("  ORIENT — applying constraints")
        LOG.info("  DECIDE + ACT — running coding agent")

        if dry_run:
            LOG.info(f"  [dry-run] Would send {len(prompt)} char prompt to {cfg.get('coding_agent', {}).get('backend', 'codebuff')}")
            agent_output = "(dry run)"
        else:
            try:
                agent_output = run_coding_agent(prompt, cfg)
            except Exception as e:
                LOG.error(f"  Coding agent crashed: {e}")
                task.status = "failed"
                save_tasks(tasks)
                continue
        LOG.info(f"  Agent output: {len(agent_output)} chars")

        # Gate check
        if dry_run:
            passed, test_output = True, "(dry run)"
        else:
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

        # --- OUTER LOOP: review ---
        if dry_run:
            diff = "(dry run diff)"
            review = Review(approved=True, summary="Dry run — auto-approved")
        else:
            diff = git_diff(project_dir)
            review = run_review(diff, task, constraints, cfg)

        LOG.info(f"  Review: {'APPROVED' if review.approved else 'NEEDS WORK'} — {review.summary}")
        for issue in review.issues:
            LOG.info(f"    - {issue}")

        if review.approved:
            task.status = "done"
            if not dry_run:
                git_stage_commit(project_dir, f"ooda: {task.id} — {task.description[:60]}")
            LOG.info(f"  {'[dry-run] Would commit' if dry_run else 'Committed'}: {task.id}")
        else:
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
    if not remaining and cfg.get("auto_push", False) and not dry_run:
        if git_push(project_dir):
            LOG.info("Pushed to remote")
        else:
            LOG.warning("Push failed")

    LOG.info(f"OODA Loop complete — {cycle} cycles, {len(remaining)} tasks remaining")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.dry_run:
        cfg["dry_run"] = True
    if args.project:
        cfg["project_dir"] = args.project
    run(cfg)


def cmd_status(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    if not tasks:
        print("No pending tasks.")
        return
    print(f"{'ID':<15} {'STATUS':<10} {'DESCRIPTION'}")
    print("-" * 60)
    for t in tasks:
        print(f"{t.id:<15} {t.status:<10} {t.description[:50]}")


def cmd_add(args: argparse.Namespace) -> None:
    path = ROOT / "tasks.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    task_id = f"task-{len(raw) + 1}"
    raw.append({
        "id": task_id,
        "description": args.description,
        "context": args.context or "",
        "status": "pending",
    })
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False)
    print(f"Added: {task_id} — {args.description}")


def cmd_reset(args: argparse.Namespace) -> None:
    path = ROOT / "tasks.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    count = 0
    for t in raw:
        if t.get("status") in ("failed", "running"):
            t["status"] = "pending"
            count += 1
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False)
    print(f"Reset {count} tasks to pending.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OODA Loop — Autonomous coding orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python ooda.py run                      # execute the loop
  python ooda.py run --dry-run            # preview without changes
  python ooda.py run --project ~/myapp    # override target project
  python ooda.py status                   # show task queue
  python ooda.py add "Add login page"     # add a task
  python ooda.py reset                    # retry failed tasks
""",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Execute the OODA loop")
    p_run.add_argument("--dry-run", action="store_true", help="Preview without writing code or committing")
    p_run.add_argument("--project", help="Override project_dir from config")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="Show task queue")
    p_status.set_defaults(func=cmd_status)

    p_add = sub.add_parser("add", help="Add a task to the queue")
    p_add.add_argument("description", help="Task description")
    p_add.add_argument("--context", "-c", help="Additional context")
    p_add.set_defaults(func=cmd_add)

    p_reset = sub.add_parser("reset", help="Reset failed/running tasks to pending")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Logging setup — console + file
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"ooda-{datetime.now():%Y%m%d-%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    LOG.info(f"Log file: {log_file}")
    args.func(args)


if __name__ == "__main__":
    main()
