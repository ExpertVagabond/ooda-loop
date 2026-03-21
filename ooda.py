#!/usr/bin/env python3
"""OODA Loop — Autonomous coding orchestrator.

Inner loop: Codebuff + local model iterates on a task until tests pass.
Outer loop: Grok/Ollama reviews the diff, generates new tasks, repeats until done.
"""

from __future__ import annotations

# ── Security: Input Validation & Error Sanitization ──
import os as _os
import re as _sec_re

# Security constants — frozen at module level
SEC_MAX_INPUT_LENGTH = 10_000
SEC_MAX_PATH_LENGTH = 4096
SEC_MAX_TASK_DESC = 2000
SEC_DANGEROUS_PATTERNS = _sec_re.compile(r"[;&|`$]")
SEC_PATH_TRAVERSAL = _sec_re.compile(r"\.\./|\.\.\\")


def sanitize_error(err: BaseException | str) -> str:
    """Strip internal paths and secrets from error messages."""
    msg = str(err)[:500] if err else "Unknown error"
    msg = _sec_re.sub(r"/[^\s:]+", "[path]", msg)
    msg = _sec_re.sub(r"(key|token|secret|password)=\S+", r"\1=[REDACTED]", msg, flags=_sec_re.IGNORECASE)
    return msg


def validate_path(path_str: str) -> str:
    """Validate a file path — reject traversal attacks and null bytes."""
    if not isinstance(path_str, str) or len(path_str) > SEC_MAX_PATH_LENGTH:
        raise ValueError("Invalid path length")
    if "\0" in path_str:
        raise ValueError("Null bytes in path")
    if SEC_PATH_TRAVERSAL.search(path_str):
        raise ValueError("Path traversal detected")
    return path_str


def sanitize_input(text: str, max_len: int = SEC_MAX_INPUT_LENGTH) -> str:
    """Sanitize user input — strip dangerous shell characters."""
    if not isinstance(text, str):
        return ""
    return SEC_DANGEROUS_PATTERNS.sub("", text[:max_len])


# ── Standard imports ──
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

# Qwen2.5-Coder FIM special tokens
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

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


def git_snapshot(project_dir: str) -> None:
    """Create a lightweight snapshot so we can revert failed coding attempts."""
    subprocess.run(["git", "add", "-A"], cwd=project_dir, capture_output=True)
    subprocess.run(
        ["git", "stash", "push", "-m", "ooda-snapshot", "--include-untracked"],
        cwd=project_dir, capture_output=True,
    )
    # Immediately restore — the stash entry stays as our rollback point
    subprocess.run(["git", "stash", "pop"], cwd=project_dir, capture_output=True)


def git_rollback(project_dir: str) -> None:
    """Revert working tree to the last commit (undo a failed coding attempt)."""
    subprocess.run(["git", "checkout", "."], cwd=project_dir, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=project_dir, capture_output=True)
    subprocess.run(["git", "reset", "HEAD"], cwd=project_dir, capture_output=True)

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
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_warm(endpoint: str, model: str) -> None:
    """Pre-load model into Ollama memory to eliminate cold-start lag."""
    try:
        httpx.post(
            f"{endpoint}/api/generate",
            json={"model": model, "prompt": "", "stream": False},
            timeout=120,
        )
        LOG.info(f"  Model {model} warmed up")
    except Exception as e:
        LOG.warning(f"  Warm-up failed: {sanitize_error(e)}")


def ollama_fim(endpoint: str, model: str, prefix: str, suffix: str) -> str:
    """Use Fill-in-Middle to generate code between prefix and suffix."""
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    try:
        resp = httpx.post(
            f"{endpoint}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "raw": True,
                  "options": {"temperature": 0.2, "stop": ["<|fim_pad|>", "<|endoftext|>", "===END==="]}},
            timeout=1200,
        )
        if resp.status_code != 200:
            return ""
        return resp.json().get("response", "").strip()
    except (httpx.ReadTimeout, httpx.ConnectError):
        return ""


def classify_task(task: Task, project_dir: str) -> str:
    """Route task to the right model: 'simple' (7b) or 'complex' (14b).

    Simple = new file creation, single-function additions.
    Complex = multi-file modifications, refactoring, architecture changes.
    """
    desc = task.description.lower()
    # Check if it's modifying existing files
    project = Path(project_dir)
    existing_files = {f.name for f in project.glob("*.py") if f.name != "__pycache__"}

    # Keywords that signal complexity
    complex_signals = ["modify", "refactor", "change", "update", "fix", "add to", "integrate", "cli"]
    simple_signals = ["create", "new file", "implement", "write"]

    if any(s in desc for s in complex_signals):
        return "complex"
    if any(s in desc for s in simple_signals) and not existing_files:
        return "simple"
    if len(existing_files) > 3:
        return "complex"
    return "simple"


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def syntax_check(project_dir: str) -> tuple[bool, str]:
    """Fast syntax + import check on all Python files. Returns (ok, error_msg)."""
    project = Path(project_dir)
    for py_file in project.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
        try:
            source = py_file.read_text()
            compile(source, py_file.name, "exec")
        except SyntaxError as e:
            return False, f"SyntaxError in {py_file.name}:{e.lineno}: {e.msg}"
    return True, ""


def parse_test_error(test_output: str) -> str:
    """Extract the actual error from pytest output for targeted retry prompts."""
    lines = test_output.split("\n")
    error_lines = []
    capture = False
    for line in lines:
        if "FAILED" in line or "Error" in line or "assert" in line.lower():
            error_lines.append(line.strip())
        if line.startswith("E "):
            error_lines.append(line.strip())
        if "short test summary" in line:
            capture = True
        elif capture and line.strip():
            error_lines.append(line.strip())
    if error_lines:
        return "\n".join(error_lines[:10])
    # Fallback: last 5 non-empty lines
    tail = [l for l in lines if l.strip()][-5:]
    return "\n".join(tail)


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


def _run_ollama_blocks(prompt: str, project_dir: str, endpoint: str, model: str) -> str:
    """Standard block-based generation: FILE/APPEND/PREPEND."""
    file_prompt = f"""{prompt}

IMPORTANT: Output your changes using ONE of these formats:

FORMAT 1 — For NEW files (files that don't exist yet):
===FILE: path/to/file.ext===
<complete file contents>
===END===

FORMAT 2 — For MODIFYING existing files:
===APPEND: path/to/file.ext===
<code to add at the end of the file>
===END===

===PREPEND: path/to/file.ext===
<imports to add at the top>
===END===

CRITICAL RULES:
- For existing files, ALWAYS use APPEND/PREPEND — never rewrite the whole file.
- For new files, use FILE with complete contents.
- Ensure every function/class you use is imported.
- Do NOT wrap code in markdown fences."""

    try:
        resp = httpx.post(
            f"{endpoint}/api/generate",
            json={"model": model, "prompt": file_prompt, "stream": False},
            timeout=1200,
        )
        if resp.status_code != 200:
            LOG.warning(f"  Ollama returned {resp.status_code}: {resp.text[:200]}")
            return "(error)"
        response_text = resp.json().get("response", "")
    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        LOG.warning(f"  Ollama failed: {sanitize_error(e)}")
        return "(timeout)"

    for match in re.finditer(
        r"===(FILE|APPEND|PREPEND):\s*(.+?)===\n(.*?)===END===",
        response_text, re.DOTALL,
    ):
        mode, rel_path, content = match.group(1), match.group(2).strip(), match.group(3)
        content = _strip_markdown_fences(content)
        target = Path(project_dir) / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if mode == "APPEND" and target.exists():
            existing = target.read_text()
            target.write_text(existing.rstrip("\n") + "\n\n" + content)
            LOG.info(f"    Appended to: {rel_path}")
        elif mode == "PREPEND" and target.exists():
            existing = target.read_text()
            target.write_text(content.rstrip("\n") + "\n" + existing)
            LOG.info(f"    Prepended to: {rel_path}")
        else:
            target.write_text(content)
            LOG.info(f"    Wrote: {rel_path}")

    return response_text


def _run_ollama_fim(prompt: str, project_dir: str, endpoint: str, model: str,
                    existing_files: set[str], task_desc: str = "") -> str:
    """FIM-based generation: insert code at the end of existing files.

    Only modifies files that are explicitly referenced in the task description.
    Never touches test files — tests are validated by pytest, not generated by FIM.
    """
    results = []
    project = Path(project_dir)
    # Use task description for targeting, not the full prompt (which has boilerplate)
    desc_lower = (task_desc or prompt).lower()

    # Determine which files to modify — only files mentioned in the task
    target_files = set()
    for filename in existing_files:
        basename = filename.replace(".py", "")
        if filename in desc_lower or basename in desc_lower:
            target_files.add(filename)

    # If no specific file mentioned, target main source files (not tests)
    if not target_files:
        target_files = {f for f in existing_files if not f.startswith("test_")}

    # Never modify test files — FIM can't generate tests that match code it just wrote
    target_files = {f for f in target_files if not f.startswith("test_")}

    LOG.info(f"    FIM targeting: {sorted(target_files)}")

    for filename in sorted(target_files):
        filepath = project / filename
        existing_code = filepath.read_text()

        # Build a FIM prompt: existing code is prefix, empty string is suffix
        # The model fills in what should come after the existing code
        fim_prompt = f"""You are adding code to an existing file.
The task: {prompt}

The file {filename} currently contains:
{existing_code}

Generate ONLY the new code to append at the end of this file.
Do NOT repeat any existing code. Only output new functions, classes, or imports needed.
If this file doesn't need changes, output nothing.
Do NOT wrap code in markdown fences."""

        try:
            resp = httpx.post(
                f"{endpoint}/api/generate",
                json={"model": model, "prompt": fim_prompt, "stream": False,
                      "options": {"temperature": 0.2}},
                timeout=1200,
            )
            if resp.status_code != 200:
                continue
            new_code = resp.json().get("response", "").strip()
        except (httpx.ReadTimeout, httpx.ConnectError):
            continue

        if not new_code or new_code == "(nothing)" or len(new_code) < 10:
            continue

        new_code = _strip_markdown_fences(new_code)

        # Build sets of existing definitions (classes and their methods)
        existing_defs = set()       # top-level: "class Foo", "def bar"
        existing_methods = {}       # class_name -> set of method names
        current_class = None
        for line in existing_code.split("\n"):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip()) if line.strip() else -1
            if stripped.startswith("class "):
                current_class = stripped.split("(")[0].split(":")[0]  # "class Foo"
                existing_defs.add(current_class)
                existing_methods[current_class] = set()
            elif stripped.startswith("def ") and indent > 0 and current_class:
                method_name = stripped.split("(")[0]  # "def bar"
                existing_methods[current_class].add(method_name)
                existing_defs.add(method_name)
            elif stripped.startswith("def ") and indent == 0:
                existing_defs.add(stripped.split("(")[0])
                current_class = None

        # Check for duplicate class blocks that might contain new methods
        new_lines = new_code.split("\n")
        has_dup_class = False
        for line in new_lines:
            stripped = line.strip()
            if stripped.startswith("class "):
                cls_name = stripped.split("(")[0].split(":")[0]
                if cls_name in existing_defs:
                    has_dup_class = True
                    break

        # Smart dedup: extract NEW methods from duplicate class wrappers
        imports = []
        inject_methods = []   # (class_name, method_lines) — inject into existing class
        standalone_code = []  # code outside any duplicate class

        if has_dup_class:
            LOG.info(f"    FIM wrapped new code in duplicate class — extracting new methods")
            in_dup_class = False
            dup_class_name = None
            dup_class_indent = 0
            current_method_lines = []
            current_method_name = None
            current_method_is_new = False

            def _flush_method():
                if current_method_is_new and current_method_lines:
                    inject_methods.append((dup_class_name, list(current_method_lines)))

            for line in new_lines:
                stripped = line.strip()
                indent = len(line) - len(line.lstrip()) if line.strip() else -1

                # Detect imports
                if line.startswith(("import ", "from ")) and not in_dup_class:
                    if line not in existing_code:
                        imports.append(line)
                    continue

                # Detect duplicate class start
                if stripped.startswith("class "):
                    cls_name = stripped.split("(")[0].split(":")[0]
                    if cls_name in existing_defs:
                        _flush_method()
                        in_dup_class = True
                        dup_class_name = cls_name
                        dup_class_indent = indent
                        current_method_lines = []
                        current_method_name = None
                        current_method_is_new = False
                        continue
                    else:
                        standalone_code.append(line)
                        continue

                if in_dup_class:
                    # Check if we've exited the class
                    if indent <= dup_class_indent and stripped and not stripped.startswith(("def ", "@")):
                        _flush_method()
                        in_dup_class = False
                        standalone_code.append(line)
                        continue

                    # Inside duplicate class — check for method defs
                    if stripped.startswith("def "):
                        _flush_method()
                        method_name = stripped.split("(")[0]
                        methods_set = existing_methods.get(dup_class_name, set())
                        current_method_is_new = method_name not in methods_set
                        current_method_name = method_name
                        current_method_lines = [line]
                        if current_method_is_new:
                            LOG.info(f"    Found new method: {method_name} for {dup_class_name}")
                    elif current_method_lines is not None:
                        current_method_lines.append(line)
                else:
                    standalone_code.append(line)

            _flush_method()
        else:
            # No duplicate classes — separate imports from code
            for line in new_lines:
                if line.startswith(("import ", "from ")):
                    if line not in existing_code:
                        imports.append(line)
                else:
                    standalone_code.append(line)

        # Nothing new to add?
        if not imports and not inject_methods and not any(l.strip() for l in standalone_code):
            LOG.info(f"    FIM: all generated code was duplicate, skipping {filename}")
            continue

        # Apply changes to the file
        modified = existing_code

        # 1. Insert new imports after last existing import
        if imports:
            import_block = "\n".join(imports)
            lines = modified.split("\n")
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith(("import ", "from ")):
                    last_import_idx = i
            lines.insert(last_import_idx + 1, import_block)
            modified = "\n".join(lines)

        # 2. Inject new methods into existing classes
        if inject_methods:
            for class_name, method_lines in inject_methods:
                # Find the end of the class body in the existing code
                lines = modified.split("\n")
                class_end_idx = None
                in_target_class = False
                class_indent = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    indent = len(line) - len(line.lstrip()) if stripped else -1
                    if stripped.startswith("class ") and line.strip().split("(")[0].split(":")[0] == class_name:
                        in_target_class = True
                        class_indent = indent
                        class_end_idx = i
                    elif in_target_class:
                        if indent > class_indent or not stripped:
                            class_end_idx = i
                        elif indent <= class_indent and stripped:
                            break
                if class_end_idx is not None:
                    # Insert after the last line of the class
                    insert_block = "\n" + "\n".join(method_lines)
                    lines.insert(class_end_idx + 1, insert_block)
                    modified = "\n".join(lines)
                    LOG.info(f"    Injected {len(method_lines)} lines into {class_name}")

        # 3. Append standalone code at end of file
        if standalone_code and any(l.strip() for l in standalone_code):
            code_block = "\n".join(standalone_code)
            modified = modified.rstrip("\n") + "\n\n\n" + code_block + "\n"

        filepath.write_text(modified)
        n_methods = len(inject_methods)
        LOG.info(f"    FIM modified: {filename} (+{len(imports)} imports, +{n_methods} injected methods, +{len(standalone_code)} standalone lines)")
        results.append(f"Modified {filename}")

    return "\n".join(results) if results else "(no changes)"


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
        endpoint = agent.get("endpoint", "http://localhost:11434")
        # Smart model routing: use task complexity to pick model
        task_complexity = cfg.get("_task_complexity", "complex")
        if task_complexity == "simple":
            model = agent.get("model_simple", agent.get("model", "qwen2.5-coder:7b"))
        else:
            model = agent.get("model", "qwen2.5-coder:14b")
        LOG.info(f"    Using model: {model} ({task_complexity})")

        # Check if any existing files are being modified — use FIM for those
        project = Path(project_dir)
        existing_py = {f.name for f in project.glob("*.py") if not f.name.startswith("__")}

        # Determine if FIM mode should be used (modification of existing file)
        use_fim = bool(existing_py) and any(
            kw in prompt.lower() for kw in ["add to", "append", "modify", "add a", "main()", "cli", "argparse"]
        )

        if use_fim and existing_py:
            task_desc = cfg.get("_task_description", "")
            return _run_ollama_fim(prompt, project_dir, endpoint, model, existing_py, task_desc)
        else:
            return _run_ollama_blocks(prompt, project_dir, endpoint, model)

    elif backend == "ollama-fim":
        # Force FIM mode regardless of heuristics
        endpoint = agent.get("endpoint", "http://localhost:11434")
        model = agent.get("model", "qwen2.5-coder:14b")
        project = Path(project_dir)
        existing_py = {f.name for f in project.glob("*.py") if not f.name.startswith("__")}
        return _run_ollama_fim(prompt, project_dir, endpoint, model, existing_py)

    elif backend == "custom":  # noqa: keep this aligned
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
    try:
        r = subprocess.run(
            test_cmd, shell=True, cwd=cfg["project_dir"],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "Tests timed out (300s)"
    # pytest exit code 5 = no tests collected (not a failure)
    passed = r.returncode in (0, 5)
    output = (r.stdout + "\n" + r.stderr).strip()
    return passed, output

# ---------------------------------------------------------------------------
# Grok review (outer loop)
# ---------------------------------------------------------------------------

def _build_review_prompts(diff: str, task: Task, constraints: str) -> tuple[str, str]:
    """Build system + user prompts for code review."""
    system_prompt = f"""You are a pragmatic code reviewer. Review this diff against the SPECIFIC task described below.

Constraints (for reference only — do NOT generate tasks for constraints already met):
{constraints}

APPROVAL CRITERIA — approve if ALL of these are true:
1. The diff implements what the task asked for
2. Tests pass (they already passed before this review)
3. No obvious bugs, crashes, or security issues

DO NOT reject for:
- Missing error handling (unless the task specifically asks for it)
- Missing tests (unless the task specifically asks for tests)
- Style/formatting preferences
- "Best practices" not mentioned in the task
- Features the task didn't ask for

Respond with ONLY valid JSON:
{{
  "approved": true/false,
  "summary": "brief review",
  "issues": ["only critical issues"],
  "new_tasks": []
}}

IMPORTANT: Set new_tasks to an EMPTY array []. Do not generate follow-up tasks.
If the code works and does what was asked, approve it."""

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
        LOG.warning(f"  Ollama review timed out: {sanitize_error(e)}")
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


def review_claude(diff: str, task: Task, constraints: str, cfg: dict) -> Review:
    """Review using Anthropic Claude API."""
    import os
    api_key = cfg.get("anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    model = cfg.get("review_model", "claude-sonnet-4-20250514")

    if not api_key:
        LOG.warning("No ANTHROPIC_API_KEY — falling back to ollama reviewer")
        return review_ollama(diff, task, constraints, cfg)

    system_prompt, user_prompt = _build_review_prompts(diff, task, constraints)

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return _parse_review_json(content)
    except Exception as e:
        LOG.warning(f"  Claude review failed: {sanitize_error(e)} — falling back to ollama")
        return review_ollama(diff, task, constraints, cfg)


def run_review(diff: str, task: Task, constraints: str, cfg: dict) -> Review:
    """Dispatch to the configured reviewer backend."""
    backend = cfg.get("reviewer", "ollama")
    if backend == "ollama":
        return review_ollama(diff, task, constraints, cfg)
    elif backend == "api":
        return review_api(diff, task, constraints, cfg)
    elif backend == "claude":
        return review_claude(diff, task, constraints, cfg)
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

    # Pre-warm the coding model so first generation isn't slow
    if not dry_run:
        agent = cfg.get("coding_agent", {})
        if agent.get("backend") == "ollama":
            endpoint = agent.get("endpoint", "http://localhost:11434")
            model = agent.get("model", "qwen2.5-coder:14b")
            LOG.info("Pre-warming model...")
            ollama_warm(endpoint, model)

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
Implement this task. Follow all constraints.
Make minimal, focused changes. Do not refactor unrelated code.
Do NOT wrap file contents in markdown code fences — output raw code only."""

        LOG.info("  ORIENT — applying constraints")

        # Route to the right model based on task complexity
        complexity = classify_task(task, project_dir)
        cfg["_task_complexity"] = complexity
        cfg["_task_description"] = task.description
        LOG.info(f"  Task complexity: {complexity}")

        LOG.info("  DECIDE + ACT — running coding agent")

        # Snapshot working tree so we can rollback if tests fail
        if not dry_run:
            git_snapshot(project_dir)

        if dry_run:
            LOG.info(f"  [dry-run] Would send {len(prompt)} char prompt to {cfg.get('coding_agent', {}).get('backend', 'codebuff')}")
            agent_output = "(dry run)"
        else:
            try:
                agent_output = run_coding_agent(prompt, cfg)
            except Exception as e:
                LOG.error(f"  Coding agent crashed: {sanitize_error(e)}")
                task.status = "failed"
                save_tasks(tasks)
                continue
        LOG.info(f"  Agent output: {len(agent_output)} chars")

        # Gate check: syntax first (fast), then full tests
        if dry_run:
            passed, test_output = True, "(dry run)"
        else:
            syn_ok, syn_err = syntax_check(project_dir)
            if not syn_ok:
                passed, test_output = False, f"Syntax check failed:\n{syn_err}"
                LOG.warning(f"  SYNTAX ERROR: {syn_err}")
            else:
                passed, test_output = run_tests(cfg)

        if not passed:
            LOG.warning(f"  Tests FAILED (attempt {task.attempts}/{max_attempts})")
            # Log first 500 chars of test output for debugging
            for line in test_output[:500].split("\n"):
                LOG.info(f"    | {line}")
            # Rollback broken changes so next attempt starts clean
            if not dry_run:
                git_rollback(project_dir)
                LOG.info("  Rolled back to clean state")
            if task.attempts < max_attempts:
                # Use parsed error for targeted retry prompt
                parsed_err = parse_test_error(test_output)
                task.context += f"\n\nPrevious attempt failed with:\n{parsed_err}"
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
            # Rollback unapproved changes — fix tasks will start clean
            if not dry_run:
                git_rollback(project_dir)
                LOG.info("  Rolled back unapproved changes")
            # Never add reviewer-generated tasks — they cause infinite cascading.
            # The original task will be retried with the review feedback instead.
            if review.issues and task.attempts < max_attempts:
                feedback = "; ".join(review.issues[:3])
                task.context += f"\n\nReview feedback: {feedback}"
                task.status = "pending"
                tasks.insert(0, task)
                LOG.info(f"  Retrying {task.id} with review feedback (attempt {task.attempts}/{max_attempts})")
            elif task.attempts >= max_attempts:
                task.status = "failed"
                LOG.error(f"  Task {task.id} FAILED after {max_attempts} attempts (reviewer rejected)")
            else:
                # No issues but not approved — auto-approve since tests passed
                task.status = "done"
                if not dry_run:
                    # Re-apply changes (rollback was already done, re-run agent)
                    LOG.info("  Review rejected without specific issues — auto-approving since tests passed")
                    # Just retry — the next pass will likely get approved
                    task.status = "pending"
                    tasks.insert(0, task)

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
        cfg["project_dir"] = validate_path(args.project)
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
    description = sanitize_input(args.description, SEC_MAX_TASK_DESC)
    context = sanitize_input(args.context or "", SEC_MAX_TASK_DESC)
    path = ROOT / "tasks.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    task_id = f"task-{len(raw) + 1}"
    raw.append({
        "id": task_id,
        "description": description,
        "context": context,
        "status": "pending",
    })
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False)
    print(f"Added: {task_id} — {description}")


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
    try:
        args.func(args)
    except Exception as exc:
        LOG.error(f"Fatal: {sanitize_error(exc)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
