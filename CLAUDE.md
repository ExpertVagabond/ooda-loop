# OODA Loop — Autonomous coding orchestrator

Observe-Orient-Decide-Act loop: local LLM codes, reviewer LLM validates, tests gate merges.

## Stack
- **Rust (2024 edition)** — Core orchestrator binary (`ooda`)
- **Python 3.11+** — Lightweight companion script (ooda.py)
- **Key Rust deps:** tokio, clap, reqwest, serde, serde_yaml, tracing, regex
- **Key Python deps:** httpx, pyyaml

## Key Commands
```bash
# Build
cargo build --release             # Build Rust binary
cargo test                        # Run Rust tests
cargo clippy -- -D warnings       # Lint

# Run
cargo run -- run                  # Execute OODA loop on configured project
cargo run -- run --dry-run        # Preview without executing
cargo run -- run --project /path  # Override project directory
cargo run -- status               # Show task queue
cargo run -- add "task desc"      # Add a task
cargo run -- reset                # Reset failed tasks to pending

# Python companion
python3 ooda.py                   # Alternative Python entry point
```

## Project Structure
```
src/
  main.rs              — CLI parser, OODA loop, coding agent, reviewer, task queue
Cargo.toml             — Rust package config
pyproject.toml         — Python companion config
ooda.py                — Python companion script
config.yaml            — Runtime configuration (agents, reviewer, test gate, loop controls)
tasks.yaml             — Active task queue (create from tasks.example.yaml)
tasks.example.yaml     — Example task definitions
constraints/
  system.md            — System-level constraints for the orchestrator
  coding.md            — Coding agent prompt constraints
  review.md            — Reviewer prompt constraints
logs/                  — Execution logs
decks/                 — Pitch decks
```

## Configuration (config.yaml)
- `project_dir` — Target project to work on
- `coding_agent.backend` — `ollama`, `codebuff`, `aider`, or `custom`
- `coding_agent.model` — e.g., `qwen2.5-coder:14b`
- `reviewer` — `ollama`, `claude`, or `api` (Grok/OpenRouter)
- `test_command` — Gate command (pytest, npm test, cargo test)
- `max_attempts_per_task` — Retries before marking failed (default 3)
- `max_cycles` — Total loop iterations (default 20)

## Environment Variables
- `ANTHROPIC_API_KEY` — When using Claude as reviewer
- `XAI_API_KEY` — When using Grok as reviewer
- `OPENROUTER_API_KEY` — When using OpenRouter as reviewer
- Ollama endpoint defaults to `http://localhost:11434`

## Architecture
- **Observe:** Read task queue from tasks.yaml
- **Orient:** Select next pending task, load constraints
- **Decide:** Send task + context to coding agent (Ollama/codebuff/aider)
- **Act:** Apply code changes, run test gate, send to reviewer
- Reviewer can approve, reject (retry), or spawn new sub-tasks
- Loop continues until all tasks done or max_cycles reached
