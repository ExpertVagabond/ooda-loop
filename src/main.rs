use clap::{Parser, Subcommand};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::info;

#[derive(Parser)]
#[command(name = "ooda", about = "OODA Loop — autonomous coding orchestrator")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Execute the OODA loop
    Run {
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        project: Option<String>,
    },
    /// Show task queue
    Status,
    /// Add a task
    Add {
        description: String,
        #[arg(long, short)]
        context: Option<String>,
    },
    /// Reset failed/running tasks to pending
    Reset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Task {
    id: String,
    description: String,
    #[serde(default)]
    context: String,
    #[serde(default = "default_pending")]
    status: String,
    #[serde(default)]
    attempts: u32,
}

fn default_pending() -> String {
    "pending".into()
}

#[derive(Debug)]
struct Review {
    approved: bool,
    summary: String,
    issues: Vec<String>,
    new_tasks: Vec<Task>,
}

#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default)]
    project_dir: String,
    #[serde(default = "default_max_attempts")]
    max_attempts_per_task: u32,
    #[serde(default = "default_max_cycles")]
    max_cycles: u32,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    auto_push: bool,
    #[serde(default)]
    test_command: String,
    #[serde(default)]
    coding_agent: CodingAgent,
    #[serde(default = "default_reviewer")]
    reviewer: String,
    #[serde(default = "default_ollama_endpoint")]
    reviewer_endpoint: String,
    #[serde(default = "default_reviewer_model")]
    reviewer_model: String,
    #[serde(default)]
    xai_api_key: String,
    #[serde(default = "default_xai_base")]
    review_api_base: String,
    #[serde(default = "default_review_model")]
    review_model: String,
}

fn default_max_attempts() -> u32 {
    3
}
fn default_max_cycles() -> u32 {
    20
}
fn default_reviewer() -> String {
    "ollama".into()
}
fn default_ollama_endpoint() -> String {
    "http://localhost:11434".into()
}
fn default_reviewer_model() -> String {
    "qwen2.5-coder:7b".into()
}
fn default_xai_base() -> String {
    "https://api.x.ai/v1".into()
}
fn default_review_model() -> String {
    "grok-3-mini".into()
}

#[derive(Debug, Deserialize, Default)]
struct CodingAgent {
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default = "default_ollama_endpoint")]
    endpoint: String,
    #[serde(default = "default_reviewer_model")]
    model: String,
    #[serde(default)]
    command: String,
}

fn default_backend() -> String {
    "codebuff".into()
}

fn root_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Validate that a project directory path is safe (exists, is a directory, no null bytes).
fn validate_project_dir(dir: &str) -> Result<PathBuf, String> {
    if dir.contains('\0') {
        return Err("project_dir contains null byte".into());
    }
    let path = PathBuf::from(dir);
    if !path.exists() {
        return Err(format!("project_dir does not exist: {dir}"));
    }
    if !path.is_dir() {
        return Err(format!("project_dir is not a directory: {dir}"));
    }
    Ok(path)
}

fn load_config() -> Config {
    let paths = [PathBuf::from("config.yaml"), root_dir().join("config.yaml")];
    for p in &paths {
        if p.exists()
            && let Ok(text) = std::fs::read_to_string(p)
            && let Ok(mut cfg) = serde_yaml::from_str::<Config>(&text)
        {
            // Resolve env vars for API keys
            if cfg.xai_api_key.starts_with('$') {
                cfg.xai_api_key =
                    std::env::var(cfg.xai_api_key.trim_start_matches('$')).unwrap_or_default();
            }
            // Cap max_cycles to prevent runaway loops
            cfg.max_cycles = cfg.max_cycles.min(100);
            cfg.max_attempts_per_task = cfg.max_attempts_per_task.min(10);
            return cfg;
        }
    }
    Config {
        project_dir: ".".into(),
        max_attempts_per_task: 3,
        max_cycles: 20,
        dry_run: false,
        auto_push: false,
        test_command: "echo 'no tests configured'".into(),
        coding_agent: CodingAgent::default(),
        reviewer: "ollama".into(),
        reviewer_endpoint: default_ollama_endpoint(),
        reviewer_model: default_reviewer_model(),
        xai_api_key: String::new(),
        review_api_base: default_xai_base(),
        review_model: default_review_model(),
    }
}

fn load_tasks() -> Vec<Task> {
    let path = PathBuf::from("tasks.yaml");
    if !path.exists() {
        let example = PathBuf::from("tasks.example.yaml");
        if example.exists() {
            let _ = std::fs::copy(&example, &path);
        } else {
            return Vec::new();
        }
    }
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let tasks: Vec<Task> = serde_yaml::from_str(&text).unwrap_or_default();
    tasks
        .into_iter()
        .filter(|t| t.status == "pending")
        .collect()
}

fn save_tasks(tasks: &[Task]) {
    let yaml = serde_yaml::to_string(tasks).unwrap_or_default();
    let _ = std::fs::write("tasks.yaml", yaml);
}

fn load_constraints() -> String {
    let dir = PathBuf::from("constraints");
    if !dir.exists() {
        return String::new();
    }
    let mut parts = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        files.sort_by_key(|e| e.file_name());
        for entry in files {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("md")
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                let name = p
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_uppercase();
                parts.push(format!("# {name}\n\n{text}"));
            }
        }
    }
    parts.join("\n\n---\n\n")
}

// Git helpers
fn git_diff(project_dir: &str) -> String {
    let _ = Command::new("git")
        .args(["add", "-A"])
        .current_dir(project_dir)
        .output();
    let out = Command::new("git")
        .args(["diff", "--cached"])
        .current_dir(project_dir)
        .output();
    match out {
        Ok(o) => {
            let diff = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if diff.is_empty() {
                let status = Command::new("git")
                    .args(["status", "--porcelain"])
                    .current_dir(project_dir)
                    .output();
                status
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|_| "(no changes)".into())
            } else {
                diff
            }
        }
        Err(_) => "(git diff failed)".into(),
    }
}

fn git_commit(project_dir: &str, message: &str) -> bool {
    let _ = Command::new("git")
        .args(["add", "-A"])
        .current_dir(project_dir)
        .output();
    Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(project_dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn git_push(project_dir: &str) -> bool {
    Command::new("git")
        .args(["push"])
        .current_dir(project_dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// Project context
fn get_project_context(project_dir: &str, max_chars: usize) -> String {
    let out = Command::new("find")
        .args([".", "-not", "-path", "./.git/*", "-type", "f"])
        .current_dir(project_dir)
        .output();
    let files: Vec<String> = out
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .split('\n')
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let mut parts = Vec::new();
    let mut total = 0;
    if !files.is_empty() {
        let tree = format!("## Existing files\n{}", files.join("\n"));
        total += tree.len();
        parts.push(tree);
    }

    let code_exts = [
        "py", "js", "ts", "rs", "rb", "go", "java", "c", "h", "yaml", "json", "toml",
    ];
    for f in &files {
        let p = Path::new(project_dir).join(f.trim_start_matches("./"));
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !code_exts.contains(&ext) || !p.is_file() {
            continue;
        }
        let content = match std::fs::read_to_string(&p) {
            Ok(c) if c.len() <= 2000 => c,
            _ => continue,
        };
        let block = format!("\n## {f}\n```\n{content}\n```");
        if total + block.len() > max_chars {
            break;
        }
        total += block.len();
        parts.push(block);
    }

    if parts.is_empty() {
        "(empty project)".into()
    } else {
        parts.join("\n")
    }
}

fn strip_fences(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("```") {
        let lines: Vec<&str> = trimmed.lines().collect();
        let start = 1;
        let end = if lines.last().map(|l| l.trim()) == Some("```") {
            lines.len() - 1
        } else {
            lines.len()
        };
        lines[start..end].join("\n") + "\n"
    } else {
        trimmed.to_string() + "\n"
    }
}

// Coding agent
fn run_coding_agent(prompt: &str, cfg: &Config) -> String {
    let project_dir = &cfg.project_dir;
    match cfg.coding_agent.backend.as_str() {
        "codebuff" => {
            let out = Command::new("codebuff")
                .args(["--message", prompt])
                .current_dir(project_dir)
                .output();
            out.map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default()
        }
        "aider" => {
            let out = Command::new("aider")
                .args(["--message", prompt, "--yes-always", "--no-git"])
                .current_dir(project_dir)
                .output();
            out.map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default()
        }
        "ollama" => {
            let file_prompt = format!(
                "{prompt}\n\nIMPORTANT: Output your changes as file blocks. For each file, use this exact format:\n\n===FILE: path/to/file.ext===\n<file contents>\n===END===\n\nOnly output files that need to be created or modified."
            );
            let client = reqwest::blocking::Client::new();
            let resp = client.post(format!("{}/api/generate", cfg.coding_agent.endpoint))
                .json(&json!({"model": cfg.coding_agent.model, "prompt": file_prompt, "stream": false}))
                .timeout(std::time::Duration::from_secs(300))
                .send();
            let response_text = resp
                .ok()
                .and_then(|r| r.json::<serde_json::Value>().ok())
                .and_then(|v| v["response"].as_str().map(|s| s.to_string()))
                .unwrap_or_default();

            let re = Regex::new(r"===FILE:\s*(.+?)===\n([\s\S]*?)===END===").unwrap();
            let project_canon = std::fs::canonicalize(project_dir)
                .unwrap_or_else(|_| PathBuf::from(project_dir));
            for cap in re.captures_iter(&response_text) {
                let rel_path = cap[1].trim();
                // Reject paths with traversal or null bytes
                if rel_path.contains("..") || rel_path.contains('\0') || rel_path.starts_with('/') {
                    info!("    SKIPPED (unsafe path): {rel_path}");
                    continue;
                }
                let target = Path::new(project_dir).join(rel_path);
                // Verify resolved path stays within project directory
                if let Ok(canon) = std::fs::canonicalize(
                    target.parent().unwrap_or(Path::new(project_dir))
                ) {
                    if !canon.starts_with(&project_canon) {
                        info!("    SKIPPED (path escape): {rel_path}");
                        continue;
                    }
                }
                let content = strip_fences(&cap[2]);
                if let Some(parent) = target.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&target, &content);
                info!("    Wrote: {rel_path}");
            }
            response_text
        }
        "custom" => {
            // Write prompt to a temp file to avoid shell injection via {prompt}
            let prompt_file = Path::new(project_dir).join(".ooda-prompt.tmp");
            let _ = std::fs::write(&prompt_file, prompt);
            let prompt_path = prompt_file.to_string_lossy().to_string();
            let cmd_str = cfg
                .coding_agent
                .command
                .replace("{prompt_file}", &prompt_path)
                .replace("{prompt}", &prompt.replace('\'', "'\\''"))
                .replace("{project_dir}", project_dir);
            let out = Command::new("sh")
                .args(["-c", &cmd_str])
                .current_dir(project_dir)
                .output();
            let _ = std::fs::remove_file(&prompt_file);
            out.map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default()
        }
        other => {
            info!("Unknown backend: {other}");
            String::new()
        }
    }
}

fn run_tests(cfg: &Config) -> (bool, String) {
    let out = Command::new("sh")
        .args(["-c", &cfg.test_command])
        .current_dir(&cfg.project_dir)
        .output();
    match out {
        Ok(o) => {
            let text = format!(
                "{}\n{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            )
            .trim()
            .to_string();
            (o.status.success(), text)
        }
        Err(e) => (false, format!("test exec failed: {e}")),
    }
}

// Review
fn parse_review_json(content: &str) -> Review {
    let trimmed = content.trim();
    let text = if trimmed.starts_with("```") {
        let inner = trimmed.split_once('\n').map(|x| x.1).unwrap_or(trimmed);
        inner
            .rsplit_once("```")
            .map(|(left, _)| left)
            .unwrap_or(inner)
    } else {
        trimmed
    };
    let re = Regex::new(r"\{[\s\S]*\}").unwrap();
    let json_str = re.find(text).map(|m| m.as_str()).unwrap_or("{}");
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(v) => {
            let new_tasks: Vec<Task> = v["new_tasks"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|t| Task {
                            id: t["id"].as_str().unwrap_or("fix").to_string(),
                            description: t["description"]
                                .as_str()
                                .unwrap_or("Fix issues")
                                .to_string(),
                            context: t["context"].as_str().unwrap_or("").to_string(),
                            status: "pending".into(),
                            attempts: 0,
                        })
                        .collect()
                })
                .unwrap_or_default();
            Review {
                approved: v["approved"].as_bool().unwrap_or(false),
                summary: v["summary"].as_str().unwrap_or("").to_string(),
                issues: v["issues"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                new_tasks,
            }
        }
        Err(_) => Review {
            approved: true,
            summary: "Parse failed — auto-approved".into(),
            issues: vec![],
            new_tasks: vec![],
        },
    }
}

fn build_review_prompts(diff: &str, task: &Task, constraints: &str) -> (String, String) {
    let system = format!(
        "You are a senior code reviewer. Review this diff against the task spec.\nApply these constraints:\n\n{constraints}\n\nRespond with ONLY valid JSON:\n{{\"approved\": true/false, \"summary\": \"brief review\", \"issues\": [\"issue1\"], \"new_tasks\": [{{\"id\": \"task-N\", \"description\": \"what\", \"context\": \"why\"}}]}}\n\nIf complete and correct, set approved=true and new_tasks=[]."
    );
    let user = format!(
        "## Task\nID: {}\nDescription: {}\nContext: {}\n\n## Diff\n```\n{}\n```",
        task.id,
        task.description,
        task.context,
        &diff[..diff.len().min(12000)]
    );
    (system, user)
}

fn review_ollama(diff: &str, task: &Task, constraints: &str, cfg: &Config) -> Review {
    let (system, user) = build_review_prompts(diff, task, constraints);
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/api/chat", cfg.reviewer_endpoint))
        .json(&json!({
            "model": cfg.reviewer_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": false, "options": {"temperature": 0.2}
        }))
        .timeout(std::time::Duration::from_secs(300))
        .send();
    let content = resp
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| v["message"]["content"].as_str().map(|s| s.to_string()))
        .unwrap_or_default();
    parse_review_json(&content)
}

fn review_api(diff: &str, task: &Task, constraints: &str, cfg: &Config) -> Review {
    if cfg.xai_api_key.is_empty() {
        return Review {
            approved: true,
            summary: "No reviewer API key".into(),
            issues: vec![],
            new_tasks: vec![],
        };
    }
    let (system, user) = build_review_prompts(diff, task, constraints);
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/chat/completions", cfg.review_api_base))
        .header("Authorization", format!("Bearer {}", cfg.xai_api_key))
        .json(&json!({
            "model": cfg.review_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2
        }))
        .timeout(std::time::Duration::from_secs(120))
        .send();
    let content = resp
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| {
            v["choices"][0]["message"]["content"]
                .as_str()
                .map(|s| s.to_string())
        })
        .unwrap_or_default();
    parse_review_json(&content)
}

fn run_review(diff: &str, task: &Task, constraints: &str, cfg: &Config) -> Review {
    match cfg.reviewer.as_str() {
        "ollama" => review_ollama(diff, task, constraints, cfg),
        "api" => review_api(diff, task, constraints, cfg),
        other => {
            info!("Unknown reviewer '{other}' — auto-approving");
            Review {
                approved: true,
                summary: format!("Unknown: {other}"),
                issues: vec![],
                new_tasks: vec![],
            }
        }
    }
}

// Omega Architecture: SkipFire — only re-observe when codebase has changed
fn has_project_changed(project_dir: &str) -> bool {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(project_dir)
        .output();
    match output {
        Ok(o) => !o.stdout.is_empty(),
        Err(_) => true, // assume changed if we can't check
    }
}

// Omega Architecture: Queue metrics (TankBuffer pattern)
#[derive(Debug, Default)]
struct QueueMetrics {
    tasks_out: u64,
    tasks_completed: u64,
    tasks_failed: u64,
    tasks_requeued: u64,
    tasks_spawned: u64,
    peak_depth: usize,
    skip_fire_hits: u64,
    skip_fire_misses: u64,
}

// Main loop
fn ooda_run(cfg: Config) {
    // Validate project directory before starting
    if let Err(e) = validate_project_dir(&cfg.project_dir) {
        info!("Invalid project_dir: {e}");
        return;
    }
    let constraints = load_constraints();
    let mut tasks = load_tasks();
    let max_attempts = cfg.max_attempts_per_task;
    let max_cycles = cfg.max_cycles;
    let dry_run = cfg.dry_run;
    let mut cycle = 0u32;

    info!(
        "OODA Loop starting — {} tasks, project: {}",
        tasks.len(),
        cfg.project_dir
    );
    if dry_run {
        info!("DRY RUN — no code will be written");
    }

    // Omega Architecture: metrics + skip-fire state
    let mut metrics = QueueMetrics::default();
    let mut cached_context: Option<String> = None;

    while !tasks.is_empty() && cycle < max_cycles {
        cycle += 1;
        metrics.peak_depth = metrics.peak_depth.max(tasks.len());
        let mut task = tasks.remove(0);
        task.status = "running".into();
        task.attempts += 1;
        metrics.tasks_out += 1;

        info!("[cycle {cycle}] Task {}: {}", task.id, task.description);

        // OBSERVE — with Omega SkipFire (only re-scan when codebase changed)
        let project_changed = has_project_changed(&cfg.project_dir);
        let project_context = match (&cached_context, project_changed) {
            (Some(cached), false) => {
                metrics.skip_fire_hits += 1;
                info!("  OBSERVE — skip-fire (no changes since last scan)");
                cached.clone()
            }
            _ => {
                metrics.skip_fire_misses += 1;
                info!("  OBSERVE — scanning codebase (changes detected)");
                let ctx = get_project_context(&cfg.project_dir, 4000);
                cached_context = Some(ctx.clone());
                ctx
            }
        };

        let prompt = format!(
            "## CONSTRAINTS\n{constraints}\n\n## EXISTING CODEBASE\n{project_context}\n\n## TASK\n{}\n\n## CONTEXT\n{}\n\n## INSTRUCTIONS\nImplement this task. Follow all constraints. Write tests if none exist.\nMake minimal, focused changes. Do not refactor unrelated code.\nDo NOT wrap file contents in markdown code fences — output raw code only.",
            task.description, task.context
        );

        info!("  ORIENT + DECIDE + ACT — running coding agent");
        let agent_output = if dry_run {
            info!(
                "  [dry-run] Would send {} char prompt to {}",
                prompt.len(),
                cfg.coding_agent.backend
            );
            "(dry run)".to_string()
        } else {
            run_coding_agent(&prompt, &cfg)
        };
        info!("  Agent output: {} chars", agent_output.len());

        // Test gate
        let (passed, test_output) = if dry_run {
            (true, "(dry run)".into())
        } else {
            run_tests(&cfg)
        };

        if !passed {
            info!(
                "  Tests FAILED (attempt {}/{})",
                task.attempts, max_attempts
            );
            if task.attempts < max_attempts {
                let truncated = &test_output[..test_output.len().min(2000)];
                task.context += &format!("\n\nPrevious attempt failed tests:\n{truncated}");
                task.status = "pending".into();
                tasks.insert(0, task);
                metrics.tasks_requeued += 1;
                continue;
            } else {
                task.status = "failed".into();
                metrics.tasks_failed += 1;
                info!("  Task {} FAILED after {max_attempts} attempts", task.id);
                save_tasks(&tasks);
                continue;
            }
        }

        info!("  Tests PASSED — staging for review");

        // Review
        let (_diff, review) = if dry_run {
            (
                "(dry run)".to_string(),
                Review {
                    approved: true,
                    summary: "Dry run".into(),
                    issues: vec![],
                    new_tasks: vec![],
                },
            )
        } else {
            let d = git_diff(&cfg.project_dir);
            let r = run_review(&d, &task, &constraints, &cfg);
            (d, r)
        };

        info!(
            "  Review: {} — {}",
            if review.approved {
                "APPROVED"
            } else {
                "NEEDS WORK"
            },
            review.summary
        );
        for issue in &review.issues {
            info!("    - {issue}");
        }

        if review.approved {
            task.status = "done".into();
            metrics.tasks_completed += 1;
            if !dry_run {
                let msg = format!(
                    "ooda: {} — {}",
                    task.id,
                    &task.description[..task.description.len().min(60)]
                );
                git_commit(&cfg.project_dir, &msg);
            }
            info!(
                "  {}: {}",
                if dry_run {
                    "[dry-run] Would commit"
                } else {
                    "Committed"
                },
                task.id
            );
        } else {
            let new_count = review.new_tasks.len() as u64;
            for nt in review.new_tasks {
                tasks.push(nt);
            }
            metrics.tasks_spawned += new_count;
            info!("  Added {} new tasks from review", review.issues.len());
        }

        save_tasks(&tasks);
    }

    let remaining: Vec<_> = tasks.iter().filter(|t| t.status == "pending").collect();
    if remaining.is_empty() && cfg.auto_push && !dry_run && git_push(&cfg.project_dir) {
        info!("Pushed to remote");
    }
    info!(
        "OODA Loop complete — {cycle} cycles, {} tasks remaining",
        remaining.len()
    );

    // Omega Architecture: print queue metrics
    info!(
        "  Queue: {} processed, {} completed, {} failed, {} requeued, {} spawned (peak depth: {})",
        metrics.tasks_out, metrics.tasks_completed, metrics.tasks_failed,
        metrics.tasks_requeued, metrics.tasks_spawned, metrics.peak_depth
    );
    let total_sf = metrics.skip_fire_hits + metrics.skip_fire_misses;
    if total_sf > 0 {
        info!(
            "  Observe skip-fire: {}/{} skipped ({:.0}%)",
            metrics.skip_fire_hits, total_sf,
            metrics.skip_fire_hits as f64 / total_sf as f64 * 100.0
        );
    }
}

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();

    match args.cmd {
        Cmd::Run { dry_run, project } => {
            let mut cfg = load_config();
            if dry_run {
                cfg.dry_run = true;
            }
            if let Some(p) = project {
                cfg.project_dir = p;
            }
            ooda_run(cfg);
        }
        Cmd::Status => {
            let tasks = load_tasks();
            if tasks.is_empty() {
                println!("No pending tasks.");
                return;
            }
            println!("{:<15} {:<10} DESCRIPTION", "ID", "STATUS");
            println!("{}", "-".repeat(60));
            for t in &tasks {
                println!(
                    "{:<15} {:<10} {}",
                    t.id,
                    t.status,
                    &t.description[..t.description.len().min(50)]
                );
            }
        }
        Cmd::Add {
            description,
            context,
        } => {
            let path = PathBuf::from("tasks.yaml");
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let mut raw: Vec<Task> = serde_yaml::from_str(&text).unwrap_or_default();
            let id = format!("task-{}", raw.len() + 1);
            raw.push(Task {
                id: id.clone(),
                description: description.clone(),
                context: context.unwrap_or_default(),
                status: "pending".into(),
                attempts: 0,
            });
            if let Ok(yaml) = serde_yaml::to_string(&raw) {
                let _ = std::fs::write(&path, yaml);
            }
            println!("Added: {id} — {description}");
        }
        Cmd::Reset => {
            let path = PathBuf::from("tasks.yaml");
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let mut raw: Vec<Task> = serde_yaml::from_str(&text).unwrap_or_default();
            let mut count = 0;
            for t in &mut raw {
                if t.status == "failed" || t.status == "running" {
                    t.status = "pending".into();
                    count += 1;
                }
            }
            if let Ok(yaml) = serde_yaml::to_string(&raw) {
                let _ = std::fs::write(&path, yaml);
            }
            println!("Reset {count} tasks to pending.");
        }
    }
}
