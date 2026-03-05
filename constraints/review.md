## Review Criteria (for Grok reviewer)

When reviewing a diff, evaluate against these criteria:

### P0 — Must fix before merge
- Security vulnerabilities (injection, hardcoded secrets, auth bypass)
- Data loss risk (missing migrations, destructive operations without backup)
- Broken tests or untested critical paths

### P1 — Should fix
- Missing error handling at system boundaries
- Violations of coding standards from constraints
- Performance issues (N+1 queries, unbounded loops, memory leaks)

### P2 — Nice to fix
- Style inconsistencies with existing codebase
- Missing edge case handling
- Suboptimal but functional implementations

### Task generation rules
- Only generate new tasks for P0 and P1 issues
- Each task must be specific and actionable
- Include the file path and line range in the task context
- Max 3 new tasks per review cycle (focus on highest priority)
