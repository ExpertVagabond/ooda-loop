## Coding Standards (adapted from HVE-Core)

### General
- No dead code. No commented-out code. No TODO comments without a task ID.
- Functions do ONE thing. Max 40 lines per function.
- Max 3 parameters per function. Use an options object beyond that.
- No magic numbers or strings — use named constants.
- Error handling at system boundaries only. Trust internal code.

### Security
- Never hardcode secrets, tokens, or keys
- Validate all external input (user input, API responses, file reads)
- Use parameterized queries for any database access
- No eval(), exec(), or dynamic code execution

### Testing (TDD)
- Write tests BEFORE or alongside implementation, never after
- Tests must be deterministic — no flaky tests, no sleep()
- Test the interface, not the implementation
- One assertion per test preferred. Multiple assertions only when testing a single logical outcome.
- Name tests: `test_<what>_<condition>_<expected>`

### Git
- One logical change per commit
- Commit message: imperative mood, under 72 chars
- Never force push. Never amend published commits.
