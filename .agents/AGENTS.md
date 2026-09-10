# Agent Rules

## tmux Best Practices
When dealing with long-running tasks, background jobs, or using the `experiment-queue`, ALWAYS launch these tasks inside a detached `tmux` session instead of relying purely on background `nohup` or `&`. 

- **Creation**: Use `tmux new-session -d -s <session_name> "<command>; read"` to create a persistent session.
- **Verification**: Always run `tmux ls` to verify that the session was created successfully and is actively running.
- **Checking Output**: You can use `tmux capture-pane -p -t <session_name>` to view the latest output of a running tmux session without attaching.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them.
If a simpler approach exists, say so.
If something is unclear, stop. Name what's confusing.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No silent fallbacks, prefer failure over obscure behavior.
No backwards compatibility by default, only if user requests explicitely.
No features beyond what was asked.
No abstractions for single-use code.
No “flexibility” that wasn't requested.
No error handling for impossible scenarios.
If 200 lines could be 50, rewrite it.
3. Surgical Changes
Touch only what you must. Clean up only your own mess.

Don't “improve” adjacent code or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice dead code, mention it — don't delete it.
4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

“Add validation” → “Write tests, then make them pass”
“Fix the bug” → “Reproduce it in a test, then fix”
“Refactor X” → “Ensure tests pass before and after”

