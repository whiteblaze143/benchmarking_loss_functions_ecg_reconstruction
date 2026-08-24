# Agent Rules

## tmux Best Practices
When dealing with long-running tasks, background jobs, or using the `experiment-queue`, ALWAYS launch these tasks inside a detached `tmux` session instead of relying purely on background `nohup` or `&`. 

- **Creation**: Use `tmux new-session -d -s <session_name> "<command>; read"` to create a persistent session.
- **Verification**: Always run `tmux ls` to verify that the session was created successfully and is actively running.
- **Checking Output**: You can use `tmux capture-pane -p -t <session_name>` to view the latest output of a running tmux session without attaching.
