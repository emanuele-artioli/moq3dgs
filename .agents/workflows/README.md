# Project workflows (slash commands)

Host-wide slash prompts live in `~/.agent-rules/workflows/` and are linked into
Cursor (`~/.cursor/commands/`) and Antigravity
(`~/.gemini/config/global_workflows/`) by `~/.agent-rules/scripts/install.py`.

Project-specific ones go here as plain `<name>.md` files (filename = `/name`).
`.cursor/commands` in this repo is a symlink onto this directory, so Cursor and
Antigravity share one tree. Claude Code does **not** read this path — its slash
surface is skills under `.claude/skills/`. Promote a workflow to a skill if it
must also auto-trigger for Claude.
