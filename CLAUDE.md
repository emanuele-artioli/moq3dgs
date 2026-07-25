# MOQ3DGS — Claude Code

@AGENTS.md

That import is the whole point of this file: `AGENTS.md` holds the project's
real rule set, read directly by every other agent, and Claude Code is the only
one that will not look at it on its own. Add nothing here that is not
specific to Claude Code — if it applies to any other agent, it belongs in
`AGENTS.md` (and if it applies to every project on this host, in
`~/.agent-rules/AGENTS.md`).

## Claude-specific notes

- Host-wide rules reach this session twice: once from `~/.claude/CLAUDE.md`,
  which Claude loads on its own, and once from the `host-rules` block inside
  `AGENTS.md`. The block exists for cloud agents that never see this host's
  home directory, so the duplication is deliberate, not drift.
- Hooks for this project are wired in `.claude/settings.json`: repo state at
  session start and the `rm` guard before every `Bash` call. They call the
  shared scripts in `~/.agent-rules/scripts/`; the values they enforce come
  from `.agent-guards.json` in this repo.
- Sessions spawned with `spawn_task` chips get their own worktree under
  `.claude/worktrees/`. Sessions started any other way share this checkout's
  HEAD — see the concurrency rules in `AGENTS.md` before running two at once.
