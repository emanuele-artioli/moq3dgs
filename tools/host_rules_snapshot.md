# Global environment notes

These apply across all projects/sessions on this host, not just one repo's
CLAUDE.md. **This file is the register of things that have gone wrong more
than once** — if a mistake happens twice, it belongs here, phrased as the rule
that prevents it rather than the story of the failure.

# Shared agent rules — single source of truth

Imported by reference (`@` syntax) from each coding agent's own rules file —
currently `~/.claude/CLAUDE.md` and `~/.gemini/GEMINI.md`. Edit **only this
file** for anything that should apply to every agent on this host. Put
agent-specific mechanics (tool names, invocation syntax, that agent's own
conventions) in the importing file instead, not here — this file stays
tool-agnostic so every importer can use it as-is.

## The host

Shared remote Linux **GPU server, no root/sudo/apt**, headless. Home is
`/home/itec/emanuele`. Install extra tooling with conda (Miniconda at
`/usr/local/miniconda3`) into a *separate* env — never into a project's
pinned env, because several forked third-party models are version-sensitive
and a stray `pip install` silently breaks them. Being headless, save media
and plots to disk; `cv2.imshow()`/`plt.show()` never works here.

## Python dependency management

Manage Python packages through `pyproject.toml`, not ad-hoc `pip install` in
the terminal. `environment.yaml` is reserved for bootstrapping heavy
CUDA/GPU binaries only (drivers, PyTorch wheels, compiled packages) — never
fall back to a `requirements.txt` file.

## GitHub CLI (gh)

`gh` is installed at `~/emanuele/bin/gh` (on `PATH` in every shell on this
host) and authenticated as `emanuele-artioli` via `gh auth login`
(credentials in `~/.config/gh/hosts.yml`, not tied to any one project).
Available in every project on this host — install/auth doesn't need
repeating.

**Use it proactively after every push to a repo with GitHub Actions (or
any CI):** don't assume a push landed cleanly or guess at failures from
job/step names alone.

- `gh run list --branch <branch> --limit 3` — find the run a push triggered
- `gh run view <run-id> --json status,conclusion -q '.status'` (poll) or
  `gh run watch <run-id>` — wait for it to finish (`gh run watch` can
  itself flake with a transient "Bad credentials" on the annotations
  call; a `gh run view <run-id>` after that still shows the real job
  status, so don't treat a `run watch` crash as the run having failed)
- `gh run view <run-id> --log-failed` — **the real fix for CI debugging.**
  The unauthenticated GitHub REST API only exposes job/step names and
  conclusions, never log content (log downloads 403 "Must have admin
  rights" even on public repos without an authenticated token) — that
  API alone means guessing at root causes from symptoms. Authenticated
  `gh` gives the exact failing line immediately.

Also usable the same way for `gh pr view`, `gh issue view`, `gh pr create`,
etc. wherever a GitHub-authenticated operation is needed — this isn't
CI-specific.

## Git — never destroy work you have not read

These repos get worked on by several agents at once (Claude sessions,
Antigravity, Codex, Copilot), and unmerged work has genuinely been lost here
before: a complete HNeRV baseline once sat in a forgotten worktree.

- **Read a branch before deleting it.** `git log main..<branch>` and
  `git diff main...<branch> --stat`. If it is not empty,
  `git tag archive/<branch> <branch>` and push the tag *before* deleting.
  Tags are free and make a triage mistake recoverable.
- **A worktree with uncommitted changes never gets `--force`d away.**
  Commit the changes onto that worktree's own branch, tag it, then remove
  the worktree. `git worktree remove` refusing is a warning, not an obstacle
  to route around.
- **"Superseded" needs proof, not a guess.** Compare with `git patch-id`, or
  diff the files against `main` — a branch whose commit message matches one
  on main may still hold changes main never got.
- **A branch alone does not isolate a session** — two agents in one checkout
  share one HEAD. Isolation needs a worktree *and* a branch.

## Research code — tests are a failsafe, not a formality

Cover envisioned behavior and plausible misuse of code we own. Skip tests for
unreachable branches, third-party library behavior, and errors a caller
cannot produce; this is research code and boilerplate slows the iteration
that actually matters. **A test that exists only to raise a coverage number
is a defect** — it makes the gate lie about what is verified. If deleting
padding drops the gate, lower the gate to the honest number and ratchet it
back up as real tests land.

The tests that pay for themselves here are the ones that check *the paper's
claim*, not just that the code runs: an experiment whose result violates the
thing the paper asserts should fail loudly and be marked uncitable, rather
than being caught later by a careful human reading a table.

## Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day. Any job expected to run over
an hour checkpoints at least every 60 minutes of wall clock — independent of
its epoch/step cadence — and its resume path is verified *before* it is
relied on. Long scripts also append a progress line to their log at least
every 10 minutes, so a silent hang is visible in minutes rather than hours.
Launch detached; never attached to a shell an SSH drop takes with it.

## Plan mode: split complex plans into parallel-agent waves

When a plan has multiple pieces of work that don't share state, don't execute
it as one linear sequence. Split it into workstreams and hand each to a
subagent working in its own git worktree (a shared checkout with only a
different branch is not isolation — two sessions in one worktree share a
single HEAD). Group workstreams into **waves** ordered by dependency: a wave
starts only once every workstream it depends on has reported results back,
and every workstream within a wave launches together, not one at a time.

**Why:** validated on a multi-part refactor — this surfaced cross-workstream
issues at each wave boundary instead of at the end, and kept parallel agents
from clobbering each other's changes.

**How to apply:** worth it for genuinely multi-part, multi-file tasks where
pieces are largely independent. Skip it for small or sequential tasks — one
file, one clear order of steps — where waves are pure coordination overhead.

## Waiting for long-running commands — never hand-roll a waiter

⛔ **Never write `until ! pgrep -f <pattern>; do sleep N; done` (or any
self-written poll loop) to wait for a job.** The harness runs the loop via
`bash -c "<the whole command string>"`, and that string *contains* the
pattern — so `pgrep -f` matches the watcher's own process and the condition
can never become true. The job finishes, the watcher spins until timeout, and
the completion goes unnoticed. This has already burned >1h of wall clock.
Escaping tricks (`[p]attern`, `pgrep -P`) technically work but are still the
wrong answer: the harness already reports completion, so there is nothing to
poll for.

Pick by duration, not by habit:

- **Finishes in < 10 min** → foreground `Bash` with an explicit `timeout`
  (ms, max 600000). Output arrives in one piece and the harness kills it at
  the deadline, so it cannot hang forever.
- **Longer than that** (GPU restoration, full evaluation passes, big
  backfills) → `Bash` with `run_in_background: true`. It detaches, survives
  across turns, and **re-invokes Claude on exit** with the path to its
  output file. Read that file; do not poll for it.
- **Need progress while it runs** → `Monitor`, with a filter that matches
  failure signatures too (`Traceback|Error|FAILED|Killed|OOM`), not just the
  success marker — a success-only filter stays silent through a crash, and
  silence is indistinguishable from "still running."

`conda run -n <env> …` is not a solution to this. It is still a foreground
command subject to the same 10-minute cap, and without
`--no-capture-output` it buffers all output until exit — so on a long job it
shows nothing and then gets killed. Use it for env activation if convenient,
never as a completion-waiting strategy.

Note: `Monitor`'s progress-matching depends on the logging cadence described
in the shared "Long jobs must checkpoint" rule above — a job that goes quiet
for more than ~10 minutes gives Monitor nothing fresh to match, which looks
identical to a hang.

Same trap, different tool: **`ScheduleWakeup` is not a wait-for-completion
mechanism.** It exists solely to self-pace `/loop` dynamic-mode iterations.
A background agent or background `Bash` job already triggers a notification
the moment it finishes — there is nothing to poll for. Don't call
`ScheduleWakeup` "just to wait" for one; it also fails outright when used
this way (it requires a `prompt` unless `stop: true`), so the mistake
surfaces immediately rather than silently wasting a turn — still worth not
repeating.
