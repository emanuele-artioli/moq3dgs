#!/usr/bin/env python3
"""Keep this project's agent rule files consistent with AGENTS.md.

`AGENTS.md` at the repository root is the one file anyone edits by hand. It is
read natively by Cursor, Antigravity, Copilot's cloud agent and code review,
and Codex. Claude Code is the only holdout -- it reads `CLAUDE.md` and has no
AGENTS.md fallback -- so `CLAUDE.md` is a thin wrapper that `@`-imports
AGENTS.md and adds whatever is Claude-only.

This script maintains the three things that cannot be hand-written:

    AGENTS.md `host-rules` block   host-wide rules, inlined for the agents
                                   that cannot import them
    .cursor/rules/cursor-harness.mdc  Cursor's own harness rules (Cursor has
                                   no user-level rules file, so they have to
                                   be delivered per project)
    .github/copilot-instructions.md   a pointer, for the Copilot surfaces that
                                   read nothing but this path

Why the host rules are inlined rather than imported: Claude Code and
Antigravity load `~/.agent-rules/AGENTS.md` themselves, but Copilot's cloud
agent and Cursor's cloud agents run on machines that have never seen this
host's home directory. Anything they must obey has to be committed into the
repository.

Why `--check` still works on CI: when `~/.agent-rules/AGENTS.md` is not
reachable, the generated blocks are left exactly as committed instead of being
regenerated from nothing. That is what replaced the old
`tools/host_rules_snapshot.md`, which existed only to make CI reproducible and
is now deleted on sight.

Usage:
    python tools/sync_agent_rules.py            # write the generated files
    python tools/sync_agent_rules.py --check    # exit 1 if any is out of date

CANONICAL COPY: this file is hand-edited only at
~/.agent-rules/scripts/sync_agent_rules.py and vendored (physically copied,
not symlinked) into each project's tools/ directory by
~/.agent-rules/scripts/vendor-sync-agent-rules.sh. It cannot be centralized
the way the hooks are (referenced by absolute path) because CI runners have no
access to ~/.agent-rules/ at all -- this script must stay a real,
self-contained file inside each project's own repo. See
~/.agent-rules/README.md for the full explanation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

AGENTS_MD = REPO_ROOT / "AGENTS.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
CURSOR_RULE = REPO_ROOT / ".cursor" / "rules" / "cursor-harness.mdc"
COPILOT_MD = REPO_ROOT / ".github" / "copilot-instructions.md"

HOST_DIR = Path.home() / ".agent-rules"
HOST_RULES = HOST_DIR / "AGENTS.md"
HOST_CURSOR_HARNESS = HOST_DIR / "harness" / "cursor.md"

# Files the previous layout generated, now superseded. Antigravity reads the
# root AGENTS.md natively since v1.20.3, and Copilot's cloud agent and code
# review read it too, so a per-agent copy of the same prose is dead weight.
OBSOLETE = (
    REPO_ROOT / ".agents" / "rules",
    REPO_ROOT / ".github" / "instructions",
    REPO_ROOT / "tools" / "host_rules_snapshot.md",
)

HOST_BLOCK = re.compile(
    r"\n*<!-- host-rules:start.*?<!-- host-rules:end -->\n?", re.DOTALL
)
COPILOT_CRITICAL = re.compile(
    r"<!--\s*copilot-critical:start\s*-->(.*?)<!--\s*copilot-critical:end\s*-->",
    re.DOTALL,
)
IMPORT_LINE = re.compile(r"^@(/\S+)\s*$", re.MULTILINE)


def resolve_imports(text: str, _depth: int = 0) -> str:
    """Expand `@/absolute/path` import lines, recursively.

    The host rules file is meant to be a leaf with no imports of its own, but
    if one is ever added, an unresolved `@/path` line inlined into a project
    would be meaningless to every agent that does not speak Claude's import
    syntax. Bounded to 4 hops, matching Claude Code's own limit.
    """
    if _depth >= 4:
        return text

    def _expand(match: re.Match[str]) -> str:
        path = Path(match.group(1))
        if not path.is_file():
            return match.group(0)
        return resolve_imports(path.read_text(), _depth + 1)

    return IMPORT_LINE.sub(_expand, text)


def host_block() -> str | None:
    """The generated host-rules block, or None when the host file is absent."""
    if not HOST_RULES.is_file():
        return None
    text = resolve_imports(HOST_RULES.read_text()).strip()
    # Demote the host file's headings so they nest under our own section.
    text = re.sub(r"^#", "##", text, flags=re.MULTILINE)
    return (
        "<!-- host-rules:start — GENERATED from ~/.agent-rules/AGENTS.md by\n"
        "     tools/sync_agent_rules.py. Do not edit inside this block; edit the\n"
        "     host file and re-run the script. Everything above the marker is\n"
        "     this project's own, hand-edited. -->\n\n"
        "# Host-wide rules\n\n"
        "These apply to every project on this host. Claude Code and Antigravity\n"
        "load them from `~/.agent-rules/AGENTS.md` directly; they are inlined\n"
        "here for the agents that cannot — notably cloud agents, which run on\n"
        "machines that have never seen this host's home directory.\n\n"
        f"{text}\n\n"
        "<!-- host-rules:end -->"
    )


def with_host_block(agents_md: str, block: str) -> str:
    """AGENTS.md with its host-rules block replaced (or appended)."""
    body = HOST_BLOCK.sub("\n", agents_md).rstrip()
    return f"{body}\n\n{block}\n"


def cursor_rule(harness: str) -> str:
    """Cursor's harness rules as an always-applied project rule."""
    return (
        "---\n"
        "description: Cursor-specific harness mechanics for this host\n"
        "alwaysApply: true\n"
        "---\n\n"
        "<!-- GENERATED from ~/.agent-rules/harness/cursor.md by\n"
        "     tools/sync_agent_rules.py — DO NOT EDIT. Cursor has no user-level\n"
        "     rules file, so these have to be delivered per project. -->\n\n"
        f"{harness.strip()}\n"
    )


def copilot_pointer(agents_md: str) -> str:
    """A pointer file for the Copilot surfaces that read nothing else.

    Copilot's cloud agent and code review read `AGENTS.md` natively; Copilot
    Chat in the IDE and on github.com read only this path. Rather than keep a
    second full copy in sync, this points at the real file and repeats only
    the handful of rules the project marks as must-not-break.
    """
    critical = COPILOT_CRITICAL.search(agents_md)
    body = (
        "<!-- GENERATED by tools/sync_agent_rules.py — DO NOT EDIT.\n"
        "     Edit AGENTS.md and re-run the script. -->\n\n"
        "# Copilot instructions\n\n"
        "The authoritative instructions for this repository are in **`AGENTS.md`\n"
        "at the repository root** — this project's own rules first, host-wide\n"
        "rules in the `host-rules` block at the end. Copilot's cloud agent and\n"
        "code review read that file directly. This file exists only for the\n"
        "surfaces that do not: Copilot Chat in the IDE and on github.com.\n\n"
        "**Read `AGENTS.md` before making any change.**\n"
    )
    if critical:
        body += (
            "\nThe rules below are repeated here because they are the ones that\n"
            "cause real damage when missed — they are not the whole set.\n\n"
            f"{critical.group(1).strip()}\n"
        )
    return body


def targets() -> dict[Path, str]:
    """Path -> intended content, for every file this script owns.

    Files whose source is unreachable (no `~/.agent-rules/`, i.e. CI) are
    omitted, so `--check` compares only what it can legitimately verify.
    """
    if not AGENTS_MD.is_file():
        raise FileNotFoundError(AGENTS_MD)

    agents_md = AGENTS_MD.read_text()
    planned: dict[Path, str] = {}

    block = host_block()
    if block is not None:
        agents_md = with_host_block(agents_md, block)
        planned[AGENTS_MD] = agents_md

    if HOST_CURSOR_HARNESS.is_file():
        planned[CURSOR_RULE] = cursor_rule(HOST_CURSOR_HARNESS.read_text())

    planned[COPILOT_MD] = copilot_pointer(agents_md)
    return planned


def obsolete_present() -> list[Path]:
    return [path for path in OBSOLETE if path.exists()]


def remove(path: Path) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink()
        path.rmdir()
    else:
        path.unlink()


def claude_wrapper_ok() -> bool:
    """Whether CLAUDE.md still imports AGENTS.md.

    Advisory. Claude Code reads no AGENTS.md of its own, so a CLAUDE.md that
    has lost its `@AGENTS.md` line means Claude sessions silently run with no
    project rules at all -- the exact failure this layout exists to prevent.
    """
    if not CLAUDE_MD.is_file():
        return False
    return "@AGENTS.md" in CLAUDE_MD.read_text()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale generated files instead of rewriting them",
    )
    args = parser.parse_args()

    try:
        planned = targets()
    except FileNotFoundError as missing:
        print(f"error: {missing} not found", file=sys.stderr)
        return 1

    stale = [
        path
        for path, content in planned.items()
        if (path.read_text() if path.is_file() else None) != content
    ]
    dead = obsolete_present()

    if args.check:
        if stale or dead:
            print("Agent rule files are out of date with AGENTS.md:", file=sys.stderr)
            for path in stale:
                print(f"  stale:    {path.relative_to(REPO_ROOT)}", file=sys.stderr)
            for path in dead:
                print(f"  obsolete: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
            print("Run: python tools/sync_agent_rules.py", file=sys.stderr)
            return 1
    else:
        for path in stale:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(planned[path])
        for path in dead:
            remove(path)

    if not claude_wrapper_ok():
        print(
            "warning: CLAUDE.md does not import AGENTS.md — Claude Code sessions "
            "will load no project rules. Add `@AGENTS.md` to CLAUDE.md.",
            file=sys.stderr,
        )

    changed = len(stale) + len(dead)
    print(f"agent rules: {changed} file(s) updated" if changed else "agent rules: up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
