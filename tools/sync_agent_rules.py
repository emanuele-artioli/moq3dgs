#!/usr/bin/env python3
"""Keep this project's agent rule files consistent with AGENTS.md.

`AGENTS.md` at the repository root is the one file anyone edits by hand. It is
read natively by Cursor, Antigravity, Copilot's cloud agent and code review,
and Codex. Claude Code is the only holdout -- it reads `CLAUDE.md` and has no
AGENTS.md fallback -- so `CLAUDE.md` is a thin wrapper that `@`-imports
AGENTS.md and adds whatever is Claude-only.

This script maintains everything that cannot be hand-written:

    AGENTS.md `host-rules` block   host-wide rules, inlined for the agents
                                   that cannot import them
    .cursor/rules/cursor-harness.mdc  Cursor's own harness rules (Cursor has
                                   no user-level rules file, so they have to
                                   be delivered per project)
    .github/copilot-instructions.md   a pointer, for the Copilot surfaces that
                                   read nothing but this path
    .claude/project-core.md        AGENTS.md minus the host block and minus
                                   every path-scoped section; what CLAUDE.md
                                   imports
    .claude/rules/<slug>.md        one per path-scoped section, deferred by
                                   `paths:` frontmatter
    .github/instructions/<slug>.instructions.md
                                   the same sections for Copilot Chat,
                                   deferred by `applyTo:` frontmatter

Why the host rules are inlined rather than imported: Claude Code and
Antigravity load `~/.agent-rules/AGENTS.md` themselves, but Copilot's cloud
agent and Cursor's cloud agents run on machines that have never seen this
host's home directory. Anything they must obey has to be committed into the
repository.

Why AGENTS.md is complete but Claude's copy is split (the `scope:` markers):
AGENTS.md stays the whole rule set in one file because Cursor, Codex,
Antigravity and Copilot's cloud agent read it eagerly and have no way to defer
part of it. Claude and Copilot Chat *do* have a way -- `paths:` and `applyTo:`
frontmatter -- so a section that only matters when touching `src/` is
delivered to them as a rule file that loads on demand instead of costing
context in every session. Same source, same intent; only the moment of
delivery differs. Mark a section by putting

    <!-- scope: src/**, tests/** -->

immediately above its `## ` heading in AGENTS.md. Sections with no marker are
always-on for everyone.

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
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

AGENTS_MD = REPO_ROOT / "AGENTS.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
CLAUDE_CORE = REPO_ROOT / ".claude" / "project-core.md"
CLAUDE_RULES = REPO_ROOT / ".claude" / "rules"
CURSOR_RULE = REPO_ROOT / ".cursor" / "rules" / "cursor-harness.mdc"
COPILOT_MD = REPO_ROOT / ".github" / "copilot-instructions.md"
COPILOT_RULES = REPO_ROOT / ".github" / "instructions"

# What CLAUDE.md must import for a Claude session to see this project's rules.
CLAUDE_IMPORT = "@.claude/project-core.md"

HOST_DIR = Path.home() / ".agent-rules"
HOST_RULES = HOST_DIR / "AGENTS.md"
HOST_CURSOR_HARNESS = HOST_DIR / "harness" / "cursor.md"

# Files the previous layout generated, now superseded. Antigravity reads the
# root AGENTS.md natively since v1.20.3, and Copilot's cloud agent and code
# review read it too, so a per-agent copy of the same prose is dead weight.
# `.github/instructions/` is deliberately NOT here any more: it came back with
# a different job, carrying only the path-scoped sections under `applyTo:`.
OBSOLETE = (
    REPO_ROOT / ".agents" / "rules",
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
SCOPE_MARKER = re.compile(r"^<!--\s*scope:\s*(?P<globs>[^>]+?)\s*-->[ \t]*$", re.MULTILINE)
SECTION_HEADING = re.compile(r"^## .+$", re.MULTILINE)

# Every file this script writes carries this, so a generated rule file that no
# longer corresponds to a `scope:` marker can be recognised and swept.
BANNER = "GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py"


class Scoped(NamedTuple):
    """One `## ` section of AGENTS.md that only applies to matching files."""

    slug: str
    title: str
    globs: tuple[str, ...]
    body: str


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


def slugify(title: str) -> str:
    """A filename stem from a section heading."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def split_scoped(body: str) -> tuple[str, list[Scoped]]:
    """Split project rules into the always-on core and the path-scoped sections.

    A section is scoped by a `<!-- scope: glob, glob -->` comment sitting on its
    own line immediately above a `## ` heading; it runs to the next `## `
    heading or the end of the text. The marker and its section are cut out of
    the core, which is what Claude loads in every session.
    """
    core = body
    scoped: list[Scoped] = []
    cuts: list[tuple[int, int]] = []

    for marker in SCOPE_MARKER.finditer(body):
        heading = SECTION_HEADING.search(body, marker.end())
        if heading is None or body[marker.end() : heading.start()].strip():
            raise ValueError(
                f"scope marker at offset {marker.start()} is not immediately "
                "above a '## ' heading; move it or drop it"
            )
        following = SECTION_HEADING.search(body, heading.end())
        end = following.start() if following else len(body)
        # Stop at the next scope marker too, not just the next heading: when two
        # scoped sections are adjacent the marker sits *above* the heading, so
        # ending at the heading would swallow the next section's marker into
        # this one's body and make the two cut ranges overlap -- which silently
        # ate an unrelated heading from the core.
        next_marker = SCOPE_MARKER.search(body, heading.end())
        if next_marker is not None and next_marker.start() < end:
            end = next_marker.start()
        title = body[heading.start() : heading.end()].lstrip("# ").strip()
        globs = tuple(
            glob.strip() for glob in marker.group("globs").split(",") if glob.strip()
        )
        if not globs:
            raise ValueError(f"scope marker above '{title}' lists no globs")
        scoped.append(
            Scoped(
                slug=slugify(title),
                title=title,
                globs=globs,
                body=body[heading.start() : end].strip(),
            )
        )
        cuts.append((marker.start(), end))

    for start, end in reversed(cuts):
        core = core[:start] + core[end:]
    return re.sub(r"\n{3,}", "\n\n", core).strip() + "\n", scoped


def claude_core(core: str, scoped: list[Scoped]) -> str:
    """The always-on slice of the project rules, for CLAUDE.md to import."""
    pointer = ""
    if scoped:
        listed = "\n".join(
            f"- **{item.title}** — loads when you touch `{'`, `'.join(item.globs)}`"
            for item in scoped
        )
        pointer = (
            "\n## Rules that load on demand\n\n"
            "These sections of `AGENTS.md` are not in this file. They arrive "
            "automatically the moment you read a file they cover, so you do not "
            "need to fetch them by hand — but if you are reasoning about one of "
            "these areas *without* opening its files, read the matching file "
            "under `.claude/rules/` first.\n\n"
            f"{listed}\n"
        )
    return (
        f"<!-- {BANNER}\n"
        "     AGENTS.md minus two things a Claude session gets elsewhere: the\n"
        "     host-rules block (already loaded from ~/.claude/CLAUDE.md) and\n"
        "     every `scope:`-marked section (delivered on demand via\n"
        "     .claude/rules/). CLAUDE.md imports this instead of AGENTS.md so a\n"
        "     session pays context only for what it needs every time. -->\n\n"
        f"{core}{pointer}"
    )


def claude_rule(item: Scoped) -> str:
    """One path-scoped section as a Claude project rule."""
    globs = "\n".join(f'  - "{glob}"' for glob in item.globs)
    return (
        f"---\npaths:\n{globs}\n---\n\n"
        f"<!-- {BANNER}\n"
        f"     The '{item.title}' section. Scoped so it costs no context until\n"
        "     Claude reads a file it actually governs. -->\n\n"
        f"{item.body}\n"
    )


def copilot_rule(item: Scoped) -> str:
    """The same section for Copilot Chat, which scopes with `applyTo:`."""
    return (
        f'---\napplyTo: "{",".join(item.globs)}"\n---\n\n'
        f"<!-- {BANNER}\n"
        f"     The '{item.title}' section. Copilot's cloud agent and code review\n"
        "     read the whole of AGENTS.md; this copy is for Copilot Chat, which\n"
        "     reads only .github/. -->\n\n"
        f"{item.body}\n"
    )


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

    # The tiered delivery. Split what a project hand-wrote -- never the host
    # block, which Claude already has from ~/.claude/CLAUDE.md and which is
    # dropped from the core slice for exactly that reason.
    core, scoped = split_scoped(HOST_BLOCK.sub("\n", agents_md).rstrip() + "\n")
    planned[CLAUDE_CORE] = claude_core(core, scoped)
    for item in scoped:
        planned[CLAUDE_RULES / f"{item.slug}.md"] = claude_rule(item)
        planned[COPILOT_RULES / f"{item.slug}.instructions.md"] = copilot_rule(item)
    return planned


def orphans(planned: dict[Path, str]) -> list[Path]:
    """Generated rule files whose `scope:` marker is gone from AGENTS.md.

    Only files carrying the banner are swept, so a hand-written rule dropped
    into either directory is left alone.
    """
    found: list[Path] = []
    for directory in (CLAUDE_RULES, COPILOT_RULES):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.md")):
            if path not in planned and BANNER in path.read_text():
                found.append(path)
    return found


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
    """Whether CLAUDE.md still imports the generated core slice.

    Advisory. Claude Code reads no AGENTS.md of its own, so a CLAUDE.md that
    has lost its import line means Claude sessions silently run with no project
    rules at all -- the exact failure this layout exists to prevent. A plain
    `@AGENTS.md` also counts: it is the pre-tiering layout, correct but
    expensive, and worth flagging rather than failing.
    """
    if not CLAUDE_MD.is_file():
        return False
    return CLAUDE_IMPORT in CLAUDE_MD.read_text()


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
    except ValueError as bad_marker:
        print(f"error: AGENTS.md: {bad_marker}", file=sys.stderr)
        return 1

    stale = [
        path
        for path, content in planned.items()
        if (path.read_text() if path.is_file() else None) != content
    ]
    dead = obsolete_present() + orphans(planned)

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
            f"warning: CLAUDE.md does not import `{CLAUDE_IMPORT}` — Claude Code "
            "sessions will load no project rules, or will load the whole of "
            "AGENTS.md including the host block it already has.",
            file=sys.stderr,
        )

    changed = len(stale) + len(dead)
    print(f"agent rules: {changed} file(s) updated" if changed else "agent rules: up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
