---
paths:
  - "tests/**"
  - "pytest.ini"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Testing' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Testing

`pytest tests/ -v` (`pyproject.toml`'s `[tool.pytest.ini_options]` sets
`testpaths = ["tests"]`, `asyncio_mode = "auto"`). There is no coverage-gate script here
yet, unlike pointstream/presley's `check_coverage_gate.py` — see "Gaps" below.

Write a unit test or integration script alongside new core logic (MoQ packetization,
spatial clustering, frustum culling), not after the fact. Research code, so keep tests
honest and thin: cover envisioned behavior and plausible misuse of code we own, and skip
padding tests that exist only to move a coverage number — see the shared testing rule
below for the full reasoning.
