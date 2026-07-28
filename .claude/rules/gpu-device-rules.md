---
paths:
  - "src/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'GPU / device rules' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## GPU / device rules

- **GPU 0 on this host is often occupied by other processes** (the README's own note).
  `moq3dgs-client` defaults `--device` to `cuda:1` for rendering; `moq3dgs-server`
  defaults to `cpu` for WFPS preprocessing (pass `--device cuda:0`/`cuda:1` explicitly to
  move it onto a GPU). This is intentional and contradicts one of the old rule files
  (Antigravity's "always fall back to a single available `cuda:0`, never hardcode
  `cuda:1`") — that guidance matched neither the shipped code nor the README and was
  dropped; see the handoff report for how this conflict was resolved.
- Code should still degrade gracefully when a specific `cuda:N` is unavailable —
  `render/rasterizer.py` already branches on `means.device.type == "cpu"`; extend that
  pattern rather than assuming a specific device index is always present.
- Multi-GPU assignment (which worker gets which device) belongs in CLI flags or scripts,
  not hardcoded inside library modules under `src/moq3dgs/`.
