---
paths:
  - "src/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Coding style' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Coding style

- Python 3.10+ strict typing everywhere; every function documents its behavioral *why*,
  not just its shapes/types (see `decorators.py` / `models.py` for the target style).
- All client↔server messages are Pydantic `BaseModel`s in `src/moq3dgs/models.py` — never
  raw dicts across the transport boundary.
- Tag resource affinity explicitly with the decorators in `src/moq3dgs/decorators.py`:
  `@network_bound` for QUIC transport / MoQ packetization / socket I/O, `@gpu_bound` for
  PyTorch inference / rasterization / tensor math.
- If rewriting CUDA/C++ rasterizer code or a long Python class, output the complete,
  non-truncated file — no `# ... rest unchanged` elisions.
