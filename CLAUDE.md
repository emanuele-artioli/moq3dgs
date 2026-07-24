# MOQ3DGS

Viewport-aware 3D Gaussian Splatting (3DGS) streamed to browsers over Media-over-QUIC
(MoQ): the server partitions a pre-trained 3DGS scene into spatial clusters and streams
them with dynamic, viewport-driven priority; the client replays a camera trace (or takes
live 6-DOF input), subscribes only to the clusters it needs, and renders/persists frames.
Research-stage streaming-transport work. No companion paper repo — unlike
pointstream/presley, no nested Overleaf-style git checkout exists in this tree.

**This file is the only rule file to edit by hand.** `AGENTS.md`,
`.agents/rules/moq3dgs.md` (Antigravity) and
`.github/instructions/moq3dgs.instructions.md` (Copilot) are *generated* from it by
`tools/sync_agent_rules.py`, which also inlines the host-wide `~/.claude/CLAUDE.md` that
only Claude Code loads automatically. Edit CLAUDE.md, then re-run the script. There is no
pre-commit hook enforcing this here yet (this repo doesn't use pre-commit at all — see
"Gaps" below), so re-run `python3 tools/sync_agent_rules.py` by hand after editing this
file and before committing.

## Entry point

Everything runs inside the `3dgs_moq` conda env. `environment.yaml` bootstraps the
CUDA/PyTorch binaries; `pyproject.toml` is the source of truth for everything else
(`pip install -e ".[dev]"` after activating).

```
conda activate 3dgs_moq
moq3dgs-server --scene assets/bicycle.ply --clusters 64 --port 4433
moq3dgs-client --trace assets/bicycle_trace.json --device cuda:1
```

- **Verified default assets**: `assets/bicycle.ply` (the server's own `--scene`
  default) and `assets/bicycle_trace.json` (the README's Quick Start `--trace`, and the
  only trace file that actually exists in `assets/`). Do **not** use
  `/home/itec/emanuele/3dgs_moq/assets/train_scene/point_cloud.ply` or
  `.../traces/eval_trace_01.json` — both old rule files named these, but that directory
  does not exist anywhere on this host; it looks like a stale or hallucinated path built
  from the conda env name rather than the repo path. `src/moq3dgs/client_app.py`'s own
  `DEFAULT_TRACE = "assets/eval_trace.json"` is *also* dead — no such file exists in
  `assets/` — so always pass `--trace` explicitly.
- `assets/bicycle_traces` is a symlink to
  `~/Datasets/EyeNavGS_Rutgers_Dataset/dataset/bicycle` (external, gitignored) — a real
  multi-frame trace source if `bicycle_trace.json` alone isn't enough.
- **Real (non-synthetic) end-to-end network tests must pass `--scene`/`--trace`
  explicitly**, pointing at the real assets above. Don't fall back to
  `scripts/generate_test_scene.py` / `scripts/generate_test_trace.py`'s synthetic mock
  data for anything meant to validate actual transport behavior — those two scripts
  exist specifically to make fast, GPU-free unit/integration tests possible.
- `scripts/run_e2e.py --scene assets/bicycle.ply --trace assets/bicycle_trace.json`
  runs server + client together in one process for a full-pipeline smoke test.

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

## MoQ transport mapping (the protocol)

Every data structure crossing the client/server boundary maps onto this hierarchy — both
old rule files and the README agreed on this:

| MoQ concept           | 3DGS mapping                                              |
|------------------------|-------------------------------------------------------------|
| **Broadcast**          | The entire 3DGS scene                                       |
| **Track**              | A spatial volume (octree leaf / semantic region)             |
| **Group**              | A density cluster (K-means) within a Track                   |
| **Subgroup (Object)**  | Importance tier / LoD bitrate ladder                          |

- Importance tiers (`ImportanceTier` in `src/moq3dgs/models.py`): `LARGE` (top 5% WFPS
  coverage) → `MEDIUM` (next 15%) → `SMALL` (remaining 80%). Each tier ships at two
  quality levels — Quality 0 (base geometry + opacity + SH0, self-contained) and
  Quality 1 (adds SH1-3 high-frequency detail) — base geometry always holds priority over
  SH refinements, to avoid "dumb volume" hard-edge artifacts during LoD transitions.
- **State & caching**: never resend a splat cluster already on the client. The server
  publishes a coordinate → Track/Group manifest; the client keeps a persistent local
  cache and issues `SUBSCRIBE` updates only for clusters the frustum calculator flags as
  missing or under-quality.
- **Dynamic priority** ranges 0 (critical) → 255 (droppable), recomputed from: distance
  to camera, frustum position (centre > periphery > behind), and tier (base geometry >
  SH refinement).
- **Modularity**: keep MoQ transport (`src/moq3dgs/transport/`) decoupled from 3DGS
  rendering (`src/moq3dgs/render/`) — communicate via the async queues and Pydantic
  models in `models.py`, not direct calls across the boundary.

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

## Testing

`pytest tests/ -v` (`pyproject.toml`'s `[tool.pytest.ini_options]` sets
`testpaths = ["tests"]`, `asyncio_mode = "auto"`). There is no coverage-gate script here
yet, unlike pointstream/presley's `check_coverage_gate.py` — see "Gaps" below.

Write a unit test or integration script alongside new core logic (MoQ packetization,
spatial clustering, frustum culling), not after the fact. Research code, so keep tests
honest and thin: cover envisioned behavior and plausible misuse of code we own, and skip
padding tests that exist only to move a coverage number — see the shared testing rule
below for the full reasoning.

## Gaps / deliberately left alone

- **No pre-commit**: no `.pre-commit-config.yaml`, no installed `.git/hooks`, no CI
  workflow anywhere in this repo. A `sync-agent-rules` pre-commit entry was **not**
  added — introducing pre-commit as a new dev-tool dependency wasn't this pass's call to
  make. Until it (or a CI job) is adopted, run `python3 tools/sync_agent_rules.py
  --check` by hand before committing rule changes.
- `test_gsplat.py` at the repo root (outside `tests/`) is a standalone `gsplat` CUDA
  smoke test hardcoded to `cuda:1` — it is not part of the pytest suite
  (`testpaths = ["tests"]` excludes it) and isn't meant to be.
- `client.log` / `server.log` at the repo root are stray run logs from past manual runs
  (already covered by the generic `*.log` gitignore rule) — safe to delete, not tracked.

## This tooling is meant to evolve

`.claude/` (this file, `agents/`, `skills/`, `hooks/`, `settings.json`) is part of the
working setup, not frozen — if a convention gets misapplied twice, or a repeated
debugging workflow (e.g. MoQ packetization) turns out to want its own skill, add it then.
Edits to `settings.json`/hooks take effect next session; CLAUDE.md loads fresh each
session.
