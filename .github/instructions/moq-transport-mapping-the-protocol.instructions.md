---
applyTo: "src/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'MoQ transport mapping (the protocol)' section. Copilot's cloud agent and code review
     read the whole of AGENTS.md; this copy is for Copilot Chat, which
     reads only .github/. -->

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
