# 3DGS-MoQ: Viewport-Aware 3D Gaussian Splatting over Media-over-QUIC

Streams pre-trained 3D Gaussian Splatting scenes to clients over Media-over-QUIC
(MoQ), with viewport-aware spatial partitioning and dynamic priority scheduling.

## Architecture

```
Server                              Client
┌──────────────────────┐            ┌──────────────────────┐
│  Scene Loader (PLY)  │            │  Trace Replayer      │
│  Spatial Partitioner │──manifest─>│  Frustum Calculator  │
│  MoQ Publisher       │──clusters─>│  Subscription Mgr    │
│  Priority Scheduler  │<─viewport──│  Splat Cache         │
└──────────────────────┘            │  GPU Renderer        │
                                    │  Disk Writer         │
                                    └──────────────────────┘
```

### MoQ Transport Mapping

| MoQ Concept   | 3DGS Mapping                                      |
|---------------|----------------------------------------------------|
| **Broadcast** | Entire 3DGS scene                                  |
| **Track**     | Spatial volume (octree node / K-means cluster)     |
| **Group**     | Chunk of Gaussians within a volume                 |
| **Object 0**  | Base: position + opacity + DC colour (SH0)         |
| **Object 1**  | Enhancement: SH1-SH3 + scale refinements           |

## Setup

```bash
# Create conda environment (CUDA + PyTorch)
conda env create -f environment.yaml
conda activate 3dgs_moq

# Or install directly via pip (requires PyTorch pre-installed)
pip install -e ".[dev]"
```

## Quick Start

```bash
# Terminal 1: Start server
moq3dgs-server --scene /path/to/train_scene --clusters 64 --port 4433

# Terminal 2: Start client
moq3dgs-client --trace /path/to/eval_trace_01.json --output ./output
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
src/moq3dgs/
├── models.py           # Pydantic data contracts
├── decorators.py       # @network_bound, @gpu_bound
├── scene/
│   ├── loader.py       # PLY scene loading
│   ├── clustering.py   # K-means + octree partitioning
│   └── lod.py          # LoD splitting (base / enhancement)
├── viewport/
│   ├── frustum.py      # View frustum culling
│   ├── priority.py     # Dynamic priority (0-255)
│   └── trace.py        # Camera trace replay
├── transport/
│   ├── manifest.py     # MoQ manifest generation
│   ├── protocol.py     # Binary wire protocol
│   ├── server.py       # Async MoQ publisher
│   └── client.py       # Async MoQ subscriber
├── render/
│   ├── cache.py        # Persistent splat cache
│   ├── rasterizer.py   # GPU rasterisation wrapper
│   └── writer.py       # Frame-to-disk writer
├── server_app.py       # Server entry point
└── client_app.py       # Client entry point
```

## Priority Scheme

Priority `0` (highest) → `255` (lowest), computed from:

1. **Distance to camera** — closer = higher priority
2. **Frustum position** — centre = highest, periphery = medium, behind = lowest
3. **LoD layer** — base (Object 0) > enhancement (Object 1)

## License

MIT
