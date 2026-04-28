#!/usr/bin/env python3
"""End-to-end integration test: server + client with a real PLY scene.

Starts a server in a background task, runs the client through a camera
trace, renders frames via CPU fallback, and saves everything to disk.

Usage::

    python scripts/run_e2e.py --scene assets/bicycle.ply --trace assets/eval_trace.json

This script is useful for:
- Verifying the pipeline works with actual 3DGS data.
- Producing rendered frames for visual inspection.
- Benchmarking transport throughput and render latency.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import structlog

from moq3dgs.transport.client import MoQClient
from moq3dgs.transport.server import create_server


async def run_e2e(
    scene_path: str,
    trace_path: str,
    output_dir: str,
    num_clusters: int = 32,
    port: int = 14435,
    device: str = "cpu",
    width: int = 640,
    height: int = 480,
    frame_delay: float = 0.1,
    no_render: bool = False,
) -> dict:
    """Run a complete server + client session."""
    # 1. Start server
    server = create_server(
        scene_path=scene_path,
        num_clusters=num_clusters,
        host="127.0.0.1",
        port=port,
    )
    await server.start()

    try:
        # 2. Create and connect client
        client = MoQClient(
            host="127.0.0.1",
            port=port,
            output_dir=output_dir,
            device=device,
            image_width=width,
            image_height=height,
            render_enabled=not no_render,
        )

        await client.connect()
        await client.start_receiving()

        # 3. Run the trace
        summary = await client.run_trace(
            trace_path=trace_path,
            inter_frame_delay=frame_delay,
        )

        await client.disconnect()
        return summary

    finally:
        await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end MoQ 3DGS test")
    parser.add_argument("--scene", type=str, default="assets/bicycle.ply")
    parser.add_argument("--trace", type=str, default="assets/eval_trace.json")
    parser.add_argument("--output", type=str, default="output/e2e_test")
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--port", type=int, default=14435)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-delay", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Limit trace to first N frames for quick testing.")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (transport-only benchmark).")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log_level)
        ),
    )

    # Optionally truncate the trace for quick testing
    if args.max_frames:
        import json
        trace = json.loads(Path(args.trace).read_text())
        frames = trace.get("frames", trace) if isinstance(trace, dict) else trace
        frames = frames[:args.max_frames]
        # Write truncated trace to a temp location
        truncated = Path(args.output) / "_truncated_trace.json"
        truncated.parent.mkdir(parents=True, exist_ok=True)
        truncated.write_text(json.dumps({"frames": frames}))
        args.trace = str(truncated)

    summary = asyncio.run(run_e2e(
        scene_path=args.scene,
        trace_path=args.trace,
        output_dir=args.output,
        num_clusters=args.clusters,
        port=args.port,
        device=args.device,
        width=args.width,
        height=args.height,
        frame_delay=args.frame_delay,
        no_render=args.no_render,
    ))

    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETE")
    print("=" * 60)
    print(f"  Frames rendered:   {summary['frames_rendered']}")
    print(f"  Clusters received: {summary['clusters_received']}")
    print(f"  Bytes received:    {summary['bytes_received'] / 1e6:.2f} MB")
    print(f"  Cache entries:     {summary['cache_entries_final']}")
    print(f"  Elapsed:           {summary['elapsed_seconds']:.1f}s")
    print(f"  Output:            {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
