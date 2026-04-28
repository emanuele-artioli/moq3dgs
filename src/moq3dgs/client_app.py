"""Client entry point: connect → replay trace → render → save.

Usage::

    python -m moq3dgs.client_app --trace assets/eval_trace.json --output ./output

Or via the installed console script::

    moq3dgs-client --trace assets/eval_trace.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

from moq3dgs.transport.client import run_client_trace

DEFAULT_TRACE = "assets/eval_trace.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MoQ client."""
    parser = argparse.ArgumentParser(
        description="3DGS-MoQ Client — replays a camera trace, receives "
        "streamed Gaussians, renders and saves frames to disk",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server address.",
    )
    parser.add_argument(
        "--port", type=int, default=4433,
        help="Server port.",
    )
    parser.add_argument(
        "--trace", type=str, default=DEFAULT_TRACE,
        help="Path to the camera movement trace JSON.",
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Directory to save rendered frames and metrics.",
    )
    parser.add_argument(
        "--client-id", type=str, default="client-0",
        help="Client identifier.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device for rendering (e.g. cuda:0, cuda:1, cpu).",
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable rendering (transport-only benchmark).",
    )
    parser.add_argument(
        "--frame-delay", type=float, default=0.05,
        help="Seconds to wait between frames for data arrival.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Client main entry point."""
    args = parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log_level)
        ),
    )

    try:
        summary = asyncio.run(
            run_client_trace(
                host=args.host,
                port=args.port,
                trace_path=args.trace,
                output_dir=args.output,
                client_id=args.client_id,
                device=args.device,
                image_width=args.width,
                image_height=args.height,
                render_enabled=not args.no_render,
                inter_frame_delay=args.frame_delay,
            )
        )
        print(f"\n✓ Trace complete: {summary['frames_rendered']} frames rendered, "
              f"{summary['clusters_received']} clusters received, "
              f"{summary['bytes_received'] / 1e6:.1f} MB")
    except ConnectionRefusedError:
        print(f"✗ Could not connect to server at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nClient interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
