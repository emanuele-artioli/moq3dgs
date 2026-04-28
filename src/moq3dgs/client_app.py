"""Client entry point: connect → replay trace → subscribe → render → save.

Usage::

    python -m moq3dgs.client_app --trace /path/to/trace.json --output ./output

Or via the installed console script::

    moq3dgs-client --trace /path/to/trace.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

from moq3dgs.transport.client import run_client_trace

DEFAULT_TRACE = "/home/itec/emanuele/3dgs_moq/assets/traces/eval_trace_01.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MoQ client."""
    parser = argparse.ArgumentParser(
        description="3DGS-MoQ Client — replays a camera trace and collects streamed Gaussians",
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
        help="Directory to save received frames and metrics.",
    )
    parser.add_argument(
        "--client-id", type=str, default="client-0",
        help="Client identifier.",
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
        asyncio.run(
            run_client_trace(
                host=args.host,
                port=args.port,
                trace_path=args.trace,
                output_dir=args.output,
                client_id=args.client_id,
            )
        )
    except KeyboardInterrupt:
        print("\nClient interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
