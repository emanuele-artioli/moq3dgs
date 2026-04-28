"""Server entry point: load scene → partition → serve.

Usage::

    python -m moq3dgs.server_app --scene /path/to/scene --port 4433

Or via the installed console script::

    moq3dgs-server --scene /path/to/scene
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

from moq3dgs.transport.server import create_server

DEFAULT_SCENE = "/home/itec/emanuele/3dgs_moq/assets/train_scene/point_cloud.ply"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MoQ server."""
    parser = argparse.ArgumentParser(
        description="3DGS-MoQ Server — streams spatially-partitioned Gaussians over MoQ/QUIC",
    )
    parser.add_argument(
        "--scene", type=str, default=DEFAULT_SCENE,
        help="Path to the 3DGS scene directory or .ply file.",
    )
    parser.add_argument(
        "--clusters", type=int, default=64,
        help="Number of spatial clusters for K-means partitioning.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address.",
    )
    parser.add_argument(
        "--port", type=int, default=4433,
        help="Bind port.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Server main entry point."""
    args = parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log_level)
        ),
    )

    server = create_server(
        scene_path=args.scene,
        num_clusters=args.clusters,
        host=args.host,
        port=args.port,
    )

    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\nShutting down server…")
        sys.exit(0)


if __name__ == "__main__":
    main()
