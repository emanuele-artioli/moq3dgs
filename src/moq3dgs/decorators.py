"""Resource tagging decorators for the execution DAG.

These decorators annotate functions with their primary resource affinity
so that schedulers and profilers can separate CPU/network-bound work from
GPU-bound work.
"""

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def network_bound(func: F) -> F:
    """Tag a function as QUIC transport / MoQ packetization / socket I/O bound.

    The scheduler uses this to decide whether the function can run
    concurrently with GPU kernels or must wait for network readiness.
    """
    func.is_network_bound = True  # type: ignore[attr-defined]
    return func


def gpu_bound(func: F) -> F:
    """Tag a function as PyTorch inference / rasterization / tensor-math bound.

    The scheduler uses this to pin the function to a specific CUDA device
    and avoid interleaving GPU work that would cause sync stalls.
    """
    func.is_gpu_bound = True  # type: ignore[attr-defined]
    return func
