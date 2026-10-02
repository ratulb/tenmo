"""Wall-clock timing helper (moved from tenmo/common_utils.mojo).

Layer-0 leaf: stdlib only.
"""

from std.time import perf_counter_ns


@always_inline("nodebug")
def now() -> Float64:
    """Seconds since an arbitrary origin (monotonic `perf_counter`)."""
    return Float64(perf_counter_ns()) / 1e9
