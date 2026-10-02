"""Layer-0 host<->device transfer helpers over `Layout` + `DeviceState`.

All helpers work purely in terms of `Layout` + `DeviceState` — no
NDBuffer/Tensor in sight. Strided CPU regions are gathered element-by-element
using the layout's flat-index math.
"""

from std.memory import unsafe_memcpy

from ..shared.layout import Layout
from ..shared.buffers import Buffer
from .device import DeviceState


@always_inline
def _flat_index(layout: Layout, n: Int) -> Int:
    """Flat buffer index of the n-th logical element of `layout`."""
    var rem = n
    var flat = layout.offset
    for k in range(layout.rank() - 1, -1, -1):
        var coord = rem % layout.shape[k]
        rem //= layout.shape[k]
        flat += coord * layout.strides[k]
    return flat


def host_to_device[
    dtype: DType,
](
    dst: DeviceState[dtype],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    layout: Layout,
    sync: Bool = False,
) raises:
    """Copy a CPU region addressed by `layout` into `dst`'s contiguous buffer.

    The CPU data lives at flat indices `layout.offset + Σ coord_k * stride_k`
    (i.e. the layout describes a possibly-strided view over the host array).
    `dst` is always written contiguously (index 0..numel-1).
    """
    with dst.buffer.map_to_host() as host_buffer:
        var device_ptr = host_buffer.unsafe_ptr()
        var numels = layout.numel()
        comptime storage = DType.uint8 if dtype == DType.bool else dtype
        for n in range(numels):
            var flat = _flat_index(layout, n)
            comptime if dtype == DType.bool:
                device_ptr[unsafe_offset=n] = Scalar[storage](
                    UInt8(1) if src[unsafe_offset=flat].cast[DType.bool]() else UInt8(0)
                )
            else:
                device_ptr[unsafe_offset=n] = rebind[Scalar[storage]](
                    src[unsafe_offset=flat]
                )
    if sync:
        dst.sync()


def device_to_host[
    dtype: DType,
](
    src: DeviceState[dtype],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    numels: Int,
    sync: Bool = False,
) raises:
    """Copy `src`'s contiguous device buffer to `dst` (indices 0..numels-1).

    bool is converted back from its uint8 storage to `Scalar[dtype]`.
    """
    with src.buffer.map_to_host() as host_buffer:
        var src_ptr = host_buffer.unsafe_ptr()
        comptime if dtype == DType.bool:
            for i in range(numels):
                dst[unsafe_offset=i] = Scalar[dtype](
                    host_buffer[i].cast[DType.uint8]() == UInt8(1)
                )
        else:
            unsafe_memcpy(
                dest=dst,
                src=src_ptr.unsafe_bitcast[Scalar[dtype]](),
                count=numels,
            )
    if sync:
        src.sync()


def device_to_host_strided[
    dtype: DType,
](
    src: DeviceState[dtype],
    layout: Layout,
    sync: Bool = False,
) raises -> Buffer[dtype]:
    """Gather the logical (possibly strided) view of `src` described by
    `layout` into a fresh contiguous CPU `Buffer` (indices 0..numel-1).

    bool is converted back from its uint8 storage to `Scalar[dtype]`.
    """
    var numels = layout.numel()
    comptime storage = DType.uint8 if dtype == DType.bool else dtype
    var out = Buffer[dtype](numels)
    with src.buffer.map_to_host() as host_buffer:
        for n in range(numels):
            var flat = _flat_index(layout, n)
            comptime if dtype == DType.bool:
                out[n] = Scalar[dtype](
                    host_buffer[flat].cast[DType.uint8]() == UInt8(1)
                )
            else:
                out[n] = rebind[Scalar[dtype]](host_buffer[flat])
    if sync:
        src.sync()
    return out


def materialize_contiguous[
    dtype: DType,
](
    src: DeviceState[dtype],
    layout: Layout,
    sync: Bool = False,
) raises -> DeviceState[dtype]:
    """Materialise the logical (possibly strided) view of `src` described by
    `layout` into a fresh contiguous `DeviceState`.

    This is the GPU-side equivalent of the tenmo-layer
    `NDBuffer.contiguous_device_state()`: a strided device view is gathered
    (via a CPU round-trip) into an independent, contiguous device buffer.
    """
    var numels = layout.numel()
    var out = DeviceState[dtype](numels, src.gpu)

    # Stage through a CPU buffer in the *storage* dtype (uint8 for bool) so
    # the source and destination pointer types fold to one consistent type.
    comptime storage = DType.uint8 if dtype == DType.bool else dtype

    var cpu = Buffer[storage](numels)
    with src.buffer.map_to_host() as host_buffer:
        for n in range(numels):
            var flat = _flat_index(layout, n)
            cpu[n] = host_buffer[flat]

    with out.buffer.map_to_host() as host_buffer:
        var device_ptr = host_buffer.unsafe_ptr()
        var cpu_ptr = cpu.unsafe_ptr()
        unsafe_memcpy(
            dest=device_ptr,
            src=cpu_ptr.unsafe_bitcast[Scalar[storage]](),
            count=numels,
        )

    if sync:
        out.sync()
    return out
