from max.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.sys import simd_width_of
from ..shared.layout import Layout
from ..gpu.device import DeviceState
from .kernel_helpers import elementwise_launch_config


def sgd_step_no_momentum_kernel[
    dtype: DType,
](
    param: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements_: Int64,
    lr: Scalar[dtype],
    weight_decay: Scalar[dtype],
):
    var num_elements = Int(num_elements_)
    var gtid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = gtid
    while i < num_elements:
        var p = param[unsafe_offset=i]
        var g = grad[unsafe_offset=i]
        if weight_decay > 0:
            g += p * weight_decay
        param[unsafe_offset=i] = p - lr * g
        i += stride


def sgd_step_momentum_kernel[
    dtype: DType,
](
    param: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    vel: Pointer[Scalar[dtype], MutAnyOrigin],
    num_elements_: Int64,
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
    weight_decay: Scalar[dtype],
):
    var num_elements = Int(num_elements_)
    var gtid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = gtid
    while i < num_elements:
        var p = param[unsafe_offset=i]
        var g = grad[unsafe_offset=i]
        var v = vel[unsafe_offset=i]
        if weight_decay > 0:
            g += p * weight_decay
        v = momentum * v + g
        vel[unsafe_offset=i] = v
        param[unsafe_offset=i] = p - lr * v
        i += stride


@fieldwise_init
struct SGDKernel[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def launch_no_momentum(
        param_layout: Layout,
        param_device_state: DeviceState[Self.dtype],
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        num_elements: Int,
        lr: Scalar[Self.dtype],
        weight_decay: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var (blocks, tpb) = elementwise_launch_config(num_elements, simdwidth)
        ref param_ds = param_device_state
        ref gpu = param_ds.get_gpu()
        var ctx = gpu[]
        ref param_buf = param_ds.device_buffer()
        ref grad_ds = grad_device_state
        ref grad_buf = grad_ds.device_buffer()
        var compiled = ctx.compile_function[
            sgd_step_no_momentum_kernel[Self.dtype],
        ]()
        ctx.enqueue_function(
            compiled,
            param_buf,
            grad_buf,
            Int64(num_elements),
            lr,
            weight_decay,
            grid_dim=blocks,
            block_dim=tpb,
        )
        if sync:
            ctx.synchronize()

    @staticmethod
    def launch_momentum(
        param_layout: Layout,
        param_device_state: DeviceState[Self.dtype],
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        vel_layout: Layout,
        vel_device_state: DeviceState[Self.dtype],
        num_elements: Int,
        lr: Scalar[Self.dtype],
        momentum: Scalar[Self.dtype],
        weight_decay: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var (blocks, tpb) = elementwise_launch_config(num_elements, simdwidth)
        ref param_ds = param_device_state
        ref gpu = param_ds.get_gpu()
        var ctx = gpu[]
        ref param_buf = param_ds.device_buffer()
        ref grad_ds = grad_device_state
        ref grad_buf = grad_ds.device_buffer()
        ref vel_ds = vel_device_state
        ref vel_buf = vel_ds.device_buffer()
        var compiled = ctx.compile_function[
            sgd_step_momentum_kernel[Self.dtype],
        ]()
        ctx.enqueue_function(
            compiled,
            param_buf,
            grad_buf,
            vel_buf,
            Int64(num_elements),
            lr,
            momentum,
            weight_decay,
            grid_dim=blocks,
            block_dim=tpb,
        )
        if sync:
            ctx.synchronize()
