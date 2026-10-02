from max.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.math import sqrt
from std.sys import simd_width_of
from ..shared.layout import Layout
from ..gpu.device import DeviceState
from .kernel_helpers import elementwise_launch_config


def adamw_step_kernel[
    dtype: DType,
](
    param: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    m: Pointer[Scalar[dtype], MutAnyOrigin],
    v: Pointer[Scalar[dtype], MutAnyOrigin],
    num_elements_: Int64,
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    weight_decay: Scalar[dtype],
    bias_correction1: Scalar[dtype],
    bias_correction2: Scalar[dtype],
):
    var num_elements = Int(num_elements_)
    var gtid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var one_minus_beta1 = Scalar[dtype](1) - beta1
    var one_minus_beta2 = Scalar[dtype](1) - beta2
    var i = gtid
    while i < num_elements:
        var p = param[unsafe_offset=i]
        var g = grad[unsafe_offset=i]
        var mm = beta1 * m[unsafe_offset=i] + one_minus_beta1 * g
        var vv = beta2 * v[unsafe_offset=i] + one_minus_beta2 * g * g
        m[unsafe_offset=i] = mm
        v[unsafe_offset=i] = vv
        var m_hat = mm / bias_correction1
        var v_hat = vv / bias_correction2
        param[unsafe_offset=i] = p - lr * (
            m_hat / (sqrt(v_hat) + eps) + weight_decay * p
        )
        i += stride


@fieldwise_init
struct AdamWKernel[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def launch(
        param_layout: Layout,
        param_device_state: DeviceState[Self.dtype],
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        m_layout: Layout,
        m_device_state: DeviceState[Self.dtype],
        v_layout: Layout,
        v_device_state: DeviceState[Self.dtype],
        num_elements: Int,
        lr: Scalar[Self.dtype],
        beta1: Scalar[Self.dtype],
        beta2: Scalar[Self.dtype],
        eps: Scalar[Self.dtype],
        weight_decay: Scalar[Self.dtype],
        bias_correction1: Scalar[Self.dtype],
        bias_correction2: Scalar[Self.dtype],
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
        ref m_ds = m_device_state
        ref m_buf = m_ds.device_buffer()
        ref v_ds = v_device_state
        ref v_buf = v_ds.device_buffer()
        var compiled = ctx.compile_function[
            adamw_step_kernel[Self.dtype],
        ]()
        ctx.enqueue_function(
            compiled,
            param_buf,
            grad_buf,
            m_buf,
            v_buf,
            Int64(num_elements),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            grid_dim=blocks,
            block_dim=tpb,
        )
        if sync:
            ctx.synchronize()
