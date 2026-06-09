from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from tenmo.ndbuffer import NDBuffer


def _sgd_launch_config(num_elements: Int) -> Tuple[Int, Int]:
    var tpb = 256 if num_elements >= 256 else num_elements
    var blocks = (num_elements + tpb - 1) // tpb
    return (tpb, blocks)


def sgd_step_no_momentum_kernel[
    dtype: DType,
](
    param: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
    lr: Scalar[dtype],
    weight_decay: Scalar[dtype],
):
    var gtid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = gtid
    while i < num_elements:
        var p = param[i]
        var g = grad[i]
        if weight_decay > 0:
            g += p * weight_decay
        param[i] = p - lr * g
        i += stride


def sgd_step_momentum_kernel[
    dtype: DType,
](
    param: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    vel: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    num_elements: Int,
    lr: Scalar[dtype],
    momentum: Scalar[dtype],
    weight_decay: Scalar[dtype],
):
    var gtid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = gtid
    while i < num_elements:
        var p = param[i]
        var g = grad[i]
        var v = vel[i]
        if weight_decay > 0:
            g += p * weight_decay
        v = momentum * v + g
        vel[i] = v
        param[i] = p - lr * v
        i += stride


@fieldwise_init
struct SGDStep[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def launch_no_momentum(
        param_ndb: NDBuffer[dtype],
        grad_ndb: NDBuffer[dtype],
        num_elements: Int,
        lr: Scalar[dtype],
        weight_decay: Scalar[dtype],
        sync: Bool = False,
    ) raises:
        var (tpb, blocks) = _sgd_launch_config(num_elements)
        ref param_ds = param_ndb.device_state.value()
        ref gpu = param_ds.get_gpu()
        var ctx = gpu[]
        ref param_buf = param_ds.device_buffer()
        ref grad_ds = grad_ndb.device_state.value()
        ref grad_buf = grad_ds.device_buffer()
        var compiled = ctx.compile_function[
            sgd_step_no_momentum_kernel[dtype],
            sgd_step_no_momentum_kernel[dtype],
        ]()
        ctx.enqueue_function(
            compiled,
            param_buf,
            grad_buf,
            num_elements,
            lr,
            weight_decay,
            grid_dim=blocks,
            block_dim=tpb,
        )
        if sync:
            ctx.synchronize()

    @staticmethod
    def launch_momentum(
        param_ndb: NDBuffer[dtype],
        grad_ndb: NDBuffer[dtype],
        vel_ndb: NDBuffer[dtype],
        num_elements: Int,
        lr: Scalar[dtype],
        momentum: Scalar[dtype],
        weight_decay: Scalar[dtype],
        sync: Bool = False,
    ) raises:
        var (tpb, blocks) = _sgd_launch_config(num_elements)
        ref param_ds = param_ndb.device_state.value()
        ref gpu = param_ds.get_gpu()
        var ctx = gpu[]
        ref param_buf = param_ds.device_buffer()
        ref grad_ds = grad_ndb.device_state.value()
        ref grad_buf = grad_ds.device_buffer()
        ref vel_ds = vel_ndb.device_state.value()
        ref vel_buf = vel_ds.device_buffer()
        var compiled = ctx.compile_function[
            sgd_step_momentum_kernel[dtype],
            sgd_step_momentum_kernel[dtype],
        ]()
        ctx.enqueue_function(
            compiled,
            param_buf,
            grad_buf,
            vel_buf,
            num_elements,
            lr,
            momentum,
            weight_decay,
            grid_dim=blocks,
            block_dim=tpb,
        )
        if sync:
            ctx.synchronize()
