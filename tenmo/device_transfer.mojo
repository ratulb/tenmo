from .tensor import Tensor
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
    ArgumentType,
)
from .shared.mnemonics import AddTensor
from .shared.panic import panic
from .gradbox import Gradbox
from .gpu.device import Device, CPU, GPU
from .shared.shapes import Shape
from std.sys import has_accelerator
from .ancestry import Ancestor


struct Flow(RegisterPassable & Equatable, ImplicitlyCopyable):
    var direction: Int
    comptime Cpu2Gpu = Flow(0)
    comptime Gpu2Cpu = Flow(1)
    comptime UnMoved = Flow(-1)

    def __init__(out self, direction: Int = 0):
        self.direction = direction
        if direction < -1 or direction > 1:
            panic(
                "Invalid direction type. Must be '0 → Cpu2Gpu', '1 → Gpu2Cpu',"
                " or '-1 → UnMoved'"
            )

    def __init__(out self, *, copy: Self):
        self.direction = copy.direction

    def __eq__(self, other: Self) -> Bool:
        return self.direction == other.direction

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct DeviceTransferBwdArg(ArgumentType):
    var flow: Flow
    var device: Device
    # Source shape at forward time. Backward needs only this (for the
    # gradbox sanity check) — never the source buffer — so the transfer
    # node runs with needs_parent_data=False and frees the source early.
    var shape: Shape


@fieldwise_init
struct DeviceTransferBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        # Transfer nodes never clear the gradbox.
    ):
        var bwd_arg = (
            output.ancestry().backward_fn().get[DeviceTransferBwdArg]()
        )
        var (flow, device, src_shape) = (
            bwd_arg.flow,
            bwd_arg.device,
            bwd_arg.shape,
        )
        var gradbox = output.gradients()
        var ancestor_ref = output.ancestry().get(0)
        debug_assert(
            src_shape == gradbox.shape(),
            "DeviceTransferBackward: gradbox shape and ancestor shape mismatch",
        )

        if flow == Flow.UnMoved:
            ancestor_ref.update_grad(gradbox, AddTensor, None)
        else:
            comptime if has_accelerator():
                if flow == Flow.Cpu2Gpu:
                    try:
                        var cpu_grad = Gradbox[Self.dtype](
                            gradbox.buffer().to_cpu(sync=False)
                        )
                        ancestor_ref.update_grad(cpu_grad, AddTensor, None)
                    except e:
                        panic(
                            "DeviceTransferBackward: GPU→CPU transfer failed: "
                            + String(e)
                        )
                        ancestor_ref.update_grad(gradbox, AddTensor, None)
                else:
                    try:
                        var gpu_grad = Gradbox[Self.dtype](
                            gradbox.buffer().to_gpu(device.kind[GPU]),
                        )
                        ancestor_ref.update_grad(gpu_grad, AddTensor, None)
                    except e:
                        panic(
                            "DeviceTransferBackward: CPU->GPU transfer failed: "
                            + String(e)
                        )
                        ancestor_ref.update_grad(gradbox, AddTensor, None)
            else:
                ancestor_ref.update_grad(gradbox, AddTensor, None)

        parent_ids.append(ancestor_ref._id)


@fieldwise_init
struct DeviceTransfer[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """Handles tensor transfers between CPU and GPU devices.
    Autograd support included.

    ## Grad Flow Rules

    The `stop_grad` parameter on `to_gpu()` and `to_cpu()` controls whether
    the transfer registers a backward node in the compute graph.

    **Rule 1 — Default behaviour (`stop_grad=False`):**
        The transfer is transparent to autograd. Gradients tunnel through
        device boundaries as if the transfer never happened. The origin
        tensor receives gradients exactly as it would from a purely
        same-device computation.

        Example: A(CPU) -> to_gpu() -> ops -> loss.backward()
                 => A.grad is populated.

    **Rule 2 — `stop_grad=True` severs the graph at that boundary:**
        The destination tensor becomes a new leaf on the target device.
        No backward node is registered for the transfer. Gradients
        accumulate on the destination leaf and never cross back to the
        source tensor.

        Example: A(CPU) -> to_gpu(stop_grad=True) -> B(GPU leaf)
                 -> ops -> loss.backward()
                 => B.grad is populated, A.grad is untouched.

    **Rule 3 — Each transfer is an independent boundary:**
        In a multi-hop chain (CPU->GPU->CPU or longer), each transfer
        applies Rule 1 or Rule 2 independently. Grad flow is only as
        wide as the narrowest `stop_grad=True` cut in the chain.

        Example: A(CPU) -> to_gpu() -> ops -> to_cpu(stop_grad=False)
                 -> ops -> loss.backward()
                 => A.grad is populated (both boundaries are transparent).

        Example: A(CPU) -> to_gpu(stop_grad=True) -> ops
                 -> to_cpu(stop_grad=False) -> loss.backward()
                 => B(GPU leaf).grad is populated, A.grad is untouched.

    **Rule 4 — When both transfers are `stop_grad=True`:**
        Each transfer creates a fresh leaf. Backward only reaches the
        last leaf in the chain. Earlier leaves, including the origin,
        are completely isolated and their grad buffers are never touched.
        Do not assert on grad buffers of isolated tensors.

        Example: A(CPU) -> to_gpu(stop_grad=True) -> ops
                 -> to_cpu(stop_grad=True) -> D(CPU leaf)
                 -> loss.backward()
                 => D.grad is populated, B.grad and A.grad are untouched.

    **Rule 5 — Intended use of `stop_grad=True` in training:**
        Transfer weights to GPU once at the start of training using
        `stop_grad=True`, making them native GPU leaves. Run the entire
        training loop on GPU. Transfer weights back to CPU after training
        using `to_cpu(stop_grad=True)` if persistence is needed.
        This avoids a cross-device grad transfer on every backward pass.
    """

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        device: Device,
        requires_grad: Optional[Bool] = None,
        stop_grad: Bool = False,
        sync: Bool = True,
    ) raises -> Tensor[Self.dtype]:
        var (code, ndb) = self.buffer.to_device(device, sync=sync)
        # Either CPU->CPU or GPU->GPU on same device — no transfer needed
        if code == -1:
            if sync:
                var out = self.copy()
                comptime if has_accelerator():
                    if out.is_on_gpu():
                        out.buffer.sync()
                return out
            return self.copy()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad and Self.dtype.is_numeric():
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                if not stop_grad:
                    var backwardFn: BackwardFn
                    if device.is_cpu():
                        # Forward was GPU->CPU
                        backwardFn = BackwardFn(
                            DeviceTransferBwdArg(
                                Flow.Gpu2Cpu,
                                self.buffer.device_state.value()
                                .get_gpu()
                                .into(),
                                self.shape(),
                            ),
                            DeviceTransferBackward[Self.dtype](),
                        )
                    else:
                        # Forward was CPU->GPU
                        backwardFn = BackwardFn(
                            DeviceTransferBwdArg(
                                Flow.Cpu2Gpu, CPU().into(), self.shape()
                            ),
                            DeviceTransferBackward[Self.dtype](),
                        )
                    backwardFn.needs_parent_data = False
                    out.add_ancestry(backwardFn^, self)

        return out^

    @always_inline
    @staticmethod
    def forward(
        self: Gradbox[Self.dtype],
        device: Device,
    ) raises -> Gradbox[Self.dtype]:
        var (code, ndb) = self.buffer().to_device(device, sync=False)
        # Either CPU->CPU or GPU->GPU on same device — no transfer needed
        if code == -1:
            return self
        return Gradbox[Self.dtype](ndb^)
