from tenmo import Tensor
from backpropagation import (
    DeviceTransferArgs,
)
from mnemonics import AddTensor
from common_utils import panic
from gradbox import Gradbox
from device import Device
from std.sys import has_accelerator


struct Flow(RegisterPassable & Equatable, ImplicitlyCopyable):
    var direction: Int
    comptime Cpu2Gpu = Flow(0)
    comptime Gpu2Cpu = Flow(1)
    comptime UnMoved = Flow(-1)

    fn __init__(out self, direction: Int = 0):
        self.direction = direction
        if direction < -1 or direction > 1:
            panic(
                "Invalid direction type. Must be '0 → Cpu2Gpu', '1 → Gpu2Cpu',"
                " or '-1 → UnMoved'"
            )

    fn __copyinit__(out self, copy: Self):
        self.direction = copy.direction

    fn __eq__(self, other: Self) -> Bool:
        return self.direction == other.direction

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


struct DeviceTransferBackward[dtype: DType](ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.fn_arg().arg[DeviceTransferArgs]
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        debug_assert(
            ancestor.shape() == gradbox.shape(),
            "DeviceTransferBackward: gradbox shape and ancestor shape mismatch",
        )

        if bwd_arg.flow == Flow.UnMoved:
            return [(ancestor^, gradbox, AddTensor)]

        comptime if has_accelerator():
            if bwd_arg.flow == Flow.Cpu2Gpu:
                # Forward was CPU→GPU, backward transfers grad GPU→CPU
                try:
                    return [
                        (
                            ancestor^,
                            Gradbox[Self.dtype](gradbox.buffer.to_cpu()),
                            AddTensor,
                        )
                    ]
                except e:
                    panic(
                        "DeviceTransferBackward: GPU→CPU transfer failed: "
                        + String(e)
                    )
                    return [(ancestor^, gradbox, AddTensor)]  # unreachable
            else:
                # Forward was GPU→CPU, backward transfers grad CPU→GPU
                try:
                    return [
                        (
                            ancestor^,
                            Gradbox[Self.dtype](
                                gradbox.buffer.to_gpu(bwd_arg.gpu.value()),
                                share=False,
                            ),
                            AddTensor,
                        )
                    ]
                except e:
                    panic(
                        "DeviceTransferBackward: CPU→GPU transfer failed: "
                        + String(e)
                    )
                    return [(ancestor^, gradbox, AddTensor)]  # unreachable

        return [(ancestor^, gradbox, AddTensor)]


@fieldwise_init
struct DeviceTransfer[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        device: Device,
        requires_grad: Optional[Bool] = None,
    ) raises -> Tensor[Self.dtype]:
        var (code, ndb) = self.buffer.to_device(device)
        # Either CPU→CPU or GPU→GPU on same device — no transfer needed
        if code == -1:
            return self
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var bwd_arg: DeviceTransferArgs
                if device.is_cpu():
                    # Forward was GPU→CPU
                    bwd_arg = DeviceTransferArgs(
                        Flow.Gpu2Cpu,
                        self.buffer.device_state.value().get_gpu(),
                    )
                else:
                    # Forward was CPU→GPU
                    bwd_arg = DeviceTransferArgs(
                        Flow.Cpu2Gpu
                    )
                out.fnArg = Optional(bwd_arg^.into_arg[Self.dtype]())
                out.add_ancestry(self)

        return out^

    @always_inline
    @staticmethod
    fn forward(
        self: Gradbox[Self.dtype],
        device: Device,
    ) raises -> Gradbox[Self.dtype]:
        var (code, ndb) = self.buffer.to_device(device)
        # Either CPU→CPU or GPU→GPU on same device — no transfer needed
        if code == -1:
            return self
        return Gradbox[Self.dtype](ndb^, share=False)


fn main() raises:
    pass
