from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_DEVICE_TRANSFER,
)
from mnemonics import AddTensor
from common_utils import panic
from gradbox import Gradbox
from device import Device, GPU
from common_utils import panic
from sys import has_accelerator

@register_passable
struct Flow(Equatable, ImplicitlyCopyable):
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

    fn __copyinit__(out self, existing: Self):
        self.direction = existing.direction

    fn __eq__(self, other: Self) -> Bool:
        return self.direction == other.direction

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

struct DeviceTransferBackward[dtype: DType](ImplicitlyCopyable):
    var flow: Flow
    var gpu: Optional[GPU]

    fn __init__(out self):
        self.flow = Flow.UnMoved
        self.gpu = None

    fn __init__(out self, flow: Flow):
        self.flow = flow
        self.gpu = None

    fn __init__(out self, flow: Flow, gpu: GPU):
        self.flow = flow
        self.gpu = gpu

    comptime TAG = BACKWARD_DEVICE_TRANSFER

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        @parameter
        if has_accelerator():
            if output.is_on_gpu():
                return self.backward_gpu(output)
            else:
                return self.backward_cpu(output)
        else:
            return self.backward_cpu(output)

    fn backward_cpu(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        debug_assert(
            ancestor.shape() != gradbox.shape(),
            "DeviceTransferBackward: gradbox shape and ancestor shape mismatch",
        )

        return [(ancestor^, gradbox, AddTensor)]

    fn backward_gpu(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        debug_assert(
            ancestor.shape() != gradbox.shape(),
            "DeviceTransferBackward: gradbox shape and ancestor shape mismatch",
        )
        var parent_gradbox: Gradbox[Self.dtype]
        if self.flow == Flow.UnMoved:
            # Just pass through incoming grad
            parent_gradbox = gradbox^
        elif self.flow == Flow.Cpu2Gpu:
            try:
                parent_gradbox = Gradbox[Self.dtype](gradbox^.buffer^.to_cpu())
            except e:
                print(e)
                panic(
                    "DeviceTransferBackward -> backward: error transferring"
                    " gradbox from GPU to CPU"
                )
                parent_gradbox = gradbox
                # Not reachable - make the compiler happy
        else:
            try:
                parent_gradbox = Gradbox[Self.dtype](
                    gradbox^.buffer^.to_gpu(self.gpu.value())
                )
            except e:
                print(e)
                panic(
                    "DeviceTransferBackward -> backward: error transferring"
                    " gradbox from CPU to GPU"
                )
                parent_gradbox = gradbox
                # Not reachable - make the compiler happy

        return [(ancestor^, parent_gradbox, AddTensor)]


@fieldwise_init
@register_passable
struct DeviceTransfer[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        device: Device,
        requires_grad: Optional[Bool] = None,
    ) raises -> Tensor[Self.dtype]:
        var (code, ndb) = self.buffer.to_device(device)
        # Either CPU -> CPU or GPU -> GPU - but same devices
        if code == -1:
            return self
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            print("DeviceTransfer -> grad_required: ", grad_required, "self.requires_grad: ", self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn: DeviceTransferBackward[Self.dtype]
                if device.is_cpu():
                    backward_fn = DeviceTransferBackward[Self.dtype](
                        Flow.Gpu2Cpu, self.buffer.device_state.value().get_gpu()
                    )
                else:
                    backward_fn = DeviceTransferBackward[Self.dtype](
                        Flow.Cpu2Gpu
                    )
                out.backwardFn = Optional(backward_fn^.into_backward_fn())
                out.add_ancestry(self)

        return out^


from common_utils import now
from testing import assert_true
from shapes import Shape

from gpu.host import DeviceContext
from device import GPU
from random import random_si64
from common_utils import now


fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(10, requires_grad=True)
    var A_g = A.to_gpu()
    print("A_g.requires_grad:", A_g.requires_grad)
    print("A_g is on gpu:", A_g.is_on_gpu())
    var B = A_g * 42
    B.backward()
    B.grad().print()
    A_g.grad().print()
    A.grad().print()
