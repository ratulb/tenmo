from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_DEVICE_TRANSFER,
)
from mnemonics import AddTensor
from common_utils import panic
from gradbox import Gradbox
from device import Device


@fieldwise_init
@register_passable
struct DeviceTransferBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_DEVICE_TRANSFER

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        debug_assert(
            ancestor.shape() != gradbox.shape(),
            "DeviceTransferBackward: gradbox shape and ancestor shape mismatch",
        )
        # Just pass through incoming grad
        return [(ancestor^, gradbox, AddTensor)]


@fieldwise_init
@register_passable
struct DeviceTransfer[dtype: DType](Copyable):
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
        var out = Tensor[Self.dtype](
            ndb^, requires_grad=requires_grad.or_else(False)
        )

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                backward_fn = DeviceTransferBackward[
                    Self.dtype
                ]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
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
    _ = """a = Tensor[dtype].arange(5000000)
    b = Tensor[dtype].arange(5000000)
    start = now()
    r1 = a - b
    print("CPU took: ", (now() - start) * 1000, "ms")
    start = now()
    ag = a.to_gpu()
    bg = b.to_gpu()
    print("to_gpu took: ", (now() - start) * 1000, "ms")
    start = now()
    r2 = ag - bg
    print("Overall GPU took: ", (now() - start) * 1000, "ms")
    assert_true(r1.all_close(r2))
    print()"""
    print("The meaty part")

    _ = """A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    print("A's id: ", A.id())
    a = A.to_gpu(requires_grad=True)

    expected = Tensor[dtype].full(Shape.of(3, 3), 2) + 42
    b = a + 42
    assert_true(b.all_close(expected), "Scalar add assertion failed")
    b.ancestry().print()
    b.backward()
    A.grad().print()"""
    var times = 1000
    var start = now()
    var s = random_si64(10, 20)
    for _ in range(times):
        ctx = DeviceContext()
        sn = random_si64(10, 20)
        same = s == sn
        print(same)

    end1 = now() - start

    start = now()

    ctx = DeviceContext()
    for _ in range(times):
        sn = random_si64(10, 20)
        same = s == sn
        ctx1 = ctx
        print(same, ctx1.id())

    end2 = now() - start

    start = now()
    device = GPU()
    for _ in range(times):
        sn = random_si64(10, 20)
        same = s == sn
        device1 = device
        print(same, device1.id)

    end3 = now() - start

    print("Time taken for ctx creation: ", end1 * 1000, "ms")
    print("Time taken for ctx copy: ", end2 * 1000, "ms")
    print("Time taken for Device copy: ", end3 * 1000, "ms")
