from tenmo import Tensor
from device_transfer import DeviceTransfer, DeviceTransferBackward
from sys import has_accelerator
from device import GPU
from multiplication import MultiplyScalar, MultiplyBackwardScalar
from mnemonics import AddTensor

fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(6, requires_grad=True)
    A.grad().print()

    @parameter
    if has_accelerator():
        var device = GPU().into()
        var A_gpu = DeviceTransfer[dtype].forward(A, device)
        print("No GPU now: ", A_gpu.is_on_gpu())
        print("A_gpu's Buffer's size: ", len(A_gpu.buffer.buffer))
        print(
            "A_gpu's DeviceState's DeviceBuffer's size: ",
            len(A_gpu.buffer.device_state.value().buffer),
        )
        print("A_gpu has gradbox: ", A_gpu.has_grad())
        print("A_gpu's gradbox is on gpu: ", A_gpu.gradbox[].is_on_gpu())
        A_gpu.gradbox[].print()
        var B_gpu = MultiplyScalar[dtype].forward(A_gpu, 91)
        print("printing A_gpu.gradbox[] after multiply forward: ")
        A_gpu.gradbox[].print()
        print("=============B_gpu============")
        B_gpu.print()
        B_gpu.gradbox[].print()
        B_gpu.seed_grad(1)
        print("printing A_gpu.gradbox[] after B_gpu seeding")
        A_gpu.gradbox[].print()

        var backward_handler = MultiplyBackwardScalar[dtype](91)
        var backward_result = backward_handler.backward(B_gpu)
        print("A_gpu.gradbox[] after backward")
        A_gpu.gradbox[].print()
        print("=============Calling multiply backward======")

        var receiver = backward_result[0][0]
        var gradbox = backward_result[0][1]
        receiver.print()
        gradbox.print()

        print(A_gpu.id(), receiver.id())

        receiver.update_grad[AddTensor](gradbox)
        print("===========Post update grad========")
        receiver.gradbox[].print()
        A_gpu.gradbox[].print()
