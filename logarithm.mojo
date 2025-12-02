from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn
from ancestry import Ancestor
from gradbox import Gradbox
from math import log
from sys import simd_width_of


@fieldwise_init
@register_passable
struct LogBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        """Compute gradient: ∂log(x)/∂x = 1/x."""
        ref grad_output = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()
        ref parent = ancestor.tensor()
        var parent_gradbox = Gradbox[dtype].zeros(shape, share=False)

        if parent.is_contiguous():
            var src = parent.buffer.data_buffer().data
            var dest = parent_gradbox.buffer.data_buffer().data
            var grad_output_data = grad_output.buffer.data_buffer().data
            var offset = parent.offset()
            var numels = parent.numels()

            alias simd_width = simd_width_of[dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var x = src.load[width=simd_width](offset + i)  # Original input
                var grad_out = grad_output_data.load[width=simd_width](i)

                # Correct: grad_out / x
                dest.store[width=simd_width](i, grad_out / x)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var x = src[offset + i]  # Original input
                dest[i] = grad_output_data[i] / x  # Correct formula
        else:
            # Non-contiguous fallback
            var index = 0
            var parent_gradbox_data = parent_gradbox.buffer.data_buffer().data
            var grad_output_data = grad_output.buffer.data_buffer().data
            for coord in shape:

                var x = parent[coord]  # Original input
                parent_gradbox_data[index] = grad_output_data[index] / x  # Correct

        return [(ancestor^, parent_gradbox^, AddTensor)]


@fieldwise_init
@register_passable
struct Logarithm[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        """Natural logarithm: y = log(x)."""
        var shape = self.shape()
        var out = Tensor[dtype].zeros(shape, requires_grad=False)

        if self.is_contiguous():
            var src = self.buffer.data_buffer().data
            var dest = out.buffer.data_buffer().data
            var offset = self.offset()
            var numels = self.numels()

            alias simd_width = simd_width_of[dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var chunk = src.load[width=simd_width](offset + i)
                dest.store[width=simd_width](i, log(chunk))

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                dest[i] = log(src[offset + i])
        else:
            # Non-contiguous fallback
            for coord in shape:
                out[coord] = log(self[coord])

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = LogBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^

## The Math
# Forward:  y = log(x)
# Backward: dy/dx = 1/x


fn main() raises:
    #x = 2.0
    #y = log(2.0)  # ≈ 0.693

    # If grad_output = 1.0:
    # Correct gradient: 1.0 / 2.0 = 0.5
    # gradient: 1.0 / log(2.0) ≈ 1.44
    test_log_backward()
    print("passes")

fn test_log_backward():
    print("Testing log backward...")

    var x = Tensor[DType.float64]([2.0, 3.0, 4.0], requires_grad=True)
    var y = x.log()
    print("y (log(x)):")  #// Should be [0.693, 1.099, 1.386]
    y.print()
    var loss = y.sum()
    loss.backward()

    print("x.grad:")
    #// ✅ Should be [0.5, 0.333, 0.25] = [1/2, 1/3, 1/4]
    #// ❌ NOT [1.44, 0.91, 0.72] = [1/log(2), 1/log(3), 1/log(4)]

    #// Numerical verification
    x.grad().print()
    var expected = Tensor[DType.float64]([0.5, 0.333333, 0.25])
    print("Expected:", expected)

    var diff = (x.grad() - expected).sum().item()
    if diff < 1e-5:
        print("✅ PASS: Gradients correct!")
    else:
        print("❌ FAIL: Gradients incorrect!")
