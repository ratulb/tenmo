from tenmo import Tensor
from mnemonics import AddTensor, DROPOUT
from backpropagation import BackwardFnArg, BufferArg, BACKWARD_DROPOUT
from gradbox import Gradbox
from common_utils import panic
from ndbuffer import NDBuffer
from buffers import Buffer
from std.sys import simd_width_of
from net import Module, Layer
from std.random import random_float64, seed
from ancestors_newest import AncestorRef

@fieldwise_init
struct DropoutBackward[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var mask_buffer = (
            output.backward_fn_arg().get[BufferArg[Self.dtype]]().buffer
        )
        var grad_output = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # Gradient flows through the same mask that was used in forward
        # grad_input = grad_output * mask
        # Since mask already has scale baked in, we just multiply
        var grad_input = grad_output * Tensor[Self.dtype](
            NDBuffer[Self.dtype](mask_buffer, grad_output.shape()),
            requires_grad=False,
        )

        return [(ancestor^, grad_input^, AddTensor)]

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]]:
        var mask_buffer = (
            output.backward_fn_arg().get[BufferArg[Self.dtype]]().buffer
        )
        var grad_output = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # Gradient flows through the same mask that was used in forward
        # grad_input = grad_output * mask
        # Since mask already has scale baked in, we just multiply
        var grad_input = grad_output * Tensor[Self.dtype](
            NDBuffer[Self.dtype](mask_buffer, grad_output.shape()),
            requires_grad=False,
        )

        return [(ancestor^, grad_input^, AddTensor)]


@fieldwise_init
struct Dropout[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """
    Optimized Dropout layer.

    1. Direct buffer manipulation (no intermediate tensors)
    2. SIMD vectorization
    3. Fused mask generation and scaling
    4. Fast random number generation
    5. Zero overhead in eval mode
    """

    var training: Bool
    var p: Scalar[Self.dtype]
    var scale: Scalar[Self.dtype]
    var seed: Int  # For reproducible randomness

    comptime TAG = DROPOUT

    fn __init__(out self, p: Scalar[Self.dtype] = Scalar[Self.dtype](0.5)):
        """Initialize Dropout layer."""
        if p < 0.0 or p >= 1.0:
            panic("Dropout probability must be in [0, 1)")

        self.training = True
        self.p = p
        self.scale = Scalar[Self.dtype](1.0) / (Scalar[Self.dtype](1.0) - p)
        self.seed = 42  # Default seed

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training
        self.p = copy.p
        self.scale = copy.scale
        self.seed = copy.seed

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if not self.training or self.p == 0.0:
            return x

        if self.p == 1.0:
            return Tensor[Self.dtype].zeros(x.shape())

        # Training mode: Apply dropout with mask saved for backward
        var output = Tensor[Self.dtype].zeros(x.shape())
        # Store mask
        var mask = Tensor[Self.dtype].zeros(x.shape())

        var x_ptr = x.data_ptr()
        var out_ptr = output.data_ptr()
        var mask_ptr = mask.data_ptr()

        var total_elements = x.numels()
        comptime simd_w = simd_width_of[Self.dtype]()

        var threshold_vec = SIMD[Self.dtype, simd_w](self.p)
        var scale_vec = SIMD[Self.dtype, simd_w](self.scale)
        var zero_vec = SIMD[Self.dtype, simd_w](0)

        # SIMD vectorized dropout
        var i = 0
        var vec_end = (total_elements // simd_w) * simd_w

        for _ in range(vec_end // simd_w):
            var x_vec = x_ptr.load[width=simd_w](i)

            # Generate random values [0, 1)
            var rand_vec = SIMD[Self.dtype, simd_w](0)

            comptime for v in range(simd_w):
                rand_vec[v] = random_float64(0.0, 1.0).cast[Self.dtype]()

            # Create mask: scale if rand > p, else 0
            var mask_vec = (rand_vec.gt(threshold_vec)).select(
                scale_vec, zero_vec
            )

            # Apply mask
            var result_vec = x_vec * mask_vec

            # Store both result and mask
            out_ptr.store[width=simd_w](i, result_vec)
            mask_ptr.store[width=simd_w](i, mask_vec)

            i += simd_w

        # Scalar tail
        for j in range(vec_end, total_elements):
            var x_val = x_ptr[j]
            var rand_val = random_float64(0.0, 1.0).cast[Self.dtype]()

            if rand_val > self.p:
                out_ptr[j] = x_val * self.scale
                mask_ptr[j] = self.scale  # Store mask value
            else:
                out_ptr[j] = 0.0
                mask_ptr[j] = 0.0  # Store mask value

        if x.requires_grad:
            output.requires_grad_(True)

            # Attach backward handler with the mask
            var backwardFnArg = BackwardFnArg[Self.dtype].from_buffer(
                BACKWARD_DROPOUT,
                mask.buffer.buffer,  # Store the mask buffer
            )

            output.add_ancestry(backwardFnArg^, x)

        return output^

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn set_seed(mut self, seed_val: Int):
        """Set random seed for reproducibility."""
        self.seed = seed_val
        seed(seed_val)

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), Self.TAG)


fn main() raises:
    print("pass")
