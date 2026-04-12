from tenmo import Tensor
from backpropagation import TileArg
from intarray import IntArray
from mnemonics import AddTensor
from shapes import Shape
from validators import Validator
from gradbox import Gradbox
from indexhelper import IndexCalculator


@fieldwise_init
struct TileBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.fn_arg().arg[TileArg]
        ref grad_out = output.gradients()[]
        var parent = output.ancestry().get(0)
        var parent_shape = bwd_arg.orig_shape
        var parent_rank = len(parent_shape)
        var repeat_rank = len(bwd_arg.repeat)

        # Handle scalar case
        if parent_rank == 0:
            var total_grad = grad_out.sum().item()
            var gradbox_parent = Gradbox[Self.dtype].full(
                Shape(), total_grad, device=grad_out.device(), share=False
            )
            return [(parent^, gradbox_parent^, AddTensor)]

        var effective_rank = max(parent_rank, repeat_rank)

        # Expand each dim into (repeat_factor, orig_dim) pairs
        var reshaped_dims = IntArray.with_capacity(effective_rank * 2)
        var reduce_axes = IntArray.with_capacity(effective_rank)

        for i in range(effective_rank):
            var parent_index = parent_rank - effective_rank + i
            var repeat_index = repeat_rank - effective_rank + i

            var orig_dim = 1 if parent_index < 0 else parent_shape[parent_index]
            var repeat_factor = (
                1 if repeat_index < 0 else bwd_arg.repeat[repeat_index]
            )

            reshaped_dims.append(repeat_factor)  # repeat dim first
            reshaped_dims.append(orig_dim)  # original dim second
            reduce_axes.append(i * 2)  # reduce along repeat axis

        var reshaped_shape = Shape(reshaped_dims)

        # Reshape grad_out → sum over repeat axes → reshape to parent shape
        var reshaped = grad_out.reshape(reshaped_shape)
        var gradbox_parent = reshaped.sum(reduce_axes, keepdims=False)
        if gradbox_parent.shape() != parent_shape:
            gradbox_parent = gradbox_parent.reshape(parent_shape)

        return [(parent^, gradbox_parent^, AddTensor)]


@fieldwise_init
struct Tile[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        repeat: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Tile — GPU-safe implementation using reshape + expand + reshape.

        Replaces element-by-element CPU loop with GPU-safe ops:
            1. Reshape input to interleaved shape: (M,N) → (1,M,1,N)
            2. Expand to repeat factors:           (1,M,1,N) → (r0,M,r1,N)
               Uses stride=0 trick — no data copy, GPU safe.
            3. Reshape to final shape:             (r0,M,r1,N) → (r0*M, r1*N)
               reshape_gpu materialises on GPU via fill.

        GPU safe: reshape → GPU safe via reshape_gpu
                  expand  → stride=0 view, GPU safe
                  reshape → GPU safe via reshape_gpu

        """
        var orig_shape = self.shape()
        var orig_rank = self.rank()
        var repeat_rank = len(repeat)
        var effective_rank = max(orig_rank, repeat_rank)

        # Build shapes for the three-step decomposition
        var interleaved_dims = IntArray.with_capacity(effective_rank * 2)
        var expand_dims = IntArray.with_capacity(effective_rank * 2)
        var final_dims = IntArray.with_capacity(effective_rank)

        for i in range(effective_rank):
            var orig_index = orig_rank - effective_rank + i
            var repeat_index = repeat_rank - effective_rank + i
            var orig_dim = 1 if orig_index < 0 else orig_shape[orig_index]
            var repeat_factor = 1 if repeat_index < 0 else repeat[repeat_index]

            interleaved_dims.append(1)  # repeat placeholder
            interleaved_dims.append(orig_dim)  # original dim
            expand_dims.append(repeat_factor)  # expand to repeat
            expand_dims.append(orig_dim)  # keep original
            final_dims.append(orig_dim * repeat_factor)  # collapsed

        var interleaved_shape = Shape(interleaved_dims)
        var expand_shape = Shape(expand_dims)
        var out_shape = Shape(final_dims)

        # Step 1: Reshape to interleaved — GPU safe via reshape_gpu
        var reshaped = self.reshape[track_grad=False](interleaved_shape)

        # Step 2: Expand using stride=0 — pure metadata, GPU safe
        var expanded = reshaped.expand[track_grad=False](expand_shape)

        # Step 3: Reshape to final shape — materialises on GPU via reshape_gpu
        var out = expanded.reshape[track_grad=False](out_shape)
        out.requires_grad_(False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var bwd_arg = TileArg(
                    repeat, orig_shape
                ).into_arg[Self.dtype]()
                out.fnArg = Optional(bwd_arg^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
