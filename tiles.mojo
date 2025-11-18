from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from intlist import IntList
from operators import AddTensor
from shapes import Shape
from validators import Validator
from gradbox import Gradbox
from ancestry import Ancestor
from indexhelper import IndexCalculator


@fieldwise_init
@register_passable
struct TileBackward[dtype: DType](ImplicitlyCopyable):
    var repeat: IntList
    var orig_shape: Shape

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward_orig(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var grad_out = output.grad().copy()
        var parent = output.ancestry().get(0)
        var parent_shape = self.orig_shape
        var parent_rank = len(parent_shape)
        var repeat_rank = len(self.repeat)
        # Handle scalar case
        if parent_rank == 0:
            var total_grad = grad_out.sum().item()
            var gradbox_parent = Gradbox[dtype].full(
                Shape(), total_grad, share=False
            )
            return [(parent^, gradbox_parent^, AddTensor)]

        # --- Handle dimension alignment ---
        var effective_rank = max(parent_rank, repeat_rank)

        # --- 1. Expand each dim into (orig_dim, repeat_factor) ---
        var reshaped_dims = IntList.with_capacity(effective_rank * 2)
        var reduce_axes = IntList.with_capacity(effective_rank)
        for i in range(effective_rank):
            var parent_index = parent_rank - effective_rank + i
            var repeat_index = repeat_rank - effective_rank + i

            var orig_dim = 1 if parent_index < 0 else parent_shape[parent_index]
            var repeat_factor = (
                1 if repeat_index < 0 else self.repeat[repeat_index]
            )

            reshaped_dims.append(orig_dim)
            reshaped_dims.append(repeat_factor)

            reduce_axes.append(
                i * 2 + 1
            )  # The repeat axis is the second one in each pair

        var reshaped_shape = Shape(reshaped_dims)

        # --- CRITICAL: Sort reduce_axes in DESCENDING order ---
        # reduce_axes.sort(asc=False) - NEVER SORT!!!!

        # --- 2. Reshape grad_out to that pattern ---
        var reshaped = grad_out.reshape(reshaped_shape)

        # --- 3. Sum over every "repeat" axis ---
        var gradbox_parent = reshaped.sum(reduce_axes, keepdims=False)
        if gradbox_parent.shape() != parent_shape:
            gradbox_parent = gradbox_parent.reshape(parent_shape)

        return [(parent^, gradbox_parent^, AddTensor)]

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref grad_out = output.gradients()[]
        var parent = output.ancestry().get(0)
        var parent_shape = self.orig_shape
        var parent_rank = len(parent_shape)
        var repeat_rank = len(self.repeat)

        # Handle scalar case
        if parent_rank == 0:
            var total_grad = grad_out.sum().item()
            var gradbox_parent = Gradbox[dtype].full(
                Shape(), total_grad, share=False
            )
            return [(parent^, gradbox_parent^, AddTensor)]

        # --- Handle dimension alignment ---
        var effective_rank = max(parent_rank, repeat_rank)

        # --- 1. Expand each dim into (orig_dim, repeat_factor) ---
        var reshaped_dims = IntList.with_capacity(effective_rank * 2)
        var reduce_axes = IntList.with_capacity(effective_rank)

        for i in range(effective_rank):
            var parent_index = parent_rank - effective_rank + i
            var repeat_index = repeat_rank - effective_rank + i

            var orig_dim = 1 if parent_index < 0 else parent_shape[parent_index]
            var repeat_factor = (
                1 if repeat_index < 0 else self.repeat[repeat_index]
            )

            # CRITICAL FIX: Put repeat_factor FIRST, then orig_dim
            reshaped_dims.append(repeat_factor)  # Repeat dimension first
            reshaped_dims.append(orig_dim)  # Original dimension second

            reduce_axes.append(
                i * 2
            )  # Reduce along the REPEAT axis (first in pair)

        var reshaped_shape = Shape(reshaped_dims)

        # --- 2. Reshape grad_out to that pattern ---
        var reshaped = grad_out.reshape(reshaped_shape)

        # --- 3. Sum over every "repeat" axis ---
        var gradbox_parent = reshaped.sum(reduce_axes, keepdims=False)
        if gradbox_parent.shape() != parent_shape:
            gradbox_parent = gradbox_parent.reshape(parent_shape)

        return [(parent^, gradbox_parent^, AddTensor)]


@fieldwise_init
@register_passable
struct Tile[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        repeat: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        """
        Tile — unified superset of torch.repeat and torch.tile.

        This operation repeats the tensor along each axis according to the given repeat factors.
        It combines the semantics of both `torch.repeat` and `torch.tile` into a single, safe,
        generalized formulation.

        ### Key Features
            - ✅ Works for any combination of tensor rank and repeat rank.
            - ✅ Never raises shape-rank mismatch errors (unlike `torch.repeat`).
            - ✅ Automatically right-aligns repeat factors (like `torch.tile`).
            - ✅ Automatically prepends implicit leading 1-dims when needed (like `torch.repeat`).
            - ✅ Always produces intuitive "broadcast-like" repetition behavior.

        ### Shape Inference
        Let:
            - `orig_shape`: shape of the input tensor (rank = R)
            - `repeat`: list of repeat factors (rank = K)
            - `effective_rank = max(R, K)`

        Then, for each axis `i` in `0..effective_rank-1`:

        orig_dim = orig_shape[i] if i < R else 1
        repeat_factor = repeat[i] if i < K else 1
        new_shape[i] = orig_dim * repeat_factor


        ### Examples

        #### Case 1 — Equal ranks
        x = [[1, 2],
             [3, 4]]           # shape (2, 2)
        x.tile(2, 3)         # → shape (4, 6)


        Case 2 — Fewer repeat dims than tensor rank

        x = [[1, 2],
             [3, 4]]           # shape (2, 2)
        x.tile(3)            # → shape (2, 6)
        # Equivalent to torch.tile(x, (3,))


        Case 3 — More repeat dims than tensor rank

        x = [[1, 2],
             [3, 4]]           # shape (2, 2)
        x.tile(2, 3, 4)      # → shape (2, 6, 8)
        # Equivalent to torch.tile(x, (2, 3, 4))


        """

        var orig_shape = self.shape()
        var orig_rank = self.rank()
        var repeat_rank = len(repeat)

        # Handle dimension alignment (PyTorch-style: align trailing dims)
        var effective_rank = max(orig_rank, repeat_rank)
        var new_shape = IntList.with_capacity(effective_rank)

        for i in range(effective_rank):
            orig_index = orig_rank - effective_rank + i
            repeat_index = repeat_rank - effective_rank + i

            orig_dim = 1 if orig_index < 0 else orig_shape[orig_index]
            repeat_factor = 1 if repeat_index < 0 else repeat[repeat_index]
            new_shape.append(orig_dim * repeat_factor)

        var out_shape = Shape(new_shape)
        var out = Tensor[dtype](out_shape, requires_grad=False)
        out_numels = out.numels()

        # --- Forward computation ---
        for flat_index in range(out_numels):
            var out_coord = IndexCalculator.index_to_coord(
                out_shape, flat_index
            )
            var src_coord = IntList.with_capacity(orig_rank)

            # Align coordinates to the trailing dimensions of the source
            coord_offset = effective_rank - orig_rank
            for d in range(orig_rank):
                src_dim = orig_shape[d]
                out_dim_index = d + coord_offset
                mapped_val = out_coord[out_dim_index] % src_dim
                src_coord.append(mapped_val)

            # Now src_coord has same rank as source tensor
            out[out_coord] = self[src_coord]

        # --- Gradient setup ---
        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                backward_fn = TileBackward[dtype](
                    repeat.copy(), orig_shape.copy()
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
