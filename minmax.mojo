from tenmo import Tensor
from operators import AddTensor
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from common_utils import panic
from validators import Validator
from utils.numerics import min_finite, max_finite
from layout.int_tuple import IntArray
from intlist import IntList
from gradbox import Gradbox
from ancestry import Ancestor

alias Gradbag[dtype: DType] = List[Tuple[IntArray, Scalar[dtype]]]


@fieldwise_init
struct MinMaxBackward[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    var axes: IntList
    var keepdims: Bool
    var gradbag: Gradbag[dtype]

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.keepdims = other.keepdims
        self.gradbag = other.gradbag.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.keepdims = other.keepdims
        self.gradbag = other.gradbag^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        # Retrieve upstream grad and saved tensors
        var gradbox = output.grad()
        var ancestor = output.ancestry().get(0)  # original input
        var mask = Gradbox[dtype].zeros(ancestor.shape(), share=False)
        for grad in self.gradbag:
            mask[grad[0]] = grad[1]
        var shape = ancestor.shape()
        var rank = shape.rank()

        # If input was scalar, just pass gradient through
        if rank == 0:
            return [(ancestor^, gradbox^, AddTensor)]

        if gradbox.shape() == Shape():
            # Scalar upstream grad → same scalar everywhere that was max
            # Build a tensor of that scalar, then mask it
            var filled = Gradbox[dtype].full(shape, gradbox.item(), share=False)
            # Apply mask: grad_contrib = filled * mask
            var grad_contrib = filled * mask
            return [(ancestor^, grad_contrib^, AddTensor)]
        else:
            # Build gradient broadcasted to input shape
            # then mask it so only argmax positions receive gradient.
            var gradbox_like_input: Gradbox[dtype]
            # Non-scalar upstream grad
            if not self.keepdims:
                gradbox_like_input = gradbox.unsqueeze(self.axes).broadcast_to(
                    shape, share=False
                )
            else:
                # keepdims=True: just broadcast to input shape
                gradbox_like_input = gradbox.broadcast_to(shape, share=False)

            # Apply mask
            var grad_contrib = mask * gradbox_like_input

            return [(ancestor^, grad_contrib^, AddTensor)]


@fieldwise_init
@register_passable
struct MinMax[dtype: DType = DType.float32]:
    @staticmethod
    fn forward[
        max: Bool, track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var out_shape = shape.compute_output_shape(normalized_axes, keepdims)
        var out = Tensor[dtype].zeros(out_shape)
        # Mask stores fractional responsibility: 1/count_of_maxes for positions that are maxima
        # Keep grad shares in gradbag which contains index, grad value(IntArray, Scalar)
        var gradbag: Gradbag[dtype] = Gradbag[dtype]()

        if out_shape == Shape():
            if rank == 0:
                # scalar input -> min/max is the value itself; mask = 1
                var v = self[IntArray()]
                out[IntArray()] = v
            elif rank == len(normalized_axes) and not keepdims:
                # reduce all dims -> scalar: find all positions equal to global max
                var first_iter = True
                var best_value = self[shape.first_index().intarray()]

                var best_positions = List[IntArray]()
                for idx in shape:
                    var cur = self[idx]
                    if first_iter:
                        best_value = cur
                        first_iter = False

                        @parameter
                        if track_grad:
                            best_positions.append(idx)
                    else:

                        @parameter
                        if max:
                            if cur > best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(idx)

                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(idx)
                                pass

                        else:
                            if cur < best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(idx)
                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(idx)
                                pass

                out[IntArray()] = best_value

                @parameter
                if track_grad:
                    # Split responsibility among ties
                    var count = len(best_positions)
                    if count > 0:
                        var inv = Scalar[dtype](1) / count
                        for p in best_positions:
                            gradbag.append((p, inv))
        else:
            # Partial reduction
            var reduced_shape = Shape(shape.axes_spans.select(normalized_axes))

            for out_idx in out_shape:
                # Track best value and all positions with that best (in the reduced block)
                @parameter
                if max:
                    best_value = min_finite[dtype]()
                else:
                    best_value = max_finite[dtype]()

                var best_positions = List[IntArray]()
                var first_iteration = True

                for red_idx in reduced_shape:
                    var full_idx = (
                        IntList(out_idx)
                        .replace(
                            normalized_axes, IntList(red_idx)
                        ) if keepdims else IntList(out_idx)
                        .insert(normalized_axes, IntList(red_idx))
                    ).intarray()
                    var cur = self[full_idx]

                    if first_iteration:
                        best_value = cur
                        first_iteration = False

                        @parameter
                        if track_grad:
                            best_positions.append(full_idx)
                    else:

                        @parameter
                        if max:
                            if cur > best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(full_idx)
                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(full_idx)
                                pass
                        else:
                            if cur < best_value:
                                best_value = cur

                                @parameter
                                if track_grad:
                                    best_positions.clear()
                                    best_positions.append(full_idx)
                            elif cur == best_value:

                                @parameter
                                if track_grad:
                                    best_positions.append(full_idx)
                                pass

                # write result to output
                out[out_idx] = best_value

                @parameter
                if track_grad:
                    # split responsibility among ties in this reduced block
                    var count = len(best_positions)
                    if count > 0:
                        var inv = Scalar[dtype](1) / count
                        for p in best_positions:
                            gradbag.append((p, inv))

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backward_fn = MinMaxBackward[dtype](
                    normalized_axes, keepdims, gradbag^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
