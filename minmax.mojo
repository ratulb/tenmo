from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from intlist import IntList
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from common_utils import panic
from validators import Validator
from utils.numerics import min_finite, max_finite

alias Gradbag[dtype: DType] = List[(IntList, Scalar[dtype])]


@fieldwise_init
struct MinMaxBackward[dtype: DType = DType.float32](Copyable & Movable):
    var axes: IntList
    var keepdims: Bool
    var gradbox: Gradbag[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        # Retrieve upstream grad and saved tensors
        var gradients = output.grad()
        var ancestor = output.ancestry().get(0)  # original input
        var mask = Tensor[dtype].zeros(ancestor.shape())
        for grad in self.gradbox:
            mask[grad[0]] = rebind[Scalar[dtype]](grad[1])
        var shape = ancestor.shape()
        var rank = shape.rank()

        # If input was scalar, just pass gradient through
        if rank == 0:
            return [(ancestor, gradients, AddTensor)]

        if gradients.shape == Shape():
            # Scalar upstream grad → same scalar everywhere that was max
            # Build a tensor of that scalar, then mask it
            var filled = Tensor[dtype].full(
                shape, gradients.item(), requires_grad=False
            )
            # Apply mask: grad_contrib = filled * mask
            var grad_contrib = filled * mask
            return [(ancestor, grad_contrib, AddTensor)]
        else:
            # Build gradient broadcasted to input shape
            # then mask it so only argmax positions receive gradient.
            var grad_like_input: Tensor[dtype]
            # Non-scalar upstream grad
            if not self.keepdims:
                grad_like_input = gradients.unsqueeze(
                    self.axes, False
                ).broadcast_to(shape, requires_grad=False)
            else:
                # keepdims=True: just broadcast to input shape
                grad_like_input = gradients.broadcast_to(
                    shape, requires_grad=False
                )

            # Apply mask
            var grad_contrib = mask * grad_like_input

            return [(ancestor, grad_contrib, AddTensor)]


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
        var shape = self.shape
        var rank = shape.rank()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var out_shape = shape.compute_output_shape(normalized_axes, keepdims)
        var out = Tensor[dtype].zeros(out_shape)
        # Mask stores fractional responsibility: 1/count_of_maxes for positions that are maxima
        # Keep grad shares in gradbox which contains index, grad value(IntList, Scalar)
        var gradbox: Gradbag[dtype] = Gradbag[dtype]()

        if out_shape == Shape():
            if rank == 0:
                # scalar input -> min/max is the value itself; mask = 1
                var v = self[IntList()]
                out[IntList()] = v
            elif rank == len(normalized_axes) and not keepdims:
                # reduce all dims -> scalar: find all positions equal to global max
                var first_iter = True
                var best_value = self[shape.first_index()]

                var best_positions = List[IntList]()
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

                out[IntList()] = best_value

                @parameter
                if track_grad:
                    # Split responsibility among ties
                    var count = len(best_positions)
                    if count > 0:
                        var inv = Scalar[dtype](1) / count
                        for p in best_positions:
                            gradbox.append((p, inv))
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

                var best_positions = List[IntList]()
                var first_iteration = True

                for red_idx in reduced_shape:
                    var full_idx = out_idx.replace(
                        normalized_axes, red_idx
                    ) if keepdims else out_idx.insert(normalized_axes, red_idx)
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
                            gradbox.append((p, inv))

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                var backward_fn = MinMaxBackward[dtype](
                    normalized_axes, keepdims, gradbox
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(self)

        return out


fn main() raises:
    pass
