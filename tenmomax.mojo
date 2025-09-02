from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from intlist import IntList
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from common_utils import panic, compute_output_shape
from validators import Validator
from utils.numerics import min_finite


@fieldwise_init
@register_passable
struct MaxBackward[dtype: DType](Copyable):
    var axes: IntList
    var keepdims: Bool

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        # Retrieve upstream grad and saved tensors
        var gradients = output.gradients()[]
        var ancestor = output.ancestry().get(0)[]  # original input
        var ancestor_mask = output.ancestry().get(1)[]  # saved argmax mask
        var mask = ancestor_mask.tensor()

        var shape = ancestor.shape()
        var rank = shape.rank()

        # If input was scalar, just pass gradient through
        if rank == 0:
            return [(ancestor, gradients, AddTensor)]

        if gradients.shape == Shape.Void:
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
                ).broadcast_to(shape)
            else:
                # keepdims=True: just broadcast to input shape
                grad_like_input = gradients.broadcast_to(shape)

            # Apply mask
            var grad_contrib = mask * grad_like_input

            return [(ancestor, grad_contrib, AddTensor)]


@fieldwise_init
@register_passable
struct MaxForward[dtype: DType]:
    @staticmethod
    fn max(
        self: Tensor[dtype],
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var shape = self.shape
        var rank = shape.rank()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var out_shape = compute_output_shape(shape, normalized_axes, keepdims)
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        var out = Tensor[dtype].zeros(out_shape)

        # Mask stores fractional responsibility: 1/count_of_maxes for positions that are maxima
        # Mask has same shape as input and requires_grad=False
        var mask = Tensor[dtype].zeros(shape, requires_grad=False)

        if out_shape == Shape.Void:
            if rank == 0:
                # scalar input -> max is the value itself; mask = 1
                var v = self[IntList.Empty]
                out[IntList.Empty] = v
            elif rank == len(normalized_axes) and not keepdims:
                # reduce all dims -> scalar: find all positions equal to global max
                var inited = False
                var max_val = self[shape.first_index()]
                var best_positions = List[IntList]()
                for idx in shape:
                    var cur = self[idx]
                    if not inited:
                        max_val = cur
                        best_positions = List[IntList]()
                        best_positions.append(idx)
                        inited = True
                    else:
                        if cur > max_val:
                            max_val = cur
                            best_positions.clear()
                            best_positions.append(idx)
                        elif cur == max_val:
                            best_positions.append(idx)
                out[IntList.Empty] = max_val

                # Split responsibility among ties
                var count = len(best_positions)
                if count > 0:
                    var inv = Scalar[dtype](1) / count
                    for p in best_positions:
                        mask[p] = inv
        else:
            # Partial reduction
            var reduced_shape = Shape(shape.axes_spans.select(normalized_axes))

            for out_idx in out_shape:
                # Track best value and all positions with that best (in the reduced block)
                var inited = False
                var max_val = min_finite[dtype]()
                var best_positions = List[IntList]()

                for red_idx in reduced_shape:
                    var full_idx = out_idx.replace(
                        normalized_axes, red_idx
                    ) if keepdims else out_idx.insert(normalized_axes, red_idx)
                    var cur = self[full_idx]
                    if not inited:
                        max_val = cur
                        best_positions = List[IntList]()
                        best_positions.append(full_idx)
                        inited = True
                    else:
                        if cur > max_val:
                            max_val = cur
                            best_positions.clear()
                            best_positions.append(full_idx)
                        elif cur == max_val:
                            best_positions.append(full_idx)

                # write max to output
                out[out_idx] = max_val

                # split responsibility among ties in this reduced block
                var count = len(best_positions)
                if count > 0:
                    var inv = Scalar[dtype](1) / count
                    for p in best_positions:
                        mask[p] = inv

        # Attach autograd info: save input and mask as ancestors so backward can use mask
        if grad_required:
            out.requires_grad_(True)
            var backward_fn = MaxBackward[dtype](
                normalized_axes.copy(), keepdims
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(
                TensorLite[dtype].of(self), TensorLite[dtype].of(mask)
            )

        return out


fn main():
    print("passes")
    a = Tensor.zeros(3, 3, requires_grad=True)
    a[0, 0] = 42
    a[1, 1] = 35
    a[2, 2] = 51
    a[2, 0] = 51
    a.print()

    print()

    max_result = a.max(IntList(1))
    print()
    max_result.print()
    max_result.backward()
    a.gradbox[].print()
