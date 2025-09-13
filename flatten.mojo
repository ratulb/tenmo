from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from shapes import Shape


@fieldwise_init
@register_passable
struct FlattenBackward[dtype: DType](Copyable):
    # no fields needed; ancestor tensor is available from output.ancestry()
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        # upstream gradients (1-D tensor)
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        
        tensor = ancestor.tensor()
        offset = tensor.offset
        tensor_shape = tensor.shape
        grad_in = Tensor[dtype].zeros(tensor_shape)

        if tensor.is_contiguous():
            n = gradients.shape.num_elements()
            for i in range(n):
                if tensor.owns_data:
                    grad_in.buffer[i] = gradients.buffer[i]
                else:
                    grad_in.buffer[i + offset] = gradients.buffer[i]

        else:
            out_shape = gradients.shape
            out_numels = out_shape.num_elements()
            for flat_idx in range(out_numels):
                idx = tensor_shape.unravel_index(flat_idx)
                grad_in[idx] = gradients[flat_idx]

        return [(ancestor, grad_in, AddTensor)]


@register_passable
struct Flatten[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        total_elems = self.shape.num_elements()
        out = Tensor[dtype].zeros(Shape([total_elems]), requires_grad=False)

        # fast contiguous path (direct buffer copy)
        if self.is_contiguous():
            for i in range(total_elems):
                if self.owns_data:
                    out[i] = self.buffer[i]
                else:
                    out[i] = self.base[].buffer[self.offset + i]
        else:
            # slow path: iterate logical indices and copy respecting offset/strides
            var flat_idx = 0
            for _, value in self:
                out[flat_idx] = value
                flat_idx += 1

        # autograd hookup
        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                backward_fn = FlattenBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    print("passes")
