from tensors import Tensor
from intlist import IntList
from shapes import Shape


fn main() raises:
    tensor = Tensor.rand(10,  init_seed= Optional(42))
    tensor.print()
    print()
    summed = sum(tensor, axes=[0], keepdim=True)
    summed.print()


fn sum[
    dtype: DType = DType.float32
](tensor: Tensor[dtype], axes: List[Int], keepdim: Bool = False) -> Tensor[
    dtype
]:
    input_shape = tensor.shape
    input_rank = input_shape.rank()
    sorted_axes = IntList.of(axes).sorted()

    spans = IntList.with_capacity(input_rank)
    for i in range(input_rank):
        if i in sorted_axes:
            if keepdim:
                spans.append(1)
            else:
                continue
        else:
            spans.append(input_shape[i])

    out_shape = Shape(spans)

    var out = Tensor[dtype].zeros(out_shape, requires_grad=tensor.requires_grad)

    #red_shape = Shape(input_shape.axes_spans.select(sorted_axes))
    red_shape = Shape(input_shape.into().select(sorted_axes))

    for out_idx in out.shape:
        var sum_val = Scalar[dtype](0)

        for red_idx in red_shape:
            var full_idx = out_idx.copy()

            if keepdim:
                # Replace values at reduced axes
                for i in range(len(sorted_axes)):
                    full_idx = full_idx.replace(sorted_axes[i], red_idx[i])
            else:
                # Insert reduction indices at correct positions
                for i in range(len(sorted_axes)):
                    full_idx = full_idx.insert(sorted_axes[i], red_idx[i])

            sum_val += tensor[full_idx]

        out[out_idx] = sum_val

    return out
