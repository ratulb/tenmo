from tensors import Tensor
from intlist import IntList
from shapes import Shape
from os import abort


fn main() raises:
    shape = Shape()
    print(shape)
    tensor1 = Tensor.of(1, 2, 3, 4, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)
    result = tensor1 + tensor2
    Tensor.walk_backward(result)
    print("*****************************")
    print("*****************************")
    print("*****************************")
    print("*****************************")
    negated = -tensor1
    negated.print()





    scalar = Tensor.scalar(42, requires_grad=True)
    result = scalar * 2

    scalar.grad[].print()  # 0.0

    Tensor.walk_backward(result)

    scalar.grad[].print()  # 2.0

    reshaped = scalar.reshape()

    reshaped.grad[].print()  # 2.0

    result2 = reshaped * 3

    Tensor.walk_backward(result2)

    scalar.grad[].print()  # 7.0

    reshaped.grad[].print()  # 5.0

    print()
    print()
    print()

    scalar.print()
    reshaped.print()
    _ = """zero = Tensor.zeros(Shape.Void, requires_grad=True)
    zero.print()

    il = IntList()
    _= il.insert(IntList(0,1,2), IntList(1,1,1))
    print("done")
    print("done")
    print("done")

    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    s = a.mean([])
    s.print()
    print("here")
    s.backward()
    a.grad[].print()
    print("there")

    scalar = Tensor.scalar(42, requires_grad=True)
    #scalar = Tensor.scalar(42)
    result = ((scalar * 3) + 2).mean()
    scalar.print()
    result.print()
    Tensor.walk_backward(result)
    scalar.grad[].print()

    t = Tensor[DType.uint8].arange(1, 24, 3, requires_grad=True)
    t.print()
    out_idx = IntList()
    sorted_axes = IntList(0)
    red_idx = IntList(0)
    for i in range(len(sorted_axes)):
        out_idx = out_idx.insert(sorted_axes[i], red_idx[i])
    print("*********")
    out_idx.print()

    print("*********")
    tensor = Tensor.of(1, 2, 3, 4, 5, 6, 7, 8)
    tensor.print()
    summed = sum(tensor, keepdim=True)
    # summed = sum(tensor, axes=[0], keepdim=False)
    summed.print()"""


fn sum[
    dtype: DType = DType.float32
](
    tensor: Tensor[dtype], axes: List[Int] = [-1], keepdim: Bool = False
) -> Tensor[dtype]:
    input_shape = tensor.shape
    input_rank = input_shape.rank()
    sorted_axes = IntList(tensor.shape.rank() - 1) if (
        len(axes) == 0 or axes == [-1]
    ) else IntList.new(axes).sorted()
    # sorted_axes = IntList.new(axes).sorted()
    for ax in sorted_axes:
        if ax < 0 or ax >= input_rank:
            abort("Invalid axis in sum: " + String(ax))

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
    red_shape = Shape(input_shape.axes_spans.select(sorted_axes))

    for out_idx in out.shape:
        var sum_val = Scalar[dtype](0)

        for red_idx in red_shape:
            if keepdim:
                full_idx = out_idx.replace(sorted_axes, red_idx)
            else:
                full_idx = out_idx.insert(sorted_axes, red_idx)

            sum_val += tensor[full_idx]

        out[out_idx] = sum_val
    return out
