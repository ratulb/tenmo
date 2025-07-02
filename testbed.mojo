from tensors import Tensor
from intlist import IntList
from shapes import Shape
from os import abort
from testing import assert_true


fn main() raises:
    A3 = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    m12 = A3.mean(axes=[1, 2], keepdims=False)  # shape: (2,)
    assert_true(m12.all_close(Tensor.d1([2.5, 6.5]).to_dtype[DType.float32]()))
    m12.backward()
    expected_grad = Tensor.d3(
        [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]
    )
    assert_true(A3.grad[].all_close(expected_grad))

    _ = """x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x.mean([]).print()
    var w = Tensor.d2([[0.1], [0.2]], requires_grad=True)
    var b = Tensor[DType.float32].d1([0.5], requires_grad=True)
    var target = Tensor.d2([[1.0], [2.0]])
    print(x.dtype, w.dtype, b.dtype, target.dtype)"""

    # var y_pred = x.matmul(w) + b

    # === Edge Cases ===
    # Case 6: Empty tensor reshape
    _ = """a = Tensor.d1([], requires_grad=True)

    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = a.reshape(3,1)
    #c = b + Tensor.scalar(100)
    c = Tensor.scalar(100) + b
    d = c.sum()
    Tensor.walk_backward(d)


    print("Shape.Void: ", Shape.Void)
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = a + Tensor.scalar(100)
    #b = Tensor.scalar(100) + a
    c = b.sum()
    Tensor.walk_backward(c)"""

    _ = """a = Tensor.d1([1, 2, 3], requires_grad=True)
    #b = a.reshape(1, 3)
    #c = a + Tensor.scalar(10)  # broadcast add
    c = Tensor.scalar(10) + a  # broadcast add
    print("\nc's ancestors\n")
    for each in c.ancestors:
        each[].print()
        print()
    d = c.sum()
    Tensor.walk_backward(d)"""

    _ = """scalar = Tensor.scalar(10)
    _ = scalar.reshape()
    _ = scalar.reshape(0)
    _ = scalar.reshape(1,1,1,1)
    reshaped = scalar.reshape(1)
    scalar.print()
    reshaped.print()"""
    _ = """a = Tensor.d2([[1, 2, 3]], requires_grad=True)
    b = a + Tensor.d1([1, 2, 3])
    s = b.sum()
    Tensor.walk_backward(s)
    return"""
    _ = """# 6. Reshape with degenerate axis: (4,) -> (1, 4) -> (4,)
    a = Tensor.d1([5, 6, 7, 8], requires_grad=True)
    b = a.reshape(Shape.of(1, 4))
    c = b.reshape(Shape.of(4,))
    d = c * Tensor.d1([1, 2, 3, 4])
    Tensor.walk_backward(d)
    a.grad[].print()
    assert_true((a.grad[] == Tensor.d1([1, 2, 3, 4])).all_true(), "reshape with (1,4) roundtrip")"""
    _ = """# 7. Reshape then broadcast in op
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    #b = a.reshape(Shape(IntList(1, 4)))
    b = a.reshape(1, 4)
    c = b + Tensor.scalar(10)  # broadcast add
    #c = b + Tensor.d1([1, 1, 1, 1])
    d = c.sum()
    print("\na\n")
    a.print()
    print("\nb\n")
    b.print()
    print("\nc\n")
    c.print()
    print("\nd\n")
    #d.print()
    Tensor.walk_backward(d)"""
    # assert_true((a == Tensor.d1([1, 1, 1, 1])).all_true(), "reshape + broadcast add + sum")

    _ = """shape = Shape()
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
    reshaped.print()"""
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


fn sum1[
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

    _ = """fn broadcast_backprop(
        upstream_grad: Tensor,
        original_tensor: Tensor,
        other_operand: Tensor
    ) -> Tensor:
        # 1. Multiply with the other operand
        grad_unreduced = upstream_grad * other_operand

        # 2. Find axes to reduce (where original had 1 but output didn't)
        reduce_axes = []
        for i in range(upstream_grad.ndim):
            if original_tensor.shape[i] == 1 and upstream_grad.shape[i] > 1:
                reduce_axes.append(i)

        # 3. Sum and reshape
        if reduce_axes:
            return grad_unreduced.sum(axes=reduce_axes, keepdims=True)
                 .reshape(original_tensor.shape)
        else:
            return grad_unreduced.reshape(original_tensor.shape)"""
