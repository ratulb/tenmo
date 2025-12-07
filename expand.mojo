from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_EXPAND
from shapes import Shape
from intarray import IntArray
from strides import Strides
from gradbox import Gradbox
from broadcasthelper import ShapeBroadcaster


@fieldwise_init
@register_passable
struct ExpandBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_EXPAND
    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        parent_shape = ancestor.shape()
        gradbox_contracted = gradbox.sum_over_broadcasted_axes(parent_shape)

        return [(ancestor^, gradbox_contracted^, AddTensor)]


@register_passable
struct Expand[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        target_shape: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        curr_shape = tensor.shape()
        shape_expanded = ShapeBroadcaster.broadcast_shape(
            curr_shape, target_shape
        )

        extra_dims = len(shape_expanded) - len(curr_shape)
        unit_shape = Shape.Unit()  # Shape(1)
        shape_padded = unit_shape * extra_dims + curr_shape
        padded_strides = (
            IntArray.filled(extra_dims, 0) + tensor.strides().intarray()
        )

        strides_expanded = IntArray.with_capacity(len(padded_strides))
        for i in range(len(shape_expanded)):
            if shape_padded[i] == 1 and shape_expanded[i] > 1:
                # Broadcasted dimension → stride 0
                strides_expanded.append(0)
            else:
                strides_expanded.append(padded_strides[i])

        strides = Strides(strides_expanded)

        offset = tensor.offset()  # keep same as current tensor

        out = Tensor[Self.dtype].build_view(
            tensor,
            shape_expanded,
            strides,
            offset,
            requires_grad=False,
        )

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_()
                var bfn = ExpandBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(bfn^)
                out.add_ancestry(tensor)

        return out^




from testing import assert_true

fn main_1() raises:
    test_deep_view_chain_backward()
    test_deep_view_chain_backward1()
    test_expand_mixed_chain()

fn test_expand_mixed_chain() raises:
    print("test_expand_mixed_chain")
    a = Tensor.arange(12, requires_grad=True)
    r = a.reshape([3, 1, 4])
    expanded = r.expand(Shape(2, 3, 2, 4))
    expanded.print()
    c = expanded.contiguous()
    c.print()
    c.backward(42)
    assert_true(a.grad() == Tensor.full(Shape(12), 168))
    a.grad().print()

fn test_deep_view_chain_backward1() raises:
    print("test_deep_view_chain_backward1")
    alias dtype = DType.float32

    a = Tensor[dtype].d2(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ],
        requires_grad=True,
    )

    v4 = a.view(shape=Shape(2, 2), strides=Strides(12, 3), offset=7)
    result = v4.sum()
    result.backward(42)

    expected_grad = Tensor[dtype].d2(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 42, 0, 0, 42, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 42, 0, 0, 42, 0],
        ]
    )

    assert_true(a.grad().as_tensor() == expected_grad)
    print("✓ Deep view chain backward pass works correctly!")

fn test_deep_view_chain_backward() raises:
    print("test_deep_view_chain_backward")


fn main() raises:
    print("test_deep_view_chain_backward")
    alias dtype = DType.float32

    # Create base tensor
    a = Tensor[dtype].d2(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ],
        requires_grad=True,
    )

    v = a.view(shape=Shape(2, 2), strides=Strides(12, 3), offset=7)
    v.print()
    # Final operation
    result = v.sum()
    result.backward(42)

    expected_grad = Tensor[dtype].d2(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 42, 42, 0, 0, 0],
            [0, 42, 42, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    a.grad().print()
    assert_true(a.grad().as_tensor() == expected_grad)
    print("✓ Deep view chain backward pass works correctly!")
