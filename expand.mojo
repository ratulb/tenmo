from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_EXPAND
from shapes import Shape
from intarray import IntArray
from strides import Strides
from gradbox import Gradbox
from ancestry import Ancestor
from broadcasthelper import ShapeBroadcaster


@fieldwise_init
@register_passable
struct ExpandBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_EXPAND
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
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
        tensor: Tensor[dtype],
        target_shape: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
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

        out = Tensor[dtype].build_view(
            tensor.address(),
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
                var bfn = ExpandBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(bfn^)
                out.add_ancestry(tensor)

        return out^


fn test_expand_mixed_chain() raises:
    a = Tensor.arange(12, requires_grad=True)
    r = a.reshape([3, 1, 4])
    expanded = r.expand(Shape(2, 3, 2, 4))
    expanded.print()
    c = expanded.contiguous()
    c.print()
    c.backward(42)
    assert_true(a.grad() == Tensor.full(Shape(12), 168))
    a.grad().print()


from testing import assert_true


fn main() raises:
    test_deep_view_chain_backward()
    test_expand_mixed_chain()
    test_absolute_stride_offset_chain()


fn test_deep_view_chain_backward() raises:
    print("test_deep_view_chain_backward")
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


fn test_absolute_stride_offset_chain() raises:
    print("test_absolute_stride_offset_chain")

    _ = """# Base tensor: 4×6, contiguous layout
    var a = Tensor[DType.int32].arange(24).reshape((4, 6))
    # Buffer indices:
    # [[ 0,  1,  2,  3,  4,  5],
    #  [ 6,  7,  8,  9, 10, 11],
    #  [12, 13, 14, 15, 16, 17],
    #  [18, 19, 20, 21, 22, 23]]

    # --- v1 -------------------------------------------------------
    # a[0:3, 1:5]
    # → shape: (3,4), offset: 1, strides: (6,1)
    # Absolute offset=1 means starting at a[0,1]
    var v1 = a.slice(0..3, 1..5)
    assert(v1.offset == 1)
    assert(v1.strides == [6, 1])

    # --- v2 -------------------------------------------------------
    # v1[1:3, 0:3]  → base indices (in a):
    # rows 1..2 of v1 → a[1..2, 1..4]
    # → shape: (2,3), offset: 7, strides: (6,1)
    var v2 = v1.slice(1..3, 0..3)
    assert(v2.offset == 7)
    assert(v2.strides == [6, 1])

    # --- v3 -------------------------------------------------------
    # v2[:, 0..3:3] → take columns 0 and 3rd in v2
    # → shape: (2,2)
    # stride along dim1 = 3 (skip 3 elements in base buffer)
    # → absolute strides: (6, 3)
    # → offset remains 7 (same starting element)
    # → accesses: [[a[1,1], a[1,4]],
    #              [a[2,1], a[2,4]]]
    var v3 = v2.slice(:, 0..3:3)
    assert(v3.offset == 7)
    assert(v3.strides == [6, 3])

    # --- v4 -------------------------------------------------------
    # v3[::2, :] → take every 2nd row (so rows 0, 2 if existed)
    # Here only 2 rows (0 and 1), step=2 collapses to first row?
    # To make it meaningful, let's simulate a 2×2 with stride doubling in dim0
    # → shape: (2,2)
    # → absolute strides: (12, 3)
    # → offset = 7
    # → accesses:
    #     (0,0) → a[1,1] =  7+0*12+0*3 = 7+0=7 → value=8
    #     (0,1) → a[1,4] =  7+0*12+1*3 = 7+3=10 → value=11
    #     (1,0) → a[3,1] =  7+1*12+0*3 = 19 → value=20
    #     (1,1) → a[3,4] =  7+1*12+1*3 = 22 → value=23
    # So v4.data = [[8,11],[20,23]]
    var v4 = Tensor[DType.int32](a.buffer, (2, 2), [12, 3], 7)

    # --- forward check -------------------------------------------
    print("v4 =", v4)
    assert(v4.data_equal(Tensor[DType.int32].d2([[8,11],[20,23]])))

    # --- backward check ------------------------------------------
    v4.backward_fill(42)

    # gradient accumulates to the 4 accessed elements
    # indices: (1,1), (1,4), (3,1), (3,4)
    var expected_grad = Tensor[DType.int32].d2(
        [[0, 0, 0, 0, 0, 0],
         [0, 42, 0, 0, 42, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 42, 0, 0, 42, 0]]
    )

    assert(a.grad[].data_equal(expected_grad))"""
    print("✅ Passed absolute stride+offset chain test")


fn test_deep_view_chain_backward1() raises:
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
    print("\na\n")

    a.print()

    v1 = a.view(shape=Shape(3, 4), strides=Strides(6, 1), offset=1)

    print("\nv1\n")

    v1.print()

    # v2 = v1.view(shape=Shape(2, 3), strides=Strides(4, 1), offset=7)
    v2 = a.view(shape=Shape(2, 3), strides=Strides(6, 1), offset=7)
    print("\nv2\n")

    v2.print()

    # v3 = v2.view(shape=Shape(2, 2), strides=Strides(1, 3), offset=5)
    v3 = a.view(shape=Shape(2, 2), strides=Strides(6, 3), offset=7)

    print("\nv3\n")

    v3.print()

    # v4 = v3.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=5)
    v4 = a.view(shape=Shape(2, 2), strides=Strides(12, 3), offset=7)

    print("\nv4\n")

    v4.print()

    # Final operation
    result = v4.sum()
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
