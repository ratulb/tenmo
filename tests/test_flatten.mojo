from tenmo import Tensor
from testing import assert_true
from shapes import Shape
from strides import Strides


fn main() raises:
    print("Running all flatten tests...")
    test_flatten_scalar()
    test_flatten_1d()
    test_flatten_2d()
    test_flatten_3d()
    test_flatten_keep_grad_chain()
    test_flatten_partial_axes()
    test_flatten_1d_to_1d()
    test_flatten_2d_to_1d()
    test_flatten_3d_to_1d()
    test_flatten_2d_partial_start_dim()
    test_flatten_3d_partial_dims()
    test_flatten_4d_complex()
    test_flatten_no_grad()
    test_flatten_with_grad_computation()
    test_flatten_requires_grad_false()
    test_flatten_requires_grad_true()
    test_flatten_grad_accumulation()
    test_flatten_view_partial_tensor()
    test_flatten_view_2d_to_1d()
    test_flatten_view_3d_to_1d()
    test_flatten_view_with_strides()
    test_flatten_view_complex_chain()
    test_flatten_view_grad_accumulation()
    test_flatten_basic_forward()
    test_flatten_start_dim()
    test_flatten_full_grad()
    test_flatten_partial_grad()
    test_flatten_no_grad_required()
    test_flatten_does_not_alias_input()
    test_flatten_after_expand()
    test_flatten_after_contiguous()
    test_flatten_view_chain()
    test_flatten_after_expand_contiguous_view_chain()
    print("All flatten tests passed!")

fn test_flatten_scalar() raises:
    print("test_flatten_scalar")
    var a = Tensor.scalar(5.0, requires_grad=True)
    var f = a.flatten()
    assert_true(f.shape() == Shape())
    assert_true(f.item() == 5.0)
    f.backward()
    assert_true(a.grad().item() == 1.0)


fn test_flatten_1d() raises:
    print("test_flatten_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var f = a.flatten()
    assert_true(f.shape() == Shape.of(3))
    assert_true((f == a))
    f.sum().backward()
    assert_true(a.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_flatten_2d() raises:
    print("test_flatten_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var f = a.flatten()
    assert_true(f.shape() == Shape.of(4))
    assert_true(f.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    f.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_flatten_3d() raises:
    print("test_flatten_3d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]],
         [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True
    )  # shape (2,2,2)
    var f = a.flatten()
    assert_true(f.shape() == Shape.of(8))
    expected_flat = Tensor.d1([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])
    assert_true(f.all_close(expected_flat))
    f.sum().backward()
    expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]],
         [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_keep_grad_chain() raises:
    print("test_flatten_keep_grad_chain")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.flatten()
    var c = b * 2.0
    var d = c.sum()
    d.backward()
    # d = sum(2 * a) → grad(a) = 2
    expected_grad = Tensor.d2([[2.0, 2.0], [2.0, 2.0]])
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_partial_axes() raises:
    print("test_flatten_partial_axes")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]],
         [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True
    )  # shape (2,2,2)
    # Flatten from axis=1 → shape becomes (2,4)
    var f = a.flatten(start_dim=1)
    assert_true(f.shape() == Shape.of(2, 4))
    f.sum().backward()
    expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]],
         [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))
#here

fn test_flatten_1d_to_1d() raises:
    print("test_flatten_1d_to_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = a.flatten()
    b.sum().backward()

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(a.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 1.0])))


fn test_flatten_2d_to_1d() raises:
    print("test_flatten_2d_to_1d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.flatten()
    b.sum().backward()

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_flatten_3d_to_1d() raises:
    print("test_flatten_3d_to_1d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.flatten()
    b.sum().backward()

    assert_true(b.shape() == Shape.of(8))
    assert_true(
        b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    )
    var expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_2d_partial_start_dim() raises:
    print("test_flatten_2d_partial_start_dim")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var b = a.flatten(start_dim=1)  # Should keep first dimension
    b.sum().backward()

    assert_true(b.shape() == Shape.of(2, 3))
    assert_true(b.all_close(Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))
    assert_true(
        a.grad().all_close(Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


fn test_flatten_3d_partial_dims() raises:
    print("test_flatten_3d_partial_dims")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.flatten(start_dim=1, end_dim=2)  # Flatten last two dimensions
    b.sum().backward()

    assert_true(b.shape() == Shape.of(2, 4))
    assert_true(
        b.all_close(Tensor.d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    )
    var expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_4d_complex() raises:
    print("test_flatten_4d_complex")
    var a = Tensor.d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ],
        requires_grad=True,
    )
    var b = a.flatten()
    b.sum().backward()

    assert_true(b.shape() == Shape.of(16))
    var expected_data = Tensor.d1(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    )
    assert_true(b.all_close(expected_data))
    var expected_grad = Tensor.d4(
        [
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_no_grad() raises:
    print("test_flatten_no_grad")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var b = a.flatten()

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(not b.requires_grad)


fn test_flatten_with_grad_computation() raises:
    print("test_flatten_with_grad_computation")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.flatten()
    var c = b * 2.0  # Additional operation after flatten
    c.sum().backward()

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(c.all_close(Tensor.d1([2.0, 4.0, 6.0, 8.0])))
    assert_true(a.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_flatten_requires_grad_false() raises:
    print("test_flatten_requires_grad_false")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.flatten(requires_grad=False)

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(not b.requires_grad)


fn test_flatten_requires_grad_true() raises:
    print("test_flatten_requires_grad_true")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var b = a.flatten(requires_grad=True)

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(b.requires_grad)


fn test_flatten_grad_accumulation() raises:
    print("test_flatten_grad_accumulation")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.flatten()

    # First backward pass
    b.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))

    # Second backward pass (should accumulate)
    b.zero_grad()
    b.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_flatten_view_2d_to_1d() raises:
    print("test_flatten_view_2d_to_1d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # Create flatten view manually using view API
    var flattened = a.view(shape=Shape(4), strides=Strides(1), offset=0)
    flattened.sum().backward()

    assert_true(flattened.shape() == Shape.of(4))
    assert_true(flattened.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_flatten_view_3d_to_1d() raises:
    print("test_flatten_view_3d_to_1d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    # Flatten 3D to 1D: shape (2, 2, 2) -> (8)
    var flattened = a.view(shape=Shape(8), strides=Strides(1), offset=0)
    flattened.sum().backward()

    assert_true(flattened.shape() == Shape.of(8))
    assert_true(
        flattened.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    )
    var expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_view_with_strides() raises:
    print("test_flatten_view_with_strides")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    # Create a strided view then flatten it
    var strided_view = a.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=0
    )
    var flattened = strided_view.view(
        shape=Shape(4), strides=Strides(1), offset=0
    )
    flattened.sum().backward()

    assert_true(flattened.shape() == Shape.of(4))
    assert_true(flattened.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(
        a.grad().all_close(Tensor.d2([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    )


fn test_flatten_view_partial_tensor() raises:
    print("test_flatten_view_partial_tensor")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    # Create view of a subset then flatten
    var subset_view = a.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=1
    )  # Take columns 1-2

    var flattened = subset_view.view(
        shape=Shape(4), strides=Strides(1), offset=0
    )

    flattened.sum().backward()
    assert_true(flattened.shape() == Shape.of(4))
    assert_true(flattened.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(
        a.grad().all_close(Tensor.d2([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    )


fn test_flatten_view_complex_chain() raises:
    print("test_flatten_view_complex_chain")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    # Complex view chain ending with flatten
    var view1 = a.view(
        shape=Shape(2, 4), strides=Strides(4, 1), offset=0
    )  # Combine last two dims
    var view2 = view1.view(
        shape=Shape(4, 2), strides=Strides(2, 1), offset=0
    )  # Reshape
    var flattened = view2.view(
        shape=Shape(8), strides=Strides(1), offset=0
    )  # Final flatten

    flattened.sum().backward()

    assert_true(flattened.shape() == Shape.of(8))
    assert_true(
        flattened.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    )
    var expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_view_grad_accumulation() raises:
    print("test_flatten_view_grad_accumulation")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var flattened = a.view(shape=Shape(4), strides=Strides(1), offset=0)

    # First backward pass
    flattened.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))

    # Second backward pass (should accumulate)
    flattened.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_flatten_basic_forward() raises:
    print("test_flatten_basic_forward")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var f = a.flatten()
    assert_true(f.shape() == Shape.of(4))
    assert_true(f.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))


fn test_flatten_start_dim() raises:
    print("test_flatten_start_dim")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )  # shape (2, 2, 2)
    var f = a.flatten(start_dim=1)
    # flatten dims 1 and 2 → (2, 4)
    assert_true(f.shape() == Shape.of(2, 4))
    assert_true(
        f.all_close(Tensor.d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    )


fn test_flatten_full_grad() raises:
    print("test_flatten_full_grad")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var f = a.flatten()
    var y = f.sum()
    y.backward()
    # Each element contributes equally (1.0)
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_flatten_partial_grad() raises:
    print("test_flatten_partial_grad")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # shape (2, 2, 2)
    var f = a.flatten(start_dim=1)  # → shape (2, 4)
    var y = f.sum()
    y.backward()
    # Gradient should be ones in original shape
    assert_true(
        a.grad().all_close(
            Tensor.d3([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
        )
    )


fn test_flatten_no_grad_required() raises:
    print("test_flatten_no_grad_required")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var f = a.flatten()
    assert_true(f.requires_grad == False)


fn test_flatten_does_not_alias_input() raises:
    print("test_flatten_does_not_alias_input")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var f = a.flatten()
    f[0] = 999.0
    # Because flatten allocates new buffer, a is unchanged
    assert_true(a[0, 0] == 1.0)


# --- View + Expand + Contiguous chain tests ---


fn test_flatten_after_expand() raises:
    print("test_flatten_after_expand")
    var base = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var exp = base.expand(Shape.of(2, 3))  # shape (2,3)
    var f = exp.flatten()
    var y = f.sum()
    y.backward()
    # Each base element was repeated twice in expand
    assert_true(base.grad().all_close(Tensor.d1([2.0, 2.0, 2.0])))


fn test_flatten_after_contiguous() raises:
    print("test_flatten_after_contiguous")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var trans = a.transpose()
    var cont = trans.contiguous()  # makes it a dense contiguous copy
    var f = cont.flatten()
    var y = f.sum()
    y.backward()
    # Contiguous copy means no aliasing → a.grad should be zeros
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_flatten_view_chain() raises:
    print("test_flatten_view_chain")
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
    )
    var v1 = a.view(Shape.of(4, 2))  # (4,2)
    var v2 = v1.view(Shape.of(2, 4))  # (2,4)
    var f = v2.flatten()  # (8,)
    var y = f.sum()
    y.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_flatten_after_expand_contiguous_view_chain() raises:
    print("test_flatten_after_expand_contiguous_view_chain")
    var base = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var exp = base.expand(Shape.of(3, 2, 2))  # (3,2,2)
    var cont = exp.contiguous()  # full copy
    var v = cont.view(Shape.of(3, 4))  # (3,4)
    var f = v.flatten()  # (12,)
    var y = f.sum()
    y.backward()
    # expand → contiguous → view → flatten should trace correctly
    assert_true(base.grad().all_close(Tensor.d2([[3.0, 3.0], [3.0, 3.0]])))
