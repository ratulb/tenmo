from tenmo.tensor import Tensor
from std.testing import assert_true
from tenmo.shapes import Shape
from tenmo.strides import Strides
from std.sys import has_accelerator


# Old tests
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
    s = f.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_flatten_2d() raises:
    print("test_flatten_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var f = a.flatten()
    assert_true(f.shape() == Shape.of(4))
    assert_true(f.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    s = f.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_flatten_3d() raises:
    print("test_flatten_3d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # shape (2,2,2)
    var f = a.flatten()
    assert_true(f.shape() == Shape.of(8))
    expected_flat = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert_true(f.all_close(expected_flat))
    s = f.sum()
    s.backward()
    expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
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
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # shape (2,2,2)
    # Flatten from axis=1 → shape becomes (2,4)
    var f = a.flatten(start_dim=1)
    assert_true(f.shape() == Shape.of(2, 4))
    s = f.sum()
    s.backward()
    expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


# here


fn test_flatten_1d_to_1d() raises:
    print("test_flatten_1d_to_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = a.flatten()
    s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(a.grad().all_close(Tensor.d1([1.0, 1.0, 1.0, 1.0])))


fn test_flatten_2d_to_1d() raises:
    print("test_flatten_2d_to_1d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.flatten()
    s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape.of(4))
    assert_true(b.all_close(Tensor.d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))


fn test_flatten_3d_to_1d() raises:
    print("test_flatten_3d_to_1d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.flatten()
    s = b.sum()
    s.backward()

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
    s = b.sum()
    s.backward()

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
    s = b.sum()
    s.backward()

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
    s = b.sum()
    s.backward()

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
    s = c.sum()
    s.backward()

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
    s = b.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))

    # Second backward pass (should accumulate)
    b.zero_grad()
    s = b.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))


fn test_flatten_view_2d_to_1d() raises:
    print("test_flatten_view_2d_to_1d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # Create flatten view manually using view API
    var flattened = a.view(shape=Shape(4), strides=Strides(1), offset=0)
    s = flattened.sum()
    s.backward()

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
    s = flattened.sum()
    s.backward()

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
    s = flattened.sum()
    s.backward()

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

    s = flattened.sum()
    s.backward()
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

    s = flattened.sum()
    s.backward()

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
    s = flattened.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))

    # Second backward pass (should accumulate)
    s = flattened.sum()
    s.backward()
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


# ═════════════════════════════════════════════════════════════════════════════
# CPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_flat_cpu_1d_noop() raises:
    print("test_flat_cpu_1d_noop")
    comptime dtype = DType.float32
    # Flatten 1D is identity
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var result = a.flatten()
    assert_true(result.shape() == Shape(4))
    assert_true(result.all_close(a))


fn test_flat_cpu_2d_full() raises:
    print("test_flat_cpu_2d_full")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.flatten()
    assert_true(result.shape() == Shape(6))
    assert_true(
        result.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    )


fn test_flat_cpu_2d_start0_end0() raises:
    print("test_flat_cpu_2d_start0_end0")
    comptime dtype = DType.float32
    # Flatten only dim 0 — shape (2,3) → (2,3) no change since 1 dim collapsed
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.flatten(0, 0)
    assert_true(result.shape() == Shape(2, 3))


fn test_flat_cpu_2d_start1_end1() raises:
    print("test_flat_cpu_2d_start1_end1")
    comptime dtype = DType.float32
    # Flatten only dim 1 — shape (2,3) → (2,3) no change
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.flatten(1, 1)
    assert_true(result.shape() == Shape(2, 3))


fn test_flat_cpu_3d_full() raises:
    print("test_flat_cpu_3d_full")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var result = a.flatten()
    assert_true(result.shape() == Shape(8))
    assert_true(
        result.all_close(
            Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        )
    )


fn test_flat_cpu_3d_start0_end1() raises:
    print("test_flat_cpu_3d_start0_end1")
    comptime dtype = DType.float32
    # Shape (2,2,2) → flatten(0,1) → (4,2)
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var result = a.flatten(0, 1)
    assert_true(result.shape() == Shape(4, 2))
    assert_true(
        result.all_close(
            Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
    )


fn test_flat_cpu_3d_start1_end2() raises:
    print("test_flat_cpu_3d_start1_end2")
    comptime dtype = DType.float32
    # Shape (2,2,2) → flatten(1,2) → (2,4)
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var result = a.flatten(1, 2)
    assert_true(result.shape() == Shape(2, 4))
    assert_true(
        result.all_close(
            Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        )
    )


fn test_flat_cpu_4d_middle() raises:
    print("test_flat_cpu_4d_middle")
    comptime dtype = DType.float32
    # Shape (2,3,4,5) → flatten(1,2) → (2,12,5)
    var a = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5))
    var result = a.flatten(1, 2)
    assert_true(result.shape() == Shape(2, 12, 5))
    assert_true(result.numels() == 120)


fn test_flat_cpu_4d_start0_end2() raises:
    print("test_flat_cpu_4d_start0_end2")
    comptime dtype = DType.float32
    # Shape (2,3,4,5) → flatten(0,2) → (24,5)
    var a = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5))
    var result = a.flatten(0, 2)
    assert_true(result.shape() == Shape(24, 5))
    assert_true(result.numels() == 120)


fn test_flat_cpu_values_preserved() raises:
    print("test_flat_cpu_values_preserved")
    comptime dtype = DType.float32
    # Verify values are preserved after flatten
    var a = Tensor[dtype].arange(6).reshape(Shape(2, 3))
    var result = a.flatten()
    for i in range(6):
        assert_true(result[[i]] == Scalar[dtype](i))


fn test_flat_cpu_no_grad() raises:
    print("test_flat_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var result = a.flatten()
    assert_true(not result.requires_grad)


fn test_flat_cpu_requires_grad_propagates() raises:
    print("test_flat_cpu_requires_grad_propagates")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten()
    assert_true(result.requires_grad)


fn test_flat_cpu_suppress_grad() raises:
    print("test_flat_cpu_suppress_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten(requires_grad=False)
    assert_true(not result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# CPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_flat_cpu_backward_2d_full() raises:
    print("test_flat_cpu_backward_2d_full")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var result = a.flatten()
    var loss = result.sum()
    loss.backward()
    # Gradient of sum through flatten is ones in original shape
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


fn test_flat_cpu_backward_3d_full() raises:
    print("test_flat_cpu_backward_3d_full")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    var result = a.flatten()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_flat_cpu_backward_3d_partial() raises:
    print("test_flat_cpu_backward_3d_partial")
    comptime dtype = DType.float32
    # flatten(1,2) — grad should still reshape back to (2,2,2)
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    var result = a.flatten(1, 2)
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_flat_cpu_backward_chain() raises:
    print("test_flat_cpu_backward_chain")
    comptime dtype = DType.float32
    # flatten → multiply → sum → backward
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten() * 3.0
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 3.0)))


fn test_flat_cpu_backward_grad_shape() raises:
    print("test_flat_cpu_backward_grad_shape")
    comptime dtype = DType.float32
    # Verify grad has same shape as original tensor
    var a = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    var result = a.flatten()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().shape() == Shape(2, 3, 4))
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


fn test_flat_cpu_backward_nonuniform_grad() raises:
    print("test_flat_cpu_backward_nonuniform_grad")
    comptime dtype = DType.float32
    # Non-uniform upstream gradient
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten()
    # Multiply by [1,2,3,4] then sum
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var loss = (result * weights).sum()
    loss.backward()
    # grad reshaped back to (2,2)
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))


fn test_flat_cpu_backward_4d_partial() raises:
    print("test_flat_cpu_backward_4d_partial")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5))
    a.requires_grad_(True)
    var result = a.flatten(1, 2)
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().shape() == Shape(2, 3, 4, 5))
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4, 5))))


# ═════════════════════════════════════════════════════════════════════════════
# GPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_flat_gpu_1d_noop() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_1d_noop")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]))
        )


fn test_flat_gpu_2d_full() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_2d_full")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(6))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            )
        )


fn test_flat_gpu_3d_full() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_3d_full")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(8))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            )
        )


fn test_flat_gpu_3d_start0_end1() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_3d_start0_end1")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.flatten(0, 1)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 2))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                )
            )
        )


fn test_flat_gpu_3d_start1_end2() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_3d_start1_end2")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.flatten(1, 2)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 4))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
            )
        )


fn test_flat_gpu_4d_middle() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_4d_middle")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5)).to_gpu()
        var result = a.flatten(1, 2)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 12, 5))
        assert_true(result.numels() == 120)


fn test_flat_gpu_values_preserved() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_values_preserved")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(6).reshape(Shape(2, 3))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten()
        var result_cpu = result.to_cpu()
        for i in range(6):
            assert_true(result_cpu[[i]] == Scalar[dtype](i))


fn test_flat_gpu_no_grad() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_no_grad")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(not result.requires_grad)


fn test_flat_gpu_requires_grad_propagates() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_requires_grad_propagates")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_flat_gpu_backward_2d_full() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_2d_full")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


fn test_flat_gpu_backward_3d_full() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_3d_full")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_flat_gpu_backward_3d_partial() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_3d_partial")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten(1, 2)
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_flat_gpu_backward_chain() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_chain")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten() * 3.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 3.0)))


fn test_flat_gpu_backward_grad_shape() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_grad_shape")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


fn test_flat_gpu_backward_nonuniform_grad() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_nonuniform_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var loss = (result * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )


fn test_flat_gpu_backward_4d_partial() raises:
    comptime if has_accelerator():
        print("test_flat_gpu_backward_4d_partial")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten(1, 2)
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().shape() == Shape(2, 3, 4, 5))
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4, 5)))
        )


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_flat_parity_2d_full_forward() raises:
    comptime if has_accelerator():
        print("test_flat_parity_2d_full_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten().all_close(a_gpu.flatten().to_cpu()))


fn test_flat_parity_3d_partial_forward() raises:
    comptime if has_accelerator():
        print("test_flat_parity_3d_partial_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten(1, 2).all_close(a_gpu.flatten(1, 2).to_cpu()))


fn test_flat_parity_4d_forward() raises:
    comptime if has_accelerator():
        print("test_flat_parity_4d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5))
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten(0, 2).all_close(a_gpu.flatten(0, 2).to_cpu()))


fn test_flat_parity_2d_backward() raises:
    comptime if has_accelerator():
        print("test_flat_parity_2d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )

        var loss_cpu = a_cpu.flatten().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.flatten().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_flat_parity_3d_backward() raises:
    comptime if has_accelerator():
        print("test_flat_parity_3d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4)).to_gpu()
        a_gpu.requires_grad_(True)

        var loss_cpu = a_cpu.flatten(1, 2).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.flatten(1, 2).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_flat_parity_chain_backward() raises:
    comptime if has_accelerator():
        print("test_flat_parity_chain_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            .to_gpu()
        )

        var loss_cpu = (a_cpu.flatten() * 2.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu.flatten() * 2.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_flat_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        print("test_flat_parity_using_zero_grad")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.flatten().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.flatten().sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    # Old tests
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
    # End of old tests
    # CPU Forward
    test_flat_cpu_1d_noop()
    test_flat_cpu_2d_full()
    test_flat_cpu_2d_start0_end0()
    test_flat_cpu_2d_start1_end1()
    test_flat_cpu_3d_full()
    test_flat_cpu_3d_start0_end1()
    test_flat_cpu_3d_start1_end2()
    test_flat_cpu_4d_middle()
    test_flat_cpu_4d_start0_end2()
    test_flat_cpu_values_preserved()
    test_flat_cpu_no_grad()
    test_flat_cpu_requires_grad_propagates()
    test_flat_cpu_suppress_grad()
    print("CPU forward passed!")

    # CPU Backward
    test_flat_cpu_backward_2d_full()
    test_flat_cpu_backward_3d_full()
    test_flat_cpu_backward_3d_partial()
    test_flat_cpu_backward_chain()
    test_flat_cpu_backward_grad_shape()
    test_flat_cpu_backward_nonuniform_grad()
    test_flat_cpu_backward_4d_partial()
    print("CPU backward passed!")

    # GPU Forward
    test_flat_gpu_1d_noop()
    test_flat_gpu_2d_full()
    test_flat_gpu_3d_full()
    test_flat_gpu_3d_start0_end1()
    test_flat_gpu_3d_start1_end2()
    test_flat_gpu_4d_middle()
    test_flat_gpu_values_preserved()
    test_flat_gpu_no_grad()
    test_flat_gpu_requires_grad_propagates()
    print("GPU forward passed!")

    # GPU Backward
    test_flat_gpu_backward_2d_full()
    test_flat_gpu_backward_3d_full()
    test_flat_gpu_backward_3d_partial()
    test_flat_gpu_backward_chain()
    test_flat_gpu_backward_grad_shape()
    test_flat_gpu_backward_nonuniform_grad()
    test_flat_gpu_backward_4d_partial()
    print("GPU backward passed!")

    # Parity
    test_flat_parity_2d_full_forward()
    test_flat_parity_3d_partial_forward()
    test_flat_parity_4d_forward()
    test_flat_parity_2d_backward()
    test_flat_parity_3d_backward()
    test_flat_parity_chain_backward()
    test_flat_parity_using_zero_grad()
    print("Parity passed!")

    print("All flatten tests passed!")
