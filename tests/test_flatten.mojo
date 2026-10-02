from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.shared.shapes import Shape
from tenmo.shared.strides import Strides
from std.sys import has_accelerator
from std.sys.defines import get_defined_string
from std.python import Python, PythonObject


# Old tests
def test_flatten_scalar() raises:
    var a = Tensor[DType.float32].scalar(5.0, requires_grad=True)
    var f = a.flatten()
    assert_true(f.shape() == Shape())
    assert_true(f.item() == 5.0)
    f.backward()
    assert_true(a.grad().item() == 1.0)


def test_flatten_1d() raises:
    var a = Tensor[DType.float32].d1([1.0, 2.0, 3.0], requires_grad=True)
    var f = a.flatten()
    assert_true(f.shape() == Shape(3))
    assert_true((f == a))
    var s = f.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor[DType.float32].d1([1.0, 1.0, 1.0])))


def test_flatten_2d() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var f = a.flatten()
    assert_true(f.shape() == Shape(4))
    assert_true(f.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    var s = f.sum()
    s.backward()
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[1.0, 1.0], [1.0, 1.0]]))
    )


def test_flatten_3d() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # shape (2,2,2)
    var f = a.flatten()
    assert_true(f.shape() == Shape(8))
    var expected_flat = Tensor[DType.float32].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    assert_true(f.all_close(expected_flat))
    var s = f.sum()
    s.backward()
    var expected_grad = Tensor[DType.float32].d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_keep_grad_chain() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var b = a.flatten()
    var c = b * 2.0
    var d = c.sum()
    d.backward()
    # d = sum(2 * a) → grad(a) = 2
    var expected_grad = Tensor[DType.float32].d2([[2.0, 2.0], [2.0, 2.0]])
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_partial_axes() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # shape (2,2,2)
    # Flatten from axis=1 → shape becomes (2,4)
    var f = a.flatten(start_dim=1)
    assert_true(f.shape() == Shape(2, 4))
    var s = f.sum()
    s.backward()
    var expected_grad = Tensor[DType.float32].d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


# here


def test_flatten_1d_to_1d() raises:
    var a = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = a.flatten()
    var s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape(4))
    assert_true(b.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d1([1.0, 1.0, 1.0, 1.0]))
    )


def test_flatten_2d_to_1d() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var b = a.flatten()
    var s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape(4))
    assert_true(b.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[1.0, 1.0], [1.0, 1.0]]))
    )


def test_flatten_3d_to_1d() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.flatten()
    var s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape(8))
    assert_true(
        b.all_close(
            Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        )
    )
    var expected_grad = Tensor[DType.float32].d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_2d_partial_start_dim() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var b = a.flatten(start_dim=1)  # Should keep first dimension
    var s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape(2, 3))
    assert_true(
        b.all_close(
            Tensor[DType.float32].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )
    )
    assert_true(
        a.grad().all_close(
            Tensor[DType.float32].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        )
    )


def test_flatten_3d_partial_dims() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.flatten(start_dim=1, end_dim=2)  # Flatten last two dimensions
    var s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape(2, 4))
    assert_true(
        b.all_close(
            Tensor[DType.float32].d2(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            )
        )
    )
    var expected_grad = Tensor[DType.float32].d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_4d_complex() raises:
    var a = Tensor[DType.float32].d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ],
        requires_grad=True,
    )
    var b = a.flatten()
    var s = b.sum()
    s.backward()

    assert_true(b.shape() == Shape(16))
    var expected_data = Tensor[DType.float32].d1(
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
    var expected_grad = Tensor[DType.float32].d4(
        [
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_no_grad() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=False
    )
    var b = a.flatten()

    assert_true(b.shape() == Shape(4))
    assert_true(b.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(not b.requires_grad)


def test_flatten_with_grad_computation() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var b = a.flatten()
    var c = b * 2.0  # Additional operation after flatten
    var s = c.sum()
    s.backward()

    assert_true(b.shape() == Shape(4))
    assert_true(b.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(c.all_close(Tensor[DType.float32].d1([2.0, 4.0, 6.0, 8.0])))
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[2.0, 2.0], [2.0, 2.0]]))
    )


def test_flatten_requires_grad_false() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var b = a.flatten(requires_grad=False)

    assert_true(b.shape() == Shape(4))
    assert_true(b.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(not b.requires_grad)


def test_flatten_requires_grad_true() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=False
    )
    var b = a.flatten(requires_grad=True)

    assert_true(b.shape() == Shape(4))
    assert_true(b.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))
    assert_true(b.requires_grad)


def test_flatten_grad_accumulation() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var b = a.flatten()

    # First backward pass
    var s = b.sum()
    s.backward()
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[1.0, 1.0], [1.0, 1.0]]))
    )

    # Second backward pass (should accumulate)
    b.zero_grad()
    s = b.sum()
    s.backward()
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[2.0, 2.0], [2.0, 2.0]]))
    )


def test_flatten_view_2d_to_1d() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    # Create flatten view manually using view API
    var flattened = a.view(shape=Shape(4), strides=Strides(1), offset=0)
    var s = flattened.sum()
    s.backward()

    assert_true(flattened.shape() == Shape(4))
    assert_true(
        flattened.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0]))
    )
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[1.0, 1.0], [1.0, 1.0]]))
    )


def test_flatten_view_3d_to_1d() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    # Flatten 3D to 1D: shape (2, 2, 2) -> (8)
    var flattened = a.view(shape=Shape(8), strides=Strides(1), offset=0)
    var s = flattened.sum()
    s.backward()

    assert_true(flattened.shape() == Shape(8))
    assert_true(
        flattened.all_close(
            Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        )
    )
    var expected_grad = Tensor[DType.float32].d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_view_with_strides() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    # Create a strided view then flatten it
    var strided_view = a.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=0
    )
    var flattened = strided_view.view(
        shape=Shape(4), strides=Strides(1), offset=0
    )
    var s = flattened.sum()
    s.backward()

    assert_true(flattened.shape() == Shape(4))
    assert_true(
        flattened.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0]))
    )
    assert_true(
        a.grad().all_close(
            Tensor[DType.float32].d2([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        )
    )


def test_flatten_view_partial_tensor() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    # Create view of a subset then flatten
    var subset_view = a.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=1
    )  # Take columns 1-2

    var flattened = subset_view.view(
        shape=Shape(4), strides=Strides(1), offset=0
    )

    var s = flattened.sum()
    s.backward()
    assert_true(flattened.shape() == Shape(4))
    assert_true(
        flattened.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0]))
    )
    assert_true(
        a.grad().all_close(
            Tensor[DType.float32].d2([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        )
    )


def test_flatten_view_complex_chain() raises:
    var a = Tensor[DType.float32].d3(
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

    var s = flattened.sum()
    s.backward()

    assert_true(flattened.shape() == Shape(8))
    assert_true(
        flattened.all_close(
            Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        )
    )
    var expected_grad = Tensor[DType.float32].d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


def test_flatten_view_grad_accumulation() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var flattened = a.view(shape=Shape(4), strides=Strides(1), offset=0)

    # First backward pass
    var s = flattened.sum()
    s.backward()
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[1.0, 1.0], [1.0, 1.0]]))
    )

    # Second backward pass (should accumulate)
    s = flattened.sum()
    s.backward()
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[2.0, 2.0], [2.0, 2.0]]))
    )


def test_flatten_basic_forward() raises:
    var a = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]])
    var f = a.flatten()
    assert_true(f.shape() == Shape(4))
    assert_true(f.all_close(Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])))


def test_flatten_start_dim() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )  # shape (2, 2, 2)
    var f = a.flatten(start_dim=1)
    # flatten dims 1 and 2 → (2, 4)
    assert_true(f.shape() == Shape(2, 4))
    assert_true(
        f.all_close(
            Tensor[DType.float32].d2(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            )
        )
    )


def test_flatten_full_grad() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var f = a.flatten()
    var y = f.sum()
    y.backward()
    # Each element contributes equally (1.0)
    assert_true(
        a.grad().all_close(Tensor[DType.float32].d2([[1.0, 1.0], [1.0, 1.0]]))
    )


def test_flatten_partial_grad() raises:
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # shape (2, 2, 2)
    var f = a.flatten(start_dim=1)  # → shape (2, 4)
    var y = f.sum()
    y.backward()
    # Gradient should be ones in original shape
    assert_true(
        a.grad().all_close(
            Tensor[DType.float32].d3(
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
            )
        )
    )


def test_flatten_no_grad_required() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=False
    )
    var f = a.flatten()
    assert_true(f.requires_grad == False)


def test_flatten_does_not_alias_input() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var f = a.flatten()
    f[0] = 999.0
    # Because flatten allocates new buffer, a is unchanged
    assert_true(a[0, 0] == 1.0)


# --- View + Expand + Contiguous chain tests ---


def test_flatten_after_expand() raises:
    var base = Tensor[DType.float32].d1([1.0, 2.0, 3.0], requires_grad=True)
    var exp = base.expand(Shape(2, 3))  # shape (2,3)
    var f = exp.flatten()
    var y = f.sum()
    y.backward()
    # Each base element was repeated twice in expand
    assert_true(
        base.grad().all_close(Tensor[DType.float32].d1([2.0, 2.0, 2.0]))
    )


def test_flatten_after_contiguous() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var trans = a.transpose()
    var cont = trans.contiguous()  # makes it a dense contiguous copy
    var f = cont.flatten()
    var y = f.sum()
    y.backward()
    # Contiguous copy means no aliasing → a.grad should be zeros
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_flatten_view_chain() raises:
    var a = Tensor[DType.float32].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
    )
    var v1 = a.view(Shape(4, 2))  # (4,2)
    var v2 = v1.view(Shape(2, 4))  # (2,4)
    var f = v2.flatten()  # (8,)
    var y = f.sum()
    y.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_flatten_after_expand_contiguous_view_chain() raises:
    var base = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var exp = base.expand(Shape(3, 2, 2))  # (3,2,2)
    var cont = exp.contiguous()  # full copy
    var v = cont.view(Shape(3, 4))  # (3,4)
    var f = v.flatten()  # (12,)
    var y = f.sum()
    y.backward()
    # expand → contiguous → view → flatten should trace correctly
    assert_true(
        base.grad().all_close(
            Tensor[DType.float32].d2([[3.0, 3.0], [3.0, 3.0]])
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# CPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_flat_cpu_1d_noop() raises:
    comptime dtype = DType.float32
    # Flatten 1D is identity
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var result = a.flatten()
    assert_true(result.shape() == Shape(4))
    assert_true(result.all_close(a))


def test_flat_cpu_2d_full() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.flatten()
    assert_true(result.shape() == Shape(6))
    assert_true(
        result.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    )


def test_flat_cpu_2d_start0_end0() raises:
    comptime dtype = DType.float32
    # Flatten only dim 0 — shape (2,3) → (2,3) no change since 1 dim collapsed
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.flatten(0, 0)
    assert_true(result.shape() == Shape(2, 3))


def test_flat_cpu_2d_start1_end1() raises:
    comptime dtype = DType.float32
    # Flatten only dim 1 — shape (2,3) → (2,3) no change
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.flatten(1, 1)
    assert_true(result.shape() == Shape(2, 3))


def test_flat_cpu_3d_full() raises:
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


def test_flat_cpu_3d_start0_end1() raises:
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


def test_flat_cpu_3d_start1_end2() raises:
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


def test_flat_cpu_4d_middle() raises:
    comptime dtype = DType.float32
    # Shape (2,3,4,5) → flatten(1,2) → (2,12,5)
    var _tmp0 = Tensor[dtype].arange(120)
    var a = _tmp0.reshape(Shape(2, 3, 4, 5))
    var result = a.flatten(1, 2)
    assert_true(result.shape() == Shape(2, 12, 5))
    assert_true(result.numels() == 120)


def test_flat_cpu_4d_start0_end2() raises:
    comptime dtype = DType.float32
    # Shape (2,3,4,5) → flatten(0,2) → (24,5)
    var _tmp0 = Tensor[dtype].arange(120)
    var a = _tmp0.reshape(Shape(2, 3, 4, 5))
    var result = a.flatten(0, 2)
    assert_true(result.shape() == Shape(24, 5))
    assert_true(result.numels() == 120)


def test_flat_cpu_values_preserved() raises:
    comptime dtype = DType.float32
    # Verify values are preserved after flatten
    var _tmp0 = Tensor[dtype].arange(6)
    var a = _tmp0.reshape(Shape(2, 3))
    var result = a.flatten()
    for i in range(6):
        assert_true(result[[i]] == Scalar[dtype](i))


def test_flat_cpu_no_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var result = a.flatten()
    assert_true(not result.requires_grad)


def test_flat_cpu_requires_grad_propagates() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten()
    assert_true(result.requires_grad)


def test_flat_cpu_suppress_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten(requires_grad=False)
    assert_true(not result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# CPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_flat_cpu_backward_2d_full() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var result = a.flatten()
    var loss = result.sum()
    loss.backward()
    # Gradient of sum through flatten is ones in original shape
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


def test_flat_cpu_backward_3d_full() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    var result = a.flatten()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


def test_flat_cpu_backward_3d_partial() raises:
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


def test_flat_cpu_backward_chain() raises:
    comptime dtype = DType.float32
    # flatten → multiply → sum → backward
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.flatten() * 3.0
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 3.0)))


def test_flat_cpu_backward_grad_shape() raises:
    comptime dtype = DType.float32
    # Verify grad has same shape as original tensor
    var _tmp0 = Tensor[dtype].arange(24)
    var a = _tmp0.reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    var result = a.flatten()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().shape() == Shape(2, 3, 4))
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


def test_flat_cpu_backward_nonuniform_grad() raises:
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


def test_flat_cpu_backward_4d_partial() raises:
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].arange(120)
    var a = _tmp0.reshape(Shape(2, 3, 4, 5))
    a.requires_grad_(True)
    var result = a.flatten(1, 2)
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().shape() == Shape(2, 3, 4, 5))
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4, 5))))


# ═════════════════════════════════════════════════════════════════════════════
# GPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_flat_gpu_1d_noop() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]))
        )


def test_flat_gpu_2d_full() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_3d_full() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_3d_start0_end1() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_3d_start1_end2() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_4d_middle() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(120)
        var _tmp1 = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a = _tmp1.to_gpu()
        var result = a.flatten(1, 2)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 12, 5))
        assert_true(result.numels() == 120)


def test_flat_gpu_values_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(6)
        var a_cpu = _tmp0.reshape(Shape(2, 3))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten()
        var result_cpu = result.to_cpu()
        for i in range(6):
            assert_true(result_cpu[[i]] == Scalar[dtype](i))


def test_flat_gpu_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(not result.requires_grad)


def test_flat_gpu_requires_grad_propagates() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_backward_2d_full() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


def test_flat_gpu_backward_3d_full() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_backward_3d_partial() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten() * 3.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 3.0)))


def test_flat_gpu_backward_grad_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


def test_flat_gpu_backward_nonuniform_grad() raises:
    comptime if has_accelerator():
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


def test_flat_gpu_backward_4d_partial() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(120)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4, 5))
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


def test_flat_parity_2d_full_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten().all_close(a_gpu.flatten().to_cpu()))


def test_flat_parity_3d_partial_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten(1, 2).all_close(a_gpu.flatten(1, 2).to_cpu()))


def test_flat_parity_4d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(120)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten(0, 2).all_close(a_gpu.flatten(0, 2).to_cpu()))


def test_flat_parity_2d_backward() raises:
    comptime if has_accelerator():
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


def test_flat_parity_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var _tmp1 = Tensor[dtype].arange(24)
        var _tmp2 = _tmp1.reshape(Shape(2, 3, 4))
        var a_gpu = _tmp2.to_gpu()
        a_gpu.requires_grad_(True)

        var loss_cpu = a_cpu.flatten(1, 2).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.flatten(1, 2).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_flat_parity_chain_backward() raises:
    comptime if has_accelerator():
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


def test_flat_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.flatten().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().clone()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.flatten().sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


def test_flatten_negative_start_dim() raises:
    # flatten(x, -2) on (2,2,2) == flatten(x, 1) -> (2,4).
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var f = a.flatten(start_dim=-2)
    assert_true(f.shape() == Shape(2, 4))
    assert_true(
        f.all_close(
            Tensor[DType.float32].d2(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            )
        )
    )


def test_flatten_negative_start_and_end_dim() raises:
    # flatten(x, 0, -1) on (2,2,2) flattens everything -> (8,).
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var f = a.flatten(start_dim=0, end_dim=-1)
    assert_true(f.shape() == Shape(8))
    assert_true(
        f.all_close(
            Tensor[DType.float32].d1(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            )
        )
    )


def test_flatten_negative_dims_backward() raises:
    # Non-uniform loss through a negative-dim flatten: grads must route
    # back to exact input positions.
    var a = Tensor[DType.float32].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    var f = a.flatten(start_dim=-2)
    var w = Tensor[DType.float32].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )
    var loss = (f * w).sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[DType.float32].d3(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
            )
        )
    )


# ============================================================================
# Out-of-range dim probes live in the MINIMAL harness
# tests/test_negdim_probes.mojo: the child performs exactly one invalid
# call and dies by the guard under test; we assert non-zero exit plus the
# exact diagnostic text. If the guard ever stops firing, the child reaches
# its own trailing panic instead and the message assertion fails. The
# harness is a separate MINIMAL file because the child JIT runs alongside
# this resident process — re-executing a full suite file risks OOM.
# Children are warm-cache recompiles; the mojo cache this process just
# built is shared.
# ============================================================================


def _spawn_negdim_probe(name: String) raises -> PythonObject:
    """Run guard probe `name` from the minimal probe harness in a child."""
    var script = (
        "__import__('subprocess').run("
        + "['pixi', 'run', 'mojo', '-I', '.', "
        + "'tests/test_negdim_probes.mojo', "
        + "'--probe-" + name + "'], "
        + "capture_output=True, text=True, timeout=1200)"
    )
    return Python.evaluate(script)


def test_flatten_dim_guards_abort_with_clear_messages() raises:
    # NOTE: children execute only under -D subprocess=1 (else vacuous
    # pass) — e.g. `pixi run mojo -I . -D subprocess=1 tests/test_flatten.mojo`.
    comptime subprocess = get_defined_string["subprocess", ""]()
    comptime if not subprocess == "":
        var start = _spawn_negdim_probe("flatten-bad-start")
        var start_out = String(start.stdout) + String(start.stderr)
        assert_true(
            String(start.returncode) != "0",
            "Flatten: bad-start probe exits non-zero",
        )
        assert_true(
            start_out.find("NDBuffer → flatten: start_dim") >= 0,
            "Flatten: bad start_dim reports a precise diagnostic",
        )
        var end = _spawn_negdim_probe("flatten-bad-end")
        var end_out = String(end.stdout) + String(end.stderr)
        assert_true(
            String(end.returncode) != "0",
            "Flatten: bad-end probe exits non-zero",
        )
        assert_true(
            end_out.find("NDBuffer → flatten: end_dim") >= 0,
            "Flatten: bad end_dim reports a precise diagnostic",
        )
    else:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
