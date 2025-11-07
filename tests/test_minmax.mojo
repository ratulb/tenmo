from tenmo import Tensor
from testing import assert_true
from intlist import IntList

fn main() raises:
    test_max_min_mixed()
    test_max_min()


fn test_max_min_mixed() raises:
    print("test_max_min_mixed")

    # Test 1: Basic max reduction along axis 1
    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )
    var max_result = a.max(IntList(1))
    var expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))

    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )

    # Reset gradients for next test
    a.zero_grad()

    # Test 2: Basic min reduction along axis 1
    var min_result = a.min(IntList(1))
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()

    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]
            )
        )
    )

    # Test 3: Max reduction along axis 0
    a.zero_grad()
    var max_axis0 = a.max(IntList(0))
    assert_true(max_axis0.all_close(Tensor.d1([51.0, 35.0, 51.0])))
    max_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
            )
        )
    )

    # Test 4: Min reduction along axis 0
    a.zero_grad()
    var min_axis0 = a.min(IntList(0))
    assert_true(min_axis0.all_close(Tensor.d1([0.0, 0.0, -5.0])))
    min_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.5, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
            )
        )
    )

    # Test 5: Global max (no axis)
    a.zero_grad()
    var global_max = a.max()
    assert_true(global_max.all_close(Tensor.scalar(51.0)))
    global_max.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )

    # Test 6: Global min (no axis)
    a.zero_grad()
    var global_min = a.min()
    assert_true(global_min.all_close(Tensor.scalar(-5.0)))
    global_min.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        )
    )

    # Test 7: Multiple axes reduction
    var b = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var max_axes_01 = b.max(IntList([0, 1]))
    assert_true(max_axes_01.all_close(Tensor.d1([7.0, 8.0])))
    max_axes_01.backward()
    assert_true(
        b.grad().all_close(
            Tensor.d3(
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]
            )
        )
    )

    # Test 8: Edge case - all same values
    var c = Tensor.d2([[5.0, 5.0], [5.0, 5.0]], requires_grad=True)

    var max_same = c.max(IntList(1))
    assert_true(max_same.all_close(Tensor.d1([5.0, 5.0])))
    max_same.backward()
    assert_true(
        c.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]]))
    )

    # Test 9: Edge case - negative infinity
    var d = Tensor.d2([[-3.4028235e38, 0.0], [1.0, 2.0]], requires_grad=True)

    var max_with_inf = d.max(IntList(1))
    assert_true(max_with_inf.all_close(Tensor.d1([0.0, 2.0])))
    max_with_inf.backward()
    assert_true(
        d.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]]))
    )

    # Test 10: Keep dimensions
    var e = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    var max_keepdim = e.max(IntList(1), keepdims=True)
    assert_true(max_keepdim.all_close(Tensor.d2([[2.0], [4.0]])))
    max_keepdim.backward()
    assert_true(
        e.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]]))
    )


fn test_max_min() raises:
    print("test_max_min")
    a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )

    max_result = a.max(IntList(1))
    expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))

    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )
    min_result = a.min([1])
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()

    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 1.0], [0.5, 1.0, 0.5], [0.5, 1.0, 0.5]]
            )
        )
    )

