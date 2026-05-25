from std.testing import assert_true, TestSuite
from tenmo.shapes import Shape
from tenmo.tensor import Tensor
from tenmo.common_utils import *

comptime dtype = DType.float32

# Test: single-axis slice on a view (offset > 0)
def test_view_slice_single_element() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var row = x[i(1), s()]  # view of row 1, offset=3, shape (3,)
    assert_true(row[0] == 4.0)
    assert_true(row[1] == 5.0)
    assert_true(row[2] == 6.0)

    var s0 = row[i(0)]
    var s1 = row[i(1)]
    var s2 = row[i(2)]
    assert_true(s0.item() == 4.0)
    assert_true(s1.item() == 5.0)
    assert_true(s2.item() == 6.0)

# Test: slice range on a view
def test_view_slice_range() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var row = x[i(1), s()]  # offset=3
    var first_two = row.slice(axis=0, start=0, end=2)
    assert_true(first_two[0] == 4.0)
    assert_true(first_two[1] == 5.0)

# Test: slice with step on a view
def test_view_slice_step() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var row = x[i(1), s()]  # offset=3
    var odds = row.slice(axis=0, start=0, end=3, step=2)
    assert_true(odds[0] == 4.0)
    assert_true(odds[1] == 6.0)

# Test: negative index slice on a view
def test_view_slice_negative() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var row = x[i(1), s()]  # offset=3, shape(3,)
    var last = row.slice(axis=0, start=-1, end=3)
    assert_true(last.item() == 6.0)

# Test: multi-axis list-style slice on a view
def test_view_multi_axis_slice() raises:
    var x = Tensor[dtype].d2([
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
    ])
    var row = x[i(1), s()]  # shape (6,), offset=6
    var sub = row.slice(axes=[0], starts=[2], ends=[5])
    assert_true(sub[0] == 8.0)
    assert_true(sub[1] == 9.0)
    assert_true(sub[2] == 10.0)

# Test: slice on view-of-view
def test_view_of_view_slice() raises:
    var x = Tensor[dtype].d3([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
    ])
    var v1 = x[i(1), s(), s()]  # shape (2,2), offset=4, values: [[5,6],[7,8]]
    var v2 = v1[i(1), s()]      # shape (2,), offset=6, values: [7,8]
    var el = v2[i(1)]            # shape (), offset=7, value: 8
    assert_true(el.item() == 8.0)

# Test: Slice-based __getitem__ on a view
def test_view_slice_syntax() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var row = x[i(1), s()]  # view offset=3, shape(3,)
    var mid = row[Slice(1, 3)]
    assert_true(mid[0] == 5.0)
    assert_true(mid[1] == 6.0)

# Test: backward gradients through view slices
def test_view_slice_backward() raises:
    var x = Tensor[dtype].d2(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True
    )
    var row = x[i(1), s()]  # view of row 1, offset=3
    var first_two = row.slice(axis=0, start=0, end=2)
    var loss = first_two.sum()
    loss.backward()
    var grad = x.grad().detach(share=True)
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([0, 0, 0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([1, 1, 0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0, 0, 0])))

# Test: backward through multi-axis slice on view
def test_view_multi_slice_backward() raises:
    var x = Tensor[dtype].d2(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    )
    x.requires_grad_(True)
    var row = x[i(1), s()]  # shape (4,), offset=4
    var sub = row.slice(axes=[0], starts=[1], ends=[3])
    var loss = sub.sum()
    loss.backward()
    var grad = x.grad().detach(share=True)
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([0, 0, 0, 0])))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0, 1, 1, 0])))
    assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0, 0, 0, 0])))

# Test: slice on transposed view
def test_view_transposed_slice() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var t = x.transpose()  # shape (3,3), strides (1,3), offset=0
    var col = t[i(1), s()]  # shape (3,), strides (3,), offset=1
    var first_two = col.slice(axis=0, start=0, end=2)
    assert_true(first_two[0] == 2.0)  # x[0,1]
    assert_true(first_two[1] == 5.0)  # x[1,1]

# Test: Non-contiguous view slice (column access)
def test_view_col_slice() raises:
    var x = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var col1 = x[s(), i(1)]  # shape (3,), strides (3,), offset=1
    var first_two = col1.slice(axis=0, start=0, end=2)
    assert_true(first_two[0] == 2.0)
    assert_true(first_two[1] == 5.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
