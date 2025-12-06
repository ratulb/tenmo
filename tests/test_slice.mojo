from tenmo import Tensor
from testing import assert_true

fn test_tensor_slice_1d_basic() raises:
    print("test_tensor_slice_1d_basic")
    var x = Tensor.d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Test basic slicing: equivalent to x[2:7]
    var slice1 = x.slice(axis=0, start=2, end=7)
    assert_true(slice1 == Tensor.d1([2, 3, 4, 5, 6]))

    # Test with step: equivalent to x[1:8:2]
    var slice2 = x.slice(axis=0, start=1, end=8, step=2)
    assert_true(slice2 == Tensor.d1([1, 3, 5, 7]))

    print("✓ Passed 1D basic slicing tests")

fn test_tensor_slice_1d_negative_indices() raises:
    print("test_tensor_slice_1d_negative_indices")
    var x = Tensor.d1([0, 1, 2, 3, 4, 5])

    # Test negative indices: equivalent to x[-3:-1]
    var slice1 = x.slice(axis=0, start=-3, end=-1)
    assert_true(slice1 == Tensor.d1([3, 4]))

    # Test mixed positive/negative: equivalent to x[1:-1]
    var slice2 = x.slice(axis=0, start=1, end=-1)
    assert_true(slice2 == Tensor.d1([1, 2, 3, 4]))

    print("✓ Passed 1D negative indices slicing tests")

fn test_tensor_slice_1d_step_sizes() raises:
    print("test_tensor_slice_1d_step_sizes")
    var x = Tensor.d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Test step size 2: equivalent to x[0:10:2]
    var slice1 = x.slice(axis=0, start=0, end=10, step=2)
    assert_true(slice1 == Tensor.d1([0, 2, 4, 6, 8]))

    # Test step size 3: equivalent to x[0:10:3]
    var slice2 = x.slice(axis=0, start=0, end=10, step=3)
    assert_true(slice2 == Tensor.d1([0, 3, 6, 9]))

    print("✓ Passed 1D step size slicing tests")

fn test_tensor_slice_2d_rows() raises:
    print("test_tensor_slice_2d_rows")
    var x = Tensor.d2([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

    # Slice rows: equivalent to x[1:3, :]
    var rows_slice = x.slice(axis=0, start=1, end=3)
    var expected_rows = Tensor.d2([[5, 6, 7, 8],
                                   [9, 10, 11, 12]])
    assert_true(rows_slice == expected_rows)

    # Slice single row: equivalent to x[0:1, :]
    var single_row = x.slice(axis=0, start=0, end=1)
    var expected_single = Tensor.d2([[1, 2, 3, 4]])
    assert_true(single_row == expected_single)

    print("✓ Passed 2D row slicing tests")

fn test_tensor_slice_2d_columns() raises:
    print("test_tensor_slice_2d_columns")
    var x = Tensor.d2([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

    # Slice columns: equivalent to x[:, 1:3]
    var cols_slice = x.slice(axis=1, start=1, end=3)
    var expected_cols = Tensor.d2([[2, 3],
                                   [6, 7],
                                   [10, 11]])
    assert_true(cols_slice == expected_cols)

    # Slice single column: equivalent to x[:, 2:3]
    var single_col = x.slice(axis=1, start=2, end=3)
    var expected_single_col = Tensor.d2([[3],
                                         [7],
                                         [11]])
    assert_true(single_col == expected_single_col)

    print("✓ Passed 2D column slicing tests")

fn test_tensor_slice_2d_both_dims() raises:
    print("test_tensor_slice_2d_both_dims")
    var x = Tensor.d2([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

    # Slice both dimensions: equivalent to x[0:2, 1:3]
    var slice1 = x.slice(axis=0, start=0, end=2)
    slice1 = slice1.slice(axis=1, start=1, end=3)
    var expected = Tensor.d2([[2, 3],
                              [6, 7]])
    assert_true(slice1 == expected)

    print("✓ Passed 2D both dimensions slicing tests")

fn test_tensor_slice_3d_basic() raises:
    print("test_tensor_slice_3d_basic")
    # Create 3D tensor with known values
    var x = Tensor.d3([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]]
    ])

    # Slice along dimension 0
    var slice_dim0 = x.slice(axis=0, start=0, end=2)
    var expected_dim0 = Tensor.d3([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ])
    assert_true(slice_dim0 == expected_dim0)

    # Slice along dimension 1
    var slice_dim1 = x.slice(axis=1, start=0, end=1)
    var expected_dim1 = Tensor.d3([
        [[1, 2, 3]],
        [[7, 8, 9]],
        [[13, 14, 15]]
    ])
    assert_true(slice_dim1 == expected_dim1)

    # Slice along dimension 2
    var slice_dim2 = x.slice(axis=2, start=0, end=2)
    var expected_dim2 = Tensor.d3([
        [[1, 2], [4, 5]],
        [[7, 8], [10, 11]],
        [[13, 14], [16, 17]]
    ])
    assert_true(slice_dim2 == expected_dim2)

    print("✓ Passed 3D basic slicing tests")

fn test_tensor_slice_with_gradients() raises:
    print("test_tensor_slice_with_gradients")
    var x = Tensor.d2([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)

    # Slice and use in computation
    var sliced = x.slice(axis=1, start=0, end=2)  # First two columns
    var loss = sliced.sum()
    loss.backward()

    # Gradients should flow only to sliced elements
    var expected_grad = Tensor.d2([[1.0, 1.0, 0.0],
                                   [1.0, 1.0, 0.0]])
    assert_true(x.grad().all_close(expected_grad))

    print("✓ Passed slicing with gradients test")

fn test_tensor_nested_slice_with_gradients() raises:
    print("test_tensor_nested_slice_with_gradients")
    var x = Tensor.d2([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)

    # Slice and use in computation
    var sliced = x.slice(axis=1, start=0, end=2)  # First two columns
    var nested = sliced.slice(axis=0, start=1, end=2)
    nested.sum().backward(42)


    # Gradients should flow only to sliced elements
    var expected_grad = Tensor.d2([[0.0, 0.0, 0.0],
                                   [42.0, 42.0, 0.0]])
    assert_true(x.grad().all_close(expected_grad))

    print("✓ Passed slicing with gradients test")


fn test_tensor_slice_edge_cases() raises:
    print("test_tensor_slice_edge_cases")
    var x = Tensor.d1([1, 2, 3, 4, 5])

    # Empty slice - start and end are not allowed to be same
    #var empty_slice = x.slice(axis=0, start=3, end=3)
    #assert_true(empty_slice == Tensor.d1([]))

    # Single element slice
    var single_slice = x.slice(axis=0, start=2, end=3)
    assert_true(single_slice == Tensor.d1([3]))

    # Full slice
    var full_slice = x.slice(axis=0, start=0, end=5)
    assert_true(full_slice == x)

    print("✓ Passed edge cases slicing tests")

fn test_tensor_slice_step_edge_cases() raises:
    print("test_tensor_slice_step_edge_cases")
    var x = Tensor.d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Step larger than slice
    var large_step = x.slice(axis=0, start=0, end=3, step=5)
    assert_true(large_step == Tensor.d1([0]))

    # Negative step (reverse)
    var reverse_slice = x.slice(axis=0, start=5, end=0, step=-1)
    assert_true(reverse_slice == Tensor.d1([5, 4, 3, 2, 1]))

    print("✓ Passed step edge cases slicing tests")

# Consolidated test function
fn main() raises:
    print("Running comprehensive tensor slice tests...")
    test_tensor_slice_1d_basic()
    test_tensor_slice_1d_negative_indices()
    test_tensor_slice_1d_step_sizes()
    test_tensor_slice_2d_rows()
    test_tensor_slice_2d_columns()
    test_tensor_slice_2d_both_dims()
    test_tensor_slice_3d_basic()
    test_tensor_slice_with_gradients()
    test_tensor_nested_slice_with_gradients()
    test_tensor_slice_edge_cases()
    test_tensor_slice_step_edge_cases()
    print("All tensor slice tests passed! ✓")

