from tensor import Tensor
from std.testing import assert_true
from std.sys import has_accelerator
from shapes import Shape
from device import CPU, GPU

# ── CPU Tests ─────────────────────────────────────────────────────────────────


fn test_onehot_cpu_1d_basic() raises:
    print("test_onehot_cpu_1d_basic")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0])
    var result = Tensor[dtype].onehot(indices, 3)
    assert_true(
        result.all_close(
            Tensor[dtype].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
    )


fn test_onehot_cpu_1d_single_class() raises:
    print("test_onehot_cpu_1d_single_class")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 0.0, 0.0])
    var result = Tensor[dtype].onehot(indices, 1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0], [1.0], [1.0]])))


fn test_onehot_cpu_1d_first_class() raises:
    print("test_onehot_cpu_1d_first_class")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 0.0, 0.0])
    var result = Tensor[dtype].onehot(indices, 4)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            )
        )
    )


fn test_onehot_cpu_1d_last_class() raises:
    print("test_onehot_cpu_1d_last_class")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([3.0, 3.0, 3.0])
    var result = Tensor[dtype].onehot(indices, 4)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
            )
        )
    )


fn test_onehot_cpu_1d_mixed() raises:
    print("test_onehot_cpu_1d_mixed")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([2.0, 0.0, 1.0, 3.0])
    var result = Tensor[dtype].onehot(indices, 4)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    )


fn test_onehot_cpu_2d_basic() raises:
    print("test_onehot_cpu_2d_basic")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d2([[0.0, 2.0], [1.0, 3.0]])
    var result = Tensor[dtype].onehot(indices, 4)
    # Shape should be (2, 2, 4)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                ]
            )
        )
    )


fn test_onehot_cpu_2d_single_row() raises:
    print("test_onehot_cpu_2d_single_row")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d2([[0.0, 1.0, 2.0]])
    var result = Tensor[dtype].onehot(indices, 3)
    assert_true(
        result.all_close(
            Tensor[dtype].d3(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
            )
        )
    )


fn test_onehot_cpu_1d_large_num_classes() raises:
    print("test_onehot_cpu_1d_large_num_classes")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 4.0, 9.0])
    var result = Tensor[dtype].onehot(indices, 10)
    # Spot check specific positions
    assert_true(result[[0, 0]] == Scalar[dtype](1.0))
    assert_true(result[[0, 1]] == Scalar[dtype](0.0))
    assert_true(result[[1, 4]] == Scalar[dtype](1.0))
    assert_true(result[[1, 3]] == Scalar[dtype](0.0))
    assert_true(result[[2, 9]] == Scalar[dtype](1.0))
    assert_true(result[[2, 8]] == Scalar[dtype](0.0))


fn test_onehot_cpu_shape() raises:
    print("test_onehot_cpu_shape")
    comptime dtype = DType.float32
    # Verify output shape is correct
    var indices = Tensor[dtype].d2([[0.0, 1.0], [2.0, 0.0], [1.0, 2.0]])
    var result = Tensor[dtype].onehot(indices, 5)
    # Input shape (3, 2) → output shape (3, 2, 5)
    assert_true(result.shape() == Shape(3, 2, 5))


fn test_onehot_cpu_all_zeros_except_one() raises:
    print("test_onehot_cpu_all_zeros_except_one")
    comptime dtype = DType.float32
    # Each row must sum to exactly 1.0
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0])
    var result = Tensor[dtype].onehot(indices, 4)
    for i in range(4):
        var row_sum = Scalar[dtype](0)
        for j in range(4):
            row_sum += result[[i, j]]
        assert_true(row_sum == Scalar[dtype](1.0))


fn test_onehot_cpu_explicit_cpu_device() raises:
    print("test_onehot_cpu_explicit_cpu_device")
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0])
    var result = Tensor[dtype].onehot(indices, 3, CPU().into())
    assert_true(result.is_on_cpu())
    assert_true(
        result.all_close(
            Tensor[dtype].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
    )


# ── GPU Tests ─────────────────────────────────────────────────────────────────


fn test_onehot_gpu_1d_basic() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_1d_basic")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 3)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )
        )


fn test_onehot_gpu_1d_mixed() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_1d_mixed")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([2.0, 0.0, 1.0, 3.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
        )


fn test_onehot_gpu_2d_basic() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_2d_basic")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d2([[0.0, 2.0], [1.0, 3.0]]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ],
                        [
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ]
                )
            )
        )


fn test_onehot_gpu_first_class() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_first_class")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 0.0, 0.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        )


fn test_onehot_gpu_last_class() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_last_class")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([3.0, 3.0, 3.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
        )


fn test_onehot_gpu_shape() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_shape")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d2(
            [[0.0, 1.0], [2.0, 0.0], [1.0, 2.0]]
        ).to_gpu()
        var result = Tensor[dtype].onehot(indices, 5)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 2, 5))


fn test_onehot_gpu_all_zeros_except_one() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_all_zeros_except_one")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        var result_cpu = result.to_cpu()
        # Each row must sum to exactly 1.0
        for i in range(4):
            var row_sum = Scalar[dtype](0)
            for j in range(4):
                row_sum += result_cpu[[i, j]]
            assert_true(row_sum == Scalar[dtype](1.0))


fn test_onehot_gpu_large_num_classes() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_large_num_classes")
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 4.0, 9.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 10)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu[[0, 0]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[0, 1]] == Scalar[dtype](0.0))
        assert_true(result_cpu[[1, 4]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[1, 3]] == Scalar[dtype](0.0))
        assert_true(result_cpu[[2, 9]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[2, 8]] == Scalar[dtype](0.0))


fn test_onehot_gpu_explicit_device() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_explicit_device")
        comptime dtype = DType.float32
        # CPU indices but explicit GPU device
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0])
        var result = Tensor[dtype].onehot(indices, 3, GPU().into())
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )
        )


fn test_onehot_gpu_override_to_cpu() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_override_to_cpu")
        comptime dtype = DType.float32
        # GPU indices but force CPU result
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 3, CPU().into())
        assert_true(result.is_on_cpu())
        assert_true(
            result.all_close(
                Tensor[dtype].d2(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )
        )


# ── CPU/GPU Parity ────────────────────────────────────────────────────────────


fn test_onehot_gpu_parity_1d() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_parity_1d")
        comptime dtype = DType.float32
        var indices_cpu = Tensor[dtype].d1([0.0, 2.0, 1.0, 3.0])
        var indices_gpu = indices_cpu.to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, 4)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, 4)
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


fn test_onehot_gpu_parity_2d() raises:
    comptime if has_accelerator():
        print("test_onehot_gpu_parity_2d")
        comptime dtype = DType.float32
        var indices_cpu = Tensor[dtype].d2([[0.0, 2.0], [1.0, 3.0]])
        var indices_gpu = indices_cpu.to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, 4)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, 4)
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


# ── Main ──────────────────────────────────────────────────────────────────────


fn main() raises:
    # CPU tests
    test_onehot_cpu_1d_basic()
    test_onehot_cpu_1d_single_class()
    test_onehot_cpu_1d_first_class()
    test_onehot_cpu_1d_last_class()
    test_onehot_cpu_1d_mixed()
    test_onehot_cpu_2d_basic()
    test_onehot_cpu_2d_single_row()
    test_onehot_cpu_1d_large_num_classes()
    test_onehot_cpu_shape()
    test_onehot_cpu_all_zeros_except_one()
    test_onehot_cpu_explicit_cpu_device()

    # GPU tests
    test_onehot_gpu_1d_basic()
    test_onehot_gpu_1d_mixed()
    test_onehot_gpu_2d_basic()
    test_onehot_gpu_first_class()
    test_onehot_gpu_last_class()
    test_onehot_gpu_shape()
    test_onehot_gpu_all_zeros_except_one()
    test_onehot_gpu_large_num_classes()
    test_onehot_gpu_explicit_device()
    test_onehot_gpu_override_to_cpu()

    # Parity tests
    test_onehot_gpu_parity_1d()
    test_onehot_gpu_parity_2d()

    print("All onehot tests passed!")
