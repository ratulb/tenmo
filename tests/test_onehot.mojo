from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.shapes import Shape
from tenmo.device import CPU, GPU

# ── CPU Tests ─────────────────────────────────────────────────────────────────


def test_onehot_cpu_1d_basic() raises:
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0])
    var result = Tensor[dtype].onehot(indices, 3)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
        )
    )


def test_onehot_cpu_1d_single_class() raises:
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 0.0, 0.0])
    var result = Tensor[dtype].onehot(indices, 1)
    assert_true(result.all_close(Tensor[dtype].d2([[1.0], [1.0], [1.0]])))


def test_onehot_cpu_1d_first_class() raises:
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 0.0, 0.0])
    var result = Tensor[dtype].onehot(indices, 4)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    )


def test_onehot_cpu_1d_last_class() raises:
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([3.0, 3.0, 3.0])
    var result = Tensor[dtype].onehot(indices, 4)
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    )


def test_onehot_cpu_1d_mixed() raises:
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


def test_onehot_cpu_2d_basic() raises:
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


def test_onehot_cpu_2d_single_row() raises:
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


def test_onehot_cpu_1d_large_num_classes() raises:
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


def test_onehot_cpu_shape() raises:
    comptime dtype = DType.float32
    # Verify output shape is correct
    var indices = Tensor[dtype].d2([[0.0, 1.0], [2.0, 0.0], [1.0, 2.0]])
    var result = Tensor[dtype].onehot(indices, 5)
    # Input shape (3, 2) → output shape (3, 2, 5)
    assert_true(result.shape() == Shape(3, 2, 5))


def test_onehot_cpu_all_zeros_except_one() raises:
    comptime dtype = DType.float32
    # Each row must sum to exactly 1.0
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0])
    var result = Tensor[dtype].onehot(indices, 4)
    for i in range(4):
        var row_sum = Scalar[dtype](0)
        for j in range(4):
            row_sum += result[[i, j]]
        assert_true(row_sum == Scalar[dtype](1.0))


def test_onehot_cpu_explicit_cpu_device() raises:
    comptime dtype = DType.float32
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0])
    var result = Tensor[dtype].onehot(indices, 3, CPU().into())
    assert_true(result.is_on_cpu())
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
        )
    )


# ── GPU Tests ─────────────────────────────────────────────────────────────────


def test_onehot_gpu_1d_basic() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_1d_mixed() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_2d_basic() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_first_class() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_last_class() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = (
            Tensor[dtype].d2([[0.0, 1.0], [2.0, 0.0], [1.0, 2.0]]).to_gpu()
        )
        var result = Tensor[dtype].onehot(indices, 5)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 2, 5))


def test_onehot_gpu_all_zeros_except_one() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_large_num_classes() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_explicit_device() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_override_to_cpu() raises:
    comptime if has_accelerator():
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


def test_onehot_gpu_parity_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices_cpu = Tensor[dtype].d1([0.0, 2.0, 1.0, 3.0])
        var indices_gpu = indices_cpu.to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, 4)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, 4)
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


def test_onehot_gpu_parity_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices_cpu = Tensor[dtype].d2([[0.0, 2.0], [1.0, 3.0]])
        var indices_gpu = indices_cpu.to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, 4)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, 4)
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


# ── Large-Size Tests ────────────────────────────────────────────────────────────


def test_onehot_cpu_large_1d() raises:
    comptime dtype = DType.float32
    var N: Int = 1000
    var C: Int = 100
    var indices_list = List[Scalar[dtype]]()
    for i in range(N):
        indices_list.append(Scalar[dtype](i % C))
    var indices = Tensor[dtype].d1(indices_list)
    var result = Tensor[dtype].onehot(indices, C)
    assert_true(result.shape() == Shape(N, C))
    # Row sums must be 1
    for i in range(N):
        var row_sum = Scalar[dtype](0)
        for j in range(C):
            row_sum += result[[i, j]]
        assert_true(row_sum == Scalar[dtype](1.0))
    # Each row has 1 at column (i % C)
    for i in range(N):
        var cls = i % C
        assert_true(result[[i, cls]] == Scalar[dtype](1.0))


def test_onehot_cpu_large_2d_batch() raises:
    comptime dtype = DType.float32
    var B: Int = 64
    var T: Int = 32
    var C: Int = 50
    var indices_data = List[Scalar[dtype]]()
    for i in range(B * T):
        indices_data.append(Scalar[dtype]((i * 7) % C))
    var indices_1d = Tensor[dtype].d1(indices_data)
    var indices_t = indices_1d.reshape(Shape(B, T))
    var result = Tensor[dtype].onehot(indices_t, C)
    assert_true(result.shape() == Shape(B, T, C))
    for b in range(B):
        for t in range(T):
            var cls = ((b * T + t) * 7) % C
            assert_true(result[[b, t, cls]] == Scalar[dtype](1.0))


def test_onehot_gpu_large_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var N: Int = 1000
        var C: Int = 100
        var indices_list = List[Scalar[dtype]]()
        for i in range(N):
            indices_list.append(Scalar[dtype](i % C))
        var indices = Tensor[dtype].d1(indices_list).to_gpu()
        var result = Tensor[dtype].onehot(indices, C)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(N, C))
        var result_cpu = result.to_cpu()
        for i in range(N):
            var row_sum = Scalar[dtype](0)
            for j in range(C):
                row_sum += result_cpu[[i, j]]
            assert_true(row_sum == Scalar[dtype](1.0))
        for i in range(N):
            var cls = i % C
            assert_true(result_cpu[[i, cls]] == Scalar[dtype](1.0))


def test_onehot_gpu_large_2d_batch() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var B: Int = 64
        var T: Int = 32
        var C: Int = 50
        var indices_data = List[Scalar[dtype]]()
        for i in range(B * T):
            indices_data.append(Scalar[dtype]((i * 7) % C))
        var indices_1d = Tensor[dtype].d1(indices_data)
        var indices_2d = indices_1d.reshape(Shape(B, T))
        var indices_gpu = indices_2d.to_gpu()
        var result = Tensor[dtype].onehot(indices_gpu, C)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(B, T, C))
        var result_cpu = result.to_cpu()
        for b in range(B):
            for t in range(T):
                var cls = ((b * T + t) * 7) % C
                assert_true(result_cpu[[b, t, cls]] == Scalar[dtype](1.0))


def test_onehot_gpu_large_parity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var N: Int = 500
        var C: Int = 80
        var indices_list = List[Scalar[dtype]]()
        for i in range(N):
            indices_list.append(Scalar[dtype]((i * 3 + 7) % C))
        var indices_cpu = Tensor[dtype].d1(indices_list)
        var indices_gpu = Tensor[dtype].d1(indices_list).to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, C)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, C)
        assert_true(result_gpu.is_on_gpu())
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
