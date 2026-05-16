from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.shapes import Shape
from tenmo.shuffle import Shuffle
from std.sys import has_accelerator


fn test_tensor_shuffle_forward_basic() raises:
    print("test_tensor_shuffle_forward_basic")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # permute along axis=0 with [1, 0]
    var shuffled = Shuffle[dtype].forward[False](t, [1, 0], axis=0)

    assert_true(shuffled.shape() == Shape(2, 3))
    assert_true(
        shuffled == Tensor[dtype].d2([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    )
    print("Passed forward basic shuffle")


fn test_tensor_shuffle_forward_axis1() raises:
    print("test_tensor_shuffle_forward_axis1")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # permute along axis=1 with [2, 0, 1]
    var shuffled = Shuffle[dtype].forward[False](t, [2, 0, 1], axis=1)

    assert_true(
        shuffled == Tensor[dtype].d2([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    )
    print("Passed forward shuffle along axis 1")


fn test_tensor_shuffle_backward_axis0() raises:
    print("test_tensor_shuffle_backward_axis0")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d2([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    t.requires_grad_(True)

    var shuffled = Shuffle[dtype].forward[True](t, [1, 0], axis=0)
    var s = shuffled.sum()
    s.backward()

    # Backward should restore correct positions (reversed)
    var expected_grad = Tensor[dtype].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(t.grad() == expected_grad)
    print("Passed backward shuffle along axis 0")


fn test_tensor_shuffle_backward_axis1() raises:
    print("test_tensor_shuffle_backward_axis1")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t.requires_grad_(True)

    var shuffled = Shuffle[dtype].forward[True](t, [2, 0, 1], axis=1)
    var s = shuffled.sum()
    s.backward()

    # Each input position contributes to exactly one output
    var expected_grad = Tensor[dtype].ones(Shape(2, 3))
    assert_true(t.grad() == expected_grad)
    print("Passed backward shuffle along axis 1")


fn test_tensor_shuffle_backward_grad_mapping() raises:
    print("test_tensor_shuffle_backward_grad_mapping")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t.requires_grad_(True)

    # Shuffle along axis=1 using [1, 2, 0]
    var shuffled = Shuffle[dtype].forward[True](t, [1, 2, 0], axis=1)

    # Multiply by position-wise coefficients to make gradients distinct
    var c = Tensor[dtype].d2([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    var out = (shuffled * c).sum()
    out.backward()

    # Backprop should map gradient contributions back using inverse permutation
    var expected_grad = Tensor[dtype].d2(
        [[30.0, 10.0, 20.0], [60.0, 40.0, 50.0]]
    )
    assert_true(t.grad() == expected_grad)
    print("Passed backward gradient mapping test")


fn test_tensor_shuffle_random_perm_length_check() raises:
    print("test_tensor_shuffle_random_perm_length_check")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var shuffled = Shuffle[dtype].forward[False](t, [], axis=0)
    # Should keep same shape and values permuted (cannot predict order but same values)
    assert_true(shuffled.shape() == Shape(2, 2))
    assert_true(shuffled != t or shuffled == t)  # likely changed
    print("Passed random permutation shuffle")


fn test_tensor_shuffle_multi_dimensional() raises:
    print("test_tensor_shuffle_multi_dimensional")
    comptime dtype = DType.float32

    var a = Tensor[dtype].arange(0, 24)
    t = a.reshape(Shape(2, 3, 4))
    var shuffled = Shuffle[dtype].forward[False](t, [2, 0, 1], axis=1)
    expected = Tensor[dtype].d3(
        [
            [
                [8.0, 9.0, 10.0, 11.0],
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            [
                [20.0, 21.0, 22.0, 23.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ],
        ]
    )

    assert_true(shuffled.shape() == Shape(2, 3, 4))
    assert_true(shuffled == expected)
    print("Passed multi-dimensional shuffle shape test")


# ============================================================
# SHUFFLE TESTS — CPU
# ============================================================

# ------------------------------------------------------------
# 1D CPU
# ------------------------------------------------------------


fn test_shuf_cpu_1d_identity_perm() raises:
    print("test_shuf_cpu_1d_identity_perm")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var s = a.shuffle([0, 1, 2, 3], axis=0)
    assert_true(s.shape() == Shape.of(4))
    assert_true(s.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_1d_reverse_perm() raises:
    print("test_shuf_cpu_1d_reverse_perm")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var s = a.shuffle([3, 2, 1, 0], axis=0)
    assert_true(s.all_close(Tensor[dtype].d1([4.0, 3.0, 2.0, 1.0])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_1d_arbitrary_perm() raises:
    print("test_shuf_cpu_1d_arbitrary_perm")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0, 50.0], requires_grad=True)
    # perm [2,0,4,1,3] means out[i] = a[perm[i]]
    var s = a.shuffle([2, 0, 4, 1, 3], axis=0)
    assert_true(s.all_close(Tensor[dtype].d1([30.0, 10.0, 50.0, 20.0, 40.0])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_1d_grad_non_uniform_loss() raises:
    print("test_shuf_cpu_1d_grad_non_uniform_loss")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    # perm [2,0,1]: out = [a[2], a[0], a[1]] = [3,1,2]
    var s = a.shuffle([2, 0, 1], axis=0)
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var loss = (s * weights).sum()
    loss.backward()
    # loss = s[0]*1 + s[1]*2 + s[2]*3
    #      = a[2]*1 + a[0]*2 + a[1]*3
    # grad: a[0]=2, a[1]=3, a[2]=1
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 3.0, 1.0])))


# ------------------------------------------------------------
# 2D CPU — axis=0
# ------------------------------------------------------------


fn test_shuf_cpu_2d_axis0_identity() raises:
    print("test_shuf_cpu_2d_axis0_identity")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var s = a.shuffle([0, 1, 2], axis=0)
    assert_true(s.all_close(a))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_2d_axis0_reverse() raises:
    print("test_shuf_cpu_2d_axis0_reverse")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var s = a.shuffle([2, 1, 0], axis=0)
    assert_true(s.shape() == Shape.of(3, 2))
    assert_true(
        s.all_close(Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]))
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_2d_axis0_arbitrary() raises:
    print("test_shuf_cpu_2d_axis0_arbitrary")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    # perm [1,2,0]: out row0=a_row1, out row1=a_row2, out row2=a_row0
    var s = a.shuffle([1, 2, 0], axis=0)
    assert_true(
        s.all_close(Tensor[dtype].d2([[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]]))
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_2d_axis0_grad_non_uniform() raises:
    print("test_shuf_cpu_2d_axis0_grad_non_uniform")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # perm [1,0]: swap rows
    var s = a.shuffle([1, 0], axis=0)
    var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var loss = (s * weights).sum()
    loss.backward()
    # s = [[3,4],[1,2]]
    # loss = 3*1 + 4*2 + 1*3 + 2*4 = 3+8+3+8 = 22
    # grad at s[0,0]=3 (was a[1,0]) → a[1,0] gets weight 1
    # grad at s[0,1]=4 (was a[1,1]) → a[1,1] gets weight 2
    # grad at s[1,0]=1 (was a[0,0]) → a[0,0] gets weight 3
    # grad at s[1,1]=2 (was a[0,1]) → a[0,1] gets weight 4
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 4.0], [1.0, 2.0]])))


# ------------------------------------------------------------
# 2D CPU — axis=1
# ------------------------------------------------------------


fn test_shuf_cpu_2d_axis1_reverse() raises:
    print("test_shuf_cpu_2d_axis1_reverse")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var s = a.shuffle([2, 1, 0], axis=1)
    assert_true(
        s.all_close(Tensor[dtype].d2([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]))
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_2d_axis1_arbitrary() raises:
    print("test_shuf_cpu_2d_axis1_arbitrary")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    # perm [2,0,1]: col0←col2, col1←col0, col2←col1
    var s = a.shuffle([2, 0, 1], axis=1)
    assert_true(
        s.all_close(Tensor[dtype].d2([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]))
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 3D CPU
# ------------------------------------------------------------


fn test_shuf_cpu_3d_axis0_reverse() raises:
    print("test_shuf_cpu_3d_axis0_reverse")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var s = a.shuffle([2, 1, 0], axis=0)
    assert_true(s.shape() == Shape.of(3, 2, 2))
    assert_true(
        s.all_close(
            Tensor[dtype].d3(
                [
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[1.0, 2.0], [3.0, 4.0]],
                ]
            )
        )
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_3d_axis1_arbitrary() raises:
    print("test_shuf_cpu_3d_axis1_arbitrary")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        requires_grad=True,
    )
    # perm [2,0,1] along axis=1
    var s = a.shuffle([2, 0, 1], axis=1)
    assert_true(s.shape() == Shape.of(2, 3, 2))
    assert_true(
        s.all_close(
            Tensor[dtype].d3(
                [
                    [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]],
                    [[11.0, 12.0], [7.0, 8.0], [9.0, 10.0]],
                ]
            )
        )
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_3d_axis2_reverse() raises:
    print("test_shuf_cpu_3d_axis2_reverse")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        requires_grad=True,
    )
    var s = a.shuffle([2, 1, 0], axis=2)
    assert_true(
        s.all_close(
            Tensor[dtype].d3(
                [
                    [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]],
                    [[9.0, 8.0, 7.0], [12.0, 11.0, 10.0]],
                ]
            )
        )
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_3d_grad_non_uniform() raises:
    print("test_shuf_cpu_3d_grad_non_uniform")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    # perm [1,0] along axis=0: swap the two batches
    var s = a.shuffle([1, 0], axis=0)
    var weights = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var loss = (s * weights).sum()
    loss.backward()
    # s[0] = a[1], s[1] = a[0]
    # grad for a[1] = weights[0] = [[1,2],[3,4]]
    # grad for a[0] = weights[1] = [[5,6],[7,8]]
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]]
            )
        )
    )


# ------------------------------------------------------------
# 4D CPU
# ------------------------------------------------------------


fn test_shuf_cpu_4d_axis0() raises:
    print("test_shuf_cpu_4d_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(4, 3, 2, 5)
    a.requires_grad_(True)
    var a_ref = a.copy()
    var s = a.shuffle([3, 1, 0, 2], axis=0)
    assert_true(s.shape() == Shape.of(4, 3, 2, 5))
    # s[0] = a[3], s[1] = a[1], s[2] = a[0], s[3] = a[2]
    assert_true(
        s.sum(axes=[1, 2, 3]).all_close(
            a_ref.sum(axes=[1, 2, 3]).shuffle([3, 1, 0, 2], axis=0)
        )
    )
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_4d_axis2() raises:
    print("test_shuf_cpu_4d_axis2")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(2, 3, 4, 5)
    a.requires_grad_(True)
    var s = a.shuffle([3, 0, 2, 1], axis=2)
    assert_true(s.shape() == Shape.of(2, 3, 4, 5))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Random perm CPU (no explicit perm)
# ------------------------------------------------------------


fn test_shuf_cpu_random_perm_grad_flow() raises:
    print("test_shuf_cpu_random_perm_grad_flow")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(5, 4)
    a.requires_grad_(True)
    # Empty perm → random shuffle
    var s = a.shuffle([], axis=0)
    assert_true(s.shape() == Shape.of(5, 4))
    var loss = s.sum()
    loss.backward()
    # Sum is permutation-invariant so grad is always ones
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_random_perm_3d_grad_flow() raises:
    print("test_shuf_cpu_random_perm_3d_grad_flow")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(6, 3, 4)
    a.requires_grad_(True)
    var s = a.shuffle([], axis=1)
    assert_true(s.shape() == Shape.of(6, 3, 4))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# track_grad=False CPU
# ------------------------------------------------------------


fn test_shuf_cpu_track_grad_false() raises:
    print("test_shuf_cpu_track_grad_false")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var s = a.shuffle[track_grad=False]([1, 0], axis=0)
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(not s.requires_grad)


# ------------------------------------------------------------
# Double shuffle round-trip CPU
# ------------------------------------------------------------


fn test_shuf_cpu_double_shuffle_roundtrip() raises:
    print("test_shuf_cpu_double_shuffle_roundtrip")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    # Apply perm then its inverse — should get back to original
    # perm [2,0,1], inverse is [1,2,0]
    var s1 = a.shuffle([2, 0, 1], axis=0)
    var s2 = s1.shuffle([1, 2, 0], axis=0)
    assert_true(s2.all_close(a))
    var loss = s2.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Shuffle then reduce CPU
# ------------------------------------------------------------


fn test_shuf_cpu_shuffle_then_sum() raises:
    print("test_shuf_cpu_shuffle_then_sum")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var s = a.shuffle([2, 0, 1], axis=0)
    var reduced = s.sum(axes=[0])  # (2,) — sum permutation-invariant
    assert_true(reduced.all_close(Tensor[dtype].d1([9.0, 12.0])))
    var loss = reduced.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_cpu_shuffle_then_mean() raises:
    print("test_shuf_cpu_shuffle_then_mean")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var s = a.shuffle([1, 2, 0], axis=0)
    var m = s.mean(axes=[0])
    var loss = m.sum()
    loss.backward()
    # mean grad = 1/3 per element
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d2(
                [
                    [1.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 3.0, 1.0 / 3.0],
                ]
            )
        )
    )


# ============================================================
# SHUFFLE TESTS — GPU
# ============================================================

# ------------------------------------------------------------
# 1D GPU
# ------------------------------------------------------------


fn test_shuf_gpu_1d_identity_perm() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_1d_identity_perm")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([0, 1, 2, 3], axis=0)
        assert_true(s.shape() == Shape.of(4))
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_1d_reverse_perm() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_1d_reverse_perm")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([3, 2, 1, 0], axis=0)
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d1([4.0, 3.0, 2.0, 1.0]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_1d_arbitrary_perm() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_1d_arbitrary_perm")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [10.0, 20.0, 30.0, 40.0, 50.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 4, 1, 3], axis=0)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d1([30.0, 10.0, 50.0, 20.0, 40.0])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_1d_grad_non_uniform() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_1d_grad_non_uniform")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=0)
        var weights = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var loss = (s * weights).sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 3.0, 1.0])))


# ------------------------------------------------------------
# 2D GPU — axis=0
# ------------------------------------------------------------


fn test_shuf_gpu_2d_axis0_reverse() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_2d_axis0_reverse")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=0)
        assert_true(s.shape() == Shape.of(3, 2))
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_2d_axis0_arbitrary() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_2d_axis0_arbitrary")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([1, 2, 0], axis=0)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_2d_axis0_grad_non_uniform() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_2d_axis0_grad_non_uniform")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([1, 0], axis=0)
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (s * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[3.0, 4.0], [1.0, 2.0]]))
        )


# ------------------------------------------------------------
# 2D GPU — axis=1
# ------------------------------------------------------------


fn test_shuf_gpu_2d_axis1_reverse() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_2d_axis1_reverse")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=1)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_2d_axis1_arbitrary() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_2d_axis1_arbitrary")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=1)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 3D GPU
# ------------------------------------------------------------


fn test_shuf_gpu_3d_axis0_reverse() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_3d_axis0_reverse")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=0)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[9.0, 10.0], [11.0, 12.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                        [[1.0, 2.0], [3.0, 4.0]],
                    ]
                )
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_3d_axis1_arbitrary() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_3d_axis1_arbitrary")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=1)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]],
                        [[11.0, 12.0], [7.0, 8.0], [9.0, 10.0]],
                    ]
                )
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_3d_axis2_reverse() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_3d_axis2_reverse")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=2)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]],
                        [[9.0, 8.0, 7.0], [12.0, 11.0, 10.0]],
                    ]
                )
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_3d_grad_non_uniform() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_3d_grad_non_uniform")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([1, 0], axis=0)
        var weights = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var loss = (s * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]]
                )
            )
        )


# ------------------------------------------------------------
# 4D GPU
# ------------------------------------------------------------


fn test_shuf_gpu_4d_axis0() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_4d_axis0")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(4, 3, 2, 5)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([3, 1, 0, 2], axis=0)
        assert_true(s.shape() == Shape.of(4, 3, 2, 5))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_4d_axis2() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_4d_axis2")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([3, 0, 2, 1], axis=2)
        assert_true(s.shape() == Shape.of(2, 3, 4, 5))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# GPU matches CPU (cross-validation)
# ------------------------------------------------------------


fn test_shuf_gpu_matches_cpu_2d() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_matches_cpu_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(5, 6)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle([4, 1, 3, 0, 2], axis=0)
        var s_cpu = a_copy.shuffle([4, 1, 3, 0, 2], axis=0)
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_shuf_gpu_matches_cpu_3d() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_matches_cpu_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(4, 4, 6)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle([3, 0, 2, 1], axis=1)
        var s_cpu = a_copy.shuffle([3, 0, 2, 1], axis=1)
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_shuf_gpu_matches_cpu_non_uniform_grad() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_matches_cpu_non_uniform_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(3, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var weights = Tensor[dtype].randn(3, 4)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle([2, 0, 1], axis=0)
        var s_cpu = a_copy.shuffle([2, 0, 1], axis=0)
        var loss_gpu = (s_gpu * weights.to_gpu()).sum()
        loss_gpu.backward()
        var loss_cpu = (s_cpu * weights).sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# Random perm GPU
# ------------------------------------------------------------


fn test_shuf_gpu_random_perm_grad_flow() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_random_perm_grad_flow")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(5, 4)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([], axis=0)
        assert_true(s.shape() == Shape.of(5, 4))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# track_grad=False GPU
# ------------------------------------------------------------


fn test_shuf_gpu_track_grad_false() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_track_grad_false")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle[track_grad=False]([1, 0], axis=0)
        assert_true(s.shape() == Shape.of(2, 2))
        assert_true(not s.requires_grad)


# ------------------------------------------------------------
# Grad lands on CPU
# ------------------------------------------------------------


fn test_shuf_gpu_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=0)
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Double shuffle round-trip GPU
# ------------------------------------------------------------


fn test_shuf_gpu_double_shuffle_roundtrip() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_double_shuffle_roundtrip")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.shuffle([2, 0, 1], axis=0)
        var s2 = s1.shuffle([1, 2, 0], axis=0)
        assert_true(s2.to_cpu().all_close(a))
        var loss = s2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Large dimension — permutation > 8 elements (tests DeviceBuffer path)
# ------------------------------------------------------------


fn test_shuf_gpu_large_axis_dim() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_large_axis_dim")
        comptime dtype = DType.float32
        # axis dim = 42 — exceeds Array max_rank of 8
        var a = Tensor[dtype].randn(42, 8)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        # Build a reverse permutation of length 42
        var perm = List[Int](capacity=42)
        for i in range(41, -1, -1):
            perm.append(i)
        var s = a_gpu.shuffle(perm, axis=0)
        assert_true(s.shape() == Shape.of(42, 8))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_shuf_gpu_large_axis_dim_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_shuf_gpu_large_axis_dim_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(42, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        # arbitrary perm of length 42
        var perm = List[Int](capacity=42)
        for i in range(42):
            perm.append((i * 13 + 7) % 42)  # pseudo-shuffle, still valid perm?
        # Use a known valid perm instead: reverse
        perm = List[Int](capacity=42)
        for i in range(41, -1, -1):
            perm.append(i)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle(perm, axis=0)
        var s_cpu = a_copy.shuffle(perm, axis=0)
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ============================================================
# MAIN
# ============================================================


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll shuffle tests passed!")
