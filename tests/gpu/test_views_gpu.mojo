from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.strides import Strides
from std.testing import (
    assert_true,
    assert_false,
    TestSuite,
)
from std.sys import has_accelerator
from tenmo.device import GPU
from tenmo.common_utils import i, newaxis, s


def test_view_into_view_2d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        assert_true(v.shape() == a.shape())
        assert_true(v.strides() == a.strides())
        assert_true(v.offset() == a.offset())
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))


def test_view_into_view_3d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
        var a3 = a.reshape(2, 3, 4)
        var a_gpu = a3.to_gpu(gpu)
        var v = a_gpu.into_view()
        assert_true(v.shape() == a3.shape())
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))


def test_view_into_view_scalar_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].scalar(42.0, requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        assert_true(v.shape() == Shape())
        v.backward(1.0)
        assert_true(a.grad().item() == 1.0)


def test_view_into_view_chain_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.into_view()
        var v2 = v1.into_view()
        var v3 = v2.into_view()
        var loss = v3.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)))


def test_view_view_reshape_2d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(1.0, 7.0, requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.view(2, 3)
        assert_true(v.shape() == Shape(2, 3))
        var v_cpu = v.to_cpu()
        assert_true(v_cpu[0, 0] == 1.0)
        assert_true(v_cpu[1, 2] == 6.0)
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))


def test_view_view_offset_1d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.view(Shape(3), offset=2)
        assert_true(v.shape() == Shape(3))
        var v_cpu = v.to_cpu()
        assert_true(v_cpu[0] == 2.0)
        assert_true(v_cpu[2] == 4.0)
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        assert_true(a.grad().all_close(expected))


def test_view_view_strides_noncontiguous_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.view(Shape(2, 2), Strides(1, 2))
        assert_false(v.is_contiguous())
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)))


def test_view_transpose_2d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.transpose()
        assert_true(v.shape() == Shape(3, 2))
        var v_cpu = v.to_cpu()
        assert_true(v_cpu[0, 0] == 1.0)
        assert_true(v_cpu[2, 1] == 6.0)
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))


def test_view_transpose_3d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
        var a3 = a.reshape(2, 3, 4)
        var a_gpu = a3.to_gpu(gpu)
        var v = a_gpu.transpose(0, 2, 1)
        assert_true(v.shape() == Shape(2, 4, 3))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))


def test_view_transpose_double_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.transpose()
        var v2 = v1.transpose()
        assert_true(v2.shape() == a.shape())
        var loss = v2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(3, 2), 1.0)))


def test_view_transpose_weighted_grad_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.transpose()
        var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var w_gpu = w.to_gpu(gpu)
        var prod = v * w_gpu
        var loss = prod.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )


def test_view_permute_2d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(1.0, 7.0, requires_grad=True)
        var a2 = a.reshape(2, 3)
        var a_gpu = a2.to_gpu(gpu)
        var v = a_gpu.permute([1, 0])
        assert_true(v.shape() == Shape(3, 2))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))


def test_view_permute_3d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
        var a3 = a.reshape(2, 3, 4)
        var a_gpu = a3.to_gpu(gpu)
        var v = a_gpu.permute([2, 0, 1])
        assert_true(v.shape() == Shape(4, 2, 3))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))


def test_view_permute_identity_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.permute([0, 1])
        assert_true(v.shape() == a.shape())
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))


def test_view_unsqueeze_2d_to_3d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.unsqueeze(0)
        assert_true(v.shape() == Shape(1, 2, 2))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))


def test_view_squeeze_3d_to_2d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.squeeze(0)
        assert_true(v.shape() == Shape(2, 2))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(1, 2, 2), 1.0)))


def test_view_unsqueeze_squeeze_chain_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.unsqueeze(1)
        var v2 = v1.squeeze(1)
        assert_true(v2.shape() == a.shape())
        var loss = v2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))


def test_view_squeeze_all_dims_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].full(Shape(1, 1, 3, 1), 5.0, requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.squeeze([])
        assert_true(v.shape() == Shape(3))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].full(Shape(1, 1, 3, 1), 1.0)
        assert_true(a.grad().all_close(expected))


def test_view_expand_1d_to_2d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.expand(4, 3)
        assert_true(v.shape() == Shape(4, 3))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


def test_view_expand_col_to_matrix_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.expand(3, 4)
        assert_true(v.shape() == Shape(3, 4))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


def test_view_expand_3d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.expand(3, 4, 2)
        assert_true(v.shape() == Shape(3, 4, 2))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[12.0, 12.0]]])))


def test_view_expand_weighted_grad_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.expand(4, 3)
        var w = Tensor[dtype].d2(
            [[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]
        )
        var w_gpu = w.to_gpu(gpu)
        var loss = (v * w_gpu).sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0]])))


def test_view_slice_rows_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
        var a2 = a.reshape(3, 4)
        var a_gpu = a2.to_gpu(gpu)
        var v = a_gpu[1:3, :]
        assert_true(v.shape() == Shape(2, 4))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        assert_true(a.grad().all_close(expected))


def test_view_slice_step_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
        var a2 = a.reshape(3, 4)
        var a_gpu = a2.to_gpu(gpu)
        var v = a_gpu[0:3:2, :]
        assert_true(v.shape() == Shape(2, 4))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        )
        assert_true(a.grad().all_close(expected))


def test_view_slice_single_element_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
        var a2 = a.reshape(3, 4)
        var a_gpu = a2.to_gpu(gpu)
        var v = a_gpu[i(1), i(2)]
        v.backward(1.0)
        var expected = Tensor[dtype].zeros(12)
        expected[6] = 1.0
        assert_true(a.grad().all_close(expected))


def test_view_slice_newaxis_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu[newaxis, s(), newaxis]
        assert_true(v.shape() == Shape(1, 3, 1))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


def test_view_flatten_3d_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
        var a3 = a.reshape(2, 3, 4)
        var a_gpu = a3.to_gpu(gpu)
        var v = a_gpu.flatten()
        assert_true(v.shape() == Shape(24))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))


def test_view_flatten_partial_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
        var a3 = a.reshape(2, 3, 4)
        var a_gpu = a3.to_gpu(gpu)
        var v = a_gpu.flatten(start_dim=1)
        assert_true(v.shape() == Shape(2, 12))
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))


def test_view_chain_into_view_then_transpose_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.into_view()
        var v2 = v1.transpose()
        var loss = v2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))


def test_view_chain_view_offset_then_transpose_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.view(Shape(2, 3), offset=0)
        var v2 = v1.transpose()
        var loss = v2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))


def test_view_chain_view_offset_multi_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 20.0, requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.view(Shape(6, 3), offset=2)
        var loss = v1.sum()
        loss.backward()
        var expected = Tensor[dtype].zeros(20)
        for i in range(2, 20):
            expected[i] = 1.0
        assert_true(a.grad().all_close(expected))


def test_view_chain_transpose_permute_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
        var a3 = a.reshape(2, 3, 4)
        var a_gpu = a3.to_gpu(gpu)
        var v1 = a_gpu.transpose(0, 2)
        var v2 = v1.permute([1, 0, 2])
        var loss = v2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))


def test_view_chain_slice_unsqueeze_expand_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
        var a2 = a.reshape(3, 4)
        var a_gpu = a2.to_gpu(gpu)
        var v1 = a_gpu[0:2, :]
        var v2 = v1.unsqueeze(0)
        var v3 = v2.expand(3, 2, 4)
        var loss = v3.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert_true(a.grad().all_close(expected))


def test_view_gradbox_zero_single_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        var loss = v.sum()
        loss.backward()
        var v_grad = v.grad()
        assert_true(v_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))


def test_view_gradbox_zero_chain_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.into_view()
        var v2 = v1.transpose()
        var loss = v2.sum()
        loss.backward()
        var v1_grad = v1.grad()
        assert_true(v1_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
        var v2_grad = v2.grad()
        assert_true(v2_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))


def test_view_gradbox_zero_complex_graph_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a1 = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a2 = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        var a1_gpu = a1.to_gpu(gpu)
        var a2_gpu = a2.to_gpu(gpu)
        var v1 = a1_gpu.into_view()
        var s1 = v1.sum()
        var v2 = a2_gpu.into_view()
        var s2 = v2.sum()
        var total = s1 + s2
        total.backward()
        var v1_grad = v1.grad()
        assert_true(v1_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
        var v2_grad = v2.grad()
        assert_true(v2_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
        assert_true(a1.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))
        assert_true(a2.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))


def test_view_gradbox_zero_two_backward_passes_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        var loss1 = v.sum()
        loss1.backward()
        assert_true(v.grad().all_close(Tensor[dtype].zeros(Shape(2, 2))))
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))


def test_view_view_mul_scalar_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        var r = v * 2.0
        var loss = r.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 2.0)))


def test_view_view_add_view_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var b = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        var va = a_gpu.into_view()
        var vb = b_gpu.into_view()
        var r = va + vb
        var loss = r.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))
        assert_true(b.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))


def test_view_view_mul_view_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var b = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        var va = a_gpu.into_view()
        var vb = b_gpu.into_view()
        var r = va * vb
        var loss = r.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]]))
        )
        assert_true(
            b.grad().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )


def test_view_view_sum_axis_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        var loss = v.sum(axes=[1])
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))


def test_view_view_broadcast_add_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var va = a_gpu.into_view()
        var bias = Tensor[dtype].d1([10.0, 20.0, 30.0], requires_grad=True)
        var bias_gpu = bias.to_gpu(gpu)
        var vbias = bias_gpu.into_view()
        var r = va + vbias
        var loss = r.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))
        assert_true(bias.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


def test_view_noncontiguous_transpose_backward_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var t = a_gpu.transpose()
        assert_false(t.is_contiguous())
        var loss = t.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))


def test_view_noncontiguous_strided_backward_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.view(Shape(2, 4), Strides(1, 2))
        assert_false(v.is_contiguous())
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(8), 1.0)))


def test_view_noncontiguous_offset_backward_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True
        )
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.view(Shape(2, 3), Strides(4, 1), offset=1)
        assert_false(v.is_contiguous())
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        )
        assert_true(a.grad().all_close(expected))


def test_view_multiple_views_same_base_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v1 = a_gpu.into_view()
        var v2 = a_gpu.view(2, 2)
        var loss = v1.sum() + v2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)))


def test_view_view_track_grad_false_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view[track_grad=False]()
        assert_false(v.requires_grad)


def test_view_view_4d_backward_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].arange(0.0, 120.0, requires_grad=True)
        var a4 = a.reshape(2, 3, 4, 5)
        var a_gpu = a4.to_gpu(gpu)
        var v = a_gpu.into_view()
        var loss = v.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(120), 1.0)))


def test_view_view_data_sharing_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var a_gpu = a.to_gpu(gpu)
        var v = a_gpu.into_view()
        a_gpu[0] = 99.0
        assert_true(v.to_cpu()[0] == 99.0)
        v[1] = 88.0
        assert_true(a[1] == 88.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
