from tensor import Tensor
from gradbox import Gradbox
from shapes import Shape
from std.testing import assert_true
from std.sys import has_accelerator
from ndbuffer import NDBuffer

comptime dtype = DType.float32


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

fn tensors_close[dtype: DType](
    a: Tensor[dtype], b: Tensor[dtype],
) raises -> Bool:
    return a.all_close[atol= Scalar[dtype](1e-5)](b)

fn gradboxes_close[dtype: DType](
    a: Gradbox[dtype], b: Gradbox[dtype],
) raises -> Bool:
    return a.all_close[atol= Scalar[dtype](1e-5)](b)


# ═══════════════════════════════════════════════════════════════════════════════
# as_gradbox — CPU
# ═══════════════════════════════════════════════════════════════════════════════

fn test_as_gradbox_cpu_1d_contiguous() raises:
    print("test_as_gradbox_cpu_1d_contiguous")
    var t = Tensor[dtype].arange(6)
    var g = t.as_gradbox()
    assert_true(g.is_on_cpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(6))
    assert_true(gradboxes_close(g, Gradbox[dtype](NDBuffer[dtype](
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    ))))
    print("passed")


fn test_as_gradbox_cpu_2d_contiguous() raises:
    print("test_as_gradbox_cpu_2d_contiguous")
    var t = Tensor[dtype].arange(6).reshape(Shape(2, 3))
    var g = t.as_gradbox()
    assert_true(g.is_on_cpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 3))
    assert_true(gradboxes_close(g, Gradbox[dtype](NDBuffer[dtype](
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    ).reshape(Shape(2, 3)))))
    print("passed")


fn test_as_gradbox_cpu_3d_contiguous() raises:
    print("test_as_gradbox_cpu_3d_contiguous")
    var t = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
    var g = t.as_gradbox()
    assert_true(g.is_on_cpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 3, 4))
    print("passed")


fn test_as_gradbox_cpu_non_contiguous_transposed() raises:
    print("test_as_gradbox_cpu_non_contiguous_transposed")
    var t = Tensor[dtype].arange(6).reshape(Shape(2, 3))
    var t_T = t.transpose()  # (3, 2) — non-contiguous
    var g = t_T.as_gradbox(contiguous=True)
    assert_true(g.is_on_cpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(3, 2))
    # verify values are correctly materialised
    var expected = Tensor[dtype].d2([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    assert_true(gradboxes_close(g, expected.as_gradbox()))
    print("passed")


fn test_as_gradbox_cpu_non_contiguous_no_materialise() raises:
    print("test_as_gradbox_cpu_non_contiguous_no_materialise")
    var t = Tensor[dtype].arange(6).reshape(Shape(2, 3))
    var t_T = t.transpose()
    var g = t_T.as_gradbox(contiguous=False)
    assert_true(g.is_on_cpu())
    assert_true(g.shape() == Shape(3, 2))
    # not contiguous — strides preserved
    assert_true(not g.is_contiguous())
    print("passed")


fn test_as_gradbox_cpu_with_offset() raises:
    print("test_as_gradbox_cpu_with_offset")
    var t = Tensor[dtype].arange(12).reshape(Shape(3, 4))
    #var sliced = t.slice(Slice(1, 3))  # rows 1..2 — offset non-zero
    var sliced = t[1:3, ::]  # rows 1..2 — offset non-zero
    var g = sliced.as_gradbox(contiguous=True)
    assert_true(g.is_on_cpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 4))
    print("passed")


fn test_as_gradbox_cpu_share_true() raises:
    print("test_as_gradbox_cpu_share_true")
    var t = Tensor[dtype].arange(4)
    var g = t.as_gradbox(share=True)
    assert_true(g.is_on_cpu())
    assert_true(g.shape() == Shape(4))
    print("passed")


fn test_as_gradbox_cpu_4d() raises:
    print("test_as_gradbox_cpu_4d")
    var t = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5))
    var g = t.as_gradbox()
    assert_true(g.is_on_cpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 3, 4, 5))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# as_tensor — CPU
# ═══════════════════════════════════════════════════════════════════════════════

fn test_as_tensor_cpu_1d_contiguous() raises:
    print("test_as_tensor_cpu_1d_contiguous")
    var g = Gradbox[dtype](NDBuffer[dtype](1.0, 2.0, 3.0, 4.0))
    var t = g.as_tensor()
    assert_true(t.is_on_cpu())
    assert_true(t.shape() == Shape(4))
    assert_true(tensors_close(t, Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])))
    print("passed")


fn test_as_tensor_cpu_2d_contiguous() raises:
    print("test_as_tensor_cpu_2d_contiguous")
    var g = Gradbox[dtype].zeros(Shape(3, 4))
    var t = g.as_tensor()
    assert_true(t.is_on_cpu())
    assert_true(t.shape() == Shape(3, 4))
    assert_true(tensors_close(t, Tensor[dtype].zeros(Shape(3, 4))))
    print("passed")


fn test_as_tensor_cpu_3d_contiguous() raises:
    print("test_as_tensor_cpu_3d_contiguous")
    var g = Gradbox[dtype].full(Shape(2, 3, 4), 7.0)
    var t = g.as_tensor()
    assert_true(t.is_on_cpu())
    assert_true(t.shape() == Shape(2, 3, 4))
    assert_true(tensors_close(t, Tensor[dtype].full(Shape(2, 3, 4), 7.0)))
    print("passed")


fn test_as_tensor_cpu_non_contiguous() raises:
    print("test_as_tensor_cpu_non_contiguous")
    var base = Tensor[dtype].arange(6).reshape(Shape(2, 3))
    var g = base.transpose().as_gradbox(contiguous=False)
    assert_true(not g.is_contiguous())
    var t = g.as_tensor()  # should materialise contiguous copy
    assert_true(t.is_on_cpu())
    assert_true(t.is_contiguous())
    assert_true(t.shape() == Shape(3, 2))
    print("passed")


fn test_as_tensor_cpu_requires_grad() raises:
    print("test_as_tensor_cpu_requires_grad")
    var g = Gradbox[dtype](NDBuffer[dtype](1.0, 2.0, 3.0))
    var t = g.as_tensor(requires_grad=True)
    assert_true(t.requires_grad)
    assert_true(t.has_grad())
    print("passed")


fn test_as_tensor_cpu_4d() raises:
    print("test_as_tensor_cpu_4d")
    var g = Gradbox[dtype].rand(Shape(2, 3, 4, 5))
    var t = g.as_tensor()
    assert_true(t.is_on_cpu())
    assert_true(t.shape() == Shape(2, 3, 4, 5))
    print("passed")


fn test_as_tensor_cpu_scalar() raises:
    print("test_as_tensor_cpu_scalar")
    var g = Gradbox[dtype].full(Shape(), 42.0)
    var t = g.as_tensor()
    assert_true(t.is_on_cpu())
    assert_true(t.item() == 42.0)
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# roundtrip — CPU
# ═══════════════════════════════════════════════════════════════════════════════

fn test_roundtrip_tensor_gradbox_tensor_cpu() raises:
    print("test_roundtrip_tensor_gradbox_tensor_cpu")
    var t = Tensor[dtype].rand(3, 4)
    var t_copy = t.copy()
    var g = t.as_gradbox()
    var t2 = g.as_tensor()
    assert_true(tensors_close(t_copy, t2))
    print("passed")


fn test_roundtrip_gradbox_tensor_gradbox_cpu() raises:
    print("test_roundtrip_gradbox_tensor_gradbox_cpu")
    var g = Gradbox[dtype].rand(Shape(4, 5))
    var g_copy = g.copy()
    var t = g.as_tensor()
    var g2 = t.as_gradbox()
    assert_true(gradboxes_close(g_copy, g2))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# as_gradbox — GPU
# ═══════════════════════════════════════════════════════════════════════════════

fn test_as_gradbox_gpu_1d_contiguous() raises:
    print("test_as_gradbox_gpu_1d_contiguous")
    var t = Tensor[dtype].arange(6).to_gpu()
    var g = t.as_gradbox()
    assert_true(g.is_on_gpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(6))
    # verify values via CPU
    var g_cpu = g.to_cpu()
    var expected = Gradbox[dtype](NDBuffer[dtype](
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
        ))
    assert_true(gradboxes_close(g_cpu, expected))
    print("passed")


fn test_as_gradbox_gpu_2d_contiguous() raises:
    print("test_as_gradbox_gpu_2d_contiguous")
    var t = Tensor[dtype].arange(6).reshape(Shape(2, 3)).to_gpu()
    var g = t.as_gradbox()
    assert_true(g.is_on_gpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 3))
    var g_cpu = g.to_cpu()
    var expected = Gradbox[dtype](NDBuffer[dtype](
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    ).reshape(Shape(2, 3)))
    assert_true(gradboxes_close(g_cpu, expected))
    print("passed")


fn test_as_gradbox_gpu_3d_contiguous() raises:
    print("test_as_gradbox_gpu_3d_contiguous")
    var t = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4)).to_gpu()
    var g = t.as_gradbox()
    assert_true(g.is_on_gpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 3, 4))
    print("passed")


fn test_as_gradbox_gpu_4d() raises:
    print("test_as_gradbox_gpu_4d")
    var t = Tensor[dtype].arange(120).reshape(Shape(2, 3, 4, 5)).to_gpu()
    var g = t.as_gradbox()
    assert_true(g.is_on_gpu())
    assert_true(g.is_contiguous())
    assert_true(g.shape() == Shape(2, 3, 4, 5))
    print("passed")


fn test_as_gradbox_gpu_contiguous_false() raises:
    print("test_as_gradbox_gpu_contiguous_false")
    var t = Tensor[dtype].arange(6).reshape(Shape(2, 3)).to_gpu()
    var g = t.as_gradbox(contiguous=False)
    assert_true(g.is_on_gpu())
    assert_true(g.shape() == Shape(2, 3))
    print("passed")


fn test_as_gradbox_gpu_values_preserved() raises:
    print("test_as_gradbox_gpu_values_preserved")
    var t_cpu = Tensor[dtype].rand(4, 5)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var g_cpu = g_gpu.to_cpu()
    assert_true(gradboxes_close(g_cpu, t_cpu.as_gradbox()))
    print("passed")


fn test_as_gradbox_gpu_large() raises:
    print("test_as_gradbox_gpu_large")
    var t_cpu = Tensor[dtype].rand(64, 128)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    assert_true(g_gpu.is_on_gpu())
    assert_true(g_gpu.shape() == Shape(64, 128))
    var g_cpu = g_gpu.to_cpu()
    assert_true(gradboxes_close(g_cpu, t_cpu.as_gradbox()))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# as_tensor — GPU
# ═══════════════════════════════════════════════════════════════════════════════

fn test_as_tensor_gpu_1d_contiguous() raises:
    print("test_as_tensor_gpu_1d_contiguous")
    var g_cpu = Gradbox[dtype](NDBuffer[dtype](1.0, 2.0, 3.0, 4.0))
    var t_gpu = g_cpu.as_tensor().to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    assert_true(t2.is_on_gpu())
    assert_true(t2.shape() == Shape(4))
    var t2_cpu = t2.to_cpu()
    assert_true(tensors_close(
        t2_cpu, Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    ))
    print("passed")


fn test_as_tensor_gpu_2d_contiguous() raises:
    print("test_as_tensor_gpu_2d_contiguous")
    var g_cpu = Gradbox[dtype].zeros(Shape(3, 4))
    var t_gpu = g_cpu.as_tensor().to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    assert_true(t2.is_on_gpu())
    assert_true(t2.shape() == Shape(3, 4))
    assert_true(tensors_close(t2.to_cpu(), Tensor[dtype].zeros(Shape(3, 4))))
    print("passed")


fn test_as_tensor_gpu_3d() raises:
    print("test_as_tensor_gpu_3d")
    var t_cpu = Tensor[dtype].rand(2, 3, 4)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    assert_true(t2.is_on_gpu())
    assert_true(t2.shape() == Shape(2, 3, 4))
    assert_true(tensors_close(t2.to_cpu(), t_cpu))
    print("passed")


fn test_as_tensor_gpu_requires_grad() raises:
    print("test_as_tensor_gpu_requires_grad")
    var t_cpu = Tensor[dtype].rand(3, 4)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor(requires_grad=True)
    assert_true(t2.requires_grad)
    assert_true(t2.is_on_gpu())
    print("passed")


fn test_as_tensor_gpu_large() raises:
    print("test_as_tensor_gpu_large")
    var t_cpu = Tensor[dtype].rand(64, 128)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    assert_true(t2.is_on_gpu())
    assert_true(t2.shape() == Shape(64, 128))
    assert_true(tensors_close(t2.to_cpu(), t_cpu))
    print("passed")


fn test_as_tensor_gpu_4d() raises:
    print("test_as_tensor_gpu_4d")
    var t_cpu = Tensor[dtype].rand(2, 3, 4, 5)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    assert_true(t2.is_on_gpu())
    assert_true(t2.shape() == Shape(2, 3, 4, 5))
    assert_true(tensors_close(t2.to_cpu(), t_cpu))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# roundtrip — GPU
# ═══════════════════════════════════════════════════════════════════════════════

fn test_roundtrip_tensor_gradbox_tensor_gpu() raises:
    print("test_roundtrip_tensor_gradbox_tensor_gpu")
    var t_cpu = Tensor[dtype].rand(3, 4)
    var t_gpu = t_cpu.to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    assert_true(t2.is_on_gpu())
    assert_true(tensors_close(t2.to_cpu(), t_cpu))
    print("passed")


fn test_roundtrip_gradbox_tensor_gradbox_gpu() raises:
    print("test_roundtrip_gradbox_tensor_gradbox_gpu")
    var g_cpu = Gradbox[dtype].rand(Shape(4, 5))
    var t_gpu = g_cpu.as_tensor().to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t2 = g_gpu.as_tensor()
    var g2 = t2.as_gradbox()
    assert_true(g2.is_on_gpu())
    assert_true(gradboxes_close(g2.to_cpu(), g_cpu))
    print("passed")


fn test_roundtrip_cpu_to_gpu_to_cpu() raises:
    print("test_roundtrip_cpu_to_gpu_to_cpu")
    var t_cpu = Tensor[dtype].rand(5, 6)
    var g_cpu = t_cpu.as_gradbox()
    var t_gpu = g_cpu.as_tensor().to_gpu()
    var g_gpu = t_gpu.as_gradbox()
    var t_back = g_gpu.as_tensor().to_cpu()
    assert_true(tensors_close(t_back, t_cpu))
    print("passed")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    print("=== as_gradbox / as_tensor CPU tests ===")
    test_as_gradbox_cpu_1d_contiguous()
    test_as_gradbox_cpu_2d_contiguous()
    test_as_gradbox_cpu_3d_contiguous()
    test_as_gradbox_cpu_non_contiguous_transposed()
    test_as_gradbox_cpu_non_contiguous_no_materialise()
    test_as_gradbox_cpu_with_offset()
    test_as_gradbox_cpu_share_true()
    test_as_gradbox_cpu_4d()
    test_as_tensor_cpu_1d_contiguous()
    test_as_tensor_cpu_2d_contiguous()
    test_as_tensor_cpu_3d_contiguous()
    test_as_tensor_cpu_non_contiguous()
    test_as_tensor_cpu_requires_grad()
    test_as_tensor_cpu_4d()
    test_as_tensor_cpu_scalar()
    test_roundtrip_tensor_gradbox_tensor_cpu()
    test_roundtrip_gradbox_tensor_gradbox_cpu()
    print("=== All CPU tests passed ===\n")

    comptime if has_accelerator():
        print("=== as_gradbox / as_tensor GPU tests ===")
        test_as_gradbox_gpu_1d_contiguous()
        test_as_gradbox_gpu_2d_contiguous()
        test_as_gradbox_gpu_3d_contiguous()
        test_as_gradbox_gpu_4d()
        test_as_gradbox_gpu_contiguous_false()
        test_as_gradbox_gpu_values_preserved()
        test_as_gradbox_gpu_large()
        test_as_tensor_gpu_1d_contiguous()
        test_as_tensor_gpu_2d_contiguous()
        test_as_tensor_gpu_3d()
        test_as_tensor_gpu_requires_grad()
        test_as_tensor_gpu_large()
        test_as_tensor_gpu_4d()
        test_roundtrip_tensor_gradbox_tensor_gpu()
        test_roundtrip_gradbox_tensor_gradbox_gpu()
        test_roundtrip_cpu_to_gpu_to_cpu()
        print("=== All GPU tests passed ===")
    else:
        print("No GPU available — skipping GPU tests")
