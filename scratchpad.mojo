# run_tensor.mojo  (sits at repo root, outside tenmo/)
from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.reduction_kernel import ProductArg
from tenmo.intarray import IntArray
from std.testing import assert_true
from tenmo.numpy_interop import test_to_ndarray
from tenmo.common_utils import *
from tenmo.strides import Strides
from tenmo.mnemonics import *
from tenmo.reduction_kernel import Reduction
from tenmo.unary_ops_kernel import UnaryOpsKernel
from std.random.philox import Random as PhiloxRandom
from std.pathlib import Path
from std.python import Python
from std.collections import Counter

@fieldwise_init
struct Review(ImplicitlyCopyable, Movable):
    var rating: Int
    var comment: String


def main_wip() raises:

    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
    var s = Tensor[dtype].scalar(2.0, requires_grad=True)
    var ss = Tensor[dtype].d1([2.0], requires_grad=True)

    # All of these work:
    var _r1 = a.dot(b)           # vector · vector  (existing)
    var _r2 = a.dot(s)           # vector · scalar  (broadcasts s → [2,2,2])
    var _r3 = s.dot(a)           # scalar · vector  (broadcasts s → [1,2,3] shape)
    var _r4 = a.dot(2.0)         # tensor · Scalar literal

    var _r5 = a.dot(ss)

fn main() raises:
    var undscore_sep = StringSlice("_")
    var dot_sep = StringSlice(".txt")
    var directory = Path("/home/tenmoomnet/aclImdb/train/pos")
    var reviews = List[Review](capacity=12500)
    for item in directory.listdir():
        ref name = item.name()
        var split = name.split(undscore_sep)
        var rating = split[1].split(dot_sep)[0]
        var complete_path = directory.joinpath(name)
        var review = complete_path^.read_text()

        reviews.append(Review(Int(rating), review))

    ref last = reviews[-1]

    print(last.comment)
    print(last.rating)
    print(len(reviews))

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 10)
    print(counter[1]) # output: 2
    counter.subtract(other)
    print(counter[3]) # output: -8


    comptime dtype = DType.float32

    var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]], True)
    var oh1 = Tensor[dtype]([1, 0, 1])
    var r = oh1.matmul[mode=vm](a)
    r.print()

    var selected = a.gather(IntArray(2, 0))
    selected = selected.sum(axes=[0])
    selected.print()

    var layer_2_delta = Tensor[dtype].d1([0.9])
    var layer_2 = Tensor[dtype].randn(5, 1)
    var layer_1_delta = layer_2_delta.matmul[mode=vm](layer_2.transpose())
    layer_1_delta.print()

    var v1 = Tensor[dtype].d1([1, 2, 3])
    var v2 = Tensor[dtype].d1([1, 2, 3])
    var v = v1.dot(v2)
    print()
    layer_2.print()
    v.print()




fn main_1() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].scalar(3)
    assert_true(t.all_close(Tensor[DType.float32].scalar(3)))
    print(max(SIMD[dtype, 3](0), SIMD[dtype, 3](-9, 2, -32)))
    var rng = PhiloxRandom(seed=42, subsequence=UInt64(1), offset=0)
    var rand_f32 = rng.step_uniform()
    print(rand_f32)
    _ = """var srt = UnaryOpsKernel.launch[SQRT](NDBuffer[dtype](1, -9, 25))
    srt.print()"""

    t.print()


fn main_3() raises:
    var a = NDBuffer[DType.float32](1.0, 2.0, 3.0)
    a.print()


fn test_prd_cpu_bwd_all_positive_1d() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2, 3, 4], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    # grad_x[i] = product / x[i] = 24 / x[i]
    a.grad().print()
    assert_true(a.grad().all_close[atol=1e-4](Tensor[dtype].d1([12, 8, 6])))


fn test_prd_cpu_bwd_all_positive_1d_orig() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var result = a.buffer.product_cpu(IntArray(0), False)
    var arg = result[1]
    # Check excl_product was stored
    print("excl_product is some:", arg.excl_product.__bool__())
    if arg.excl_product:
        var excl = arg.excl_product.value()
        print("excl[0]:", excl.get(0))
        print("excl[1]:", excl.get(1))
        print("excl[2]:", excl.get(2))


fn test_complex_mixed_ops_backward() raises:
    print("test_complex_mixed_ops_backward")
    comptime dtype = DType.float32

    a = Tensor[dtype].d2(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], requires_grad=True
    )
    v1 = a.view(shape=Shape(2, 4), strides=Strides(4, 1), offset=2)
    v1.print()

    v2 = v1.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)

    v3 = v2.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=0)

    c = v3.contiguous()

    s = c.mean()

    s.backward(42)

    grad = a.grad().as_tensor()
    grad.print()
    result = grad[Slice(0, 1, None), Slice(2, None, None)]
    result.print()
    assert_true(result == Tensor[dtype].d2([[10.5, 10.5]]))


fn test_contig_cpu_1d_slice_view() raises:
    print("test_contig_cpu_1d_slice_view")
    comptime dtype = DType.float32
    # Create a non-contiguous view via transpose of a 2D then flatten — or use
    # a known strided view via unsqueeze + squeeze to produce offset
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    # transpose produces non-contiguous strides
    var t = a.transpose()  # (3,2), non-contiguous
    # var row = t.squeeze([1])                  # squeeze won't help here — use sum to get grad
    var c = t.contiguous()
    assert_true(c.shape() == Shape(3, 2))
    # Values: transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
    c.print()
    assert_true(
        c.all_close(Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    )
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_expand_1d_to_2d_new_batch_dim() raises:
    print("test_expand_1d_to_2d_new_batch_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)  # shape (3,)
    var e = a.expand(4, 3)  # shape (4,3)
    assert_true(e.shape() == Shape.of(4, 3))
    # Every row is [1, 2, 3]
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    assert_true(e.all_close(expected))
    es = e.sum()
    es.backward()
    # Each element of a was broadcast 4 times → grad = 4.0
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


fn test_tanh_contiguous_vs_non_contiguous() raises:
    print("test_tanh_contiguous_vs_non_contiguous")

    comptime dtype = DType.float32
    var x_contig = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
    var y_contig = x_contig.tanh[track_grad=True]()

    # Create non-contiguous tensor (e.g., via slice or transpose)
    var x_large = Tensor[dtype].d2([[0.0, 99.0], [1.0, 99.0], [-1.0, 99.0]])
    var x_non_contig = x_large[:, slice(0, 1)]  # Slice to get non-contiguous
    x_non_contig.requires_grad_(True)
    var y_non_contig = x_non_contig.tanh[track_grad=True]()

    # Both should give same results
    y_contig.print()
    y_contig.unsqueeze[track_grad=False](-1).print()
    y_non_contig.print()
    assert_true(
        y_contig.unsqueeze[track_grad=False](-1).all_close[atol=1e-5](
            y_non_contig
        ),
        "Contiguous and non-contiguous should match",
    )


fn test_matmul_2d_non_contiguous_both_views() raises:
    print("test_matmul_2d_non_contiguous_both_views")
    comptime dtype = DType.float32
    var base = Tensor[dtype].d2(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        requires_grad=True,
    )
    print("past this point 1")
    var AA = base.view(shape=Shape(2, 2), strides=Strides(8, 1), offset=0)

    print("past this point 2")
    var BB = base.view(shape=Shape(2, 2), strides=Strides(4, 1), offset=10)

    print("past this point 3")
    var C = AA.matmul(BB)
    # A = [[1,2],[9,10]], B = [[11,12],[15,16]]
    var expected = Tensor[dtype].d2([[41.0, 44.0], [249.0, 268.0]]).float()
    C.print()
    assert_true(C.all_close(expected))

    # validate_matmul_2d_grads(AA, BB, C)


fn test_vector_matrix_with_vector_view() raises:
    print("test_vector_matrix_with_vector_view")
    comptime dtype = DType.float32
    var base_v = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var v_view = base_v.view(
        shape=Shape(3), strides=Strides(1), offset=1
    )  # [1,2,3]
    var M = Tensor[dtype].d2(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], requires_grad=True
    )
    var r = v_view.matmul[mode=vm](M)
    var loss = r.sum()
    loss.backward()

    # [1,2,3] @ [[1,0],[0,1],[0,0]] = [1,2]
    r.print()
    assert_true(r.all_close(Tensor[dtype].d1([1.0, 2.0])))
    # Gradients should flow to viewed portion [1,2,3]
    assert_true(
        base_v.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0, 0.0, 0.0]))
    )


fn test_matrix_vector_with_matrix_view() raises:
    print("test_matrix_vector_with_matrix_view")
    comptime dtype = DType.float32
    var base_M = Tensor[dtype].d1(
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0], requires_grad=True
    )
    # var M_view = base_M.view(shape=Shape(2, 2), strides=Strides(3, 1), offset=2)
    var M_view = base_M.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)
    var v = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var r = M_view.matmul[mode=mv](v)
    var loss = r.sum()
    loss.backward()

    # M_view = [[1,2],[3,4]] @ [1,2] = [5,11]
    r.print()
    assert_true(r.all_close(Tensor[dtype].d1([5.0, 11.0])))
    # Gradients should flow only to viewed portion [1,2,3,4]
    base_M.grad().print()
    assert_true(
        base_M.grad().all_close(
            Tensor[dtype].d1([0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0])
        )
    )


fn test_slice_backward_chained() raises:
    """Test gradient flow through chained slices."""
    print("test_slice_backward_chained")

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        requires_grad=True,
    )

    var sliced1 = x[:, 1:3]  # Get middle 2 columns
    var sliced2 = sliced1[1:2, :]  # Get row 1 from that
    var loss = sliced2.sum()  # Sum of [6, 7]
    loss.backward()
    # Only element at [1, 1] and [1, 2] should have gradient
    var expected_grad = Tensor[dtype].d2(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    x.grad().print()
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_rlv_cpu_noncontig_transposed() raises:
    print("test_rlv_cpu_noncontig_transposed")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    var t = a.transpose()  # non-contiguous view
    var out = t.relu()
    var loss = out.sum()
    loss.backward()
    # transpose relu forward: max(0, [[−1,3],[2,−4]]) = [[0,3],[2,0]]
    assert_true(
        out.contiguous().all_close(Tensor[dtype].d2([[0.0, 3.0], [2.0, 0.0]]))
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
    a.grad().print()
    assert_true(
        a.grad().all_close(Tensor.d2([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    )


fn test_tensor_dot() raises:
    print("test_tensor_dot")
    comptime dtype = DType.float32
    a = Tensor[dtype].scalar(5, requires_grad=True)
    b = Tensor[dtype].scalar(15, requires_grad=True)
    c = a.matmul(b)
    c.backward()
    assert_true(a.grad().item() == 15)
    assert_true(b.grad().item() == 5)

    d = a.into_view()
    e = d.matmul(b)
    e.backward()
    assert_true(a.grad().item() == 30)
    assert_true(b.grad().item() == 10)
    assert_true(d.grad().item() == 0)

    a = Tensor[dtype].arange(10, requires_grad=True)
    b = a[5::2]
    c = Tensor[dtype].d1([3, 4, 5])
    d = b.matmul(c)
    d.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].d1([0, 0, 0, 0, 0, 3, 0, 4, 0, 5]))
    )


fn main_2() raises:
    test_complex_mixed_ops_backward()
    test_contig_cpu_1d_slice_view()
    test_expand_1d_to_2d_new_batch_dim()
    test_tanh_contiguous_vs_non_contiguous()
    test_matmul_2d_non_contiguous_both_views()
    test_vector_matrix_with_vector_view()
    test_matrix_vector_with_matrix_view()
    test_slice_backward_chained()
    test_tensor_dot()


def main_5() raises:
    var shape = Shape(2, 3, 4)
    var s = shape[0:-2] + [1]
    print(s)
