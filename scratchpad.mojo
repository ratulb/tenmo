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
from tenmo.nlp import IMDBTextCleaner, DefaultTokenizer
from tenmo.buffers import Buffer
from bpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

comptime dtype = DType.float32

def main() raises:
    test_shape_compute_output_shape_single_axis()

fn test_shape_compute_output_shape_single_axis() raises:
    print("test_shape_compute_output_shape_single_axis")
    var s = Shape(2, 3, 4)
    var axes = IntArray.with_capacity(1)
    axes.append(1)  # Reduce middle axis
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(
        result_no_keep.rank() == 2, "result without keepdims should have rank 2"
    )
    assert_true(result_no_keep[0] == 2, "first dim unchanged")
    assert_true(result_no_keep[1] == 4, "third dim unchanged")
    assert_true(
        result_keep.rank() == 3, "result with keepdims should have rank 3"
    )
    assert_true(result_keep[1] == 1, "reduced dim should be 1")
    print("test_shape_compute_output_shape_single_axis passed")

def main_100() raises:
    _="""var v = Tensor[dtype].d1([1, 2, 3])
    var m = Tensor[dtype].d2([[4, 5, 6], [7,8,9],[10, 11, 12]])
    var r = v.matmul[mode=vm](m)
    r.print()"""
    var strs: List[UInt8] = [1, 6, 9]
    var xxx = strs[::]
    print(xxx)

    var tok = RegexTokenizer()
    tok.train("hello world", 256 + 8)

    # tok.vocab_size gives total token count
    # tok.vocab.get(id) gives bytes for that id
    print("tok.vocab_size: ", tok.bpe.vocab_size)

    var word2index = Dict[String, Int]()
    for token_id in range(tok.bpe.vocab_size):
        var token_bytes = tok.bpe.vocab.get(token_id)
        var key = String(from_utf8_lossy=token_bytes[::])
        #var key = String(from_utf8_lossy=Span[UInt8](token_bytes))
        word2index[key] = token_id

    # Look up a token by its text
    if "hello" in word2index:
        print(word2index["hello"]) # Does not print!
    if "world" in word2index:
        print(word2index["world"]) # Does not print!


    _="""for item in word2index.items():
        print(item.key, item.value)"""

fn main_70() raises:
    var b = Buffer[dtype]([Scalar[dtype](1), Scalar[dtype](21), Scalar[dtype](9)])
    var r = b.load[3](0)
    print(r+r)

fn test_mcpy_cpu_2d_fuse_sum_duplicate_indices() raises:
    print("mcpy_cpu_2d_fuse_sum_duplicate_indices")
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var idx = IntArray()
    idx.append(0)
    idx.append(0)
    idx.append(0)
    # row 0 three times: [3, 6]
    var result = a.gather(idx, axis=0, fuse_sum=True)
    assert_true(result.all_close(Tensor[dtype].d1([3.0, 6.0])))

def main_60() raises:
    test_mcpy_cpu_2d_fuse_sum_duplicate_indices()

def main_80() raises:
    var t = BasicTokenizer()
    t.train("aaabdaaabac", 256 + 3)
    var ids = t.encode("aaabdaaabac")
    print(ids)
    # [258, 100, 258, 97, 99]
    print(t.decode(ids))
    # aaabdaaabac

    var rt = RegexTokenizer()
    rt.train("The quick brown fox jumps over the lazy dog.", 256 + 16)
    ids = t.encode("The quick brown fox jumps over the lazy dog.")
    print(len(ids))
    # 27 (pretokenization prevents cross-word merges)
    print(rt.decode(ids))


    var gpt4 = GPT4Tokenizer()
    gpt4.load_tiktoken("/home/tenmoomnet/bpe/data/o200k_base.tiktoken")
    ids = gpt4.encode("Hello, world!")
    print(len(ids))
    # 4
    print(gpt4.decode(ids))
    # Hello, world!



def main_40() raises:
    var cleaner = IMDBTextCleaner()
    var tokenizer = DefaultTokenizer.from_file("the-verdict.txt", cleaner, 3, 1)
    #var url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    #var tokenizer = DefaultTokenizer.from_url(url, cleaner, 1, 1)
    print(len(tokenizer))
    var text = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."

    var ids = tokenizer.encode(text)
    print(tokenizer.decode(ids))
    #from importlib.metadata import version
    #import tiktoken
    var importlib = Python.import_module("importlib.metadata")
    var tiktoken = Python.import_module("tiktoken")
    var version = importlib.version
    print("tiktoken version:", version("tiktoken"))

    var enc      = tiktoken.get_encoding("cl100k_base")
    #var enc      = tiktoken.get_encoding("gpt2")
    print(enc.decode(enc.encode(text)))
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    #integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    integers = tokenizer.encode(text)
    print(integers)
    var py = Python.import_module("builtins")
    integers1 = enc.encode(py.str(text),allowed_special={"<|endoftext|>"})
    print(integers1)
    strings = tokenizer.decode(integers)
    print(strings)

fn main_30() raises:
    var t = Tensor[dtype].scalar(3)
    assert_true(t.all_close(Tensor[DType.float32].scalar(3)))
    print(max(SIMD[dtype, 3](0), SIMD[dtype, 3](-9, 2, -32)))
    var rng = PhiloxRandom(seed=42, subsequence=UInt64(1), offset=0)
    var rand_f32 = rng.step_uniform()
    print(rand_f32)
    _ = """var srt = UnaryOpsKernel.launch[SQRT](NDBuffer[dtype](1, -9, 25))
    srt.print()"""

    t.print()


fn test_prd_cpu_bwd_all_positive_1d() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
    var a = Tensor[dtype].d1([2, 3, 4], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    # grad_x[i] = product / x[i] = 24 / x[i]
    a.grad().print()
    assert_true(a.grad().all_close[atol=1e-4](Tensor[dtype].d1([12, 8, 6])))


fn test_prd_cpu_bwd_all_positive_1d_orig() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
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

