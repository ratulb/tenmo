from tenmo import (
    Tensor,
    NDBuffer,
    Shape,
    SGD,
    Buffer,
    Strides,
    IntArray,
    Gradbox,
)
from std.testing import assert_true, TestSuite
from tenmo.common_utils import s, i, panic, Epsilon
from std.sys.defines import get_defined_string
from tenmo.matrixshapevalidator import MatrixShapeValidator
from std.sys import has_accelerator, simd_width_of
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.sys.intrinsics import PrefetchLocality
from tenmo.kernels.matmul_kernel import MatmulNdGpu
from std.sys import prefetch, PrefetchOptions
from std import math
from tenmo.device import GPU
from std.utils.numerics import max_finite, min_finite, neg_inf
from tenmo.numpy_interop import to_ndarray, from_ndarray
from std.random import seed
from std.sys.defines import get_defined_string
from tenmo.mnemonics import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
)
import tenmo
from tenmo.net import Linear
from std.collections import Set
from tenmo.backpropagation import ArgumentType
from tenmo.shared import Reduction

comptime dtype = DType.float32


trait Argument:
    comptime new_type: Optional[DType]


@fieldwise_init
struct NewTypeArg[orig_type: DType]:
    @staticmethod
    def convert[
        new_dtype: DType, //
    ](var gradbox: Gradbox[new_dtype]) -> Gradbox[Self.orig_type]:
        return gradbox.to_dtype[Self.orig_type]()


def main() raises:
    var inputs = Tensor[dtype].d2(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )

    inputs.print()


def main1() raises:
    comptime dtype = DType.float32
    var B, T, C = 4, 8, 2
    var X = Tensor[dtype].rand(B, T, C)
    var bow = Tensor[dtype].zeros(B, T, C)
    for b in range(B):
        for t in range(T):
            var x_prev = X[i(b), s(0, t + 1), s()]
            bow.fill(x_prev.mean(axes=[0]), i(b), i(t), s())

    var wei = Tensor[dtype].ones(T, T)
    wei = wei.tril[False]()
    wei = wei / wei.sum(axes=[1], keepdims=True)
    # wei.print()

    var bow2 = wei.matmul(X)
    assert_true(bow.all_close(bow2))
    var tril = Tensor[dtype].ones(T, T).tril()
    wei = Tensor[dtype].zeros(T, T)
    wei = wei.masked_fill(tril == 0, neg_inf[dtype]())
    wei.print()
    wei = wei.softmax(axes=[-1])
    print()
    wei.print()
    var bow3 = wei.matmul(X)
    assert_true(bow.all_close(bow3))


def attention() raises:
    inputs = Tensor[dtype].d2(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )
    _ = """query = inputs[i(1), s()]
    attn_scores_2 = Tensor[dtype].empty(inputs.shape()[0])
    for i, x_i in enumerate(inputs.slices()):
        attn_scores_2[i] = x_i.dot(query).item()

    attn_scores_2.print()

    var res: Float32 = 0
    row_0 = inputs[i(0), s()]
    for idx, element in enumerate(row_0):
        res += element[1] * query[idx]
    print(res)
    print(row_0.dot(query).item())
    var attn_weights_2 = attn_scores_2.softmax(axes=[0])
    attn_weights_2.print()
    attn_weights_2.sum().print()

    var context_vec_2 = Tensor[dtype].zeros(query.shape())
    for i, x_i in enumerate(inputs.slices()):
        context_vec_2 += attn_weights_2[i] * x_i

    context_vec_2.print()

    attn_scores = tenmo.empty(6, 6)
    for i, x_i in enumerate(inputs.slices()):
        for j, x_j in enumerate(inputs.slices()):
            attn_scores[i, j] = tenmo.dot(x_i, x_j).item()

    attn_scores.print()

    attn_scores = inputs.matmul(inputs.transpose())
    attn_weights = attn_scores.softmax(axes=[-1])
    attn_weights.print()
    attn_weights[i(1), s()].sum().print()
    attn_weights.sum(axes=[-1]).print()
    attn_weights.matmul(inputs).print()

    x_2 = inputs[i(1), s()]
    d_in = inputs.shape()[1]
    d_out = 2
    seed(123)
    var init_seed = 123
    W_query = Tensor[dtype].rand(
        d_in, d_out, init_seed=init_seed + 42, requires_grad=False
    )
    W_key = Tensor[dtype].rand(
        d_in, d_out, init_seed=init_seed + (-9), requires_grad=False
    )
    W_value = Tensor[dtype].rand(
        d_in, d_out, init_seed=init_seed + 101, requires_grad=False
    )

    print()
    W_query.print()
    print()
    W_key.print()
    print()
    W_value.print()

    query_2 = x_2.matmul(W_query)
    key_2 = x_2.matmul(W_key)
    value_2 = x_2.matmul(W_value)
    print("query_2 and key_2\n")
    query_2.print()
    key_2.print()

    keys = inputs.matmul(W_key)
    values = inputs.matmul(W_value)
    print("keys.shape:", keys.shape())
    print("values.shape:", values.shape())

    keys_2 = keys[i(1), s()]
    attn_score_22 = query_2.dot(keys_2)
    attn_score_22.print()

    attn_scores_2 = query_2.matmul(keys.transpose())
    attn_scores_2.print()

    d_k = Float32(keys.shape()[-1])
    attn_weights_2 = (attn_scores_2 / Float32(d_k**0.5)).softmax(axes=[-1])
    attn_weights_2.print()

    context_vec_2 = attn_weights_2.matmul(values)
    context_vec_2.print()"""
    seed(42)
    d_in = inputs.shape()[1]
    d_out = 2
    var sa_v1 = SelfAttention_v1[dtype](d_in, d_out)
    sa_v1(inputs).print()


@fieldwise_init
struct SelfAttention_v1[dtype: DType](ImplicitlyCopyable & Movable):
    var W_query: Tensor[Self.dtype]
    var W_key: Tensor[Self.dtype]
    var W_value: Tensor[Self.dtype]

    def __init__(out self, d_in: Int, d_out: Int):
        self.W_query = Tensor[Self.dtype].rand(d_in, d_out)
        self.W_key = Tensor[Self.dtype].rand(d_in, d_out)
        self.W_value = Tensor[Self.dtype].rand(d_in, d_out)

    def __call__(
        self, x: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var keys = x.matmul(self.W_key)
        var queries = x.matmul(self.W_query)
        var values = x.matmul(self.W_value)
        var attn_scores = queries.matmul(keys.transpose())
        var denom = Scalar[Self.dtype](math.sqrt(keys.shape()[-1]))
        var attn_weights = (attn_scores / denom).softmax(axes=[-1])
        var context_vec = attn_weights.matmul(values)
        return context_vec^


struct SelfAttention_v2[dtype: DType](ImplicitlyCopyable & Movable):
    var W_query: Linear[Self.dtype]
    var W_key: Linear[Self.dtype]
    var W_value: Linear[Self.dtype]

    def __init__(out self, d_in: Int, d_out: Int):
        self.W_query = Linear[Self.dtype](
            d_in, d_out, init_method="xavier", bias_zero=True
        )
        self.W_key = Linear[Self.dtype](
            d_in, d_out, init_method="xavier", bias_zero=True
        )
        self.W_value = Linear[Self.dtype](
            d_in, d_out, init_method="xavier", bias_zero=True
        )

    def __call__(
        mut self, mut x: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var keys = self.W_key(x)
        var queries = self.W_query(x)
        var values = self.W_value(x)
        var attn_scores = queries.matmul(keys.transpose())
        var denom = Scalar[Self.dtype](math.sqrt(keys.shape()[-1]))
        var attn_weights = (attn_scores / denom).softmax(axes=[-1])
        var context_vec = attn_weights.matmul(values)
        return context_vec^
