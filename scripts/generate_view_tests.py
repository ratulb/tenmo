"""Generate view tests for CPU and GPU.

Usage:
    python3 scripts/generate_view_tests.py

Output:
    - Appends CPU test functions to stdout (pipe to append to tests/test_views.mojo)
    - Writes GPU test file to tests/gpu/test_views_gpu.mojo

Each test is self-contained (no helpers) so the GPU chunk extractor
(generate_gpu_test_suite.py) can parse it correctly.
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_FILE = os.path.join(REPO, "tests", "gpu", "test_views_gpu.mojo")

# ============================================================
# Test catalog: tuples of (cpu_fn_body, gpu_fn_body)
# Each is a self-contained Mojo function definition.
# ============================================================

tests_cpu = []
tests_gpu = []


def add_test(name, cpu_body, gpu_body):
    cpu_fn = f"def test_view_{name}_cpu() raises:\n"
    cpu_fn += "    " + cpu_body.replace("\n", "\n    ") + "\n"
    tests_cpu.append(cpu_fn)

    gpu_fn = f"def test_view_{name}_gpu() raises:\n"
    gpu_fn += "    " + "comptime if has_accelerator():\n"
    gpu_body_indented = "        " + gpu_body.replace("\n", "\n        ")
    gpu_fn += gpu_body_indented + "\n"
    tests_gpu.append(gpu_fn)


# ============================================================
# Test definitions
# ============================================================

# --- into_view ---
add_test(
    "into_view_2d",
    # CPU
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var v = a.into_view()
assert_true(v.shape() == a.shape())
assert_true(v.strides() == a.strides())
assert_true(v.offset() == a.offset())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
    # GPU
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
assert_true(v.shape() == a.shape())
assert_true(v.strides() == a.strides())
assert_true(v.offset() == a.offset())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
)

add_test(
    "into_view_3d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var v = a3.into_view()
assert_true(v.shape() == a3.shape())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var a_gpu = a3.to_gpu(gpu)
var v = a_gpu.into_view()
assert_true(v.shape() == a3.shape())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
)

add_test(
    "into_view_scalar",
    """comptime dtype = DType.float32
var a = Tensor[dtype].scalar(42.0, requires_grad=True)
var v = a.into_view()
assert_true(v.shape() == Shape())
assert_true(v.item() == 42.0)
v.backward(1.0)
assert_true(a.grad().item() == 1.0)""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].scalar(42.0, requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
assert_true(v.shape() == Shape())
v.backward(1.0)
assert_true(a.grad().item() == 1.0)""",
)

add_test(
    "into_view_chain",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
var v1 = a.into_view()
var v2 = v1.into_view()
var v3 = v2.into_view()
var loss = v3.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.into_view()
var v2 = v1.into_view()
var v3 = v2.into_view()
var loss = v3.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)))""",
)

# --- view (shape) ---

add_test(
    "view_reshape_2d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(1.0, 7.0, requires_grad=True)
var v = a.view(2, 3)
assert_true(v.shape() == Shape(2, 3))
assert_true(v[0, 0] == 1.0)
assert_true(v[1, 2] == 6.0)
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))""",
    """comptime dtype = DType.float32
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
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))""",
)

add_test(
    "view_offset_1d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
var v = a.view(Shape(3), offset=2)
assert_true(v.shape() == Shape(3))
assert_true(v[0] == 2.0)
assert_true(v[2] == 4.0)
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.view(Shape(3), offset=2)
assert_true(v.shape() == Shape(3))
var v_cpu = v.to_cpu()
assert_true(v_cpu[0] == 2.0)
assert_true(v_cpu[2] == 4.0)
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
assert_true(a.grad().all_close(expected))""",
)

add_test(
    "view_strides_noncontiguous",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
var v = a.view(Shape(2, 2), Strides(1, 2))
assert_false(v.is_contiguous())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.view(Shape(2, 2), Strides(1, 2))
assert_false(v.is_contiguous())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 1.0)))""",
)

# --- transpose ---

add_test(
    "transpose_2d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var v = a.transpose()
assert_true(v.shape() == Shape(3, 2))
assert_true(v[0, 0] == 1.0)
assert_true(v[2, 1] == 6.0)
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.transpose()
assert_true(v.shape() == Shape(3, 2))
var v_cpu = v.to_cpu()
assert_true(v_cpu[0, 0] == 1.0)
assert_true(v_cpu[2, 1] == 6.0)
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
)

add_test(
    "transpose_3d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var v = a3.transpose(0, 2, 1)
assert_true(v.shape() == Shape(2, 4, 3))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var a_gpu = a3.to_gpu(gpu)
var v = a_gpu.transpose(0, 2, 1)
assert_true(v.shape() == Shape(2, 4, 3))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
)

add_test(
    "transpose_double",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
var v1 = a.transpose()
var v2 = v1.transpose()
assert_true(v2.shape() == a.shape())
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(3, 2), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.transpose()
var v2 = v1.transpose()
assert_true(v2.shape() == a.shape())
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(3, 2), 1.0)))""",
)

add_test(
    "transpose_weighted_grad",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.transpose()
var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
var prod = v * w
var loss = prod.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.transpose()
var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
var w_gpu = w.to_gpu(gpu)
var prod = v * w_gpu
var loss = prod.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))""",
)

# --- permute ---

add_test(
    "permute_2d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(1.0, 7.0, requires_grad=True)
var a2 = a.reshape(2, 3)
var v = a2.permute([1, 0])
assert_true(v.shape() == Shape(3, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(1.0, 7.0, requires_grad=True)
var a2 = a.reshape(2, 3)
var a_gpu = a2.to_gpu(gpu)
var v = a_gpu.permute([1, 0])
assert_true(v.shape() == Shape(3, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))""",
)

add_test(
    "permute_3d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var v = a3.permute([2, 0, 1])
assert_true(v.shape() == Shape(4, 2, 3))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var a_gpu = a3.to_gpu(gpu)
var v = a_gpu.permute([2, 0, 1])
assert_true(v.shape() == Shape(4, 2, 3))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
)

add_test(
    "permute_identity",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.permute([0, 1])
assert_true(v.shape() == a.shape())
assert_true(v.all_close(a))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.permute([0, 1])
assert_true(v.shape() == a.shape())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
)

# --- squeeze / unsqueeze ---

add_test(
    "unsqueeze_2d_to_3d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.unsqueeze(0)
assert_true(v.shape() == Shape(1, 2, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.unsqueeze(0)
assert_true(v.shape() == Shape(1, 2, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
)

add_test(
    "squeeze_3d_to_2d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
var v = a.squeeze(0)
assert_true(v.shape() == Shape(2, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(1, 2, 2), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.squeeze(0)
assert_true(v.shape() == Shape(2, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(1, 2, 2), 1.0)))""",
)

add_test(
    "unsqueeze_squeeze_chain",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var v1 = a.unsqueeze(1)
var v2 = v1.squeeze(1)
assert_true(v2.shape() == a.shape())
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.unsqueeze(1)
var v2 = v1.squeeze(1)
assert_true(v2.shape() == a.shape())
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
)

add_test(
    "squeeze_all_dims",
    """comptime dtype = DType.float32
var a = Tensor[dtype].full(Shape(1, 1, 3, 1), 5.0, requires_grad=True)
var v = a.squeeze([])
assert_true(v.shape() == Shape(3))
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].full(Shape(1, 1, 3, 1), 1.0)
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].full(Shape(1, 1, 3, 1), 5.0, requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.squeeze([])
assert_true(v.shape() == Shape(3))
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].full(Shape(1, 1, 3, 1), 1.0)
assert_true(a.grad().all_close(expected))""",
)

# --- expand ---

add_test(
    "expand_1d_to_2d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
var v = a.expand(4, 3)
assert_true(v.shape() == Shape(4, 3))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.expand(4, 3)
assert_true(v.shape() == Shape(4, 3))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))""",
)

add_test(
    "expand_col_to_matrix",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
var v = a.expand(3, 4)
assert_true(v.shape() == Shape(3, 4))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.expand(3, 4)
assert_true(v.shape() == Shape(3, 4))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))""",
)

add_test(
    "expand_3d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
var v = a.expand(3, 4, 2)
assert_true(v.shape() == Shape(3, 4, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d3([[[12.0, 12.0]]])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.expand(3, 4, 2)
assert_true(v.shape() == Shape(3, 4, 2))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d3([[[12.0, 12.0]]])))""",
)

add_test(
    "expand_weighted_grad",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
var v = a.expand(4, 3)
var w = Tensor[dtype].d2([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0], [2.0, 1.0, 2.0]])
var loss = (v * w).sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0]])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.expand(4, 3)
var w = Tensor[dtype].d2([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0], [2.0, 1.0, 2.0]])
var w_gpu = w.to_gpu(gpu)
var loss = (v * w_gpu).sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0]])))""",
)

# --- slice / __getitem__ ---

add_test(
    "slice_rows",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var v = a2[1:3, :]
assert_true(v.shape() == Shape(2, 4))
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var a_gpu = a2.to_gpu(gpu)
var v = a_gpu[1:3, :]
assert_true(v.shape() == Shape(2, 4))
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
assert_true(a.grad().all_close(expected))""",
)

add_test(
    "slice_step",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var v = a2[0:3:2, :]
assert_true(v.shape() == Shape(2, 4))
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var a_gpu = a2.to_gpu(gpu)
var v = a_gpu[0:3:2, :]
assert_true(v.shape() == Shape(2, 4))
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
assert_true(a.grad().all_close(expected))""",
)

add_test(
    "slice_single_element",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var v = a2[i(1), i(2)]
v.backward(1.0)
var expected = Tensor[dtype].zeros(12)
expected[6] = 1.0
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var a_gpu = a2.to_gpu(gpu)
var v = a_gpu[i(1), i(2)]
v.backward(1.0)
var expected = Tensor[dtype].zeros(12)
expected[6] = 1.0
assert_true(a.grad().all_close(expected))""",
)

add_test(
    "slice_newaxis",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
var v = a[newaxis, s(), newaxis]
assert_true(v.shape() == Shape(1, 3, 1))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu[newaxis, s(), newaxis]
assert_true(v.shape() == Shape(1, 3, 1))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))""",
)

# --- flatten ---

add_test(
    "flatten_3d",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var v = a3.flatten()
assert_true(v.shape() == Shape(24))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var a_gpu = a3.to_gpu(gpu)
var v = a_gpu.flatten()
assert_true(v.shape() == Shape(24))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
)

add_test(
    "flatten_partial",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var v = a3.flatten(start_dim=1)
assert_true(v.shape() == Shape(2, 12))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var a_gpu = a3.to_gpu(gpu)
var v = a_gpu.flatten(start_dim=1)
assert_true(v.shape() == Shape(2, 12))
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
)

# --- view chains ---

add_test(
    "chain_into_view_then_transpose",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var v1 = a.into_view()
var v2 = v1.transpose()
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.into_view()
var v2 = v1.transpose()
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
)

add_test(
    "chain_view_offset_then_transpose",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
var v1 = a.view(Shape(2, 3), offset=0)
var v2 = v1.transpose()
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.view(Shape(2, 3), offset=0)
var v2 = v1.transpose()
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(6), 1.0)))""",
)

add_test(
    "chain_view_offset_multi",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 20.0, requires_grad=True)
var v1 = a.view(Shape(6, 3), offset=2)
var loss = v1.sum()
loss.backward()
var expected = Tensor[dtype].zeros(20)
for i in range(2, 20):
    expected[i] = 1.0
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 20.0, requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.view(Shape(6, 3), offset=2)
var loss = v1.sum()
loss.backward()
var expected = Tensor[dtype].zeros(20)
for i in range(2, 20):
    expected[i] = 1.0
assert_true(a.grad().all_close(expected))""",
)

add_test(
    "chain_transpose_permute",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var v1 = a3.transpose(0, 2)
var v2 = v1.permute([1, 0, 2])
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 24.0, requires_grad=True)
var a3 = a.reshape(2, 3, 4)
var a_gpu = a3.to_gpu(gpu)
var v1 = a_gpu.transpose(0, 2)
var v2 = v1.permute([1, 0, 2])
var loss = v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(24), 1.0)))""",
)

add_test(
    "chain_slice_unsqueeze_expand",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var v1 = a2[0:2, :]
var v2 = v1.unsqueeze(0)
var v3 = v2.expand(3, 2, 4)
var loss = v3.sum()
loss.backward()
var expected = Tensor[dtype].d1([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 12.0, requires_grad=True)
var a2 = a.reshape(3, 4)
var a_gpu = a2.to_gpu(gpu)
var v1 = a_gpu[0:2, :]
var v2 = v1.unsqueeze(0)
var v3 = v2.expand(3, 2, 4)
var loss = v3.sum()
loss.backward()
var expected = Tensor[dtype].d1([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
assert_true(a.grad().all_close(expected))""",
)

# --- view gradbox zero ---

add_test(
    "gradbox_zero_single",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.into_view()
var loss = v.sum()
loss.backward()
var v_grad = v.grad()
assert_true(v_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
var loss = v.sum()
loss.backward()
var v_grad = v.grad()
assert_true(v_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
)

add_test(
    "gradbox_zero_chain",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v1 = a.into_view()
var v2 = v1.transpose()
var loss = v2.sum()
loss.backward()
var v1_grad = v1.grad()
assert_true(v1_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
var v2_grad = v2.grad()
assert_true(v2_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))""",
    """comptime dtype = DType.float32
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
assert_true(v2_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))""",
)

add_test(
    "gradbox_zero_complex_graph",
    """comptime dtype = DType.float32
var a1 = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a2 = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
var v1 = a1.into_view()
var s1 = v1.sum()
var v2 = a2.into_view()
var s2 = v2.sum()
var total = s1 + s2
total.backward()
var v1_grad = v1.grad()
assert_true(v1_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
var v2_grad = v2.grad()
assert_true(v2_grad.all_close(Tensor[dtype].zeros(Shape(2, 2))))
assert_true(a1.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))
assert_true(a2.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
    """comptime dtype = DType.float32
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
assert_true(a2.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
)

add_test(
    "gradbox_zero_two_backward_passes",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.into_view()
var loss1 = v.sum()
loss1.backward()
assert_true(v.grad().all_close(Tensor[dtype].zeros(Shape(2, 2))))
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
var loss1 = v.sum()
loss1.backward()
assert_true(v.grad().all_close(Tensor[dtype].zeros(Shape(2, 2))))
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
)

# --- ops on views ---

add_test(
    "view_mul_scalar",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.into_view()
var r = v * 2.0
var loss = r.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 2.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
var r = v * 2.0
var loss = r.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 2.0)))""",
)

add_test(
    "view_add_view",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var b = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
var va = a.into_view()
var vb = b.into_view()
var r = va + vb
var loss = r.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))
assert_true(b.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
    """comptime dtype = DType.float32
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
assert_true(b.grad().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))""",
)

add_test(
    "view_mul_view",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var b = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
var va = a.into_view()
var vb = b.into_view()
var r = va * vb
var loss = r.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])))
assert_true(b.grad().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))""",
    """comptime dtype = DType.float32
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
assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])))
assert_true(b.grad().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))""",
)

add_test(
    "view_sum_axis",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var v = a.into_view()
var loss = v.sum(axes=[1])
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
var loss = v.sum(axes=[1])
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
)

add_test(
    "view_broadcast_add",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var va = a.into_view()
var bias = Tensor[dtype].d1([10.0, 20.0, 30.0], requires_grad=True)
var vbias = bias.into_view()
var r = va + vbias
var loss = r.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))
assert_true(bias.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var va = a_gpu.into_view()
var bias = Tensor[dtype].d1([10.0, 20.0, 30.0], requires_grad=True)
var bias_gpu = bias.to_gpu(gpu)
var vbias = bias_gpu.into_view()
var r = va + vbias
var loss = r.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))
assert_true(bias.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))""",
)

# --- non-contiguous views ---

add_test(
    "noncontiguous_transpose_backward",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var t = a.transpose()
assert_false(t.is_contiguous())
var loss = t.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var t = a_gpu.transpose()
assert_false(t.is_contiguous())
var loss = t.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)))""",
)

add_test(
    "noncontiguous_strided_backward",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True)
var v = a.view(Shape(2, 4), Strides(1, 2))
assert_false(v.is_contiguous())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(8), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.view(Shape(2, 4), Strides(1, 2))
assert_false(v.is_contiguous())
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(8), 1.0)))""",
)

add_test(
    "noncontiguous_offset_backward",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True)
var v = a.view(Shape(2, 3), Strides(4, 1), offset=1)
assert_false(v.is_contiguous())
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
assert_true(a.grad().all_close(expected))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.view(Shape(2, 3), Strides(4, 1), offset=1)
assert_false(v.is_contiguous())
var loss = v.sum()
loss.backward()
var expected = Tensor[dtype].d1([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
assert_true(a.grad().all_close(expected))""",
)

# --- edge cases ---

add_test(
    "multiple_views_same_base",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
var v1 = a.into_view()
var v2 = a.view(2, 2)
var loss = v1.sum() + v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v1 = a_gpu.into_view()
var v2 = a_gpu.view(2, 2)
var loss = v1.sum() + v2.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)))""",
)

add_test(
    "view_track_grad_false",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var v = a.into_view[track_grad=False]()
assert_false(v.requires_grad)""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view[track_grad=False]()
assert_false(v.requires_grad)""",
)

add_test(
    "view_4d_backward",
    """comptime dtype = DType.float32
var a = Tensor[dtype].arange(0.0, 120.0, requires_grad=True)
var a4 = a.reshape(2, 3, 4, 5)
var v = a4.into_view()
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(120), 1.0)))""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].arange(0.0, 120.0, requires_grad=True)
var a4 = a.reshape(2, 3, 4, 5)
var a_gpu = a4.to_gpu(gpu)
var v = a_gpu.into_view()
var loss = v.sum()
loss.backward()
assert_true(a.grad().all_close(Tensor[dtype].full(Shape(120), 1.0)))""",
)

add_test(
    "view_data_sharing",
    """comptime dtype = DType.float32
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
var v = a.into_view()
a[0] = 99.0
assert_true(v[0] == 99.0)
v[1] = 88.0
assert_true(a[1] == 88.0)""",
    """comptime dtype = DType.float32
var gpu = GPU()
var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
var a_gpu = a.to_gpu(gpu)
var v = a_gpu.into_view()
a_gpu[0] = 99.0
assert_true(v.to_cpu()[0] == 99.0)
v[1] = 88.0
assert_true(a[1] == 88.0)""",
)

# ============================================================
# GPU file imports + main
# ============================================================

GPU_IMPORTS = """from tenmo.tensor import Tensor
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


"""

GPU_MAIN = """

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
"""


def generate():
    # Write GPU file
    os.makedirs(os.path.dirname(GPU_FILE), exist_ok=True)
    with open(GPU_FILE, "w") as f:
        f.write(GPU_IMPORTS)
        for fn in tests_gpu:
            f.write(fn + "\n")
        f.write(GPU_MAIN)

    # Print CPU tests to stdout (for appending to test_views.mojo)
    print("#" * 72)
    print("# Generated view tests — CPU (append to tests/test_views.mojo)")
    print("#" * 72)
    print()
    for fn in tests_cpu:
        print(fn)


if __name__ == "__main__":
    generate()
