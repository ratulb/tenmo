"""GPU dtype-cast optimization tests.

Tests NDBuffer.to_dtype and Tensor.to_dtype on GPU for:
  1. Same-dtype: create_sub_buffer (zero-copy view)
  2. Cross-dtype: CastKernel.launch (single GPU kernel)
  3. Bool dtype handling (uint8 storage)

Run on a GPU box:
  ./execute.sh dtype_cast
  pixi run mojo -I . tests/test_dtype_cast_gpu.mojo
"""

from std.sys import has_accelerator, size_of
from std.testing import assert_true, TestSuite

from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.shared.shapes import Shape
from tenmo.gpu.device import DeviceState


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ═══════════════════════════════════════════════════════════════════════════════
# Same-dtype to_dtype — should be zero-copy on GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_dtype_cast_same_f32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(4096)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float32]()
        assert_true(result.to_cpu().all_close(a))


def test_dtype_cast_same_f16() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(2048).to_dtype[DType.float16]()
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float16]()
        assert_true(result.to_cpu().all_close(a))


def test_dtype_cast_same_i32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.int32].arange(1024)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.int32]()
        assert_true(result.to_cpu() == a)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-dtype to_dtype — CastKernel on GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_dtype_cast_f32_to_f16() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(4096)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float16]()
        var expected = a.to_dtype[DType.float16]()
        assert_true(result.to_cpu().all_close(expected))


def test_dtype_cast_f16_to_f32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(4096).to_dtype[DType.float16]()
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float32]()
        var expected = a.to_dtype[DType.float32]()
        assert_true(result.to_cpu().all_close(expected))


def test_dtype_cast_f32_to_f64() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(2048)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float64]()
        var expected = a.to_dtype[DType.float64]()
        assert_true(result.to_cpu().all_close(expected))


def test_dtype_cast_f64_to_f32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float64].arange(2048)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float32]()
        var expected = a.to_dtype[DType.float32]()
        assert_true(result.to_cpu().all_close(expected))


def test_dtype_cast_f16_to_f64() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(1024).to_dtype[DType.float16]()
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float64]()
        var expected = a.to_dtype[DType.float64]()
        assert_true(result.to_cpu().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# Float ↔ int conversions
# ═══════════════════════════════════════════════════════════════════════════════


def test_dtype_cast_f32_to_i32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(4096)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.int32]()
        var expected = a.to_dtype[DType.int32]()
        assert_true(result.to_cpu() == expected)


def test_dtype_cast_i32_to_f32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.int32].arange(4096)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.float32]()
        var expected = a.to_dtype[DType.float32]()
        assert_true(result.to_cpu().all_close(expected))


def test_dtype_cast_i64_to_i32() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.int64].arange(4096)
        var a_gpu = a.to_gpu()
        var result = a_gpu.to_dtype[DType.int32]()
        var expected = a.to_dtype[DType.int32]()
        assert_true(result.to_cpu() == expected)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU vs CPU round-trip equivalence
# ═══════════════════════════════════════════════════════════════════════════════


def test_dtype_cast_gpu_vs_cpu_f32_f16() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(16384)
        var a_gpu = a.to_gpu()
        var gpu_result = a_gpu.to_dtype[DType.float16]()
        var cpu_result = a.to_dtype[DType.float16]()
        assert_true(gpu_result.to_cpu().all_close(cpu_result))


def test_dtype_cast_gpu_vs_cpu_f32_f64() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(8192)
        var a_gpu = a.to_gpu()
        var gpu_result = a_gpu.to_dtype[DType.float64]()
        var cpu_result = a.to_dtype[DType.float64]()
        assert_true(gpu_result.to_cpu().all_close(cpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# NDBuffer-level tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_ndbuffer_to_dtype_same_f32() raises:
    comptime if has_accelerator():
        var state = DeviceState[DType.float32](4096)
        with state.buffer.map_to_host() as host:
            for i in range(4096):
                host[i] = Float32(i) * 0.5
        state.sync()

        var ndb = NDBuffer[DType.float32].with_device_state(state^, Shape(4096))
        var result = ndb.to_dtype[DType.float32]()

        if result.is_on_gpu():
            with result.device_state.value().buffer.map_to_host() as host:
                for i in range(4096):
                    assert_true(abs(Float64(host[i]) - Float64(i) * 0.5) < 1e-6)


def test_ndbuffer_to_dtype_cross_f32_f16() raises:
    comptime if has_accelerator():
        var state = DeviceState[DType.float32](2048)
        with state.buffer.map_to_host() as host:
            for i in range(2048):
                host[i] = Float32(i) * 0.1
        state.sync()

        var ndb = NDBuffer[DType.float32].with_device_state(state^, Shape(2048))
        var result = ndb.to_dtype[DType.float16]()

        assert_true(result.is_on_gpu())
        with result.device_state.value().buffer.map_to_host() as host:
            for i in range(2048):
                var expected = Float32(i) * 0.1
                var got = Float32(host[i])
                assert_true(abs(expected - got) < abs(expected) * 0.01 + 1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# size_of ratios — documentation tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_dtype_cast_sizeof_ratios() raises:
    comptime assert size_of[DType.float32]() == 4
    comptime assert size_of[DType.float16]() == 2
    comptime assert size_of[DType.float64]() == 8
    comptime assert size_of[DType.int32]() == 4
    comptime assert size_of[DType.int64]() == 8
    comptime assert size_of[DType.uint8]() == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Same-dtype independence — the cast result must own its storage (CPU parity)
# Fails while to_dtype returns an aliasing sub-buffer view; passes with a
# fresh device buffer (CastKernel identity copy).
# ═══════════════════════════════════════════════════════════════════════════════


def test_to_dtype_same_is_independent_gpu() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(64)
        var a_gpu = a.to_gpu()
        var b_gpu = a_gpu.to_dtype[DType.float32]()
        b_gpu += Float32(100.0)
        var expected = Tensor[DType.float32].arange(64)
        expected += Float32(100.0)
        assert_true(b_gpu.to_cpu().all_close(expected))
        # The source must be unchanged through the cast result.
        assert_true(a_gpu.to_cpu().all_close(a))


# ═══════════════════════════════════════════════════════════════════════════════
# Bool storage mapping (uint8 on GPU) + nonzero→True canonicalization
# ═══════════════════════════════════════════════════════════════════════════════


def test_to_dtype_bool_roundtrip_gpu() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].d1([True, False, True, False])
        var a_gpu = a_cpu.to_gpu()
        # Same-dtype bool through the uint8 storage path.
        var same = a_gpu.to_dtype[DType.bool]()
        assert_true(same.to_cpu() == a_cpu)
        # bool → f32 is exact.
        var as_f32 = a_gpu.to_dtype[DType.float32]()
        assert_true(
            as_f32.to_cpu().all_close(
                Tensor[DType.float32].d1([1.0, 0.0, 1.0, 0.0])
            )
        )


def test_to_dtype_f32_to_bool_nonzero_gpu() raises:
    comptime if has_accelerator():
        # CPU parity: fractional nonzero casts to True (not truncation to 0).
        var a_cpu = Tensor[DType.float32].d1([0.0, 0.5, 1.0, 2.0])
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.to_dtype[DType.bool]()
        assert_true(
            result.to_cpu()
            == Tensor[DType.bool].d1([False, True, True, True])
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Offset-view casts — slice on device (real offset), then cast
# ═══════════════════════════════════════════════════════════════════════════════


def test_to_dtype_offset_view_cast_gpu() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.float32].arange(128)
        var a_gpu = a.to_gpu()
        var v_gpu = a_gpu.slice(32, 96)
        var v_cpu = a.slice(32, 96)
        # Cross-dtype through an offset view.
        var r_gpu = v_gpu.to_dtype[DType.float16]()
        assert_true(r_gpu.to_cpu().all_close(v_cpu.to_dtype[DType.float16]()))
        # Same-dtype through an offset view.
        var s_gpu = v_gpu.to_dtype[DType.float32]()
        assert_true(s_gpu.to_cpu().all_close(v_cpu))
