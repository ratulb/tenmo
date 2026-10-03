from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from std.sys import has_accelerator
from tenmo.gpu.device import CPU, GPU

comptime F32 = DType.float32


def test_arange_device_gpu() raises:
    comptime if has_accelerator():
        print("test_arange_device_gpu")
        var a = Tensor[F32].arange(0, 5, device=GPU().into())
        var ref_cpu = Tensor[F32].arange(0, 5)
        assert_true(a.is_on_gpu())
        assert_true(a.to_cpu().all_close(ref_cpu))


def test_arange_device_default_cpu() raises:
    print("test_arange_device_default_cpu")
    var a = Tensor[F32].arange(0, 5)
    assert_true(not a.is_on_gpu())


def test_linspace_device_gpu() raises:
    comptime if has_accelerator():
        print("test_linspace_device_gpu")
        var l = Tensor[F32].linspace(0.0, 1.0, 5, device=GPU().into())
        var ref_cpu = Tensor[F32].linspace(0.0, 1.0, 5)
        assert_true(l.is_on_gpu())
        assert_true(l.to_cpu().all_close(ref_cpu))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
