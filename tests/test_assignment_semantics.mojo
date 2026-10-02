# test_assignment_semantics.mojo
# PyTorch copy-semantics parity:
#   - `var b = a`      → ALIAS (shares storage; b.id() == a.id())
#   - `t.clone()`      → independent deep copy (fresh _id, isolated storage)
#   - op results       → independent of operands (fresh _id, no storage share)
#   - view ops (transpose/reshape) still alias; contiguous() materializes

from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.shared.intarray import IntArray
from tenmo.shared.buffers import Buffer
from tenmo.numpy_interop import ndarray_ptr
from tenmo.ndbuffer import NDBuffer
from std.sys import has_accelerator
from std.python import Python
from std.testing import assert_true, TestSuite


def test_copy_init_aliases_cpu() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = a
    assert_true(b.id() == a.id())
    # Write through b, read through a
    b.buffer.data_buffer()[0] = 9.0
    assert_true(a.buffer.data_buffer()[0] == 9.0)
    # Write through a, read through b
    a.buffer.data_buffer()[2] = 7.0
    assert_true(b.buffer.data_buffer()[2] == 7.0)


def test_copy_calling_convention_aliases_cpu() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = a.copy()
    assert_true(b.id() == a.id())
    b.buffer.data_buffer()[1] = 5.0
    assert_true(a.buffer.data_buffer()[1] == 5.0)


def test_clone_independent_cpu() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var c = a.clone()
    assert_true(c.id() != a.id())
    assert_true(c.all_close(a))
    c.buffer.data_buffer()[0] = 9.0
    assert_true(a.buffer.data_buffer()[0] == 1.0)
    a.buffer.data_buffer()[1] = 5.0
    assert_true(c.buffer.data_buffer()[1] == 2.0)


def test_op_result_independent_of_operands() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([10.0, 20.0, 30.0])
    var r = a + b
    assert_true(r.id() != a.id())
    assert_true(r.id() != b.id())
    assert_true(r.all_close(Tensor[dtype].d1([11.0, 22.0, 33.0])))
    # Mutating the result never touches operands
    r.buffer.data_buffer()[0] = 99.0
    assert_true(a.buffer.data_buffer()[0] == 1.0)
    assert_true(b.buffer.data_buffer()[0] == 10.0)


def test_view_ops_still_alias() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    # transpose is a view — writes through it hit the parent storage
    var t = a.transpose(0, 1)
    t.buffer.data_buffer()[0] = 9.0
    assert_true(a.buffer.data_buffer()[0] == 9.0)
    # reshape of a contiguous shared tensor is a metadata view
    var v = a.reshape(4)
    v.buffer.data_buffer()[1] = 8.0
    assert_true(a.buffer.data_buffer()[1] == 8.0)


def test_contiguous_materializes_by_default() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    # owned=True (default): isolated copy even when already contiguous.
    var c = a.contiguous()
    c.buffer.data_buffer()[0] = 9.0
    assert_true(a.buffer.data_buffer()[0] == 1.0)
    # owned=False: fast-path alias when already contiguous+shared.
    var alias = a.contiguous(owned=False)
    alias.buffer.data_buffer()[0] = 9.0
    assert_true(a.buffer.data_buffer()[0] == 9.0)


def test_ndbuffer_copy_aliases_and_clone_independent() raises:
    var a = NDBuffer[DType.float32](Shape(3))
    for i in range(3):
        a.data_buffer()[i] = Float32(i)
    # copy() aliases (shared-from-birth buffers)
    var b = a.copy()
    b.data_buffer()[0] = 9.0
    assert_true(a.data_buffer()[0] == 9.0)
    # clone() is an isolated snapshot
    var c = a.clone()
    c.data_buffer()[0] = 7.0
    assert_true(a.data_buffer()[0] == 9.0)
    assert_true(c.data_buffer()[0] == 7.0)


def test_copy_init_aliases_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = a
        assert_true(b.id() == a.id())
        b.buffer.data_buffer()[0] = 9.0
        assert_true(a.buffer.data_buffer()[0] == 9.0)


def test_clone_independent_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var c = a.clone()
        assert_true(c.id() != a.id())
        c.buffer.data_buffer()[0] = 9.0
        assert_true(a.buffer.data_buffer()[0] == 1.0)


def test_view_ops_callable_on_borrowed_ref() raises:
    """View ops no longer require an exclusive borrow (`mut self`).

    Every read-only view op must compile and run through a `ref`-borrowed
    tensor (regression for the Q4 sweep — this failed to compile before).
    """
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    ref r = a
    var t = r.transpose()
    assert_true(t.all_close(a))
    var s = r.slice(1, 3)
    assert_true(s.all_close(Tensor[dtype].d1([2.0, 3.0])))
    var v = r.view(2, 2)
    assert_true(v.shape() == Shape(2, 2))
    var rsh = r.reshape(2, 2)
    assert_true(rsh.shape() == Shape(2, 2))
    var sq = r.squeeze()
    assert_true(sq.shape() == Shape(4))
    var us = r.unsqueeze(0)
    assert_true(us.shape() == Shape(1, 4))
    var pm = r.permute(IntArray(0))
    assert_true(pm.shape() == Shape(4))
    var dt = r.detach()
    assert_true(not dt.requires_grad)
    var sl = r[:2]
    assert_true(sl.all_close(Tensor[dtype].d1([1.0, 2.0])))
    var cg = r.contiguous()
    assert_true(cg.all_close(a))
    var rep = r.repeat(2)
    assert_true(rep.shape() == Shape(8))


def test_buffer_clone_external() raises:
    """Buffer.clone() deep-copies external (borrowed) sources instead of
    panicking (Q1: clone lifts the old shared_copy external panic)."""
    comptime dtype = DType.float32
    var np = Python.import_module("numpy")
    var py_list = Python.list(Float32(1.0), Float32(2.0), Float32(3.0))
    var arr = np.array(py_list, dtype=np.float32)
    var ptr = ndarray_ptr[DType.float32](arr).unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    var ext = Buffer[dtype](3, ptr, copy=False)
    assert_true(not ext.is_shared(), "external source stays unshared")
    var c = ext.clone()
    assert_true(c.is_shared(), "clone owns fresh shared storage")
    assert_true(c[0] == Float32(1.0))
    assert_true(c[2] == Float32(3.0))
    c[1] = Float32(9.0)
    assert_true(ext[1] == Float32(2.0), "clone is isolated from external source")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
