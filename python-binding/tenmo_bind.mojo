"""Python bindings for Tenmo — a Mojo ML framework.

Build: pixi run mojo build python-binding/tenmo_bind.mojo -I . --emit shared-lib -o python-binding/_tenmo.so
Use:   PYTHONPATH=python-binding python -c "import tenmo; t = tenmo.tensor([1.0,2.0]); print(t.tolist())"
"""

from std.python import Python, PythonObject
from std.python.bindings import (
    PythonModuleBuilder,
    ExceptionType,
    raise_python_exception,
)
from std.python._cpython import PyObjectPtr
from std.python.numpy import copy_to_numpy_array
from tenmo.tensor import Tensor
from tenmo.net import (
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Flatten,
    Module,
    Sequential,
    mm,
)
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.optim import SGD
from tenmo.adamw import AdamW
from tenmo.mse import MSELoss
from tenmo.bceloss import BCEWithLogitsLoss, BCELoss
from tenmo.accuracy import Accuracy
from tenmo.dataloader import DataLoader, Batch
from tenmo.shared.shapes import Shape
from tenmo.shared.mnemonics import DEFAULT_INDEX_DTYPE
from tenmo.ndbuffer import NDBuffer
from tenmo.views import View
from tenmo.shared.indexhelper import Idx, NewAxis
from tenmo.shared.panic import panic
from tenmo.numpy_interop import (
    from_ndarray,
    numpy_dtype,
    mojo_dtype,
    dtype_from_string,
)
from std.ffi import c_long
from std.collections import StringDict

# ── Type aliases ─────────────────────────────────────────────────
# All 11 numpy-compatible Tensor dtypes registered as separate Python types.

comptime TF32 = Tensor[DType.float32]
comptime TI64 = Tensor[DType.int64]
comptime L32 = Linear[DType.float32]
comptime S32 = Sequential[DType.float32]
comptime M32 = Module[DType.float32]
comptime OPT32 = SGD[DType.float32]
comptime ADAMW32 = AdamW[DType.float32]
comptime CE32 = CrossEntropyLoss[DType.float32]
comptime MSE32 = MSELoss[DType.float32]
comptime BCE32 = BCELoss[DType.float32]
comptime BCEWL32 = BCEWithLogitsLoss[DType.float32]


# ── CPython helpers ──────────────────────────────────────────────

def _py_list_to_ints(py_list: PythonObject) raises -> List[Int]:
    var n = len(py_list)
    var result = List[Int](capacity=n)
    for i in range(n):
        result.append(Int(py=py_list[i]))
    return result^


def _py_list_to_tensors(
    py_list: PythonObject,
) raises -> List[Tensor[DType.float32]]:
    var n = len(py_list)
    var result = List[Tensor[DType.float32]](capacity=n)
    for i in range(n):
        result.append(
            py_list[i].downcast_value_ptr[TF32]()[]
        )
    return result^


def _tensor_shape_to_py[
    dtype: DType
](t: Tensor[dtype]) raises -> PythonObject:
    ref cpy = Python().cpython()
    var ndim = t.rank()
    var tup = cpy.PyTuple_New(ndim)
    for i in range(ndim):
        _ = cpy.PyTuple_SetItem(tup, i, cpy.PyLong_FromSsize_t(t.shape()[i]))
    return PythonObject(from_owned=tup)


# ── Generic tensor handlers (dtype-agnostic) ────────────────────
# These work identically for float32, int64, and bool tensors.


# ── Tensor init helpers ────────────────────────────────────────
# Each dtype needs its own init that returns the correct type.

def _init_tensor_generic[
    dtype: DType
](args: PythonObject, kwargs: PythonObject) raises -> Tensor[dtype]:
    """Generic Tensor init from Python list or numpy array. Preserves dtype.

    Note: CPython passes kwds=NULL to tp_init when no keyword arguments are
    given; the wrapper wraps that in a NULL-backed PythonObject, so any
    operation on kwargs must first check `kwargs._obj_ptr`.
    """
    var requires_grad = False
    if kwargs._obj_ptr and "requires_grad" in kwargs:
        requires_grad = Bool(py=kwargs["requires_grad"])
    var py_data = args[0]
    var builtins = Python.import_module("builtins")
    if builtins.hasattr(py_data, "dtype"):
        var np = Python.import_module("numpy")
        return from_ndarray[dtype](
            np.asarray(py_data, dtype=numpy_dtype(dtype)),
            requires_grad=requires_grad,
        )
    # Plain Python list: bulk-convert through numpy (nesting, ragged
    # validation and casting in C) + one from_ndarray memcpy. Ragged
    # input raises ValueError loudly out of np.asarray.
    return _list_to_tensor[dtype](py_data, requires_grad)


def _list_to_tensor[
    dtype: DType
](data: PythonObject, requires_grad: Bool) raises -> Tensor[dtype]:
    var np = Python.import_module("numpy")
    return from_ndarray[dtype](
        np.asarray(data, dtype=numpy_dtype(dtype)),
        requires_grad=requires_grad,
    )


def _generic_shape[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    return _tensor_shape_to_py[dtype](self.downcast_value_ptr[Tensor[dtype]]()[])


def _generic_ndim[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    ref cpy = Python().cpython()
    return PythonObject(
        from_owned=cpy.PyLong_FromSsize_t(
            self.downcast_value_ptr[Tensor[dtype]]()[].rank()
        )
    )


def _generic_numels[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    ref cpy = Python().cpython()
    return PythonObject(
        from_owned=cpy.PyLong_FromSsize_t(
            self.downcast_value_ptr[Tensor[dtype]]()[].numels()
        )
    )


def _generic_requires_grad[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    ref cpy = Python().cpython()
    var rg = self.downcast_value_ptr[Tensor[dtype]]()[].requires_grad
    if rg:
        return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(1)))
    else:
        return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(0)))


# __repr__/__str__ are served by the type builder's tp_repr slot, which calls
# Tensor.write_repr_to — no method-table entry needed (slots win anyway).


def _generic_numpy_dtype[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    _ = self
    return numpy_dtype(dtype)


def _scalar_to_py[
    dtype: DType
](v: Scalar[dtype]) raises -> PythonObject:
    """Marshal one tensor element to its natural Python type."""
    ref cpy = Python().cpython()
    comptime if dtype.is_floating_point():
        return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(v)))
    elif dtype == DType.bool:
        if Bool(v):
            return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(1)))
        else:
            return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(0)))
    else:
        return PythonObject(from_owned=cpy.PyLong_FromSsize_t(Int(v)))


def _tolist_build[
    dtype: DType
](data: List[Scalar[dtype]], dims: List[Int], pos: Int) raises -> PythonObject:
    """Recursively build shape-nested Python lists from row-major flat data."""
    ref cpy = Python().cpython()
    var dim0 = dims[0]
    var list_ptr = cpy.PyList_New(dim0)
    if len(dims) == 1:
        for i in range(dim0):
            _ = cpy.PyList_SetItem(
                list_ptr, i, _scalar_to_py[dtype](data[pos + i]).steal_data()
            )
    else:
        var block = 1
        var rest = List[Int]()
        for j in range(1, len(dims)):
            rest.append(dims[j])
            block *= dims[j]
        for i in range(dim0):
            _ = cpy.PyList_SetItem(
                list_ptr,
                i,
                _tolist_build[dtype](data, rest, pos + i * block)
                    .steal_data(),
            )
    return PythonObject(from_owned=list_ptr)


def _generic_tolist[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    """Converts to list, nested per shape (numpy-compatible)."""
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    if t.rank() == 0:
        return _scalar_to_py[dtype](t.item())
    var data = t.tolist()
    var dims = List[Int]()
    for i in range(t.rank()):
        dims.append(t.shape()[i])
    return _tolist_build[dtype](data, dims^, 0)


def _generic_item[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    """Returns a properly typed Python scalar."""
    _ = args
    ref cpy = Python().cpython()
    var v = self.downcast_value_ptr[Tensor[dtype]]()[].item()
    comptime if dtype.is_floating_point():
        return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(v)))
    elif dtype == DType.bool:
        if Bool(v):
            return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(1)))
        else:
            return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(0)))
    else:
        return PythonObject(from_owned=cpy.PyLong_FromSsize_t(Int(v)))


def _generic_numpy[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    """Single-copy Tensor -> numpy ndarray (C-contiguous, correct dtype).

    Fast path (contiguous + zero offset): one memcpy straight out of the
    tensor buffer, then a no-copy reshape view. Otherwise one owned
    contiguous copy first. Either way exactly one bulk copy — the old
    facade path (tolist + asarray) copied two to three times.
    """
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype == DType.bool:
        # copy_to_numpy_array has no bool flavor: nest via the C-level
        # list builder, one asarray on top.
        var np = Python.import_module("numpy")
        return np.asarray(
            _generic_tolist[dtype](self, args), numpy_dtype(dtype)
        )
    else:
        _ = args
        var arr: PythonObject
        if t.is_contiguous() and t.offset() == 0:
            var span = Span[Scalar[dtype], MutAnyOrigin](
                unsafe_ptr=t.data_ptr(), length=t.numels()
            )
            arr = copy_to_numpy_array(span)
        else:
            # track_grad=False: a numpy export never joins the autograd
            # graph — erase the BackwardFn machinery per dtype.
            var c = t.contiguous[track_grad=False]()
            var span = Span[Scalar[dtype], MutAnyOrigin](
                unsafe_ptr=c.data_ptr(), length=c.numels()
            )
            arr = copy_to_numpy_array(span)
        return arr.reshape(_tensor_shape_to_py[dtype](t))


# ── Generic per-dtype Tensor registration ────────────────────────
# Registers the common surface (metadata + typed item + nested tolist)
# for any dtype. Call once per dtype; each Tensor[dtype] must be bound
# exactly once. Extra dtype-specific methods can be appended afterwards
# via mb.type_builders[len(mb.type_builders) - 1].

def register_tensor[
    dtype: DType
](mut mb: PythonModuleBuilder, name: StringSlice[ImmStaticOrigin]) raises:
    ref b = mb.add_type[Tensor[dtype]](name)
    _ = b.def_py_init[_init_tensor_generic[dtype]]()
    _ = b.def_py_method[_generic_shape[dtype]]("shape")
    _ = b.def_py_method[_generic_ndim[dtype]]("ndim")
    _ = b.def_py_method[_generic_numels[dtype]]("numels")
    _ = b.def_py_method[_generic_requires_grad[dtype]]("requires_grad")
    _ = b.def_py_method[_generic_numpy_dtype[dtype]]("numpy_dtype")
    _ = b.def_py_method[_generic_item[dtype]]("item")
    _ = b.def_py_method[_generic_tolist[dtype]]("tolist")
    _ = b.def_py_method[_generic_numpy[dtype]]("numpy")
    # ── comparisons (dtype-generic: compare/compare_scalar; safe for
    # every dtype incl. bool — returns BoolTensor) ──
    _ = b.def_py_method[_generic_eq[dtype]]("eq")
    _ = b.def_py_method[_generic_ne[dtype]]("ne")
    _ = b.def_py_method[_generic_gt[dtype]]("gt")
    _ = b.def_py_method[_generic_lt[dtype]]("lt")
    _ = b.def_py_method[_generic_ge[dtype]]("ge")
    _ = b.def_py_method[_generic_le[dtype]]("le")
    _ = b.def_py_method[_generic_gt_tensor[dtype]]("gt_tensor")
    _ = b.def_py_method[_generic_lt_tensor[dtype]]("lt_tensor")
    _ = b.def_py_method[_generic_ge_tensor[dtype]]("ge_tensor")
    _ = b.def_py_method[_generic_le_tensor[dtype]]("le_tensor")
    _ = b.def_py_method[_generic_is_leaf[dtype]]("is_leaf")
    comptime if dtype == DType.float32 or dtype == DType.int64 or dtype == DType.float64:
        # Arithmetic + in-place surface covers float32/int64/float64.
        # Each additional dtype family costs ~2-4GB of compile-time memory
        # because def_py_method materializes every handler plus the core
        # ops it reaches. Widen one dtype at a time, building under a cap:
        #   systemd-run --scope --user -p MemoryMax=14336M pixi run mojo build -j2 ...
        _ = b.def_py_method[_generic_add[dtype]]("add")
        _ = b.def_py_method[_generic_sub[dtype]]("sub")
        _ = b.def_py_method[_generic_mul[dtype]]("mul")
        _ = b.def_py_method[_generic_add_scalar[dtype]]("add_scalar")
        _ = b.def_py_method[_generic_sub_scalar[dtype]]("sub_scalar")
        _ = b.def_py_method[_generic_mul_scalar[dtype]]("mul_scalar")
        _ = b.def_py_method[_generic_rsub_scalar[dtype]]("rsub_scalar")
        _ = b.def_py_method[_generic_truediv[dtype]]("truediv")
        _ = b.def_py_method[_generic_truediv_scalar[dtype]](
            "truediv_scalar"
        )
        _ = b.def_py_method[_generic_rtruediv_scalar[dtype]](
            "rtruediv_scalar"
        )
        _ = b.def_py_method[_generic_neg[dtype]]("neg")
        _ = b.def_py_method[_generic_pow[dtype]]("pow")
        _ = b.def_py_method[_generic_iadd[dtype]]("iadd")
        _ = b.def_py_method[_generic_isub[dtype]]("isub")
        _ = b.def_py_method[_generic_imul[dtype]]("imul")
        _ = b.def_py_method[_generic_itruediv[dtype]]("itruediv")
        _ = b.def_py_method[_generic_iadd_scalar[dtype]]("iadd_scalar")
        _ = b.def_py_method[_generic_isub_scalar[dtype]]("isub_scalar")
        _ = b.def_py_method[_generic_imul_scalar[dtype]]("imul_scalar")
        _ = b.def_py_method[_generic_itruediv_scalar[dtype]](
            "itruediv_scalar"
        )
        _ = b.def_py_method[_generic_index[dtype]]("index")
        _ = b.def_py_method[_generic_setitem_scalar[dtype]](
            "setitem_scalar"
        )
        _ = b.def_py_method[_generic_setitem_tensor[dtype]](
            "setitem_tensor"
        )


# ── Scalar marshalling (dtype-family exact) ───────────────────────

def _py_to_scalar[
    dtype: DType
](v: PythonObject) raises -> Scalar[dtype]:
    """Marshal a Python scalar to Scalar[dtype], via the CPython C-API.

    Same acceptance as the ConvertibleFromPython constructors (Python +
    numpy scalars, __int__/__float__/__bool__ carriers) with loud errors,
    minus one layer of PythonObject dynamic dispatch per conversion.
    """
    ref cpy = Python().cpython()
    var ptr = v._obj_ptr
    comptime if dtype == DType.bool:
        var truth = cpy.PyObject_IsTrue(ptr)
        if truth < 0:
            raise cpy.get_error()
        return Scalar[dtype](UInt8(1 if truth == 1 else 0))
    elif dtype.is_floating_point():
        var f = cpy.PyFloat_AsDouble(ptr)
        if f == -1.0 and cpy.PyErr_Occurred():
            raise cpy.get_error()
        return Scalar[dtype](Float64(f))
    else:
        var tmp = cpy.PyNumber_Long(ptr)
        if not tmp:
            raise cpy.get_error()
        var i = cpy.PyLong_AsSsize_t(tmp)
        var failed = cpy.PyErr_Occurred()
        cpy.Py_DecRef(tmp)
        if failed:
            raise cpy.get_error()
        return Scalar[dtype](Int64(i))


# ── Arithmetic: tensor-tensor ────────────────────────────────────

def _generic_add[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__add__[True](b))
    else:
        return PythonObject(alloc=a.__add__[False](b))


def _generic_sub[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__sub__[True](b))
    else:
        return PythonObject(alloc=a.__sub__[False](b))


def _generic_mul[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__mul__[True](b))
    else:
        return PythonObject(alloc=a.__mul__[False](b))


def _generic_truediv[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__truediv__[True](b))
    else:
        return PythonObject(alloc=a.__truediv__[False](b))


# ── Arithmetic: tensor-scalar ────────────────────────────────────

def _generic_add_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__add__[True](_py_to_scalar[dtype](args[0])))
    else:
        return PythonObject(alloc=a.__add__[False](_py_to_scalar[dtype](args[0])))


def _generic_sub_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__sub__[True](_py_to_scalar[dtype](args[0])))
    else:
        return PythonObject(alloc=a.__sub__[False](_py_to_scalar[dtype](args[0])))


def _generic_mul_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__mul__[True](_py_to_scalar[dtype](args[0])))
    else:
        return PythonObject(alloc=a.__mul__[False](_py_to_scalar[dtype](args[0])))


def _generic_truediv_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(
            alloc=a.__truediv__[True](_py_to_scalar[dtype](args[0]))
        )
    else:
        return PythonObject(
            alloc=a.__truediv__[False](_py_to_scalar[dtype](args[0]))
        )


def _generic_rsub_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__rsub__[True](_py_to_scalar[dtype](args[0])))
    else:
        return PythonObject(alloc=a.__rsub__[False](_py_to_scalar[dtype](args[0])))


def _generic_rtruediv_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(
            alloc=a.__rtruediv__[True](_py_to_scalar[dtype](args[0]))
        )
    else:
        return PythonObject(
            alloc=a.__rtruediv__[False](_py_to_scalar[dtype](args[0]))
        )


def _generic_pow[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(
            alloc=a.__pow__[True](_py_to_scalar[dtype](args[0]))
        )
    else:
        return PythonObject(
            alloc=a.__pow__[False](_py_to_scalar[dtype](args[0]))
        )


# ── Unary minus ──────────────────────────────────────────────────

def _generic_neg[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    comptime if dtype.is_floating_point():
        return PythonObject(alloc=a.__neg__[True]())
    else:
        return PythonObject(alloc=a.__neg__[False]())


# ── Comparisons → BoolTensor ─────────────────────────────────────

def _generic_eq[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.eq(b))


def _generic_ne[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.ne(b))


def _generic_gt[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.gt(_py_to_scalar[dtype](args[0])))


def _generic_lt[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.lt(_py_to_scalar[dtype](args[0])))


def _generic_ge[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.__ge__(_py_to_scalar[dtype](args[0])))


def _generic_le[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.__le__(_py_to_scalar[dtype](args[0])))


def _generic_gt_tensor[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.__gt__(b))


def _generic_lt_tensor[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.__lt__(b))


def _generic_ge_tensor[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.__ge__(b))


def _generic_le_tensor[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.__le__(b))


# ── In-place ─────────────────────────────────────────────────────

def _generic_iadd[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__iadd__(
        args[0].downcast_value_ptr[Tensor[dtype]]()[]
    )
    return PythonObject(None)


def _generic_isub[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__isub__(
        args[0].downcast_value_ptr[Tensor[dtype]]()[]
    )
    return PythonObject(None)


def _generic_imul[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__imul__(
        args[0].downcast_value_ptr[Tensor[dtype]]()[]
    )
    return PythonObject(None)


def _generic_itruediv[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__itruediv__(
        args[0].downcast_value_ptr[Tensor[dtype]]()[]
    )
    return PythonObject(None)


def _generic_iadd_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__iadd__(
        _py_to_scalar[dtype](args[0])
    )
    return PythonObject(None)


def _generic_isub_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__isub__(
        _py_to_scalar[dtype](args[0])
    )
    return PythonObject(None)


def _generic_imul_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__imul__(
        _py_to_scalar[dtype](args[0])
    )
    return PythonObject(None)


def _generic_itruediv_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy: in-place ops must mutate through the stored value.
    self.downcast_value_ptr[Tensor[dtype]]()[].__itruediv__(
        _py_to_scalar[dtype](args[0])
    )
    return PythonObject(None)


# ── Indexing / slicing ───────────────────────────────────────────
# All three handlers consume four parallel python int lists
# (kinds, starts, stops, steps) + optional trailing flags.
# Lane kinds: 0 = integer index lane, 1 = slice lane, 2 = newaxis lane.
# The wrapper pre-normalizes slices (positive step only) and pre-checks
# integer bounds so a core panic (abort) can never fire from
# Python-land input; reversed-slice and bool/fancy indexing raise in
# the wrapper (intentional deviations from core semantics).

def _index_lanes_to_idx(
    py_kinds: PythonObject,
    py_starts: PythonObject,
    py_stops: PythonObject,
    py_steps: PythonObject,
) raises -> List[Idx]:
    var kinds = _py_list_to_ints(py_kinds)
    var starts = _py_list_to_ints(py_starts)
    var stops = _py_list_to_ints(py_stops)
    var steps = _py_list_to_ints(py_steps)
    var lanes = List[Idx](capacity=len(kinds))
    for k in range(len(kinds)):
        if kinds[k] == 0:
            lanes.append(Idx(starts[k]))
        elif kinds[k] == 1:
            lanes.append(Idx(slice(starts[k], stops[k], steps[k])))
        elif kinds[k] == 2:
            lanes.append(Idx(NewAxis()))
        else:
            panic("Unknown Python-bindings index lane kind")
    return lanes^


def _generic_index[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    # No `var a =` copy-out: `downcast_value_ptr()[][]` is a DEEP copy of the
    # stored Tensor (buffer.copy()), which would sever view/buffer sharing.
    # Pass the stored pointee by reference into View.forward_list.
    var lanes = _index_lanes_to_idx(args[0], args[1], args[2], args[3])
    var track = len(args) > 4 and Int(py=args[4]) != 0
    comptime if dtype.is_floating_point():
        var requires_grad: Optional[Bool] = None
        if not track:
            requires_grad = Optional[Bool](False)
        return PythonObject(
            alloc=View[dtype].forward_list[True](
                self.downcast_value_ptr[Tensor[dtype]]()[],
                lanes,
                requires_grad,
                sync=True,
            )
        )
    else:
        return PythonObject(
            alloc=View[dtype].forward_list[False](
                self.downcast_value_ptr[Tensor[dtype]]()[],
                lanes,
                sync=True,
            )
        )


def _generic_setitem_scalar[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var lanes = _index_lanes_to_idx(args[0], args[1], args[2], args[3])
    self.downcast_value_ptr[Tensor[dtype]]()[].fill(
        _py_to_scalar[dtype](args[4]), lanes
    )
    return PythonObject(None)


def _generic_setitem_tensor[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var src = args[4].downcast_value_ptr[Tensor[dtype]]()[]
    var lanes = _index_lanes_to_idx(args[0], args[1], args[2], args[3])
    self.downcast_value_ptr[Tensor[dtype]]()[].fill(src, lanes)
    return PythonObject(None)


# ── Misc ─────────────────────────────────────────────────────────

def _generic_is_leaf[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    ref cpy = Python().cpython()
    if self.downcast_value_ptr[Tensor[dtype]]()[].is_leaf():
        return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(1)))
    else:
        return PythonObject(from_owned=cpy.PyBool_FromLong(c_long(0)))


# ── Generic floating-point tensor handlers (f32 + f64) ─────────
# Instantiated per dtype in PyInit (f32 chain + f64 chain). Integer and
# bool dtypes expose only the common surface in register_tensor.

def _tensor_sum[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    var keepdims = False
    if len(args) > 1:
        keepdims = Bool(py=args[1])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.sum(axes=axes, keepdims=keepdims))


def _tensor_mean[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    var keepdims = False
    if len(args) > 1:
        keepdims = Bool(py=args[1])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.mean(axes=axes, keepdims=keepdims))


def _tensor_product[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    var keepdims = False
    if len(args) > 1:
        keepdims = Bool(py=args[1])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.product(axes=axes, keepdims=keepdims))


def _tensor_variance[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axis = -100
    if len(args) > 0:
        axis = Int(py=args[0])
    var keepdims = False
    if len(args) > 1:
        keepdims = Bool(py=args[1])
    var unbiased = True
    if len(args) > 2:
        unbiased = Bool(py=args[2])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(
        alloc=t.variance(axis=axis, keepdims=keepdims, unbiased=unbiased)
    )


def _tensor_std[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axis = -100
    if len(args) > 0:
        axis = Int(py=args[0])
    var keepdims = False
    if len(args) > 1:
        keepdims = Bool(py=args[1])
    var unbiased = True
    if len(args) > 2:
        unbiased = Bool(py=args[2])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(
        alloc=t.std(axis=axis, keepdims=keepdims, unbiased=unbiased)
    )


def _tensor_norm[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var p_val = 2.0
    if len(args) > 0:
        p_val = Float64(py=args[0])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.norm(p=p_val))


def _tensor_max[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.max(axes=axes))


def _tensor_min[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.min(axes=axes))


def _tensor_argmax[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axis = 0
    if len(args) > 0:
        axis = Int(py=args[0])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var result = t.argmax(axis=axis)
    ref cpy = Python().cpython()
    if result.numels() == 1:
        return PythonObject(
            from_owned=cpy.PyLong_FromSsize_t(Int(result.item()))
        )
    var data = result.tolist()
    var list_ptr = cpy.PyList_New(len(data))
    for i in range(len(data)):
        _ = cpy.PyList_SetItem(
            list_ptr, i, cpy.PyLong_FromSsize_t(Int(data[i]))
        )
    return PythonObject(from_owned=list_ptr)


def _tensor_argmin[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axis = 0
    if len(args) > 0:
        axis = Int(py=args[0])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var result = t.argmin(axis=axis)
    ref cpy = Python().cpython()
    if result.numels() == 1:
        return PythonObject(
            from_owned=cpy.PyLong_FromSsize_t(Int(result.item()))
        )
    var data = result.tolist()
    var list_ptr = cpy.PyList_New(len(data))
    for i in range(len(data)):
        _ = cpy.PyList_SetItem(
            list_ptr, i, cpy.PyLong_FromSsize_t(Int(data[i]))
        )
    return PythonObject(from_owned=list_ptr)


def _tensor_softmax[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.softmax(axes=axes))


def _tensor_flatten[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var start_dim = 0
    var end_dim = -1
    if len(args) > 0:
        start_dim = Int(py=args[0])
    if len(args) > 1:
        end_dim = Int(py=args[1])
    if end_dim == -1:
        return PythonObject(
            alloc=self.downcast_value_ptr[Tensor[dtype]]()[].flatten(
                start_dim=start_dim, end_dim=None
            )
        )
    return PythonObject(
        alloc=self.downcast_value_ptr[Tensor[dtype]]()[].flatten(
            start_dim=start_dim, end_dim=end_dim
        )
    )


def _tensor_reshape[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var shape_list = _py_list_to_ints(args[0])
    return PythonObject(alloc=self.downcast_value_ptr[Tensor[dtype]]()[].reshape(Shape(shape_list)))


def _tensor_matmul[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var A = self.downcast_value_ptr[Tensor[dtype]]()[]
    var B = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=A.matmul(B))


def _tensor_clip[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var min_val = Float64(py=args[0])
    var max_val = Float64(py=args[1])
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(
        alloc=t.clip(
            Scalar[dtype](min_val),
            Scalar[dtype](max_val),
        )
    )


def _tensor_abs[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=abs(t))


# ── Shape / view ops ────────────────────────────────────────────────


def _tensor_transpose[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    return PythonObject(alloc=self.downcast_value_ptr[Tensor[dtype]]()[].transpose(axes))


def _tensor_permute[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = _py_list_to_ints(args[0])
    return PythonObject(alloc=self.downcast_value_ptr[Tensor[dtype]]()[].permute(axes))


def _tensor_squeeze[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = List[Int]()
    if len(args) > 0:
        axes = _py_list_to_ints(args[0])
    return PythonObject(alloc=self.downcast_value_ptr[Tensor[dtype]]()[].squeeze(axes))


def _tensor_unsqueeze[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var axes = _py_list_to_ints(args[0])
    return PythonObject(alloc=self.downcast_value_ptr[Tensor[dtype]]()[].unsqueeze(axes))


def _tensor_expand[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var target = _py_list_to_ints(args[0])
    return PythonObject(alloc=self.downcast_value_ptr[Tensor[dtype]]()[].expand(Shape(target)))


def _tensor_contiguous[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var owned = Bool(py=args[0]) if len(args) > 0 else True
    return PythonObject(
        alloc=self.downcast_value_ptr[Tensor[dtype]]()[].contiguous(owned=owned)
    )


def _tensor_masked_fill[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var mask = args[0].downcast_value_ptr[Tensor[DType.bool]]()[]
    var value = Scalar[dtype](Float64(py=args[1]))
    return PythonObject(alloc=t.masked_fill(mask, value))


# ── Dtype / device / detach ops ───────────────────────────────────


def _tensor_detach[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.detach())


def _tensor_is_contiguous[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(t.is_contiguous())


def _tensor_device[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var d = t.device()
    if d.is_gpu():
        return PythonObject("cuda:0")
    return PythonObject("cpu")


def _to_dtype_dispatch[
    dtype: DType
](
    src: Tensor[dtype], dt: DType
) raises -> PythonObject:
    if dt == DType.float16:
        return PythonObject(alloc=src.to_dtype[DType.float16]())
    elif dt == DType.float32:
        return PythonObject(alloc=src.to_dtype[DType.float32]())
    elif dt == DType.float64:
        return PythonObject(alloc=src.to_dtype[DType.float64]())
    elif dt == DType.int8:
        return PythonObject(alloc=src.to_dtype[DType.int8]())
    elif dt == DType.int16:
        return PythonObject(alloc=src.to_dtype[DType.int16]())
    elif dt == DType.int32:
        return PythonObject(alloc=src.to_dtype[DType.int32]())
    elif dt == DType.int64:
        return PythonObject(alloc=src.to_dtype[DType.int64]())
    elif dt == DType.uint8:
        return PythonObject(alloc=src.to_dtype[DType.uint8]())
    elif dt == DType.uint16:
        return PythonObject(alloc=src.to_dtype[DType.uint16]())
    elif dt == DType.uint32:
        return PythonObject(alloc=src.to_dtype[DType.uint32]())
    elif dt == DType.uint64:
        return PythonObject(alloc=src.to_dtype[DType.uint64]())
    elif dt == DType.bool:
        return PythonObject(alloc=src.to_dtype[DType.bool]())
    else:
        raise Error("unsupported dtype")


def _tensor_to_dtype[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var dt = dtype_from_string(String(py=args[0]))
    return _to_dtype_dispatch[dtype](t, dt)


def _tensor_zeros_like[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(
        alloc=Tensor[dtype].zeros(
            t.shape(), requires_grad=t.requires_grad
        )
    )


def _tensor_ones_like[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(
        alloc=Tensor[dtype].ones(
            t.shape(), requires_grad=t.requires_grad
        )
    )


def _tensor_triu[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var diagonal = 0
    if len(args) > 0:
        diagonal = Int(py=args[0])
    return PythonObject(alloc=t.triu(diagonal=diagonal))


def _tensor_tril[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var diagonal = 0
    if len(args) > 0:
        diagonal = Int(py=args[0])
    return PythonObject(alloc=t.tril(diagonal=diagonal))


def _tensor_cumsum[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var axis = 0
    if len(args) > 0:
        axis = Int(py=args[0])
    return PythonObject(alloc=t.cumsum(axis=axis))


def _tensor_dot[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.dot(b))


def _tensor_outer[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.outer(b))


def _tensor_gather[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var py_indices = args[0]
    var indices = List[Int]()
    for i in range(len(py_indices)):
        indices.append(Int(py=py_indices[i]))
    var axis = 0
    if len(args) > 1:
        axis = Int(py=args[1])
    return PythonObject(alloc=t.gather(indices=indices, axis=axis))


def _tensor_mse[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.mse(b))


def _tensor_bce[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.binary_cross_entropy(b))


def _tensor_bce_logits[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var a = self.downcast_value_ptr[Tensor[dtype]]()[]
    var b = args[0].downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=a.binary_cross_entropy_with_logits(b))


def _tensor_seed_grad[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    var val = Float64(1.0)
    if len(args) > 0:
        val = Float64(py=args[0])
    t.seed_grad(Scalar[dtype](val))
    return PythonObject(None)


# ── Module-level: concat / stack / where ────────────────────────────


def _concat_fn(
    tensors: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var axis = 0
    if "axis" in kwargs:
        axis = Int(py=kwargs["axis"])
    var t_list = _py_list_to_tensors(tensors)
    return PythonObject(
        alloc=Tensor[DType.float32].concat(t_list, axis=axis)
    )


def _stack_fn(
    tensors: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var axis = 0
    if "axis" in kwargs:
        axis = Int(py=kwargs["axis"])
    var t_list = _py_list_to_tensors(tensors)
    return PythonObject(
        alloc=Tensor[DType.float32].stack(t_list, axis=axis)
    )


def _where_fn(
    args: PythonObject
) raises -> PythonObject:
    var cond = args[0].downcast_value_ptr[Tensor[DType.bool]]()[]
    var ta = args[1].downcast_value_ptr[TF32]()[]
    var tb = args[2].downcast_value_ptr[TF32]()[]
    return PythonObject(alloc=TF32.where(cond, ta, tb))


def _tensor_bool_where[
    dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var cond = self.downcast_value_ptr[Tensor[DType.bool]]()[]
    var ta = args[0].downcast_value_ptr[TF32]()[]
    var tb = args[1].downcast_value_ptr[TF32]()[]
    return PythonObject(alloc=TF32.where(cond, ta, tb))



# ── Unary math / activation ops ─────────────────────────────────────


def _tensor_exp[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.exp())


def _tensor_log[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.log())


def _tensor_sqrt[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.sqrt())


def _tensor_tanh[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.tanh())


def _tensor_sigmoid[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.sigmoid())


def _tensor_relu[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.relu())


def _tensor_reciprocal[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    return PythonObject(alloc=t.reciprocal())


def _tensor_backward[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    t.backward(start_grad=Scalar[dtype](1.0))
    return PythonObject(None)


def _tensor_grad[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    var t = self.downcast_value_ptr[Tensor[dtype]]()[]
    if not t.requires_grad or not t.has_grad():
        return PythonObject(None)
    var g = t.gradients().detach().as_tensor()
    return PythonObject(alloc=g^)


def _tensor_set_requires_grad[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    var val = Bool(py=args[0])
    var ptr = self.downcast_value_ptr[Tensor[dtype]]()
    ptr[].requires_grad = val
    if val and not ptr[].has_grad():
        ptr[].init_gradbox()
    return self


def _tensor_zero_grad[
    dtype: DType
](
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject where dtype.is_floating_point():
    _ = args
    self.downcast_value_ptr[Tensor[dtype]]()[].zero_grad()
    return PythonObject(None)


# ── Linear ───────────────────────────────────────────────────────

def _init_linear_32(
    args: PythonObject, kwargs: PythonObject
) raises -> L32:
    var in_f = Int(py=args[0])
    var out_f = Int(py=args[1])
    var bias = True
    var bias_zero = True
    var init_method = "uniform"
    if "bias" in kwargs:
        bias = Bool(py=kwargs["bias"])
    if "bias_zero" in kwargs:
        bias_zero = Bool(py=kwargs["bias_zero"])
    if "init_method" in kwargs:
        init_method = String(kwargs["init_method"])
    return L32(in_f, out_f, bias=bias, bias_zero=bias_zero, init_method=init_method)


def _linear_into_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[L32]()
    return PythonObject(alloc=ptr[].into())


def _linear_weight(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[L32]()
    var l = ptr[]
    return PythonObject(alloc=l.weight)


def _linear_bias(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[L32]()
    var l = ptr[]
    if l.bias:
        var bias_tensor = l.bias.value()
        return PythonObject(alloc=bias_tensor)
    return PythonObject(None)


def _linear_train_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[L32]()[].train()
    return PythonObject(None)


def _linear_eval_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[L32]()[].eval()
    return PythonObject(None)


# ── ReLU ─────────────────────────────────────────────────────────

comptime RELU32 = ReLU[DType.float32]


def _init_relu_32(
    args: PythonObject, kwargs: PythonObject
) raises -> RELU32:
    _ = args; _ = kwargs
    return RELU32()


def _relu_into_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[RELU32]()
    return PythonObject(alloc=ptr[].into())


def _relu_train_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[RELU32]()[].train()
    return PythonObject(None)


def _relu_eval_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[RELU32]()[].eval()
    return PythonObject(None)


# ── Sigmoid ──────────────────────────────────────────────────────

comptime SIGMOID32 = Sigmoid[DType.float32]


def _init_sigmoid_32(
    args: PythonObject, kwargs: PythonObject
) raises -> SIGMOID32:
    _ = args; _ = kwargs
    return SIGMOID32()


def _sigmoid_into_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[SIGMOID32]()
    return PythonObject(alloc=ptr[].into())


def _sigmoid_train_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[SIGMOID32]()[].train()
    return PythonObject(None)


def _sigmoid_eval_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[SIGMOID32]()[].eval()
    return PythonObject(None)


# ── Tanh ─────────────────────────────────────────────────────────

comptime TANH32 = Tanh[DType.float32]


def _init_tanh_32(
    args: PythonObject, kwargs: PythonObject
) raises -> TANH32:
    _ = args; _ = kwargs
    return TANH32()


def _tanh_into_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[TANH32]()
    return PythonObject(alloc=ptr[].into())


def _tanh_train_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[TANH32]()[].train()
    return PythonObject(None)


def _tanh_eval_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[TANH32]()[].eval()
    return PythonObject(None)


# ── Flatten ──────────────────────────────────────────────────────

comptime FLATTEN32 = Flatten[DType.float32]


def _init_flatten_32(
    args: PythonObject, kwargs: PythonObject
) raises -> FLATTEN32:
    _ = args; _ = kwargs
    return FLATTEN32()


def _flatten_into_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[FLATTEN32]()
    return PythonObject(alloc=ptr[].into())


def _flatten_train_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[FLATTEN32]()[].train()
    return PythonObject(None)


def _flatten_eval_32(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[FLATTEN32]()[].eval()
    return PythonObject(None)


# ── Sequential ───────────────────────────────────────────────────

def _init_sequential_32(
    args: PythonObject, kwargs: PythonObject
) raises -> S32:
    """Sequential(modules) — takes Python list of Module or layer objects."""
    _ = kwargs
    var py_modules = args[0]
    var s = S32()
    for i in range(len(py_modules)):
        var py_m = py_modules[i]
        # Check if the object is already a Module
        # All layers have "into"; Module does not
        var builtins = Python.import_module("builtins")
        if builtins.hasattr(py_m, "into"):
            # Call .into() via Python to get a _Module, then downcast
            var py_module = py_m.into()
            var m = py_module.downcast_value_ptr[M32]()[]
            s.modules.append(m)
        else:
            var m = py_m.downcast_value_ptr[M32]()[]
            s.modules.append(m)
    return s^


def _sequential_call(
    self_: Pointer[S32, MutAnyOrigin], x: PythonObject
) raises -> PythonObject:
    var input = x.downcast_value_ptr[TF32]()[]
    return PythonObject(alloc=self_[](input))


def _sequential_train(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[S32]()[].train()
    return PythonObject(None)


def _sequential_eval(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[S32]()[].eval()
    return PythonObject(None)


def _sequential_zero_grad(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var s = self.downcast_value_ptr[S32]()
    for param in s[].parameters():
        param[].zero_grad()
    return PythonObject(None)


def _sequential_num_parameters(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    ref cpy = Python().cpython()
    return PythonObject(
        from_owned=cpy.PyLong_FromSsize_t(
            self.downcast_value_ptr[S32]()[].num_parameters()
        )
    )


# ── CrossEntropyLoss ─────────────────────────────────────────────

def _init_ce_32(
    args: PythonObject, kwargs: PythonObject
) raises -> CE32:
    var reduction = "mean"
    var ignore_index = -100
    var label_smoothing = 0.0
    var training = True
    if "reduction" in kwargs:
        reduction = String(kwargs["reduction"])
    if "ignore_index" in kwargs:
        ignore_index = Int(py=kwargs["ignore_index"])
    if "label_smoothing" in kwargs:
        label_smoothing = Float64(py=kwargs["label_smoothing"])
    if "training" in kwargs:
        training = Bool(py=kwargs["training"])
    return CE32(
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=Scalar[DType.float32](label_smoothing),
        training=training,
    )


def _ce_call_int(
    self_: Pointer[CE32, MutAnyOrigin],
    logits: PythonObject,
    target: PythonObject,
) raises -> PythonObject:
    """CE forward, class-index path: f32 logits + int64 targets.

    Dispatch by registered capsule type (bound as "forward_int"); the
    facade picks the flavor from the target dtype, so no dtype sniffing
    here. Mismatched capsules fail loud in downcast_value_ptr.
    """
    var l = logits.downcast_value_ptr[TF32]()[]
    var t = target.downcast_value_ptr[TI64]()[]
    return PythonObject(alloc=self_[](l, t))


def _ce_call_float(
    self_: Pointer[CE32, MutAnyOrigin],
    logits: PythonObject,
    target: PythonObject,
) raises -> PythonObject:
    """CE forward, probability path: f32 logits + f32 targets."""
    var l = logits.downcast_value_ptr[TF32]()[]
    var t = target.downcast_value_ptr[TF32]()[]
    return PythonObject(alloc=self_[](l, t))


def _ce_train(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[CE32]()[].train()
    return PythonObject(None)


def _ce_eval(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[CE32]()[].eval()
    return PythonObject(None)


# ── MSELoss / BCELoss / BCEWithLogitsLoss ────────────────────────
# Stateless MSE + mode-flaggable BCE pair. Plain same-dtype (f32,f32)
# calls: pred/target share the model dtype, so no per-flavor split like
# the CE class-index-vs-probability pair above. Extend per-flavor for
# float64 in the same shape.

def _init_mse_32(
    args: PythonObject, kwargs: PythonObject
) raises -> MSE32:
    _ = args, kwargs
    return MSE32()


def _mse_call(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = self
    var pred = args[0].downcast_value_ptr[TF32]()[]
    var target = args[1].downcast_value_ptr[TF32]()[]
    var loss = MSE32()
    return PythonObject(alloc=loss(pred, target))


def _init_bce_32(
    args: PythonObject, kwargs: PythonObject
) raises -> BCE32:
    _ = args, kwargs
    return BCE32()


def _bce_call(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    var pred = args[0].downcast_value_ptr[TF32]()[]
    var target = args[1].downcast_value_ptr[TF32]()[]
    var ptr = self.downcast_value_ptr[BCE32]()
    return PythonObject(alloc=ptr[](pred, target))


def _bce_train(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[BCE32]()[].train()
    return PythonObject(None)


def _bce_eval(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[BCE32]()[].eval()
    return PythonObject(None)


def _init_bcewl_32(
    args: PythonObject, kwargs: PythonObject
) raises -> BCEWL32:
    _ = args, kwargs
    return BCEWL32()


def _bcewl_call(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    var logits = args[0].downcast_value_ptr[TF32]()[]
    var target = args[1].downcast_value_ptr[TF32]()[]
    var ptr = self.downcast_value_ptr[BCEWL32]()
    return PythonObject(alloc=ptr[](logits, target))


def _bcewl_train(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[BCEWL32]()[].train()
    return PythonObject(None)


def _bcewl_eval(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[BCEWL32]()[].eval()
    return PythonObject(None)


# ── SGD ──────────────────────────────────────────────────────────

def _init_sgd_32(
    args: PythonObject, kwargs: PythonObject
) raises -> OPT32:
    """SGD(model, lr, momentum, ...). Takes a Sequential model."""
    var py_model = args[0]
    var ptr = py_model.downcast_value_ptr[S32]()
    var params = ptr[].parameters()
    var lr = Scalar[DType.float32](0.01)
    var momentum = Scalar[DType.float32](0.0)
    var weight_decay = Scalar[DType.float32](0.0)
    var clip_norm = Scalar[DType.float32](0.0)
    var clip_value = Scalar[DType.float32](0.0)
    if "lr" in kwargs:
        lr = Scalar[DType.float32](Float64(py=kwargs["lr"]))
    if "momentum" in kwargs:
        momentum = Scalar[DType.float32](
            Float64(py=kwargs["momentum"])
        )
    if "weight_decay" in kwargs:
        weight_decay = Scalar[DType.float32](
            Float64(py=kwargs["weight_decay"])
        )
    if "clip_norm" in kwargs:
        clip_norm = Scalar[DType.float32](
            Float64(py=kwargs["clip_norm"])
        )
    if "clip_value" in kwargs:
        clip_value = Scalar[DType.float32](
            Float64(py=kwargs["clip_value"])
        )
    return OPT32(
        params^,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        clip_value=clip_value,
    )


def _sgd_step(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[OPT32]()[].step()
    return PythonObject(None)


def _sgd_zero_grad(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[OPT32]()[].zero_grad()
    return PythonObject(None)


def _sgd_set_lr(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    var lr = Scalar[DType.float32](Float64(py=args[0]))
    self.downcast_value_ptr[OPT32]()[].set_lr(lr)
    return PythonObject(None)


def _sgd_get_lr(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var lr = self.downcast_value_ptr[OPT32]()[].get_lr()
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(lr)))


def _sgd_state_dict(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    return self.downcast_value_ptr[OPT32]()[].state_dict()


# ── AdamW ────────────────────────────────────────────────────────

def _init_adamw_32(
    args: PythonObject, kwargs: PythonObject
) raises -> ADAMW32:
    """AdamW(model, lr, beta1, ...). Takes a Sequential model."""
    var py_model = args[0]
    var ptr = py_model.downcast_value_ptr[S32]()
    var params = ptr[].parameters()
    var lr = Scalar[DType.float32](0.001)
    var beta1 = Scalar[DType.float32](0.9)
    var beta2 = Scalar[DType.float32](0.95)
    var eps = Scalar[DType.float32](1e-8)
    var weight_decay = Scalar[DType.float32](0.1)
    var clip_norm = Scalar[DType.float32](0.0)
    var clip_value = Scalar[DType.float32](0.0)
    if "lr" in kwargs:
        lr = Scalar[DType.float32](Float64(py=kwargs["lr"]))
    if "beta1" in kwargs:
        beta1 = Scalar[DType.float32](Float64(py=kwargs["beta1"]))
    if "beta2" in kwargs:
        beta2 = Scalar[DType.float32](Float64(py=kwargs["beta2"]))
    if "eps" in kwargs:
        eps = Scalar[DType.float32](Float64(py=kwargs["eps"]))
    if "weight_decay" in kwargs:
        weight_decay = Scalar[DType.float32](
            Float64(py=kwargs["weight_decay"])
        )
    if "clip_norm" in kwargs:
        clip_norm = Scalar[DType.float32](
            Float64(py=kwargs["clip_norm"])
        )
    if "clip_value" in kwargs:
        clip_value = Scalar[DType.float32](
            Float64(py=kwargs["clip_value"])
        )
    return ADAMW32(
        params^,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        clip_value=clip_value,
    )


def _adamw_step(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[ADAMW32]()[].step()
    return PythonObject(None)


def _adamw_zero_grad(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[ADAMW32]()[].zero_grad()
    return PythonObject(None)


def _adamw_set_lr(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    var lr = Scalar[DType.float32](Float64(py=args[0]))
    self.downcast_value_ptr[ADAMW32]()[].set_lr(lr)
    return PythonObject(None)


def _adamw_get_lr(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    var lr = self.downcast_value_ptr[ADAMW32]()[].get_lr()
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(lr)))


def _adamw_state_dict(
    mut self: PythonObject, mut args: PythonObject
) raises -> PythonObject:
    _ = args
    return self.downcast_value_ptr[ADAMW32]()[].state_dict()


# ── Accuracy (static functions) ──────────────────────────────────

def _accuracy_compute(
    pred: PythonObject, target: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var p = pred.downcast_value_ptr[TF32]()[]
    var t = target.downcast_value_ptr[TF32]()[]
    var target_i64 = t.to_dtype[DType.int64]()
    var sync = True
    if "sync" in kwargs:
        sync = Bool(py=kwargs["sync"])
    var result = Accuracy[DType.float32].compute(p, target_i64, sync)
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(result)))


def _accuracy_token(
    pred: PythonObject, target: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var p = pred.downcast_value_ptr[TF32]()[]
    var t = target.downcast_value_ptr[TF32]()[]
    var target_i64 = t.to_dtype[DType.int64]()
    var sync = True
    if "sync" in kwargs:
        sync = Bool(py=kwargs["sync"])
    var result = Accuracy[DType.float32].token_accuracy(p, target_i64, sync)
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(result)))


def _accuracy_sequence(
    pred: PythonObject, target: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var p = pred.downcast_value_ptr[TF32]()[]
    var t = target.downcast_value_ptr[TF32]()[]
    var target_i64 = t.to_dtype[DType.int64]()
    var sync = True
    if "sync" in kwargs:
        sync = Bool(py=kwargs["sync"])
    var result = Accuracy[DType.float32].sequence_accuracy(
        p, target_i64, sync
    )
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyFloat_FromDouble(Float64(result)))


# ── Module-level factory functions ───────────────────────────────

def _tensor_fn(
    data: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    """Create a tensor from a Python list or numpy array.

    For numpy arrays, the original dtype is preserved for all 11 numeric/bool types.
    For Python lists, dtype defaults to float32; pass dtype= to specify.
    """
    var requires_grad = False
    if "requires_grad" in kwargs:
        requires_grad = Bool(py=kwargs["requires_grad"])

    var builtins = Python.import_module("builtins")
    if builtins.hasattr(data, "dtype"):
        var src_dtype = mojo_dtype(data.dtype)
        if src_dtype == DType.float16:
            return PythonObject(
                alloc=from_ndarray[DType.float16](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.float32:
            return PythonObject(
                alloc=from_ndarray[DType.float32](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.float64:
            return PythonObject(
                alloc=from_ndarray[DType.float64](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.int8:
            return PythonObject(
                alloc=from_ndarray[DType.int8](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.int16:
            return PythonObject(
                alloc=from_ndarray[DType.int16](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.int32:
            return PythonObject(
                alloc=from_ndarray[DType.int32](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.int64:
            return PythonObject(
                alloc=from_ndarray[DType.int64](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.uint8:
            return PythonObject(
                alloc=from_ndarray[DType.uint8](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.uint16:
            return PythonObject(
                alloc=from_ndarray[DType.uint16](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.uint32:
            return PythonObject(
                alloc=from_ndarray[DType.uint32](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.uint64:
            return PythonObject(
                alloc=from_ndarray[DType.uint64](data, requires_grad=requires_grad)
            )
        elif src_dtype == DType.bool:
            return PythonObject(
                alloc=from_ndarray[DType.bool](data, requires_grad=requires_grad)
            )
        else:
            var np = Python.import_module("numpy")
            var np_ref = np.asarray(data, dtype=np.float32)
            return PythonObject(
                alloc=from_ndarray[DType.float32](np_ref, requires_grad=requires_grad)
            )

    var target_dtype = DType.float32
    if "dtype" in kwargs:
        target_dtype = dtype_from_string(String(py=kwargs["dtype"]))

    # Plain Python list: same bulk path as _init_tensor_generic (numpy
    # does nesting/validation/casting in C, from_ndarray memcpys once).
    if target_dtype == DType.float16:
        return PythonObject(
            alloc=_list_to_tensor[DType.float16](data, requires_grad)
        )
    elif target_dtype == DType.float32:
        return PythonObject(
            alloc=_list_to_tensor[DType.float32](data, requires_grad)
        )
    elif target_dtype == DType.float64:
        return PythonObject(
            alloc=_list_to_tensor[DType.float64](data, requires_grad)
        )
    elif target_dtype == DType.int8:
        return PythonObject(
            alloc=_list_to_tensor[DType.int8](data, requires_grad)
        )
    elif target_dtype == DType.int16:
        return PythonObject(
            alloc=_list_to_tensor[DType.int16](data, requires_grad)
        )
    elif target_dtype == DType.int32:
        return PythonObject(
            alloc=_list_to_tensor[DType.int32](data, requires_grad)
        )
    elif target_dtype == DType.int64:
        return PythonObject(
            alloc=_list_to_tensor[DType.int64](data, requires_grad)
        )
    elif target_dtype == DType.uint8:
        return PythonObject(
            alloc=_list_to_tensor[DType.uint8](data, requires_grad)
        )
    elif target_dtype == DType.uint16:
        return PythonObject(
            alloc=_list_to_tensor[DType.uint16](data, requires_grad)
        )
    elif target_dtype == DType.uint32:
        return PythonObject(
            alloc=_list_to_tensor[DType.uint32](data, requires_grad)
        )
    elif target_dtype == DType.uint64:
        return PythonObject(
            alloc=_list_to_tensor[DType.uint64](data, requires_grad)
        )
    elif target_dtype == DType.bool:
        return PythonObject(
            alloc=_list_to_tensor[DType.bool](data, requires_grad)
        )
    else:
        return PythonObject(
            alloc=_list_to_tensor[DType.float32](data, requires_grad)
        )


def _factory_for[
    dtype: DType
](
    op: String,
    shape_list: List[Int],
    requires_grad: Bool,
    start: Float64,
    stop: Float64,
    step: Float64,
) raises -> PythonObject:
    """Build a factory tensor (zeros/ones/randn/arange) for one dtype."""
    var shp = Shape(shape_list)
    if op == "zeros":
        return PythonObject(
            alloc=Tensor[dtype].zeros(shp, requires_grad=requires_grad)
        )
    if op == "ones":
        return PythonObject(
            alloc=Tensor[dtype].ones(shp, requires_grad=requires_grad)
        )
    if op == "randn":
        comptime if dtype.is_floating_point():
            return PythonObject(
                alloc=Tensor[dtype].randn(shp, requires_grad=requires_grad)
            )
        else:
            raise Error("randn requires a floating-point dtype")
    if op == "full":
        comptime if dtype == DType.bool:
            return PythonObject(
                alloc=Tensor[dtype].full(
                    shp,
                    Scalar[dtype](UInt8(1 if start != 0.0 else 0)),
                    requires_grad=requires_grad,
                )
            )
        else:
            return PythonObject(
                alloc=Tensor[dtype].full(
                    shp,
                    Scalar[dtype](start),
                    requires_grad=requires_grad,
                )
            )
    if op == "rand":
        comptime if dtype.is_floating_point():
            return PythonObject(
                alloc=Tensor[dtype].rand(
                    shp,
                    min=Scalar[dtype](start),
                    max=Scalar[dtype](stop),
                    requires_grad=requires_grad,
                )
            )
        else:
            raise Error("rand requires a floating-point dtype")
    if op == "linspace":
        comptime if dtype.is_floating_point():
            var n = shape_list[0]
            return PythonObject(
                alloc=Tensor[dtype].linspace(
                    Scalar[dtype](start),
                    Scalar[dtype](stop),
                    n,
                    requires_grad=requires_grad,
                )
            )
        else:
            raise Error("linspace requires a floating-point dtype")
    if op == "eye":
        var dim = shape_list[0]
        return PythonObject(
            alloc=Tensor[dtype].eye(dim, requires_grad=requires_grad)
        )
    # arange
    comptime if dtype == DType.bool:
        raise Error("arange does not support bool tensors")
    else:
        return PythonObject(
            alloc=Tensor[dtype].arange(
                Scalar[dtype](start),
                Scalar[dtype](stop),
                Scalar[dtype](step),
                requires_grad=requires_grad,
            )
        )


def _full_fn(
    shape: PythonObject, value: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var shape_list = _py_list_to_ints(shape)
    var val = Float64(py=value)
    var dt = _kwargs_dtype(kwargs)
    var rg = False
    if "requires_grad" in kwargs:
        rg = Bool(py=kwargs["requires_grad"])
    return _factory_dispatch("full", shape_list, rg, val, 0.0, 0.0, dt)


def _rand_fn(
    shape: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var shape_list = _py_list_to_ints(shape)
    var low = 0.0
    var high = 1.0
    if "low" in kwargs:
        low = Float64(py=kwargs["low"])
    if "high" in kwargs:
        high = Float64(py=kwargs["high"])
    var dt = _kwargs_dtype(kwargs)
    var rg = False
    if "requires_grad" in kwargs:
        rg = Bool(py=kwargs["requires_grad"])
    return _factory_dispatch("rand", shape_list, rg, low, high, 0.0, dt)


def _linspace_fn(
    start: PythonObject,
    end: PythonObject,
    steps: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    var s = Float64(py=start)
    var e = Float64(py=end)
    var n = Int(py=steps)
    var dt = _kwargs_dtype(kwargs)
    var rg = False
    if "requires_grad" in kwargs:
        rg = Bool(py=kwargs["requires_grad"])
    var shape_linspace = List[Int]()
    shape_linspace.append(n)
    return _factory_dispatch("linspace", shape_linspace, rg, s, e, 0.0, dt)


def _eye_fn(
    n: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var dim = Int(py=n)
    var dt = _kwargs_dtype(kwargs)
    var rg = False
    if "requires_grad" in kwargs:
        rg = Bool(py=kwargs["requires_grad"])
    var shape_eye = List[Int]()
    shape_eye.append(dim)
    return _factory_dispatch("eye", shape_eye, rg, 0.0, 0.0, 0.0, dt)


def _factory_dispatch(
    op: String,
    shape_list: List[Int],
    requires_grad: Bool,
    start: Float64,
    stop: Float64,
    step: Float64,
    dt: DType,
) raises -> PythonObject:
    """Route a factory call to the given resolved dtype."""
    if dt == DType.float16:
        return _factory_for[DType.float16](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.float32:
        return _factory_for[DType.float32](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.float64:
        return _factory_for[DType.float64](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.int8:
        return _factory_for[DType.int8](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.int16:
        return _factory_for[DType.int16](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.int32:
        return _factory_for[DType.int32](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.int64:
        return _factory_for[DType.int64](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.uint8:
        return _factory_for[DType.uint8](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.uint16:
        return _factory_for[DType.uint16](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.uint32:
        return _factory_for[DType.uint32](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.uint64:
        return _factory_for[DType.uint64](
            op, shape_list, requires_grad, start, stop, step
        )
    elif dt == DType.bool:
        return _factory_for[DType.bool](
            op, shape_list, requires_grad, start, stop, step
        )
    else:
        raise Error("unsupported dtype for factory function")


def _kwargs_dtype(kwargs: StringDict[PythonObject]) raises -> DType:
    """Extract the resolved dtype from factory kwargs (default float32)."""
    if "dtype" in kwargs:
        return dtype_from_string(String(py=kwargs["dtype"]))
    return DType.float32


def _zeros_fn(
    shape: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var shape_list = _py_list_to_ints(shape)
    var requires_grad = False
    if "requires_grad" in kwargs:
        requires_grad = Bool(py=kwargs["requires_grad"])
    return _factory_dispatch(
        "zeros",
        shape_list,
        requires_grad,
        0.0,
        0.0,
        0.0,
        _kwargs_dtype(kwargs),
    )


def _ones_fn(
    shape: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var shape_list = _py_list_to_ints(shape)
    var requires_grad = False
    if "requires_grad" in kwargs:
        requires_grad = Bool(py=kwargs["requires_grad"])
    return _factory_dispatch(
        "ones",
        shape_list,
        requires_grad,
        0.0,
        0.0,
        0.0,
        _kwargs_dtype(kwargs),
    )


def _randn_fn(
    shape: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var shape_list = _py_list_to_ints(shape)
    var requires_grad = False
    if "requires_grad" in kwargs:
        requires_grad = Bool(py=kwargs["requires_grad"])
    return _factory_dispatch(
        "randn",
        shape_list,
        requires_grad,
        0.0,
        0.0,
        0.0,
        _kwargs_dtype(kwargs),
    )


def _arange_fn(
    end: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    var requires_grad = False
    if "requires_grad" in kwargs:
        requires_grad = Bool(py=kwargs["requires_grad"])
    var start_val = 0.0
    var end_val = Float64(py=end)
    var step_val = 1.0
    if "start" in kwargs:
        start_val = Float64(py=kwargs["start"])
    if "step" in kwargs:
        step_val = Float64(py=kwargs["step"])
    var no_shape = List[Int]()
    return _factory_dispatch(
        "arange",
        no_shape,
        requires_grad,
        start_val,
        end_val,
        step_val,
        _kwargs_dtype(kwargs),
    )


def _matmul_fn(
    A: PythonObject, B: PythonObject
) raises -> PythonObject:
    var a = A.downcast_value_ptr[TF32]()[]
    var b = B.downcast_value_ptr[TF32]()[]
    return PythonObject(alloc=a.matmul(b))


# ── Epoch-level training/eval (DataLoader stays in Mojo) ─────────

from tenmo.dataloader import NumpyDataset, NativeLoader


def _train_epoch(
    model: PythonObject,
    criterion: PythonObject,
    optimizer: PythonObject,
    features: PythonObject,
    labels: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Run one training epoch entirely in Mojo. Returns (loss, accuracy)."""
    var batch_size = 64
    var shuffle = True
    var normalize_mean = Optional[Scalar[DType.float32]](None)
    var normalize_std = Optional[Scalar[DType.float32]](None)
    if "batch_size" in kwargs:
        batch_size = Int(py=kwargs["batch_size"])
    if "shuffle" in kwargs:
        shuffle = Bool(py=kwargs["shuffle"])
    if "normalize_mean" in kwargs:
        normalize_mean = Optional[Scalar[DType.float32]](
            Scalar[DType.float32](Float64(py=kwargs["normalize_mean"]))
        )
    if "normalize_std" in kwargs:
        normalize_std = Optional[Scalar[DType.float32]](
            Scalar[DType.float32](Float64(py=kwargs["normalize_std"]))
        )

    var ds = NumpyDataset[DType.float32, DType.int64](
        features, labels, copy=True
    )
    var loader = ds.into_loader(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    var mod_ptr = model.downcast_value_ptr[S32]()
    var crit_ptr = criterion.downcast_value_ptr[CE32]()
    var opt_ptr = optimizer.downcast_value_ptr[OPT32]()
    mod_ptr[].train()
    crit_ptr[].train()

    var total_loss = Scalar[DType.float32](0.0)
    var total_correct = Float64(0.0)
    var total_count = 0

    loader.reset()
    while loader.__has_next__():
        ref batch = loader.__next__()
        var pred = mod_ptr[](batch.features)
        var loss = crit_ptr[](pred, batch.labels)

        opt_ptr[].zero_grad()
        loss.backward()
        opt_ptr[].step()

        var bs = batch.batch_size
        total_loss += loss.item() * Scalar[DType.float32](bs)
        total_correct += Accuracy[DType.float32].compute(
            pred, batch.labels
        ) * Float64(bs)
        total_count += bs

    var avg_loss = Float64(total_loss) / Float64(total_count)
    var acc = total_correct / Float64(total_count)

    ref cpy = Python().cpython()
    var result = cpy.PyTuple_New(2)
    _ = cpy.PyTuple_SetItem(result, 0, cpy.PyFloat_FromDouble(avg_loss))
    _ = cpy.PyTuple_SetItem(result, 1, cpy.PyFloat_FromDouble(acc))
    return PythonObject(from_owned=result)


def _eval_epoch(
    model: PythonObject,
    criterion: PythonObject,
    features: PythonObject,
    labels: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Run one eval epoch entirely in Mojo. Returns (loss, accuracy)."""
    var batch_size = 64
    var normalize_mean = Optional[Scalar[DType.float32]](None)
    var normalize_std = Optional[Scalar[DType.float32]](None)
    if "batch_size" in kwargs:
        batch_size = Int(py=kwargs["batch_size"])
    if "normalize_mean" in kwargs:
        normalize_mean = Optional[Scalar[DType.float32]](
            Scalar[DType.float32](Float64(py=kwargs["normalize_mean"]))
        )
    if "normalize_std" in kwargs:
        normalize_std = Optional[Scalar[DType.float32]](
            Scalar[DType.float32](Float64(py=kwargs["normalize_std"]))
        )

    var ds = NumpyDataset[DType.float32, DType.int64](
        features, labels, copy=True
    )
    var loader = ds.into_loader(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    var mod_ptr = model.downcast_value_ptr[S32]()
    var crit_ptr = criterion.downcast_value_ptr[CE32]()
    mod_ptr[].eval()
    crit_ptr[].eval()

    var total_loss = Scalar[DType.float32](0.0)
    var total_correct = Float64(0.0)
    var total_count = 0

    loader.reset()
    while loader.__has_next__():
        ref batch = loader.__next__()
        var pred = mod_ptr[](batch.features)
        var loss = crit_ptr[](pred, batch.labels)

        var bs = batch.batch_size
        total_loss += loss.item() * Scalar[DType.float32](bs)
        total_correct += Accuracy[DType.float32].compute(
            pred, batch.labels
        ) * Float64(bs)
        total_count += bs

    var avg_loss = Float64(total_loss) / Float64(total_count)
    var acc = total_correct / Float64(total_count)

    ref cpy = Python().cpython()
    var result = cpy.PyTuple_New(2)
    _ = cpy.PyTuple_SetItem(result, 0, cpy.PyFloat_FromDouble(avg_loss))
    _ = cpy.PyTuple_SetItem(result, 1, cpy.PyFloat_FromDouble(acc))
    return PythonObject(from_owned=result)


# ── DataLoader (tensor-native, dtype-generic) ────────────────────
# Per-dtype-pair registered classes; each holds a comptime-typed
# DataLoader[sample_dtype, label_dtype] engine. Construction converts
# Python values
# (existing Tenmo Tensor or numpy array) into Tensors — one copy for
# numpy, zero extra for a clean contiguous array. Batches come back as
# a pair of Tenmo Tensors with their natural dtypes.

def _ml_tensor_from_py[dtype: DType](obj: PythonObject) raises -> Tensor[dtype]:
    """Convert a Python value into Tensor[dtype]. Existing Tenmo Tensors.
    pass through (sharing storage); numpy arrays are wrapped with one copy.
    """
    var builtins = Python.import_module("builtins")
    if builtins.hasattr(obj, "numpy_dtype"):
        return obj.downcast_value_ptr[Tensor[dtype]]()[]
    return from_ndarray[dtype](obj, requires_grad=False)


def _init_data_loader[
    sample_dtype: DType, label_dtype: DType
](args: PythonObject, kwargs: PythonObject) raises -> DataLoader[
    sample_dtype, label_dtype
]:
    """DataLoader(features, labels, batch_size, shuffle, drop_last)."""
    _ = kwargs
    var features = _ml_tensor_from_py[sample_dtype](args[0])
    var labels = _ml_tensor_from_py[label_dtype](args[1])
    var batch_size = Int(py=args[2])
    var shuffle = Bool(py=args[3])
    var drop_last = Bool(py=args[4])
    return DataLoader[sample_dtype, label_dtype](
        features^, labels^, batch_size, shuffle, drop_last
    )


def _loader_next[
    sample_dtype: DType, label_dtype: DType
](py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    """Raw ``next()``.
    Calls ``DataLoader.__next__()`` and surfaces a typed
    Python ``StopIteration`` when the epoch ends (``def_py_method`` cannot
    express a custom Python exception type)."""
    _ = args
    var self_obj = PythonObject(from_borrowed=py_self)
    try:
        ref batch = self_obj.downcast_value_ptr[
            DataLoader[sample_dtype, label_dtype]
        ]()[].__next__()
        ref cpy = Python().cpython()
        var tup = cpy.PyTuple_New(2)
        var fx_cap = PythonObject(alloc=batch.features)
        var fy_cap = PythonObject(alloc=batch.labels)
        _ = cpy.PyTuple_SetItem(tup, 0, fx_cap.steal_data())
        _ = cpy.PyTuple_SetItem(tup, 1, fy_cap.steal_data())
        return PythonObject(from_owned=tup).steal_data()
    except e:
        return raise_python_exception(e, ExceptionType("PyExc_StopIteration"))


def _loader_reset[
    sample_dtype: DType, label_dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    self.downcast_value_ptr[
        DataLoader[sample_dtype, label_dtype]
    ]()[].reset()
    return PythonObject(None)


def _loader_len[
    sample_dtype: DType, label_dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    ref cpy = Python().cpython()
    return PythonObject(
        from_owned=cpy.PyLong_FromSsize_t(
            self.downcast_value_ptr[
                DataLoader[sample_dtype, label_dtype]
            ]()[].__len__()
        )
    )


def _loader_has_next[
    sample_dtype: DType, label_dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    var ok = self.downcast_value_ptr[
        DataLoader[sample_dtype, label_dtype]
    ]()[].__has_next__()
    if ok:
        return PythonObject(
            from_owned=Python().cpython().PyBool_FromLong(c_long(1))
        )
    return PythonObject(
        from_owned=Python().cpython().PyBool_FromLong(c_long(0))
    )


def _loader_set_mode[
    sample_dtype: DType, label_dtype: DType
](mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var shuffle = Bool(py=args[0])
    self.downcast_value_ptr[
        DataLoader[sample_dtype, label_dtype]
    ]()[].set_shuffle(shuffle)
    return PythonObject(None)


def register_data_loader[
    sample_dtype: DType, label_dtype: DType
](
    mut mb: PythonModuleBuilder, name: StringSlice[ImmStaticOrigin]
) raises:
    ref b = mb.add_type[DataLoader[sample_dtype, label_dtype]](name)
    _ = b.def_py_init[_init_data_loader[sample_dtype, label_dtype]]()
    _ = b.def_py_c_method(_loader_next[sample_dtype, label_dtype], "next")
    _ = b.def_py_method[_loader_reset[sample_dtype, label_dtype]]("reset")
    _ = b.def_py_method[_loader_len[sample_dtype, label_dtype]]("length")
    _ = b.def_py_method[_loader_has_next[sample_dtype, label_dtype]](
        "has_next"
    )
    _ = b.def_py_method[_loader_set_mode[sample_dtype, label_dtype]](
        "set_mode"
    )


# ── Module entry point ───────────────────────────────────────────

@export("PyInit__tenmo")
def PyInit__tenmo() abi("C") -> PythonObject:
    try:
        var mb = PythonModuleBuilder("tenmo")

        # ── Tensors: one Python type per dtype (common surface) ──
        # Tensor[float32] and Float64Tensor carry the full autograd/math
        # surface appended below (generic handlers instantiated per dtype).
        register_tensor[DType.float32](mb, "Tensor")
        var f32_idx = len(mb.type_builders) - 1
        _ = mb.type_builders[f32_idx] \
            .def_py_method[_tensor_backward[DType.float32]]("backward") \
            .def_py_method[_tensor_grad[DType.float32]]("grad") \
            .def_py_method[_tensor_set_requires_grad[DType.float32]]("requires_grad_") \
            .def_py_method[_tensor_zero_grad[DType.float32]]("zero_grad") \
            .def_py_method[_tensor_sum[DType.float32]]("sum") \
            .def_py_method[_tensor_mean[DType.float32]]("mean") \
            .def_py_method[_tensor_max[DType.float32]]("max") \
            .def_py_method[_tensor_min[DType.float32]]("min") \
            .def_py_method[_tensor_argmax[DType.float32]]("argmax") \
            .def_py_method[_tensor_argmin[DType.float32]]("argmin") \
            .def_py_method[_tensor_softmax[DType.float32]]("softmax") \
            .def_py_method[_tensor_flatten[DType.float32]]("flatten") \
            .def_py_method[_tensor_reshape[DType.float32]]("reshape") \
            .def_py_method[_tensor_matmul[DType.float32]]("matmul") \
            .def_py_method[_tensor_clip[DType.float32]]("clip") \
            .def_py_method[_tensor_abs[DType.float32]]("abs") \
            .def_py_method[_tensor_exp[DType.float32]]("exp") \
            .def_py_method[_tensor_log[DType.float32]]("log") \
            .def_py_method[_tensor_sqrt[DType.float32]]("sqrt") \
            .def_py_method[_tensor_tanh[DType.float32]]("tanh") \
            .def_py_method[_tensor_sigmoid[DType.float32]]("sigmoid") \
            .def_py_method[_tensor_relu[DType.float32]]("relu") \
            .def_py_method[_tensor_reciprocal[DType.float32]]("reciprocal") \
            .def_py_method[_tensor_product[DType.float32]]("product") \
            .def_py_method[_tensor_variance[DType.float32]]("variance") \
            .def_py_method[_tensor_std[DType.float32]]("std") \
            .def_py_method[_tensor_norm[DType.float32]]("norm") \
            .def_py_method[_tensor_transpose[DType.float32]]("transpose") \
            .def_py_method[_tensor_permute[DType.float32]]("permute") \
            .def_py_method[_tensor_squeeze[DType.float32]]("squeeze") \
            .def_py_method[_tensor_unsqueeze[DType.float32]]("unsqueeze") \
            .def_py_method[_tensor_expand[DType.float32]]("expand") \
            .def_py_method[_tensor_contiguous[DType.float32]]("contiguous") \
            .def_py_method[_tensor_masked_fill[DType.float32]]("masked_fill") \
            .def_py_method[_tensor_detach[DType.float32]]("detach") \
            .def_py_method[_tensor_is_contiguous[DType.float32]]("is_contiguous") \
            .def_py_method[_tensor_device[DType.float32]]("device") \
            .def_py_method[_tensor_to_dtype[DType.float32]]("to_dtype") \
            .def_py_method[_tensor_zeros_like[DType.float32]]("zeros_like") \
            .def_py_method[_tensor_ones_like[DType.float32]]("ones_like") \
            .def_py_method[_tensor_triu[DType.float32]]("triu") \
            .def_py_method[_tensor_tril[DType.float32]]("tril") \
            .def_py_method[_tensor_cumsum[DType.float32]]("cumsum") \
            .def_py_method[_tensor_dot[DType.float32]]("dot") \
            .def_py_method[_tensor_outer[DType.float32]]("outer") \
            .def_py_method[_tensor_gather[DType.float32]]("gather") \
            .def_py_method[_tensor_mse[DType.float32]]("mse") \
            .def_py_method[_tensor_bce[DType.float32]]("bce") \
            .def_py_method[_tensor_bce_logits[DType.float32]]("bce_logits") \
            .def_py_method[_tensor_seed_grad[DType.float32]]("seed_grad")

        register_tensor[DType.float16](mb, "Float16Tensor")
        register_tensor[DType.float64](mb, "Float64Tensor")
        # Float64 rich surface mirrors float32 (same generic handlers).
        # The full 50-method surface OOMs at -O3 even under MemoryMax=17G,
        # so it is built with --optimization-level 1 (see run_python_tests
        # logs); slice history: slice 1 (autograd/reductions/shape) fit
        # -O3/17G, the union needs -O1.
        _ = mb.type_builders[len(mb.type_builders) - 1] \
            .def_py_method[_tensor_backward[DType.float64]]("backward") \
            .def_py_method[_tensor_grad[DType.float64]]("grad") \
            .def_py_method[_tensor_set_requires_grad[DType.float64]]("requires_grad_") \
            .def_py_method[_tensor_zero_grad[DType.float64]]("zero_grad") \
            .def_py_method[_tensor_sum[DType.float64]]("sum") \
            .def_py_method[_tensor_mean[DType.float64]]("mean") \
            .def_py_method[_tensor_max[DType.float64]]("max") \
            .def_py_method[_tensor_min[DType.float64]]("min") \
            .def_py_method[_tensor_argmax[DType.float64]]("argmax") \
            .def_py_method[_tensor_argmin[DType.float64]]("argmin") \
            .def_py_method[_tensor_softmax[DType.float64]]("softmax") \
            .def_py_method[_tensor_flatten[DType.float64]]("flatten") \
            .def_py_method[_tensor_reshape[DType.float64]]("reshape") \
            .def_py_method[_tensor_matmul[DType.float64]]("matmul") \
            .def_py_method[_tensor_clip[DType.float64]]("clip") \
            .def_py_method[_tensor_abs[DType.float64]]("abs") \
            .def_py_method[_tensor_exp[DType.float64]]("exp") \
            .def_py_method[_tensor_log[DType.float64]]("log") \
            .def_py_method[_tensor_sqrt[DType.float64]]("sqrt") \
            .def_py_method[_tensor_tanh[DType.float64]]("tanh") \
            .def_py_method[_tensor_sigmoid[DType.float64]]("sigmoid") \
            .def_py_method[_tensor_relu[DType.float64]]("relu") \
            .def_py_method[_tensor_reciprocal[DType.float64]]("reciprocal") \
            .def_py_method[_tensor_product[DType.float64]]("product") \
            .def_py_method[_tensor_variance[DType.float64]]("variance") \
            .def_py_method[_tensor_std[DType.float64]]("std") \
            .def_py_method[_tensor_norm[DType.float64]]("norm") \
            .def_py_method[_tensor_transpose[DType.float64]]("transpose") \
            .def_py_method[_tensor_permute[DType.float64]]("permute") \
            .def_py_method[_tensor_squeeze[DType.float64]]("squeeze") \
            .def_py_method[_tensor_unsqueeze[DType.float64]]("unsqueeze") \
            .def_py_method[_tensor_expand[DType.float64]]("expand") \
            .def_py_method[_tensor_contiguous[DType.float64]]("contiguous") \
            .def_py_method[_tensor_masked_fill[DType.float64]]("masked_fill") \
            .def_py_method[_tensor_detach[DType.float64]]("detach") \
            .def_py_method[_tensor_is_contiguous[DType.float64]]("is_contiguous") \
            .def_py_method[_tensor_device[DType.float64]]("device") \
            .def_py_method[_tensor_to_dtype[DType.float64]]("to_dtype") \
            .def_py_method[_tensor_zeros_like[DType.float64]]("zeros_like") \
            .def_py_method[_tensor_ones_like[DType.float64]]("ones_like") \
            .def_py_method[_tensor_triu[DType.float64]]("triu") \
            .def_py_method[_tensor_tril[DType.float64]]("tril") \
            .def_py_method[_tensor_cumsum[DType.float64]]("cumsum") \
            .def_py_method[_tensor_dot[DType.float64]]("dot") \
            .def_py_method[_tensor_outer[DType.float64]]("outer") \
            .def_py_method[_tensor_gather[DType.float64]]("gather") \
            .def_py_method[_tensor_mse[DType.float64]]("mse") \
            .def_py_method[_tensor_bce[DType.float64]]("bce") \
            .def_py_method[_tensor_bce_logits[DType.float64]]("bce_logits") \
            .def_py_method[_tensor_seed_grad[DType.float64]]("seed_grad")
        register_tensor[DType.int8](mb, "Int8Tensor")
        register_tensor[DType.int16](mb, "Int16Tensor")
        register_tensor[DType.int32](mb, "Int32Tensor")
        register_tensor[DType.int64](mb, "Int64Tensor")
        register_tensor[DType.uint8](mb, "Uint8Tensor")
        register_tensor[DType.uint16](mb, "Uint16Tensor")
        register_tensor[DType.uint32](mb, "Uint32Tensor")
        register_tensor[DType.uint64](mb, "Uint64Tensor")
        register_tensor[DType.bool](mb, "BoolTensor")
        var bool_idx = len(mb.type_builders) - 1
        _ = mb.type_builders[bool_idx] \
            .def_py_method[_tensor_bool_where[DType.bool]]("where")

        # ── Layers (float32 only) ────────────────────────────────
        _ = mb.add_type[L32]("Linear") \
            .def_py_init[_init_linear_32]() \
            .def_py_method[_linear_into_32]("into") \
            .def_py_method[_linear_weight]("weight") \
            .def_py_method[_linear_bias]("bias") \
            .def_py_method[_linear_train_32]("train") \
            .def_py_method[_linear_eval_32]("eval")

        _ = mb.add_type[RELU32]("ReLU") \
            .def_py_init[_init_relu_32]() \
            .def_py_method[_relu_into_32]("into") \
            .def_py_method[_relu_train_32]("train") \
            .def_py_method[_relu_eval_32]("eval")

        _ = mb.add_type[SIGMOID32]("Sigmoid") \
            .def_py_init[_init_sigmoid_32]() \
            .def_py_method[_sigmoid_into_32]("into") \
            .def_py_method[_sigmoid_train_32]("train") \
            .def_py_method[_sigmoid_eval_32]("eval")

        _ = mb.add_type[TANH32]("Tanh") \
            .def_py_init[_init_tanh_32]() \
            .def_py_method[_tanh_into_32]("into") \
            .def_py_method[_tanh_train_32]("train") \
            .def_py_method[_tanh_eval_32]("eval")

        _ = mb.add_type[FLATTEN32]("Flatten") \
            .def_py_init[_init_flatten_32]() \
            .def_py_method[_flatten_into_32]("into") \
            .def_py_method[_flatten_train_32]("train") \
            .def_py_method[_flatten_eval_32]("eval")

        _ = mb.add_type[M32]("_Module")

        _ = mb.add_type[S32]("Sequential") \
            .def_py_init[_init_sequential_32]() \
            .def_method[_sequential_call]("forward") \
            .def_py_method[_sequential_train]("train") \
            .def_py_method[_sequential_eval]("eval") \
            .def_py_method[_sequential_zero_grad]("zero_grad") \
            .def_py_method[_sequential_num_parameters]("num_parameters")

        _ = mb.add_type[CE32]("CrossEntropyLoss") \
            .def_py_init[_init_ce_32]() \
            .def_method[_ce_call_int]("forward_int") \
            .def_method[_ce_call_float]("forward_float") \
            .def_py_method[_ce_train]("train") \
            .def_py_method[_ce_eval]("eval")

        # MSELoss is stateless (no train/eval); facade no-ops them.
        _ = mb.add_type[MSE32]("MSELoss") \
            .def_py_init[_init_mse_32]() \
            .def_py_method[_mse_call]("forward")

        _ = mb.add_type[BCE32]("BCELoss") \
            .def_py_init[_init_bce_32]() \
            .def_py_method[_bce_call]("forward") \
            .def_py_method[_bce_train]("train") \
            .def_py_method[_bce_eval]("eval")

        _ = mb.add_type[BCEWL32]("BCEWithLogitsLoss") \
            .def_py_init[_init_bcewl_32]() \
            .def_py_method[_bcewl_call]("forward") \
            .def_py_method[_bcewl_train]("train") \
            .def_py_method[_bcewl_eval]("eval")

        _ = mb.add_type[OPT32]("SGD") \
            .def_py_init[_init_sgd_32]() \
            .def_py_method[_sgd_step]("step") \
            .def_py_method[_sgd_zero_grad]("zero_grad") \
            .def_py_method[_sgd_set_lr]("set_lr") \
            .def_py_method[_sgd_get_lr]("get_lr") \
            .def_py_method[_sgd_state_dict]("state_dict")

        _ = mb.add_type[ADAMW32]("AdamW") \
            .def_py_init[_init_adamw_32]() \
            .def_py_method[_adamw_step]("step") \
            .def_py_method[_adamw_zero_grad]("zero_grad") \
            .def_py_method[_adamw_set_lr]("set_lr") \
            .def_py_method[_adamw_get_lr]("get_lr") \
            .def_py_method[_adamw_state_dict]("state_dict")

        # ── DataLoader (tensor-native, dtype pairs) ────────────────
        # (float32, int64) covers the class-index CE path; (float32,
        # float32) covers MSE/BCE and CE-probability targets. Each extra
        # pair costs ~1-2GB of compile-time memory; widen one pair at a
        # time, building with --optimization-level 1 under a 17G cap:
        #   register_data_loader[DType.float64, DType.int64](mb, "DataLoaderF64")
        register_data_loader[DType.float32, DType.int64](mb, "DataLoader")
        register_data_loader[DType.float32, DType.float32](
            mb, "DataLoaderProb"
        )

        # ── Module-level functions ────────────────────────────────
        mb.def_function[_tensor_fn]("tensor")
        mb.def_function[_zeros_fn]("zeros")
        mb.def_function[_ones_fn]("ones")
        mb.def_function[_randn_fn]("randn")
        mb.def_function[_arange_fn]("arange")
        mb.def_function[_matmul_fn]("matmul")
        mb.def_function[_accuracy_compute]("accuracy")
        mb.def_function[_accuracy_token]("token_accuracy")
        mb.def_function[_accuracy_sequence]("sequence_accuracy")
        mb.def_function[_train_epoch]("train_epoch")
        mb.def_function[_eval_epoch]("eval_epoch")
        mb.def_function[_concat_fn]("concat")
        mb.def_function[_stack_fn]("stack")
        mb.def_function[_where_fn]("where")
        mb.def_function[_full_fn]("full")
        mb.def_function[_rand_fn]("rand")
        mb.def_function[_linspace_fn]("linspace")
        mb.def_function[_eye_fn]("eye")

        return mb.finalize()
    except:
        return PythonObject(None)
