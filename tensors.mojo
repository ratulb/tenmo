### Mojo Tensor
### Implement tensor library in mojo from first principles
from math import iota, exp, floor
from random import seed, random_float64
from algorithm import vectorize
from sys import simdwidthof
from utils.numerics import max_finite, min_finite
from os import abort
from memory import memcpy, memset, memset_zero
from shapes import Shape
from intlist import IntList
from ancestry import Ancestors
from views import TensorView
from strides import Strides
from shared import TensorLike
from common_utils_imports import *
from operators import __tensor_op_tensor__, __tensor_op_scalar__
from operators_imports import *
from walkback import *


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Representable & Writable
):
    alias Row = List[Scalar[dtype]]
    alias Rows = List[Self.Row]
    alias Block = List[Self.Rows]
    alias Blocks = List[Self.Block]
    alias Ancestor_of = TensorLike.from_tensor
    var shape: Shape
    var strides: Strides
    var data: UnsafePointer[Scalar[dtype]]
    var requires_grad: Bool
    var grad: UnsafePointer[Self]
    var ancestors: Ancestors[dtype]
    var base: UnsafePointer[Tensor[dtype]]  # Only allocated on need basis
    var backwardFn: Optional[BackwardFn[dtype]]

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(
        out self,
        shape: Shape,
        data: UnsafePointer[Scalar[dtype]],
        requires_grad: Bool = False,
    ):
        Shape.validate(shape)
        self.shape = shape
        self.strides = Strides.default(shape)
        self.requires_grad = requires_grad
        self.backwardFn = None
        self.grad = UnsafePointer[Self]()
        self.ancestors = Ancestors[dtype].untracked()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.data = data
        self.init_grad()

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape
        self.strides = Strides.default(shape)
        self.requires_grad = requires_grad
        self.base = UnsafePointer[Tensor[dtype]]()
        self.backwardFn = None
        self.grad = UnsafePointer[Self]()
        self.ancestors = Ancestors[dtype].untracked()
        if shape.ndim == 0:  # Tensor with Shape ()
            self.data = UnsafePointer[Scalar[dtype]].alloc(1)
        else:
            self.data = UnsafePointer[Scalar[dtype]].alloc(
                self.shape.num_elements()
            )

        self.init_grad()

    fn __moveinit__(out self, owned other: Self):
        self.shape = other.shape
        self.strides = other.strides
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.ancestors = other.ancestors
        self.base = other.base
        self.backwardFn = other.backwardFn

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.strides = other.strides
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.ancestors = other.ancestors
        self.base = other.base
        self.backwardFn = other.backwardFn
        self.init_grad()

    fn init_grad(mut self):
        if self.requires_grad and not self.grad.__as_bool__():
            gradients = Tensor[dtype](self.shape)
            self.grad = UnsafePointer[Self].alloc(1)
            self.grad.init_pointee_move(gradients^)
            self.zero_grad()

    fn is_contiguous(self) -> Bool:
        return True

    fn is_tensor(self) -> Bool:
        return True

    fn is_leaf(self) -> Bool:
        return self.requires_grad and self.has_backward_fn()

    fn is_view(self) -> Bool:
        return False

    fn backward(self, start_grad: Scalar[dtype] = 1.0):
        TensorLike.from_tensor(self).backward(start_grad)

    fn backward(self, seed_tensor: Tensor[dtype]):
        TensorLike.from_tensor(self).backward(seed_tensor)

    fn into_view(
        self, requires_grad: Optional[Bool] = None
    ) -> TensorView[dtype]:
        shape = self.shape
        strides = self.strides
        out = TensorView(
            self.unsafe_address(),
            self.shape,
            self.strides,
            offset=0,
            requires_grad=requires_grad.value() if requires_grad else self.requires_grad,
        )
        if self.requires_grad:
            backward_fn = ViewBackward[dtype](
                shape, strides, 0
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn permute(self, axes: IntList) -> TensorView[dtype]:
        view = self.into_view()
        permutated = view.permute(axes)
        return permutated

    fn permute(self, axes: List[Int]) -> TensorView[dtype]:
        view = self.into_view()
        permutated = view.permute(axes)
        return permutated

    # Check if it has a backward fn before calling this API
    fn backward_fn(self) -> BackwardFn[dtype]:
        return self.backwardFn.value()

    fn has_backward_fn(self) -> Bool:
        return self.backwardFn is not None

    fn rows(self) -> Int:
        if not self.rank() == 2:
            abort("Tensor → rows: tensor rank is not 2")
        return self.shape[0]

    fn cols(self) -> Int:
        if not self.rank() == 2:
            abort("Tensor → cols: tensor rank is not 2")
        return self.shape[1]

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor → __getitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("Tensor →__getitem__(*indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(IntList())"
            )
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(*indices): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor → __setitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(IntList): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn item(self) -> Scalar[dtype]:
        if (
            self.shape != Shape.Unit and self.shape.ndim != 0
        ):  # Tensor with Shape ()
            abort(
                "Tensor.item(): Only valid for scalar or singleton tensors, got"
                " shape: "
                + self.shape.__str__()
            )
        return self[0] if self.shape == Shape.Unit else self[IntList.Empty]

    fn __getitem__(self, *slices: Slice) -> TensorView[dtype]:
        # Delegate shape/strides/offset computation
        view_shape, view_strides, new_offset = (
            Validator.validate_and_compute_view_metadata(
                self.shape,
                self.strides,
                slices,
            )
        )

        out = TensorView[dtype](
            self.unsafe_address(),
            shape=view_shape,
            strides=view_strides,
            offset=new_offset,
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            backward_fn = ViewBackward[dtype](
                view_shape, view_strides, new_offset
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __getitem__(self, *indices: Idx) -> TensorView[dtype]:
        # Compute view metadata
        view_shape, view_strides, view_offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                self.shape, self.strides, indices
            )
        )

        # Handle scalar (rank-0) case
        is_scalar = len(view_shape) == 0
        shape = Shape.Void if is_scalar else view_shape
        strides = Strides.Zero if is_scalar else view_strides

        # Create view
        out = TensorView[dtype](
            self.unsafe_address(),
            shape=shape,
            strides=strides,
            offset=view_offset,
            requires_grad=self.requires_grad,
        )

        # Setup autograd
        if self.requires_grad:
            backward_fn = ViewBackward[dtype](
                shape, strides, view_offset
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    # Always use this to print grad to avoid surprises of segmentation fault!
    fn gprint(self, num_first: Int = 10, num_last: Int = 10):
        if not self.requires_grad:
            print("Tensor is non-differentiable")
        elif self.requires_grad and not self.has_grad():
            print("Requires grad but grad not initialized")
        else:
            self.grad[].print(num_first, num_last)

    fn add_ancestry(mut self, *parents: TensorLike[dtype]):
        for parent in parents:
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_copy(parent)
            self.ancestors.append(ptr)

    fn ancestry(self) -> Ancestors[dtype]:
        return self.ancestors

    fn free(owned self):
        if self.has_grad():
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
                (self.grad[].data + i).destroy_pointee()
            self.grad.free()
            log_debug(
                "Tensor__del__ → freed grad(and pointees) and self data"
                " pointees"
            )
        else:
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
            log_debug("Tensor__del__ → freed self data pointees")
        self.shape.free()
        if self.data:
            self.data.free()
        log_debug("Tensor__del__ → called free on data")
        if self.base:
            self.base[].free()
            self.base.destroy_pointee()
            self.base.free()
            log_debug("Tensor__del__ → called free on base")
        _ = self^

    fn __len__(self) -> Int:
        return self.shape.num_elements()

    fn len(self) -> Int:
        return self.shape.num_elements()

    fn size(self) -> Int:
        return self.shape.num_elements()

    fn numels(self) -> Int:
        return self.shape.num_elements()

    fn ndim(self) -> Int:
        return self.shape.ndim

    fn rank(self) -> Int:
        return self.shape.ndim

    @always_inline
    fn broadcastable(self, to: Tensor[dtype]) -> Bool:
        return self.shape.broadcastable(to.shape)

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.for_all(all_truthy)

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        fn any_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.any(any_truthy)

    fn for_all[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = self.data.load[width=simd_width](i * simd_width)
            for j in range(simd_width):
                if not pred(vector[j]):
                    return False
        for k in range(remaining):
            if not pred(self.data.load[width=1](simd_blocks * simd_width + k)):
                return False
        return True

    fn any[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = self.data.load[width=simd_width](i * simd_width)
            for j in range(simd_width):
                if pred(vector[j]):
                    return True
        for k in range(remaining):
            if pred(self.data.load[width=1](simd_blocks * simd_width + k)):
                return True
        return False

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "Tensor → all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            abort(
                "Tensor → all_close expects same shaped tensors: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector1 = self.data.load[width=simd_width](i * simd_width)
            vector2 = other.data.load[width=simd_width](i * simd_width)
            diff = abs(vector1 - vector2)
            tolerance = atol + rtol * abs(vector2)
            if (diff > tolerance).reduce_or():
                return False
        for k in range(remaining):
            value1 = self.data.load[width=1](simd_blocks * simd_width + k)
            value2 = other.data.load[width=1](simd_blocks * simd_width + k)
            if abs(value1 - value2) > atol + rtol * abs(value2):
                return False

        return True

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        if not self._requires_grad() or not self.has_grad():
            return
        if self.shape != with_tensor.shape:
            abort(
                "Tensor → seed_grad: Shapes not equal -> "
                + self.shape.__str__()
                + " ≠ "
                + with_tensor.shape.__str__()
            )
        memcpy(self.grad[].data, with_tensor.data, with_tensor.numels())

    fn seed_grad(self, value: Scalar[dtype]):
        if self.has_grad():
            self.grad[].fill(value)

    fn fill(self, value: Scalar[dtype]):
        @parameter
        fn set_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](idx, value)

        vectorize[set_value, simdwidthof[dtype]()](self.numels())

    fn __eq__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return tensor_compare_scalar[Equal](self, scalar)

    fn __ne__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return tensor_compare_scalar[NotEqual](self, scalar)

    fn __lt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return tensor_compare_scalar[LessThan](self, scalar)

    fn __le__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return tensor_compare_scalar[LessThanEqual](self, scalar)

    fn __gt__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return tensor_compare_scalar[GreaterThan](self, scalar)

    fn __ge__(self, scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        return tensor_compare_scalar[GreaterThanEqual](self, scalar)

    fn __eq__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return tensor_compare[Equal](self, other)

    fn __ne__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return tensor_compare[NotEqual](self, other)

    fn __lt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return tensor_compare[LessThan](self, other)

    fn __le__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return tensor_compare[LessThanEqual](self, other)

    fn __gt__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return tensor_compare[GreaterThan](self, other)

    fn __ge__(self, other: Tensor[dtype]) -> Tensor[DType.bool]:
        return tensor_compare[GreaterThanEqual](self, other)

    fn __eq__(self, other: TensorView[dtype]) -> Bool:
        return TensorLike.from_tensor(self).equal(TensorLike.from_view(other))

    fn __iadd__(self, other: Self):
        if self.is_leaf():
            abort(
                "Tensor → __iadd__(self, other): Cannot perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            abort(
                "Tensor → __iadd__(self, other): Dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        @parameter
        fn add_elems[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx,
                (
                    self.data.load[width=simd_width](idx)
                    + other.data.load[width=simd_width](idx)
                ),
            )

        vectorize[add_elems, simdwidthof[dtype]()](self.numels())

    fn exp(self) -> Tensor[dtype]:
        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn exp_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, exp(self.data.load[width=simd_width](idx))
            )

        vectorize[exp_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __neg__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __neg__ is for numeric data types only",
        ]()

        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn negate_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).__neg__()
            )

        vectorize[negate_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __invert__(self: Tensor[DType.bool]) -> Tensor[DType.bool]:
        result = Tensor[DType.bool](self.shape)

        @parameter
        fn invert_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, ~self.data.load[width=simd_width](idx)
            )

        vectorize[invert_elems, simdwidthof[DType.bool]()](result.numels())
        return result

    fn __abs__(self) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __abs__ is for numeric data types only",
        ]()

        result = Tensor[dtype](self.shape, requires_grad=self.requires_grad)

        @parameter
        fn absolute_value[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).__abs__()
            )

        vectorize[absolute_value, simdwidthof[dtype]()](result.numels())
        return result

    fn has_grad(self) -> Bool:
        return self.grad.__as_bool__()

    fn _requires_grad(self) -> Bool:
        return self.requires_grad

    fn grad_is_zero(self) -> Bool:
        if not self.requires_grad:
            abort(
                "Tensor → grad_is_zero: checking grad on a tensor that does"
                " have grad"
            )

        fn all_zero(val: Scalar[dtype]) -> Bool:
            return val == Scalar[dtype](0)

        return self.has_grad() and self.grad[].for_all(all_zero)

    fn zero_grad(self):
        if self.requires_grad and self.has_grad():
            memset_zero(self.grad[].data, self.grad[].numels())

    fn __str__(self) -> String:
        dims = len(self.shape)
        s = String("[")
        if dims == 1:
            s += "1D Tensor"
        elif dims == 2:
            s += "2D Tensor"
        elif dims == 3:
            s += "3D Tensor"
        elif dims == 4:
            s += "4D Tensor"
        elif dims == 5:
            s += "5D Tensor"
        else:
            s += "Tensor"
        s += self.shape.__str__()
        s += ", Type: " + dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @staticmethod
    fn full_like(
        like: Tensor[dtype], value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        shape = like.shape
        return Tensor[dtype].full(shape, value, requires_grad=requires_grad)

    @staticmethod
    fn full(
        shape: Shape, value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        tensor.fill(value)
        return tensor

    @staticmethod
    fn rand(
        *axes_spans: Int,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(tensor.numels()):  # To be vectorized
            tensor.data.store[volatile=True](
                i,
                random_float64(
                    min.cast[DType.float64](), max.cast[DType.float64]()
                ).cast[dtype](),
            )
        return tensor

    @staticmethod
    fn arange(
        *args: Scalar[dtype],
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        start: Scalar[dtype] = 0
        end: Scalar[dtype] = max_finite[dtype]()
        step: Scalar[dtype] = 1

        n = len(args)
        if n == 1:
            end = args[0]
        elif n == 2:
            start = args[0]
            end = args[1]
        elif n == 3:
            start = args[0]
            end = args[1]
            step = args[2]
        else:
            abort(
                "Tensor.arange expects 1 to 3 arguments:\n"
                + "- arange(end)\n"
                + "- arange(start, end)\n"
                + "- arange(start, end, step)\n"
                + "Got: "
                + String(len(args))
                + " argument(s)"
            )

        if step == 0:
            abort("step can not be zero")
        if (step > 0 and start >= end) or (step < 0 and start <= end):
            abort("Invalid range for the given step")
        delta = end - start
        size = floor(delta / step)
        if size <= 0:
            abort("Error: computed arange size is zero")
        count = size.__int__()
        tensor = Tensor[dtype](count, requires_grad=requires_grad)

        @parameter
        fn fill(i: Int) -> Scalar[dtype]:
            return (i * step + start) % end

        @parameter
        fn mapper[simd_width: Int](idx: Int):
            first_entry = fill(idx).cast[dtype]()
            data = SIMD[dtype, simd_width](first_entry)
            for i in range(1, simd_width):
                data[i] = fill(idx + i).cast[dtype]()
            tensor.data.store[width=simd_width](idx, data)

        vectorize[mapper, simdwidthof[dtype]()](tensor.numels())

        return tensor

    @staticmethod
    fn zeros(
        axes_spans: List[Int], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        return Self.zeros(Shape(axes_spans), requires_grad)

    @staticmethod
    fn zeros(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        shape = Shape(axes_spans)
        return Self.zeros(shape, requires_grad)

    @staticmethod
    fn zeros_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype](tensor.shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

    @staticmethod
    fn ones_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype].full(tensor.shape, 1, requires_grad=requires_grad)
        return out

    @staticmethod
    fn zeros(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        out = Tensor[dtype](shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

    @staticmethod
    fn d1(row: Self.Row, requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d1")
        # Gotchas - watch out
        if len(row) == 0:
            return Tensor[dtype].scalar(
                min_finite[dtype](), requires_grad=requires_grad
            )
        shape = Shape(IntList(len(row)))
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, row.data, len(row))
        return tensor

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntList(len(rows), len(rows[0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                abort("Tensor → d2 → not all rows equal in length")
            flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d3")
        dims = IntList(len(blocks), len(blocks[0]), len(blocks[0][0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                abort("Tensor → d3 → not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    abort("Tensor → d3 → not all rows equal in length")

                flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn d4(
        blockgrid: List[Self.Block], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d4")
        dims = IntList(
            len(blockgrid),
            len(blockgrid[0]),
            len(blockgrid[0][0]),
            len(blockgrid[0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blockgrid:
            if len(block) != dims[1]:
                abort(
                    "Tensor → d4 → not all blocks are of equal length in the"
                    " blockgrid"
                )
            for matrix in block:
                if len(matrix) != dims[2]:
                    abort(
                        "Tensor → d4 → not all matrices are of equal length"
                        " in block"
                    )
                for row in matrix:
                    if len(row) != dims[3]:
                        abort(
                            "Tensor → d4 not all rows are of equal length in"
                            " matrix"
                        )
                    flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn d5(
        blockhive: List[Self.Blocks], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "d5")
        dims = IntList(
            len(blockhive),
            len(blockhive[0]),
            len(blockhive[0][0]),
            len(blockhive[0][0][0]),
            len(blockhive[0][0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for blocks in blockhive:
            if len(blocks) != dims[1]:
                abort(
                    "Tensor → d5 → not all blocks are of equal length in the"
                    " input"
                )
            for block in blocks:
                if len(block) != dims[2]:
                    abort("Tensor → d5 → unequal block length")
                for matrix in block:
                    if len(matrix) != dims[3]:
                        abort(
                            "Tensor → d5 not all matrices are of equal length"
                            " in block"
                        )
                    for row in matrix:
                        if len(row) != dims[4]:
                            abort(
                                "Tensor → d5 not all rows are of equal length"
                                " in matrix"
                            )
                        flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn of(*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "of(*elems)")
        # shape = Shape.of(len(elems))
        shape = Shape(IntList(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of(
        elems: Self.Row,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        Validator.validate_dtype_consistency(dtype, requires_grad, "of(elems)")
        shape = Shape.of(len(elems))
        tensor = Tensor[Self.dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of[
        row_size: Int
    ](*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Validator.validate_dtype_consistency(
            dtype, requires_grad, "of[row_size]"
        )

        if not (row_size >= 1 and row_size <= len(elems)):
            abort(
                (
                    "Tensor → of[row_size] → invalid row size or not enough"
                    " elements"
                ),
            )
        num_rows = len(elems) // row_size
        axes_spans = variadic1or2(num_rows, row_size)
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(num_rows):
            for j in range(row_size):
                tensor[i, j] = elems[i * row_size + j]
        return tensor

    @staticmethod
    fn scalar(val: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        result = Tensor[dtype](Shape.Void, requires_grad=requires_grad)
        result[IntList.Empty] = val
        return result

    @staticmethod
    fn ones(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        return Self.ones(Shape(axes_spans), requires_grad)

    @staticmethod
    fn ones(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        var value: SIMD[dtype, 1]

        @parameter
        if dtype.is_floating_point():
            value = SIMD[dtype, 1](1.0)
        else:
            value = SIMD[dtype, 1](1)
        for i in range(tensor.numels()):
            tensor.data.store(i, value)
        return tensor

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        tensor_like = TensorLike.from_tensor(self)
        tensor_like.print(num_first, num_last)

    fn float(self) -> Tensor[DType.float32]:
        if dtype == DType.float32:
            return rebind[Tensor[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Tensor[DType.float64]:
        if dtype == DType.float64:
            return rebind[Tensor[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Tensor[NewType]:
        result = Tensor[NewType](self.shape, self.requires_grad)

        @parameter
        fn cast_values[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).cast[NewType]()
            )

        vectorize[cast_values, simdwidthof[NewType]()](result.numels())
        return result

    fn update_grad[opcode: Int](self, gradients: Tensor[dtype]):
        self.grad[] = __tensor_op_tensor__[dtype, opcode](
            self.grad[], gradients
        )

    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape.Void

    fn data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn unsafe_address(
        ref self,
    ) -> UnsafePointer[
        Self,
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
        return UnsafePointer(to=self).origin_cast[
            mut = Origin(__origin_of(self)).mut, origin = __origin_of(self)
        ]()

    @always_inline
    fn load[nelts: Int = 1](self, row: Int, col: Int) -> SIMD[dtype, nelts]:
        constrained[
            is_power_of_two(nelts),
            "Tensor → load: SIMD width (nelts) must be a power of 2",
        ]()

        if self.rank() != 2:
            abort("Tensor → load: supported only for 2D tensors")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + nelts > self.shape[1]
        ):
            abort("Tensor → load: Out-of-bounds access")

        addr = row * self.strides[0] + col * self.strides[1]
        return self.data.load[width=nelts, volatile=True](addr)

    @always_inline
    fn store[
        nelts: Int = 1
    ](self, row: Int, col: Int, value: SIMD[dtype, nelts]):
        constrained[
            is_power_of_two(nelts),
            "Tensor → store: SIMD width (nelts) must be a power of 2",
        ]()

        if self.rank() != 2:
            abort("Tensor → store is supported only for 2D tensors")

        if (
            row < 0
            or row >= self.shape[0]
            or col < 0
            or col + nelts > self.shape[1]
        ):
            abort("Tensor → store: out-of-bounds access")

        addr = row * self.strides[0] + col * self.strides[1]
        self.data.store[width=nelts, volatile=True](addr, value)

    fn __rtruediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __rtruediv__ is for numeric data types only",
        ]()

        var out = __tensor_op_scalar__[dtype, DivideScalar](self, scalar)

        if self.requires_grad:
            backward_fn = RightTrueDivBackwardScalar[dtype](
                scalar
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __truediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __truediv__ is for numeric data types only",
        ]()

        if scalar == Scalar[dtype](0):
            abort("Tensor → __truediv__ : canot divide by " + scalar.__str__())
        var out = __tensor_op_scalar__[dtype, DivideByScalar](
            self,
            scalar,
        )
        if self.requires_grad:
            backward_fn = TrueDivBackwardScalar[dtype](
                scalar
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __radd__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, AddScalar](self, scalar)

        if self.requires_grad:
            backward_fn = AddBackwardScalar[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, factor: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, MulScalar](
            self,
            factor,
        )

        if self.requires_grad:
            backward_fn = MulBackwardScalar[dtype](factor).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __pow__(self, exponent: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __pow__ is for numeric data types only",
        ]()

        var out = __tensor_op_scalar__[dtype, Power](
            self,
            exponent,
        )

        if self.requires_grad:
            backward_fn = ExponientionBackward[dtype](
                exponent
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __rsub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, SubtractFromScalar](self, scalar)
        if self.requires_grad:
            backward_fn = SubLeftRightBackwardScalar[dtype](
                True
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __sub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, SubtractScalar](self, scalar)

        if self.requires_grad:
            backward_fn = SubLeftRightBackwardScalar[dtype](
                False
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn reshape(self) -> Tensor[dtype]:
        if self.numels() != 1:
            abort(
                "Only tensor with single element can be reshaped to scalar"
                " tensor"
            )
        return self.reshape(Shape.Void)

    fn reshape(self, *newdims: Int) -> Tensor[dtype]:
        if len(newdims) == 1 and newdims[0] == 0:
            return self.reshape()
        shape = Validator.validate_new_shape(
            self.shape.intlist(), IntList(newdims)
        )
        return self.reshape(shape, validated=True)

    fn reshape(self, shape: List[Int]) -> Tensor[dtype]:
        new_shape = Validator.validate_new_shape(
            self.shape.intlist(), IntList.new(shape)
        )
        return self.reshape(new_shape, validated=True)

    fn reshape(
        self, new_shape: Shape, validated: Bool = False
    ) -> Tensor[dtype]:
        shape = new_shape if validated else Validator.validate_new_shape(
            self.shape.intlist(), new_shape.intlist()
        )
        if self.numels() != shape.num_elements():
            abort(
                "Tensor with "
                + String(self.numels())
                + " element(s) can't be converted to a tensor containing "
                + String(shape.num_elements())
                + " element(s)"
            )

        requires_grad = self.requires_grad
        out = Tensor[dtype](shape, self.data, requires_grad=requires_grad)

        if requires_grad:
            # Only allocate base if needed
            base = Tensor[dtype].zeros(self.shape)
            out.base = UnsafePointer[Tensor[dtype]].alloc(1)
            out.base.init_pointee_move(base^)

            backward_fn = ReshapeBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn sum_all(self) -> Scalar[dtype]:
        return sum_all(self)

    fn sum(self, axes: List[Int] = [], keepdims: Bool = False) -> Tensor[dtype]:
        return self.sum(IntList.new(axes), keepdims)

    fn sum(
        self: Self,
        axes: IntList,
        keepdims: Bool = False,
    ) -> Tensor[dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            self.shape, axes
        )
        out = TensorLike.from_tensor(self).sum(normalized_axes, keepdims)
        if self.requires_grad:
            out.requires_grad = True
            out.init_grad()
            backward_fn = SumBackward[dtype](
                normalized_axes.copy(), keepdims
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn mean(
        self, axes: List[Int] = [], keepdims: Bool = False
    ) -> Tensor[dtype]:
        return self.mean(IntList.new(axes), keepdims)

    fn mean(self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            self.shape, axes
        )
        out = TensorLike.from_tensor(self).mean(normalized_axes, keepdims)
        if self.requires_grad:
            out.requires_grad = True
            out.init_grad()
            backward_fn = MeanBackward[dtype](
                normalized_axes.copy(), keepdims
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn __add__(self, other: Self) -> Tensor[dtype]:
        if self.unsafe_address() == other.unsafe_address():
            return self.__mul__(2)
        if not self.broadcastable(other):
            abort(
                "__add__ → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_add_subtract_operation[
                Add, AddTensor, AddTensor
            ](
                other,
            )

        var out = __tensor_op_tensor__[dtype, AddTensor](self, other)

        if self.requires_grad or other.requires_grad:
            backward_fn = AddBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            if self.requires_grad:
                out.add_ancestry(Self.Ancestor_of(self))
            if other.requires_grad:
                out.add_ancestry(Self.Ancestor_of(other))

        return out

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__sub__ → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_add_subtract_operation[
                Subtract, AddTensor, SubtractTensor
            ](other)

        out = __tensor_op_tensor__[dtype, SubtractTensor](self, other)

        if self.requires_grad or other.requires_grad:
            sub_backward = SubBackward[dtype]()
            if self.requires_grad:
                out.add_ancestry(Self.Ancestor_of(self))
                sub_backward.negate(False)
            if other.requires_grad:
                out.add_ancestry(Self.Ancestor_of(other))
                sub_backward.negate(True)
            backward_fn = sub_backward.into_backward_fn()
            out.backwardFn = Optional(backward_fn)

        return out

    fn broadcast_mul_operation(
        self: Self,
        other: Self,
    ) -> Tensor[dtype]:
        out = self.broadcast_op(other, scalar_ops[dtype, Multiply])
        requires_grad = self.requires_grad or other.requires_grad
        if requires_grad:
            backward_fn = BroadcastBackward[
                dtype, AddTensor, AddTensor, True
            ]().into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))
            out.add_ancestry(Self.Ancestor_of(other))

        return out

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__mul__(self * other) → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_mul_operation(other)

        var out = __tensor_op_tensor__[dtype, MulTensor](
            self,
            other,
        )

        if self.requires_grad or other.requires_grad:
            backward_fn = MultiplyBackward[dtype]().into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))
            out.add_ancestry(Self.Ancestor_of(other))

        return out

    fn broadcast_add_subtract_operation[
        Element_Wise_Op: Int, Tensor_Op_First: Int, Tensor_Op_Second: Int
    ](self, other: Self) -> Tensor[dtype]:
        var out = self.broadcast_op(other, scalar_ops[dtype, Element_Wise_Op])

        if self.requires_grad or other.requires_grad:
            backward_fn = BroadcastBackward[
                dtype, Tensor_Op_First, Tensor_Op_Second, False
            ]().into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))
            out.add_ancestry(Self.Ancestor_of(other))

        return out

    fn __iadd__(self, value: Scalar[dtype]):
        if self.is_leaf():
            abort(
                "Cannot perform in-place operation on a leaf tensor requiring"
                " grad."
            )

        @parameter
        fn add_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](self.numels())

    fn broadcast_to(self, target_shape: Shape) -> Tensor[dtype]:
        if not self.shape.broadcastable(target_shape):
            abort(
                "Tensor → broadcast_to: shape "
                + self.shape.__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        mask = self.shape.broadcast_mask(target_shape)
        out = Tensor[dtype](target_shape, requires_grad=self.requires_grad)

        for idx in target_shape:
            src_idx = self.shape.translate_index(idx, mask, target_shape)
            out[idx] = self[src_idx]

        return out

    fn broadcast_mask(self, broadcast_shape: Shape) -> IntList:
        return self.shape.broadcast_mask(broadcast_shape)

    fn translate_index(
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        return self.shape.translate_index(indices, mask, broadcast_shape)

    fn broadcast_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        if self.shape.rank() == 0 or other.shape.rank() == 0:
            return self.broadcast_scalar_op(other, op)
        else:
            return self.broadcast_tensor_op(other, op)

    fn broadcast_scalar_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        # Decide result shape
        result_shape = other.shape if self.shape.rank() == 0 else self.shape
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

        for indices in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[indices]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[indices]
            )
            result[indices] = op(self_val, other_val)

        return result

    fn broadcast_tensor_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        result_shape = Shape.broadcast_shape(self.shape, other.shape)
        mask1 = self.broadcast_mask(result_shape)
        mask2 = other.broadcast_mask(result_shape)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

        for indices in result_shape:
            self_indices = self.translate_index(indices, mask1, result_shape)
            other_indices = other.translate_index(indices, mask2, result_shape)
            result[indices] = op(self[self_indices], other[other_indices])

        return result

    fn backward_contribution(
        self,
        other: Tensor[dtype],
        upstream_grad: Tensor[dtype],
        do_multiply: Bool,
    ) -> Tensor[dtype]:
        var grad_contrib: Tensor[dtype]
        if upstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                self.shape, upstream_grad.item(), requires_grad=False
            )
        else:
            grad_contrib = (
                upstream_grad * other if do_multiply else upstream_grad
            )
            if grad_contrib.shape != self.shape:
                axes = self.broadcast_mask(grad_contrib.shape).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)
            grad_contrib.requires_grad = False

        return grad_contrib

    fn view(
        self,
        shape: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> TensorView[dtype]:
        return self.view(
            shape=Shape(shape), offset=offset, requires_grad=requires_grad
        )

    fn view(
        self,
        shape: Shape,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> TensorView[dtype]:
        if offset < 0 or offset >= self.numels():
            abort(
                "Tensor → view(shape): offset out of bounds: offset => "
                + String(offset)
                + "and self.numels() => "
                + String(self.numels())
            )
        # _requires_grad = requires_grad.value() if requires_grad else self.requires_grad
        if shape == self.shape and offset == 0:  # Tensor offset is always 0
            return self.into_view(requires_grad=requires_grad)
        if shape.num_elements() + offset > self.numels():
            abort("Tensor → view(shape): shape numels exceeds base tensor size")
        strides = Strides.default(shape)
        out = TensorView(
            self.unsafe_address(),
            shape,
            strides,
            offset=offset,
            requires_grad=requires_grad.value() if requires_grad else self.requires_grad,
        )
        if self.requires_grad:
            backward_fn = ViewBackward[dtype](
                shape, strides, offset
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out

    fn view(
        self,
        shape: List[Int],
        strides: List[Int],
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> TensorView[dtype]:
        return self.view(
            shape=Shape(shape),
            strides=Strides(strides),
            offset=offset,
            requires_grad=requires_grad,
        )

    fn view(
        self,
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> TensorView[dtype]:
        if offset < 0 or offset >= self.numels():
            abort("Tensor → view: offset out of bounds")

        if strides.rank() != shape.rank():
            abort("Tensor → view: shape and strides must have same rank")

        var min_index = offset
        var max_index = offset
        for i in range(shape.rank()):
            stride = strides[i]
            extent = (shape[i] - 1) * stride
            if extent > 0:
                max_index += extent
            else:
                min_index += extent

        if min_index < 0 or max_index >= self.numels():
            abort("Tensor → view: requested view accesses out-of-bounds data")

        return TensorView(
            self.unsafe_address(),
            shape,
            strides,
            offset=offset,
            requires_grad=requires_grad.value() if requires_grad else self.requires_grad,
        )

    fn mse(self, target: Tensor[dtype]) -> Tensor[dtype]:
        return ((self - target) ** 2).mean()

    fn matmul[
        simd_width: Int = simdwidthof[dtype]()
    ](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
        Validator.validate_matrix_shapes(A, B)
        rows_a = A.rows()
        cols_a = A.cols()
        cols_b = B.cols()
        C = Tensor[dtype].zeros(rows_a, cols_b)
        for i in range(rows_a):
            for j in range(cols_a):
                scalar_a = A.load(i, j)

                @parameter
                fn mul_add[simdwidth: Int](k: Int):
                    vectorized_a = SIMD[dtype, simdwidth](scalar_a)
                    vector_b = B.load[simdwidth](j, k)
                    product = vectorized_a * vector_b
                    offset = i * cols_b + k
                    C.data.store[width=simdwidth](
                        offset,
                        C.data.load[width=simdwidth](offset) + product,
                    )

                vectorize[mul_add, simd_width](cols_b)
        requires_grad = A.requires_grad or B.requires_grad
        if requires_grad:
            C.requires_grad = True
            C.init_grad()
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            C.backwardFn = Optional(backward_fn)
            C.add_ancestry(Self.Ancestor_of(A))
            C.add_ancestry(Self.Ancestor_of(B))

        return C

    fn matmul(self, other: TensorView[dtype]) -> Self:
        Validator.validate_matrix_shapes(self, other)
        return self.matmul(UnsafePointer(to=other), validated=True)

    fn matmul[
        simd_width: Int = simdwidthof[dtype]()
    ](
        A: Tensor[dtype],
        V: UnsafePointer[TensorView[dtype]],
        validated: Bool = False,
    ) -> Tensor[dtype]:
        B = V[]
        if not validated:
            Validator.validate_matrix_shapes(A, B)
        rows_a = A.shape[0]
        cols_a = A.shape[1]
        cols_b = B.shape[1]
        packed = B.strides[1] == 1

        C = Tensor[dtype].zeros(rows_a, cols_b)
        for i in range(0, rows_a):
            for j in range(0, cols_b, simd_width):
                mbatch = min(simd_width, cols_b - j)
                var accum = SIMD[dtype, simd_width](0)

                for k in range(0, cols_a):
                    scalar_a = A.load(i, k)

                    if packed and mbatch == simd_width:
                        simd_vector = B.load[simd_width](k, j)
                        accum += simd_vector * scalar_a
                    else:
                        # mbatch < simd_width or scattered B cols
                        for step in range(0, mbatch):
                            scalar_b = B.load(k, j + step)
                            accum[step] += scalar_a * scalar_b

                if mbatch == simd_width:
                    C.store[simd_width](i, j, accum)
                else:
                    for step in range(0, mbatch):
                        C.store(i, j + step, accum[step])

        requires_grad = A.requires_grad or B.requires_grad
        if requires_grad:
            C.requires_grad = True
            C.init_grad()
            backward_fn = MatmulBackward[dtype]().into_backward_fn()
            C.backwardFn = Optional(backward_fn)
            C.add_ancestry(Self.Ancestor_of(A))
            C.add_ancestry(TensorLike(V))

        return C

    fn transpose(
        self, *axes: Int, requires_grad: Optional[Bool] = None
    ) -> TensorView[dtype]:
        transpose_axes = IntList.with_capacity(len(axes))
        for axis in transpose_axes:
            transpose_axes.append(axis)
        return self.transpose(transpose_axes, requires_grad)

    fn transpose(
        self, axes: List[Int] = [], requires_grad: Optional[Bool] = None
    ) -> TensorView[dtype]:
        return self.transpose(IntList.new(axes))

    fn transpose(
        self, axes: IntList, requires_grad: Optional[Bool] = None
    ) -> TensorView[dtype]:
        shape = self.shape
        var normalized_axes = Validator.validate_axes(
            axes if len(axes)
            > 0 else IntList.range_list(shape.rank()).reversed(),
            shape,
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)
        out = self.view(
            new_shape,
            new_strides,
            offset=0,
            requires_grad=requires_grad.value() if requires_grad else self.requires_grad,
        )
        if self.requires_grad:
            backward_fn = TransposeBackward[dtype](
                normalized_axes
            ).into_backward_fn()

            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(Self.Ancestor_of(self))

        return out


fn main() raises:
    a = Tensor.rand(3, 4, 5)
    v = a[:, :, :]
    print(v.is_contiguous())


from testing import assert_true
