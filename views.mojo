from tensors import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from os import abort
from shared import TensorLike
from memory import memcpy


fn main():
    pass


from testing import assert_true


fn test_into_tensor_full_view_copy() raises:
    print("test_into_tensor_full_view_copy")
    var t = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var v = t.view(Shape.of(2, 2))
    var out = v.into_tensor()
    assert_true(out.all_close(t))
    assert_true(out.data != t.data)  # Ensure deep copy


fn test_into_tensor_transposed_view() raises:
    print("test_into_tensor_transposed_view")
    var t = Tensor.d2([[1, 2, 3], [4, 5, 6]])
    var v = t.transpose()
    var out = v.into_tensor()
    assert_true(out.all_close(Tensor.d2([[1, 4], [2, 5], [3, 6]])))


fn test_into_tensor_offset_view() raises:
    print("test_into_tensor_offset_view")
    var t = Tensor.d1([0, 1, 2, 3, 4, 5])
    var v = t.view(Shape.of(2), offset=3)
    var out = v.into_tensor()
    assert_true(out.all_close(Tensor.d1([3, 4])))


fn test_into_tensor_scalar_view() raises:
    print("test_into_tensor_scalar_view")
    var t = Tensor.scalar(42)
    var v = t.view(Shape.Void)
    var out = v.into_tensor()
    assert_true(out.shape == Shape.Void)
    assert_true(out.item() == 42)


fn test_into_tensor_empty_view() raises:
    print("test_into_tensor_empty_view")
    var t = Tensor[DType.float32](Shape.of(0, 3))
    var v = t.view(Shape.of(0, 3))
    var out = v.into_tensor()
    assert_true(out.shape == Shape.of(0, 3))
    # assert_true(out.data.len() == 0)


fn test_into_tensor_grad_flag_true() raises:
    print("test_into_tensor_grad_flag_true")
    var t = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = t.view(Shape.of(2, 2))
    var out = v.into_tensor()
    assert_true(out.requires_grad == True)


fn test_into_tensor_grad_flag_false() raises:
    print("test_into_tensor_grad_flag_false")
    var t = Tensor.d2([[1, 2], [3, 4]], requires_grad=False)
    var v = t.view(Shape.of(2, 2))
    var out = v.into_tensor()
    assert_true(out.requires_grad == False)


fn test_into_tensor_large_contiguous_copy() raises:
    print("test_into_tensor_large_contiguous_copy")
    N = 1024 * 1024
    var t = Tensor[DType.float32](Shape.of(N))
    for i in range(N):
        t[i] = i
    var v = t.view(Shape.of(N))
    var out = v.into_tensor()
    assert_true(out.shape == Shape.of(N))
    assert_true(out[123456] == 123456)


_ = """fn test_into_tensor_isolated_memory() raises:
    print("test_into_tensor_isolated_memory")
    var t = Tensor.d1([1, 2, 3, 4])
    var v = t.slice(1, 3)  # [2, 3]
    var out = v.into_tensor()
    v[0] = 999
    assert_true(out.all_close(Tensor.d1([2, 3])))  # Unaffected by view mutation

fn test_into_tensor_strided_view_rows() raises:
    print("test_into_tensor_strided_view_rows")
    var t = Tensor.d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var v = t.slice_rows(0, 4, 2)  # Should give rows [0, 2]
    var out = v.into_tensor()
    assert_true(out.all_close(Tensor.d2([[1, 2], [5, 6]])))

fn test_into_tensor_contiguous_slice_1d() raises:
    print("test_into_tensor_contiguous_slice_1d")
    var t = Tensor.d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    var v = t.slice(2, 7)
    var out = v.into_tensor()
    assert_true(out.all_close(Tensor.d1([2, 3, 4, 5, 6])))



fn test_into_tensor_nested_view() raises:
    print("test_into_tensor_nested_view")
    var t = Tensor.d1([10, 20, 30, 40, 50])
    var v1 = t.slice(1, 5)           # [20, 30, 40, 50]
    var v2 = v1.slice(1, 3)          # [30, 40]
    var out = v2.into_tensor()
    assert_true(out.all_close(Tensor.d1([30, 40])))"""


struct TensorView[dtype: DType = DType.float32](
    Sized & Copyable & Movable & Stringable & Representable & Writable
):
    alias Blank: TensorView[dtype] = Self(
        UnsafePointer[Tensor[dtype]](), Shape.Void, Strides(IntList.Empty), 0
    )
    var base_tensor: UnsafePointer[Tensor[dtype]]
    var shape: Shape
    var strides: Strides
    var offset: Int
    var requires_grad: Bool

    fn __init__(
        out self,
        base_tensor: UnsafePointer[Tensor[dtype]],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Bool = False,
    ):
        self.base_tensor = base_tensor
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.requires_grad = requires_grad

    fn __moveinit__(out self, owned other: Self):
        self.base_tensor = other.base_tensor
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self.requires_grad = other.requires_grad

    fn __copyinit__(out self, other: Self):
        self.base_tensor = other.base_tensor
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset
        self.requires_grad = other.requires_grad

    fn is_contiguous(self) -> Bool:
        # return self.offset == 0 and self.strides.is_contiguous(self.shape)
        return self.strides.is_contiguous(self.shape)

    fn into_tensorlike(self) -> TensorLike[dtype]:
        return TensorLike[dtype](UnsafePointer(to=self))

    # Index calculation: flat offset into underlying tensor's data[]
    fn index_offset(self, indices: IntList) -> Int:
        if not indices.len() == self.shape.rank():
            abort("TensorView → index_offset → rank mismatch")
        var flat_idx = self.offset
        for i in range(indices.len()):
            flat_idx += indices[i] * self.strides[i]
        return flat_idx

    # Element access
    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        return self.base_tensor[].data.load[volatile=True](
            self.index_offset(indices)
        )

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        self.base_tensor[].data.store[volatile=True](
            self.index_offset(indices), value
        )

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        return self.base_tensor[].data.load[volatile=True](
            self.index_offset(IntList(indices))
        )

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        self.base_tensor[].data.store[volatile=True](
            self.index_offset(IntList(indices)), value
        )

    fn has_grad(self) -> Bool:
        return self.base_tensor[].has_grad()

    fn _requires_grad(self) -> Bool:
        return self.requires_grad

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn id(self) -> Int:
        return Int(self.address())

    fn is_view(self) -> Bool:
        return True

    fn is_tensor(self) -> Bool:
        return False

    fn into_view(self) -> TensorView[dtype]:
        return self

    fn into_tensor(self) -> Tensor[dtype]:
        out = Tensor[dtype](self.shape, requires_grad=self.requires_grad)
        numels = self.shape.num_elements()

        if self.is_contiguous():
            # Fast path: single memcpy from base tensor
            memcpy[Scalar[dtype]](
                out.data, self.base_tensor[].data + self.offset, numels
            )
            return out

        # Slow path: general indexing using shape
        rank = self.shape.rank()
        indices = IntList.filled(rank, 0)

        for _ in range(numels):
            # Copy value at current index from view to out
            out[indices] = self[indices]

            # Increment multi-dimensional index (manual shape walker)
            var carry = True
            for dim in reversed(range(rank)):
                if carry:
                    indices[dim] += 1
                    if indices[dim] >= self.shape[dim]:
                        indices[dim] = 0  # Carry over
                        carry = True
                    else:
                        carry = False
        return out

    fn seed_grad(self, value: Scalar[dtype]):
        self.base_tensor[].seed_grad(value)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        self.base_tensor[].seed_grad(with_tensor)

    fn __str__(self) -> String:
        dims = len(self.shape)
        s = String("[")
        if dims == 1:
            s += "1D View"
        elif dims == 2:
            s += "2D View"
        elif dims == 3:
            s += "3D View"
        elif dims == 4:
            s += "4D View"
        elif dims == 5:
            s += "5D View"
        else:
            s += "View"
        s += self.shape.__str__()
        s += ", Type: " + self.dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __len__(self) -> Int:
        return self.numels()

    fn len(self) -> Int:
        return self.numels()

    fn size(self) -> Int:
        return self.numels()

    fn numels(self) -> Int:
        return self.shape.num_elements()

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        tensor_like = self.into_tensorlike()
        tensor_like.print(num_first, num_last)
