from memory import UnsafePointer, memcpy, Pointer
from os import abort
from tensors import Tensor


@register_passable
struct Ancestors[dtype: DType = DType.float32](Sized & Copyable):
    var ancestors: UnsafePointer[UnsafePointer[Tensor[dtype]]]
    var size: Int
    var capacity: Int

    fn __init__(out self):
        self.ancestors = UnsafePointer[UnsafePointer[Tensor[dtype]]]()
        self.capacity = 0
        self.size = 0

    @always_inline("nodebug")
    fn __init__(out self, *addresses: UnsafePointer[Tensor[dtype]]):
        self.ancestors = UnsafePointer[UnsafePointer[Tensor[dtype]]].alloc(
            len(addresses)
        )
        self.size = len(addresses)
        self.capacity = len(addresses)
        for idx in range(len(addresses)):
            (self.ancestors + idx)[] = addresses[idx]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        self.ancestors = UnsafePointer[UnsafePointer[Tensor[dtype]]].alloc(
            existing.capacity
        )
        memcpy(self.ancestors, existing.ancestors, existing.size)

    @staticmethod
    fn none() -> Ancestors[dtype]:
        return Self()

    @staticmethod
    fn with_capacity(capacity: Int) -> Ancestors[dtype]:
        array = Self()
        array.ancestors = UnsafePointer[UnsafePointer[Tensor[dtype]]].alloc(
            capacity
        )
        array.capacity = capacity
        array.size = 0
        return array

    @always_inline("nodebug")
    # fn __del__(owned self):
    fn free(owned self):
        if self.ancestors:
            print("Ancestors __del__ is kicking in alright")
            for idx in range(len(self)):
                (self.ancestors + idx).destroy_pointee()
            self.ancestors.free()

    @always_inline("nodebug")
    fn get(self, idx: Int) -> UnsafePointer[Tensor[dtype]]:
        if idx < 0 or idx >= len(self):
            abort("Ancestors get -> Out-of-bounds read")
        address = (self.ancestors + idx)[]
        if address.__as_bool__() == False:
            abort("Ancestors get -> Uninitialized ancestor address")
        return address

    fn __len__(self) -> Int:
        return self.size

    fn append_all(mut self, *addresses: UnsafePointer[Tensor[dtype]]):
        for address in addresses:
            self.append(address)

    fn append(mut self, address: UnsafePointer[Tensor[dtype]]):
        if self.size == self.capacity:
            new_capacity = max(1, self.capacity * 2)
            self.resize(new_capacity)
        (self.ancestors + self.size)[] = address
        self.size += 1

    fn resize(mut self, new_capacity: Int):
        self.reserve(new_capacity)

    fn reserve(mut self, new_capacity: Int):
        if new_capacity <= self.capacity:
            return
        new_ancestors = UnsafePointer[UnsafePointer[Tensor[dtype]]].alloc(
            new_capacity
        )
        if self.size > 0:
            memcpy(new_ancestors, self.ancestors, self.size)
        if self.ancestors:
            for i in range(len(self)):
                (self.ancestors + i).destroy_pointee()
            self.ancestors.free()
        self.ancestors = new_ancestors
        self.capacity = new_capacity

    fn print(self) raises -> None:
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i).__str__(), end=" ")
        print()

    fn __iter__(ref self) -> _AncestorsIter[self.dtype, __origin_of(self)]:
        return _AncestorsIter[self.dtype](0, Pointer(to=self))

    fn __reversed__(
        ref self,
    ) -> _AncestorsIter[self.dtype, __origin_of(self), False]:
        return _AncestorsIter[self.dtype, forward=False](
            len(self), Pointer(to=self)
        )


struct _AncestorsIter[
    dtype: DType, origin: Origin[False], forward: Bool = True
](Sized & Copyable):
    var index: Int
    var src: Pointer[Ancestors[dtype], origin]

    fn __init__(out self, idx: Int, src: Pointer[Ancestors[dtype], origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> UnsafePointer[Tensor[dtype]]:
        @parameter
        if forward:
            self.index += 1
            return self.src[].get(self.index - 1)
        else:
            self.index -= 1
            return self.src[].get(self.index)

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


fn main():
    tensor = Tensor.rand(5, 3)
    print(tensor.address())
    print(Int(tensor.address()))
