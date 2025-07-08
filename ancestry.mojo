from memory import UnsafePointer, memcpy, Pointer
from os import abort
from shared import TensorLike
from common_utils import log_debug


#@register_passable
struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var ancestors: UnsafePointer[UnsafePointer[TensorLike[dtype]]]
    var size: Int
    var capacity: Int

    fn __init__(out self):
        self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]]()
        self.capacity = 0
        self.size = 0

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]].alloc(
            existing.capacity
        )
        #memcpy(self.ancestors, existing.ancestors, existing.size)
        for idx in range(existing.size):
            (self.ancestors + idx)[] = (existing.ancestors + idx)[]

    fn __moveinit__(out self, owned existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]].alloc(
            existing.capacity
        )
        #memcpy(self.ancestors, existing.ancestors, existing.size)
        for idx in range(existing.size):
            (self.ancestors + idx)[] = (existing.ancestors + idx)[]



    @staticmethod
    fn untracked() -> Ancestors[dtype]:
        return Self()

    @staticmethod
    fn with_capacity(capacity: Int) -> Ancestors[dtype]:
        array = Self()
        array.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]].alloc(
            capacity
        )
        array.capacity = capacity
        array.size = 0
        return array

    @always_inline("nodebug")
    # fn __del__(owned self):
    fn free(owned self):
        if self.ancestors:
            log_debug("Ancestors __del__ is kicking in alright")
            for idx in range(len(self)):
                (self.ancestors + idx).destroy_pointee()
            self.ancestors.free()

    @always_inline("nodebug")
    fn get(self, idx: Int) -> UnsafePointer[TensorLike[dtype]]:
        if idx < 0 or idx >= len(self):
            abort("Ancestors get -> Out-of-bounds read")
        address = (self.ancestors + idx)[]
        if address.__as_bool__() == False:
            abort("Ancestors get -> Uninitialized ancestor address")
        return address

    fn __len__(self) -> Int:
        return self.size

    fn append(mut self, address_: UnsafePointer[TensorLike[dtype]]):
        if self.size == self.capacity:
            new_capacity = max(1, self.capacity * 2)
            self.resize(new_capacity)
        (self.ancestors + self.size)[] = address_
        self.size += 1

    fn add_ancestry(mut self, tensor_likes: VariadicListMem[TensorLike[dtype]]):
        for tensor_like in tensor_likes:
            if tensor_like._requires_grad():
                self.append(tensor_like.address())

    fn resize(mut self, new_capacity: Int):
        self.reserve(new_capacity)

    fn reserve(mut self, new_capacity: Int):
        if new_capacity <= self.capacity:
            return
        new_ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]].alloc(
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

    fn print(self) -> None:
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i)[].id().__str__(), end=" ")
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

    fn __next__(mut self) -> UnsafePointer[TensorLike[dtype]]:
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

from tensors import Tensor

fn populate_ancestry[dtype: DType = DType.float32](*tensor_likes: TensorLike[dtype]) -> Ancestors[dtype]:
    #ancestors1 = Ancestors[dtype].untracked()
    ancestors1 = Ancestors[dtype].with_capacity(2)
    #ancestors1.add_ancestry(tensor_likes)
    for each in tensor_likes:
        ancestors1.append(each.address())
    print("ok1")
    ancestors1.print()
    return ancestors1


fn main():
    ancestors = Ancestors[DType.float32].untracked()
    print("ok0")
    ancestors.print()
    t1 = Tensor([1,2,3], requires_grad=True)
    t2 = Tensor([4,2,3], requires_grad=True)
    ancestors2 = populate_ancestry(t1.into_tensorlike(),t2.into_tensorlike())
    print("ok2")
    ancestors2.print()
    copied = ancestors2
    print("ok3")
    copied.print()
