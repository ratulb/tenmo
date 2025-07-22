from memory import memcpy, Pointer
from os import abort
from shared import TensorLike
from common_utils import log_debug



fn main() raises:
    pass


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var ancestors: UnsafePointer[UnsafePointer[TensorLike[dtype]]]
    var size: Int
    var capacity: Int

    fn __init__(out self):
        self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]]()
        self.capacity = 0
        self.size = 0

    fn __init__(out self, capacity: Int):
        self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]].alloc(
            capacity
        )
        self.capacity = capacity
        self.size = 0

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        if existing.size > 0:
            self.ancestors = UnsafePointer[
                UnsafePointer[TensorLike[dtype]]
            ].alloc(existing.size)
            memcpy(self.ancestors, existing.ancestors, existing.size)
        else:
            self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]]()

    fn __moveinit__(out self, owned existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        if existing.size > 0:
            self.ancestors = UnsafePointer[
                UnsafePointer[TensorLike[dtype]]
            ].alloc(existing.size)
            memcpy(self.ancestors, existing.ancestors, existing.size)
        else:
            self.ancestors = UnsafePointer[UnsafePointer[TensorLike[dtype]]]()

    @staticmethod
    fn untracked() -> Ancestors[dtype]:
        return Self()

    @always_inline("nodebug")
    # fn __del__(owned self):
    fn free(owned self):
        if self.ancestors:
            log_debug("Ancestors __del__ called")
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

    fn append(mut self, addr: UnsafePointer[TensorLike[dtype]]):
        if self.size == self.capacity:
            new_capacity = max(1, self.capacity * 2)
            self.resize(new_capacity)
        (self.ancestors + self.size)[] = addr
        self.size += 1

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
            each = self.get(i)[]
            inner_id = each.inner_id()
            print(String(inner_id), end=" ")
            # print(self.get(i).__str__(), end=" ")
        print()

    fn __contains__(self, tensor_like: TensorLike[dtype]) -> Bool:
        for i in range(len(self)):
            tensor_like_inside = self.get(i)[]
            inner_id = tensor_like_inside.inner_id()
            if inner_id == tensor_like.inner_id():
                return True
        return False

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
