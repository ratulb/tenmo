from memory import memcpy, Pointer
from shared import TensorLite
from common_utils import log_debug, panic


fn main() raises:
    pass


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var ancestors: List[UnsafePointer[TensorLite[dtype]]]

    fn __init__(out self):
        self.ancestors = List[UnsafePointer[TensorLite[dtype]]]()

    fn __init__(out self, capacity: Int):
        self.ancestors = List[UnsafePointer[TensorLite[dtype]]](
            capacity=capacity
        )

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.ancestors = existing.ancestors.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.ancestors = existing.ancestors^

    @staticmethod
    fn untracked() -> Ancestors[dtype]:
        return Self()

    fn free(deinit self):
        if self.ancestors:
            log_debug("Ancestors __del__ called")
            for idx in range(len(self)):
                if self.ancestors[idx].__as_bool__() == False:
                    continue
                tensor_ptr = self.ancestors[idx][].inner_address()
                if tensor_ptr.__as_bool__() == False:
                    continue
                tensor_ptr.destroy_pointee()
                tensor_ptr.free()
                self.ancestors[idx].destroy_pointee()
                self.ancestors[idx].free()
            self.ancestors.clear()

    fn get(self, idx: Int) -> UnsafePointer[TensorLite[dtype]]:
        if idx < 0 or idx >= len(self.ancestors):
            panic("Ancestors get → Out-of-bounds read")
        address = self.ancestors[idx]
        if address.__as_bool__() == False:
            panic("Ancestors get → Uninitialized ancestor address")
        return address

    fn __len__(self) -> Int:
        return len(self.ancestors)

    fn __bool__(self) -> Bool:
        """Checks whether this contains any entry or not.

        Returns:
            `False` if there is no entry, `True` otherwise.
        """
        return len(self) > 0

    fn append(mut self, addr: UnsafePointer[TensorLite[dtype]]):
        self.ancestors.append(addr)

    @no_inline
    fn print(self, id: Bool = True) -> None:
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            each = self.get(i)
            instance = each[]
            inner_id = instance.inner_id()
            if id:
                print(inner_id, end=" ")
            else:
                print(each.__str__(), end=" ")
        print()

    fn __contains__(self, tensor_like: TensorLite[dtype]) -> Bool:
        for each in self.ancestors:
            entry = each[]
            inner_id = entry.inner_id()
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

    fn __next__(mut self) -> UnsafePointer[TensorLite[dtype]]:
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
