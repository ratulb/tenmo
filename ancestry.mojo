from memory import Pointer
from shared import TensorLite
from common_utils import log_debug


fn main() raises:
    pass


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var ancestors: List[TensorLite[dtype]]

    fn __init__(out self):
        self.ancestors = List[TensorLite[dtype]]()

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
            for tli in self:
                tli.free()
            self.ancestors.clear()

    @always_inline
    fn get(self, idx: Int) -> TensorLite[dtype]:
        return self.ancestors[idx]

    fn __len__(self) -> Int:
        return len(self.ancestors)

    @always_inline
    fn append(mut self, tli: TensorLite[dtype]):
        self.ancestors.append(tli)

    @no_inline
    fn print(self, id: Bool = True) -> None:
        var total = len(self)
        print("Ancestors[", total, "] = ", end="")

        for tli in self.ancestors:
            if id:
                print(tli.inner_id(), end=" ")
            else:
                print(tli.inner_address().__str__(), end=" ")
        print()

    fn __contains__(self, tensor_like: TensorLite[dtype]) -> Bool:
        for tli in self.ancestors:
            if tli.inner_id() == tensor_like.inner_id():
                return True
        return False

    fn __iter__(ref self) -> AncestorsIter[self.dtype, __origin_of(self)]:
        return AncestorsIter[self.dtype](0, Pointer(to=self))

    fn __reversed__(
        ref self,
    ) -> AncestorsIter[self.dtype, __origin_of(self), False]:
        return AncestorsIter[self.dtype, forward=False](
            len(self), Pointer(to=self)
        )

struct AncestorsIter[dtype: DType, origin: Origin[False], forward: Bool = True](
    Sized & Copyable
):
    var index: Int
    var src: Pointer[Ancestors[dtype], origin]

    fn __init__(out self, idx: Int, src: Pointer[Ancestors[dtype], origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> TensorLite[dtype]:
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
