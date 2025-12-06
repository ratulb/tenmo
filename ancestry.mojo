from memory import Pointer
from tenmo import Tensor


fn main() raises:
    pass


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var ancestors: List[Tensor[dtype]]

    fn __init__(out self):
        self.ancestors = List[Tensor[dtype]]()

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.ancestors = existing.ancestors.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.ancestors = existing.ancestors^

    @staticmethod
    fn untracked() -> Ancestors[dtype]:
        return Self()

    @always_inline
    fn __del__(deinit self):
        self.ancestors.clear()

    fn get(ref self, idx: Int) -> ref[self.ancestors] Tensor[dtype]:
        return self.ancestors[idx]

    fn __len__(self) -> Int:
        return len(self.ancestors)

    fn __bool__(self) -> Bool:
        return len(self) > 0

    @always_inline
    fn append(mut self, var ancestor: Tensor[dtype]):
        self.ancestors.append(ancestor^)

    @no_inline
    fn print(self):
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i).id(), end=" ")
        print()

    fn __iter__(ref self) -> AncestorIterator[self.dtype, origin_of(self)]:
        return AncestorIterator[self.dtype](0, Pointer(to=self))


struct AncestorIterator[dtype: DType, origin: Origin[False]](
    Sized & ImplicitlyCopyable
):
    var index: Int
    var src: Pointer[Ancestors[dtype], origin]

    fn __init__(out self, idx: Int, src: Pointer[Ancestors[dtype], origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Tensor[dtype]:
        self.index += 1
        return self.src[].get(self.index - 1)

    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index
