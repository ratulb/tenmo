from std.memory import Pointer
from tenmo import Tensor
from gradbox import Gradbox
from backpropagation import BackwardFnArg
from ndbuffer import NDBuffer
from common_utils import panic
from shapes import Shape
from std.os.atomic import Consistency, fence

# ========================================
# Ancestor
# ========================================


struct Ancestor[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Lightweight handle for ancestry tracking.

    Carries everything needed for:
    1. Graph traversal        — _id
    2. Grad routing           — requires_grad, gradbox ptr
    3. Data (CPU+GPU)   — nd_buffer (shared NDBuffer, refcount bump only)
    4. Backward compute       — nd_buffer (shape/strides/data for matmul etc.)
    5. Backward invoke        — backwardFnArg, parents

    What is NOT here vs full Tensor copy:
    - No new gradbox allocation
    - No recursive Tensor ancestor chain copy
    """

    var _id: UInt
    var requires_grad: Bool
    var gradbox: UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]
    var nd_buffer: NDBuffer[Self.dtype]
    var parents: Optional[Ancestors[Self.dtype]]
    var on_gpu: Bool

    fn __init__(out self):
        self._id = 0
        self.requires_grad = False
        self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.nd_buffer = NDBuffer[Self.dtype]()
        self.parents = None
        self.on_gpu = False

    fn __copyinit__(out self, copy: Self):
        self._id = copy._id
        self.requires_grad = copy.requires_grad
        self.gradbox = copy.gradbox
        self.nd_buffer = copy.nd_buffer.copy()
        self.parents = copy.parents.copy()
        self.on_gpu = copy.on_gpu
        if self.gradbox:
            _ = (
                self.gradbox[]
                ._refcount[]
                .fetch_add[ordering=Consistency.MONOTONIC](1)
            )

    fn __moveinit__(out self, deinit take: Self):
        self._id = take._id
        self.requires_grad = take.requires_grad
        self.gradbox = take.gradbox
        self.nd_buffer = take.nd_buffer^
        self.parents = take.parents^
        self.on_gpu = take.on_gpu

    fn has_ancestry(self) -> Bool:
        return (
            self.parents is not None and len(self.parents.value()) > 0
        )

    fn shape(ref self) -> ref[self.nd_buffer.shape] Shape:
        return self.nd_buffer.shape

    fn buffer(ref self) -> ref[self.nd_buffer] NDBuffer[Self.dtype]:
        return self.nd_buffer

    fn gradients(self) -> UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]:
        return self.gradbox

    fn ancestry(
        ref self,
    ) -> ref[self.parents.value()] Ancestors[Self.dtype]:
        if self.parents == None:
            panic("Ancestor → ancestry: ancestors not initialized")
        return self.parents.value()

    fn is_on_gpu(self) -> Bool:
        return self.on_gpu

    @staticmethod
    fn from_tensor(ref tensor: Tensor[Self.dtype]) -> Ancestor[Self.dtype]:
        var out = Ancestor[Self.dtype]()
        out._id = tensor._id
        out.requires_grad = tensor.requires_grad
        out.on_gpu = tensor.is_on_gpu()
        out.nd_buffer = tensor.buffer.copy()
        if tensor.ancestors:
            out.parents = tensor.ancestors.copy()
        # Only bump and store if gradbox exists!
        if tensor.gradbox:
            _ = (
                tensor.gradbox[]
                ._refcount[]
                .fetch_add[ordering=Consistency.MONOTONIC](1)
            )
            out.gradbox = tensor.gradbox

        return out^

    fn __del__(deinit self):
        if not self.gradbox:
            return
        if (
            self.gradbox[]
            ._refcount[]
            .fetch_sub[ordering=Consistency.RELEASE](1)
            != 1
        ):
            return
        fence[ordering=Consistency.ACQUIRE]()
        self.gradbox.destroy_pointee()
        self.gradbox.free()


# ========================================
# Ancestors — new name for Ancestors
# holds List[Ancestor] instead of List[Tensor]
# ========================================


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var refs: List[Ancestor[Self.dtype]]
    var backwardFnArg: BackwardFnArg[Self.dtype]

    fn __init__(out self, var backwardFnArg: BackwardFnArg[Self.dtype]):
        self.refs = {}
        self.backwardFnArg = backwardFnArg^

    fn __copyinit__(out self, copy: Self):
        self.refs = copy.refs.copy()
        self.backwardFnArg = copy.backwardFnArg.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.refs = take.refs^
        self.backwardFnArg = take.backwardFnArg^

    fn backward_fn_arg(
        ref self,
    ) -> ref[self.backwardFnArg] BackwardFnArg[Self.dtype]:
        return self.backwardFnArg

    fn set_backward_fn_arg(
            mut self, var backwardFnArg: BackwardFnArg[Self.dtype]
    ):
        self.backwardFnArg = backwardFnArg^

    @always_inline
    fn append(mut self, ref parent: Tensor[Self.dtype]):
        self.refs.append(Ancestor[Self.dtype].from_tensor(parent))

    @always_inline
    fn __del__(deinit self):
        self.refs.clear()

    fn get(ref self, idx: Int) -> ref[self.refs] Ancestor[Self.dtype]:
        return self.refs[idx]

    fn tensor(ref self, idx: Int) -> Tensor[Self.dtype]:
        ref ancestor = self.get(idx)
        return Tensor[Self.dtype](ancestor.buffer(), requires_grad=ancestor.requires_grad)

    fn __len__(self) -> Int:
        return len(self.refs)

    fn __bool__(self) -> Bool:
        return len(self) > 0

    @no_inline
    fn print(self):
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i)._id, end=" ")
        print()

    fn __iter__(
        ref self,
    ) -> AncestorIterator[Self.dtype, origin_of(self)]:
        return AncestorIterator[Self.dtype](0, Pointer(to=self))


# ========================================
# AncestorIterator — yields Ancestor
# ========================================


struct AncestorIterator[dtype: DType, origin: ImmutOrigin](
    Sized & ImplicitlyCopyable
):
    var index: Int
    var src: Pointer[Ancestors[Self.dtype], Self.origin]

    fn __init__(
        out self,
        idx: Int,
        src: Pointer[Ancestors[Self.dtype], Self.origin],
    ):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Ancestor[Self.dtype]:
        self.index += 1
        return self.src[].get(self.index - 1)

    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


fn main() raises:
    print("Does pass indeed")
