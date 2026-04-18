from std.memory import Pointer
from tenmo import Tensor
from gradbox import Gradbox
from backpropagation import BackwardFnArg
from ndbuffer import NDBuffer
from common_utils import panic
from shapes import Shape
from std.os.atomic import Consistency, fence

# ========================================
# AncestorRef
# ========================================


struct AncestorRef[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Lightweight handle for the new ancestry system.
    Replaces full Tensor storage in ParentRefs.

    Carries everything needed for:
    1. Graph traversal        — id
    2. Grad routing           — requires_grad, gradbox ptr
    3. Data alive (CPU+GPU)   — nd_buffer (shared NDBuffer, refcount bump only)
    4. Backward compute       — nd_buffer (shape/strides/data for matmul etc.)
    5. Backward invoke        — backwardFnArg, parent_refs

    What is NOT here vs full Tensor copy:
    - No new gradbox allocation
    - No recursive Tensor ancestor chain copy
    - No BackwardFnArg heap copy at ancestry append time
    """

    var id: UInt
    var requires_grad: Bool
    var gradbox: UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]
    var nd_buffer: NDBuffer[Self.dtype]
    var parent_refs: Optional[ParentRefs[Self.dtype]]
    var backwardFnArg: Optional[BackwardFnArg[Self.dtype]]
    var on_gpu: Bool

    fn __init__(out self):
        self.id = 0
        self.requires_grad = False
        self.gradbox = UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]()
        self.nd_buffer = NDBuffer[Self.dtype]()
        self.parent_refs = None
        self.backwardFnArg = None
        self.on_gpu = False

    fn __copyinit__(out self, copy: Self):
        self.id = copy.id
        self.requires_grad = copy.requires_grad
        self.gradbox = copy.gradbox
        self.nd_buffer = copy.nd_buffer.copy()
        self.parent_refs = copy.parent_refs.copy()
        self.backwardFnArg = copy.backwardFnArg.copy()
        self.on_gpu = copy.on_gpu
        if self.gradbox:
            _ = (
                self.gradbox[]
                ._refcount[]
                .fetch_add[ordering=Consistency.MONOTONIC](1)
            )

    fn __moveinit__(out self, deinit take: Self):
        self.id = take.id
        self.requires_grad = take.requires_grad
        self.gradbox = take.gradbox
        self.nd_buffer = take.nd_buffer^
        self.parent_refs = take.parent_refs^
        self.backwardFnArg = take.backwardFnArg^
        self.on_gpu = take.on_gpu

    fn has_ancestry(self) -> Bool:
        return (
            self.parent_refs is not None and len(self.parent_refs.value()) > 0
        )

    fn has_backward_fn_arg(self) -> Bool:
        return self.backwardFnArg is not None

    fn shape(ref self) -> ref[self.nd_buffer.shape] Shape:
        return self.nd_buffer.shape

    fn backward_fn_arg(
        ref self,
    ) -> ref[self.backwardFnArg.value()] BackwardFnArg[Self.dtype]:
        return self.backwardFnArg.value()

    fn buffer(ref self) -> ref[self.nd_buffer] NDBuffer[Self.dtype]:
        return self.nd_buffer

    fn gradients(self) -> UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]:
        return self.gradbox

    fn ancestry(
        ref self,
    ) -> ref[self.parent_refs.value()] ParentRefs[Self.dtype]:
        if self.parent_refs == None:
            panic("AncestorRef → parents: parent_refs not initialized")
        return self.parent_refs.value()

    fn is_on_gpu(self) -> Bool:
        return self.on_gpu

    @staticmethod
    fn from_tensor(ref tensor: Tensor[Self.dtype]) -> AncestorRef[Self.dtype]:
        var out = AncestorRef[Self.dtype]()
        out.id = tensor._id
        out.requires_grad = tensor.requires_grad
        out.on_gpu = tensor.is_on_gpu()
        out.nd_buffer = tensor.buffer.copy()
        out.backwardFnArg = tensor.backwardFnArg.copy()
        if tensor.parent_refs:
            out.parent_refs = tensor.parent_refs.copy()
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
# ParentRefs — new name for Ancestors
# holds List[AncestorRef] instead of List[Tensor]
# ========================================


struct ParentRefs[dtype: DType](Sized & Copyable & Movable):
    var refs: List[AncestorRef[Self.dtype]]

    fn __init__(out self):
        self.refs = {}

    fn __copyinit__(out self, copy: Self):
        self.refs = copy.refs.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.refs = take.refs^

    @staticmethod
    fn empty() -> ParentRefs[Self.dtype]:
        return Self()

    @always_inline
    fn __del__(deinit self):
        self.refs.clear()

    fn get(ref self, idx: Int) -> ref[self.refs] AncestorRef[Self.dtype]:
        return self.refs[idx]

    fn __len__(self) -> Int:
        return len(self.refs)

    fn __bool__(self) -> Bool:
        return len(self) > 0

    @no_inline
    fn print(self):
        total = len(self)
        print("ParentRefs[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i).id, end=" ")
        print()

    fn __iter__(
        ref self,
    ) -> ParentRefsIterator[Self.dtype, origin_of(self)]:
        return ParentRefsIterator[Self.dtype](0, Pointer(to=self))


# ========================================
# ParentRefsIterator — yields AncestorRef
# ========================================


struct ParentRefsIterator[dtype: DType, origin: ImmutOrigin](
    Sized & ImplicitlyCopyable
):
    var index: Int
    var src: Pointer[ParentRefs[Self.dtype], Self.origin]

    fn __init__(
        out self,
        idx: Int,
        src: Pointer[ParentRefs[Self.dtype], Self.origin],
    ):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> AncestorRef[Self.dtype]:
        self.index += 1
        return self.src[].get(self.index - 1)

    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


fn main() raises:
    print("Does pass indeed")
