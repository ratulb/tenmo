from std.memory import Pointer
from .tensor import Tensor
from .gradbox import Gradbox
from .backpropagation import BackwardFnArg
from .ndbuffer import NDBuffer
from .common_utils import panic, i, s
from .shapes import Shape
from std.atomic import Ordering, fence


struct Ancestor[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Lightweight handle for ancestry tracking.

    Carries everything needed for:
    1. Graph traversal        — _id
    2. Grad routing           — requires_grad, gradbox ptr
    3. Data (CPU+GPU)   — ndb (shared NDBuffer, refcount bump only)
    4. Backward compute       — ndb (shape/strides/data for matmul etc.)
    5. Backward invoke        — backwardFnArg, parents

    What is NOT here vs full Tensor copy:
    - No new gradbox allocation
    - No recursive Tensor ancestor chain copy
    """

    var _id: UInt
    var requires_grad: Bool
    var gradbox: Optional[UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]]
    var ndb: NDBuffer[Self.dtype]
    var parents: Optional[Ancestors[Self.dtype]]

    def __init__(out self):
        self._id = 0
        self.requires_grad = False
        self.gradbox = {}
        self.ndb = NDBuffer[Self.dtype]()
        self.parents = None

    def __init__(out self, *, copy: Self):
        self._id = copy._id
        self.requires_grad = copy.requires_grad
        self.gradbox = copy.gradbox
        self.ndb = copy.ndb.copy()
        self.parents = copy.parents.copy()
        if self.gradbox:
            _ = (
                self.gradbox.unsafe_value()[]
                ._refcount[]
                .fetch_add[ordering=Ordering.RELAXED](1)
            )

    def __init__(out self, *, deinit take: Self):
        self._id = take._id
        self.requires_grad = take.requires_grad
        self.gradbox = take.gradbox
        self.ndb = take.ndb^
        self.parents = take.parents^

    def has_ancestry(ref self) -> Bool:
        return self.parents is not None and len(self.parents.value()) > 0

    def shape(ref self) -> ref[self.ndb.shape] Shape:
        return self.ndb.shape

    def buffer(ref self) -> NDBuffer[Self.dtype]:
        return self.ndb.copy()

    def gradients(ref self) -> UnsafePointer[Gradbox[Self.dtype], MutAnyOrigin]:
        return self.gradbox.unsafe_value()

    def ancestry(
        ref self,
    ) -> ref[self.parents.value()] Ancestors[Self.dtype]:
        if self.parents == None:
            panic("Ancestor → ancestry: ancestors not initialized")
        return self.parents.value()

    def is_on_gpu(self) -> Bool:
        return self.ndb.is_on_gpu()

    @staticmethod
    def from_tensor(ref tensor: Tensor[Self.dtype]) -> Ancestor[Self.dtype]:
        var out = Ancestor[Self.dtype]()
        out._id = tensor._id
        out.requires_grad = tensor.requires_grad
        out.ndb = tensor.buffer.copy()
        if tensor.ancestors:
            out.parents = tensor.ancestors.copy()
        # Only bump and store if gradbox exists!
        if tensor.gradbox:
            _ = (
                tensor.gradbox.unsafe_value()[]
                ._refcount[]
                .fetch_add[ordering=Ordering.RELAXED](1)
            )
            out.gradbox = tensor.gradbox

        return out^

    def update_grad(
        ref self,
        ref incoming: Gradbox[Self.dtype],
        op_code: Int,
        extra_arg: Optional[BackwardFnArg[Self.dtype]] = None,
    ):
        """Apply incoming gradient to this ancestor's gradbox.

        Args:
            incoming:  Upstream gradient from backward handler.
            op_code:   How to accumulate — AddTensor, SubtractTensor,
                       ZeroGrad, ScatterAddTensor.
            extra_arg: Type-erased argument for ops that need additional
                       context beyond the gradient itself. Currently used
                       by ScatterAddTensor to carry GatherArg (indices + axis).
                       Future ops can store any ArgumentType here.
        """
        if not self.requires_grad or not self.gradbox:
            return

        if op_code == AddTensor:
            self.gradbox.unsafe_value()[] += incoming

        elif op_code == SubtractTensor:
            self.gradbox.unsafe_value()[] -= incoming

        elif op_code == ZeroGrad:
            self.gradbox.unsafe_value()[].zero_grad()

        elif op_code == ScatterAddTensor:
            ref arg = extra_arg.value().get[GatherArg]()
            Filler[Self.dtype].scatter_add(
                self.gradbox.unsafe_value()[].buffer,
                incoming.buffer,
                arg.indices,
                arg.axis,
            )
            # Zero padding row grad if set
            if arg.padding_idx:
                self.gradbox.unsafe_value()[].fill(
                    0.0, i(arg.padding_idx.value()), s()
                )
        else:
            print("Ancestor → update_grad: unknown op_code", String(op_code))

    def __del__(deinit self):
        if self.gradbox:
            if (
                self.gradbox.unsafe_value()[]
                ._refcount[]
                .fetch_sub[ordering=Ordering.RELEASE](1)
                != 1
            ):
                return
            fence[ordering=Ordering.ACQUIRE]()
            self.gradbox.unsafe_value().destroy_pointee()
            self.gradbox.unsafe_value().free()


# ========================================
# Ancestors — holds List[Ancestor] + BackwardFnArg
# ========================================


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    var origins: List[Ancestor[Self.dtype]]
    var backwardFnArg: BackwardFnArg[Self.dtype]

    def __init__(out self, var backwardFnArg: BackwardFnArg[Self.dtype]):
        self.origins = {}
        self.backwardFnArg = backwardFnArg^

    def __init__(out self, *, copy: Self):
        self.origins = copy.origins.copy()
        self.backwardFnArg = copy.backwardFnArg.copy()

    def __init__(out self, *, deinit take: Self):
        self.origins = take.origins^
        self.backwardFnArg = take.backwardFnArg^

    def backward_fn_arg(
        ref self,
    ) -> ref[self.backwardFnArg] BackwardFnArg[Self.dtype]:
        return self.backwardFnArg

    def set_backward_fn_arg(
        mut self, var backwardFnArg: BackwardFnArg[Self.dtype]
    ):
        self.backwardFnArg = backwardFnArg^

    @always_inline
    def append(mut self, ref parent: Tensor[Self.dtype]):
        self.origins.append(Ancestor[Self.dtype].from_tensor(parent))

    @always_inline
    def __del__(deinit self):
        self.origins.clear()

    def get(ref self, idx: Int) -> ref[self.origins] Ancestor[Self.dtype]:
        return self.origins[idx]

    def tensor(ref self, idx: Int) -> Tensor[Self.dtype]:
        ref ancestor = self.get(idx)
        return Tensor[Self.dtype](
            ancestor.buffer(), requires_grad=ancestor.requires_grad
        )

    def __len__(self) -> Int:
        return len(self.origins)

    def __bool__(self) -> Bool:
        return len(self) > 0

    @no_inline
    def print(self):
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i)._id, end=" ")
        print()

    def __iter__(
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

    def __init__(
        out self,
        idx: Int,
        src: Pointer[Ancestors[Self.dtype], Self.origin],
    ):
        self.src = src
        self.index = idx

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) -> Ancestor[Self.dtype]:
        self.index += 1
        return self.src[].get(self.index - 1).copy()

    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        return len(self.src[]) - self.index
