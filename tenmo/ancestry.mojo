from std.memory import Pointer
from .tensor import Tensor
from .gradbox import Gradbox
from .backpropagation import BackwardFnArg
from .ndbuffer import NDBuffer
from .common_utils import panic, i, s
from .shapes import Shape


struct Ancestor[dtype: DType](ImplicitlyCopyable & Movable):
    var _id: UInt
    var requires_grad: Bool
    var gradbox: Optional[Gradbox[Self.dtype]]
    var ndb: Optional[NDBuffer[Self.dtype]]
    var parents: Optional[Ancestors[Self.dtype]]

    def __init__(out self):
        self._id = 0
        self.requires_grad = False
        self.gradbox = {}
        self.ndb = {}
        self.parents = None

    def __init__(out self, *, copy: Self):
        self._id = copy._id
        self.requires_grad = copy.requires_grad
        self.gradbox = copy.gradbox
        self.ndb = copy.ndb.copy()
        self.parents = copy.parents.copy()

    def __init__(out self, *, deinit move: Self):
        self._id = move._id
        self.requires_grad = move.requires_grad
        self.gradbox = move.gradbox
        self.ndb = move.ndb^
        self.parents = move.parents^

    def has_ancestry(ref self) -> Bool:
        return self.parents is not None and len(self.parents.value()) > 0

    def shape(ref self) -> ref[self.ndb.value().shape] Shape:
        return self.ndb.value().shape

    def buffer(ref self) -> ref[self.ndb.value()] NDBuffer[Self.dtype]:
        return self.ndb.value()

    def gradients(ref self) -> ref[self.gradbox.value()] Gradbox[Self.dtype]:
        return self.gradbox.value()

    def ancestry(
        mut self,
    ) -> ref[self.parents.value()] Ancestors[Self.dtype]:
        if self.parents == None:
            panic("Ancestor → ancestry: ancestors not initialized")
        return self.parents.value()

    def strides(ref self) -> ref[self.ndb.value().strides] Strides:
        return self.ndb.value().strides

    def offset(self) -> Int:
        return self.ndb.value().offset

    def max_index(self) -> Int:
        ref ndb = self.ndb.value()
        return ndb.max_index()

    def is_on_gpu(self) -> Bool:
        if self.ndb:
            return self.ndb.value().is_on_gpu()
        return False

    def update_grad(
        mut self,
        ref incoming: Gradbox[Self.dtype],
        op_code: Int,
        extra_arg: Optional[BackwardFnArg[Self.dtype]] = None,
    ):
        if not self.requires_grad or not self.gradbox:
            return

        ref gradbox = self.gradbox.value()

        if op_code == AddTensor:
            gradbox += incoming

        elif op_code == SubtractTensor:
            gradbox -= incoming

        elif op_code == ZeroGrad:
            gradbox.zero_grad()

        elif op_code == ScatterAddTensor:
            ref arg = extra_arg.value().get[GatherArg]()
            Filler[Self.dtype].scatter_add(
                gradbox.buffer(),
                incoming.buffer(),
                arg.indices,
                arg.axis,
            )
            if arg.padding_idx:
                gradbox.fill(0.0, i(arg.padding_idx.value()), s())
        else:
            print("Ancestor → update_grad: unknown op_code", String(op_code))

    def __del__(deinit self):
        pass


struct Ancestors[dtype: DType](Sized & Copyable & Movable):
    """Ancestors — holds List[Ancestor] + BackwardFnArg."""

    var origins: List[Ancestor[Self.dtype]]
    var backwardFnArg: BackwardFnArg[Self.dtype]

    def __init__(out self, var backwardFnArg: BackwardFnArg[Self.dtype]):
        self.origins = {}
        self.backwardFnArg = backwardFnArg^

    def __init__(out self, *, copy: Self):
        self.origins = copy.origins.copy()
        self.backwardFnArg = copy.backwardFnArg.copy()

    def __init__(out self, *, deinit move: Self):
        self.origins = move.origins^
        self.backwardFnArg = move.backwardFnArg^

    def backward_fn_arg(
        ref self,
    ) -> ref[self.backwardFnArg] BackwardFnArg[Self.dtype]:
        return self.backwardFnArg

    def set_backward_fn_arg(
        mut self, var backwardFnArg: BackwardFnArg[Self.dtype]
    ):
        self.backwardFnArg = backwardFnArg^

    @always_inline
    def append(mut self, var ancestor: Ancestor[Self.dtype]):
        self.origins.append(ancestor^)

    @always_inline
    def get(mut self, idx: Int) -> ref[self.origins] Ancestor[Self.dtype]:
        return self.origins[idx]

    @always_inline
    def ref_get(ref self, idx: Int) -> ref[self.origins] Ancestor[Self.dtype]:
        return self.origins[idx]

    def __len__(self) -> Int:
        return len(self.origins)

    def __bool__(self) -> Bool:
        return len(self) > 0

    @no_inline
    def print(self):
        total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.ref_get(i)._id, end=" ")
        print()

    def __iter__(
        ref self,
    ) -> AncestorIterator[Self.dtype, origin_of(self)]:
        return AncestorIterator[Self.dtype](0, Pointer(to=self))


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
        return self.src[].ref_get(self.index - 1).copy()

    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        return len(self.src[]) - self.index
