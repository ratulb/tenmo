from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic, Ordering, fence
from std.sys import size_of
from .gradbox import Gradbox
from .backpropagation import (
    BackwardFn,
    CopyFn,
    DestroyerFn,
)

from .gather import GatherArg
from .ndbuffer import NDBuffer, NDBufferLite
from .shared.indexhelper import i, s
from .shared.panic import panic
from .shared.shapes import Shape
from .shared.strides import Strides
from .filler import Filler
from .shared.mnemonics import (
    AddTensor,
    SubtractTensor,
    ZeroGrad,
    ScatterAddTensor,
)


struct Ancestor[dtype: DType](ImplicitlyCopyable):
    var _id: UInt
    var requires_grad: Bool
    var gradbox: Optional[Gradbox[Self.dtype]]
    var ndb: NDBufferLite[Self.dtype]
    var parents: Optional[Ancestors[Self.dtype]]

    def __init__(out self):
        self._id = 0
        self.requires_grad = False
        self.gradbox = {}
        self.ndb = NDBufferLite[Self.dtype]()
        self.parents = None

    def __init__(out self, *, copy: Self):
        self._id = copy._id
        self.requires_grad = copy.requires_grad
        self.gradbox = copy.gradbox
        self.ndb = copy.ndb
        self.parents = copy.parents

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
        self,
    ) -> ref[self.parents.value()] Ancestors[Self.dtype]:
        if self.parents == None:
            panic("Ancestor → ancestry: ancestors not initialized")
        return self.parents.value()

    def strides(ref self) -> ref[self.ndb.value().strides] Strides:
        return self.ndb.value().strides

    def offset(self) -> Int:
        return self.ndb.value().offset

    def max_storage_index(self) -> Int:
        ref ndb = self.ndb.value()
        return ndb.max_storage_index()

    def is_on_gpu(self) -> Bool:
        if not self.ndb.is_empty():
            return self.ndb.value().is_on_gpu()
        return False

    def update_grad(
        mut self,
        ref incoming: Gradbox[Self.dtype],
        op_code: Int,
        extra_arg: Optional[BackwardFn] = None,
    ):
        if not self.requires_grad or not self.gradbox:
            return

        # Handler-boundary invariant: every gradient payload handed to a
        # parent must be contiguous with zero offset. Debug builds fail
        # loud here; release builds erase this check entirely.
        debug_assert(
            incoming.buffer().is_contiguous()
            and incoming.buffer().offset == 0,
            "Ancestor.update_grad: incoming gradbox must be contiguous"
            " with zero offset",
        )

        comptime if Self.dtype.is_numeric():
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
                    gradbox.fill(0, i(arg.padding_idx.value()), s())
            else:
                print(
                    "Ancestor → update_grad: unknown op_code", String(op_code)
                )


def make_node_copier[dtype: DType]() -> CopyFn:
    def copy_node(
        src: Pointer[UInt8, MutAnyOrigin],
    ) -> Pointer[UInt8, MutAnyOrigin]:
        var dst = unsafe_alloc[Ancestor[dtype]](1)
        # Loading through the pointer invokes Ancestor.__init__(copy:) —
        # gradbox/parents storage is *shared* (Buffer/Ancestors refcount
        # bumps; the Gradbox's shape/strides are deep-copied) + an ndb copy
        # snapshot (data shared via Buffer bump, shape/strides deep-copied).
        dst.unsafe_write(src.unsafe_bitcast[Ancestor[dtype]]()[])
        return dst.unsafe_bitcast[UInt8]().as_unsafe_any_origin()

    return copy_node


def make_node_destroyer[dtype: DType]() -> DestroyerFn:
    def destroy_node(p: Pointer[UInt8, MutAnyOrigin]) -> None:
        p.unsafe_bitcast[Ancestor[dtype]]().unsafe_deinit_pointee()
        p.unsafe_bitcast[Ancestor[dtype]]().unsafe_free()

    return destroy_node


@fieldwise_init
struct ParentNode(ImplicitlyCopyable):
    """DType-erased storage slot for one graph node.

    The ancestry list inside `Ancestry` stores nodes as dtype-erased blobs so
    one backward traversal can mix dtypes (e.g. an f32 leaf reached through an
    f16 cast op). Each slot holds the byte pointer to a true `Ancestor[dtype]`
    plus the copy/destroy function pointers instantiated at wrap time — the
    same pattern as `BackwardFn`. Field access only happens after `as[dtype]()`
    restores the true type, guarded by a debug_assert on the stored tag.
    """
    var ptr: Pointer[UInt8, MutUntrackedOrigin]
    var tag: DType
    var destroy_fn: DestroyerFn
    var copy_fn: CopyFn

    def __deinit__(deinit self):
        self.destroy_fn(self.ptr.as_unsafe_any_origin())

    def __init__(out self, *, deinit move: Self):
        self.ptr = move.ptr
        self.tag = move.tag
        self.destroy_fn = move.destroy_fn
        self.copy_fn = move.copy_fn

    def __init__(out self, *, copy: Self):
        self.tag = copy.tag
        self.destroy_fn = copy.destroy_fn
        self.copy_fn = copy.copy_fn
        self.ptr = self.copy_fn(
            copy.ptr.as_unsafe_any_origin()
        ).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()  # fresh node via Ancestor.__init__(copy:) — storage shared, data not

    @always_inline
    def as[dtype: DType](ref self) -> Ancestor[dtype]:
        """Restore the true typed node (fresh copy; tag-checked in debug)."""
        debug_assert(
            self.tag == dtype,
            (
                "ParentNode.as: stored node dtype tag does not match requested"
                " type"
            ),
        )
        return self.ptr.unsafe_bitcast[Ancestor[dtype]]()[].copy()


@always_inline
def to_parent_node[dtype: DType](var ancestor: Ancestor[dtype]) -> ParentNode:
    """Wrap a typed node into an erased storage slot (takes ownership)."""
    var p = unsafe_alloc[Ancestor[dtype]](1)
    p.unsafe_write(ancestor^)
    return ParentNode(
        p.unsafe_bitcast[UInt8](),
        dtype,
        make_node_destroyer[dtype](),
        make_node_copier[dtype](),
    )


struct Ancestry(Sized & Movable):
    """Heap-only record behind the Ancestors handle.

    Deliberately NON-generic: both fields are dtype-erased — the node list is
    `List[ParentNode]` (each slot wraps a true `Ancestor[dtype]`, restored at
    the owner dtype via Ancestors.get/ref_get) + a type-erased `BackwardFn`.
    The `dtype` parameter used to be carried here (before node erasure
    made the node list erased) but is now vestigial — only `Ancestors[dtype]`
    needs a dtype, for its typed accessors. Never used as a value type outside
    Ancestors — always constructed directly into the combined
    [Atomic refcount | Ancestry] heap allocation (formerly the public
    Ancestors struct; merged when node storage was erased).
    """

    var origins: List[ParentNode]
    var backwardFn: BackwardFn

    def __init__(out self, var backwardFn: BackwardFn):
        self.origins = {}
        self.backwardFn = backwardFn^

    def __init__(out self, *, deinit move: Self):
        self.origins = move.origins^
        self.backwardFn = move.backwardFn^

    def __len__(self) -> Int:
        # Kept: satisfies the Sized bound required by size_of() at
        # allocation time. Not used by callers — they go through Ancestors.
        return len(self.origins)


struct Ancestors[dtype: DType](Sized & ImplicitlyCopyable):
    """Refcounted heap handle to an autograd node's parent list + backward function.

    Single combined heap allocation: [Atomic | Ancestry]. Lets every op
    snapshot its parents' ancestry in O(1) (shared DAG) instead of deep-copying
    the whole subtree — kills the O(n²) forward/backward graph copies.
    Copies bump a single atomic refcount; the
    pointee is destroyed when the last handle drops. Unlike Gradbox — which
    shed its own combined [Atomic | NDBuffer] handle in the storage
    cleanup and now inherits refcounting from its NDBuffer's Buffer — Ancestors
    must keep the combined allocation: its payload (List[ParentNode] +
    BackwardFn) is not itself refcounted, so inline fields could not share
    storage across copies. Formerly split across the Ancestors payload +
    SharedAncestors handle; the payload's methods are now forwarded here
    directly.
    """

    var _ptr: Optional[Pointer[Ancestry, MutUntrackedOrigin]]
    var _refcount: Optional[Pointer[Atomic[UInt64], MutUntrackedOrigin]]

    def __init__(out self, var backwardFn: BackwardFn):
        var ref_size = size_of[Atomic[UInt64]]()
        var anc_size = size_of[Ancestry]()
        var alloc_base = unsafe_alloc[UInt8](ref_size + anc_size)
        var ref_ptr = alloc_base.unsafe_bitcast[Atomic[UInt64]]()
        ref_ptr[] = Atomic[UInt64](1)
        var anc_ptr = alloc_base.unsafe_offset(ref_size).unsafe_bitcast[
            Ancestry
        ]()
        anc_ptr.unsafe_write(Ancestry(backwardFn^))
        self._ptr = anc_ptr
        self._refcount = ref_ptr

    def __init__(out self, *, deinit move: Self):
        self._ptr = move._ptr
        self._refcount = move._refcount
        _ = move._ptr = {}
        _ = move._refcount = {}

    def __init__(out self, *, copy: Self):
        self._ptr = copy._ptr
        self._refcount = copy._refcount
        if copy._refcount:
            _ = copy._refcount.unsafe_value()[].fetch_add[
                ordering=Ordering.RELAXED
            ](1)

    def __deinit__(deinit self):
        if self._refcount == None or self._ptr == None:
            return
        if (
            self._refcount.unsafe_value()[].fetch_sub[
                ordering=Ordering.RELEASE
            ](1)
            != 1
        ):
            return
        fence[ordering=Ordering.ACQUIRE]()
        self._ptr.unsafe_value().unsafe_deinit_pointee()
        var alloc_start = self._refcount.unsafe_value().unsafe_bitcast[UInt8]()
        alloc_start.unsafe_free()
        _ = self._refcount = {}
        _ = self._ptr = {}

    @always_inline
    def value(ref self) -> ref[self] Ancestry:
        """Reference to the heap record inside the combined [Atomic | Ancestry] allocation.
        (Gradbox.buffer(), by contrast, dereferences nothing — it returns the inline NDBuffer field).
        """
        return self._ptr.unsafe_value()[]

    # Accessors moved up from the former Ancestry payload

    def backward_fn(
        ref self,
    ) -> ref[self.value().backwardFn] BackwardFn:
        return self.value().backwardFn

    def set_backward_fn(mut self, var backwardFn: BackwardFn):
        self.value().backwardFn = backwardFn^

    @always_inline
    def append(mut self, var ancestor: Ancestor[Self.dtype]):
        self.value().origins.append(to_parent_node[Self.dtype](ancestor^))

    @always_inline
    def append_erased[pt: DType](mut self, var ancestor: Ancestor[pt]):
        """Append a node whose dtype differs from this handle's owner.

        Storage is dtype-erased (ParentNode), so the only thing the owner
        typing constrained was the wrapper signature. Used by cross-dtype
        edges — currently only ToDtypeBackward's registration.
        """
        self.value().origins.append(to_parent_node[pt](ancestor^))

    @always_inline
    def get(ref self, idx: Int) -> Ancestor[Self.dtype]:
        return self.value().origins[idx].as[Self.dtype]()

    def get_erased(
        ref self,
        idx: Int,
    ) -> ref[self.value().origins.unsafe_get(idx)] ParentNode:
        """Escape hatch: fetch a stored slot WITHOUT owner-dtype unwrapping.

        get()/ref_get() restore nodes at the OWNER's dtype, which fails the
        tag assert for cross-dtype edges (ToDtypeBackward: an f32 parent
        stored under an f16 output). Callers restore the true type themselves
        via `.as[T]()` on the returned reference.
        """
        return self.value().origins[idx]

    def __len__(self) -> Int:
        return len(self.value().origins)

    def __bool__(self) -> Bool:
        return self._ptr != None

    @no_inline
    def print(self):
        var total = len(self)
        print("Ancestors[", total, "] = ", end="")
        for i in range(total):
            print(self.get(i)._id, end=" ")
        print()

    def __iter__(
        ref self,
    ) -> AncestorIterator[Self.dtype, origin_of(self)]:
        return AncestorIterator[Self.dtype](0, Pointer(to=self))


struct AncestorIterator[dtype: DType, origin: ImmOrigin](
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
        return self.src[].get(self.index - 1)

    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        return len(self.src[]) - self.index
