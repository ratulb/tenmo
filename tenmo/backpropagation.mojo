"""Backpropagation — Autograd Dispatch and Backward Operations.

This module provides the core infrastructure for Tenmo's autograd system:

1. **Type-erased handler** — `BackwardFnHandle` + `make_backward_fn_handle`
   lift a comptime-known backward handler into a raw-pointer call pointer.
   `BackwardFn` stores that erased pointer (plus a type-erased argument
   payload) so the ancestor graph never references a graph type.
2. **Backward dispatcher** — `Backward.invoke` calls the erased handler
   pointer stored on each node directly (no jump table).


Related:
  - [README_AUTOGRAD.md](https://github.com/ratulb/tenmo/blob/main/README_AUTOGRAD.md) — Full autograd architecture
  - [ancestry.mojo](https://github.com/ratulb/tenmo/blob/main/tenmo/ancestry.mojo) — Ancestor and Ancestors types
  - [gradbox.mojo](https://github.com/ratulb/tenmo/blob/main/tenmo/gradbox.mojo) — Gradient storage with refcounting
"""

from std.memory.alloc import unsafe_alloc
from .ancestry import Ancestor
from .shared.buffers import Buffer
from .ndbuffer import NDBuffer
from .shared.intarray import IntArray


trait ArgumentType(ImplicitlyCopyable & Deinitable):
    pass


trait BackwardFnType(ImplicitlyCopyable & Deinitable):
    """Conformance trait for backward handlers.

    Each backward handler struct implements this trait and pins its own
    ``dtype`` via ``comptime datatype = Self.dtype``. The stored call pointer
    inside ``BackwardFn`` is fully type-erased (raw ``Pointer``
    arguments) so the trait's signature never leaks an ``Ancestor[dtype]``
    reference into the ancestor graph — breaking the type-level recursion that
    stalled GPU codegen.
    """

    comptime datatype: DType

    @staticmethod
    def backward(
        var output: Ancestor[Self.datatype],
        mut parent_ids: List[UInt],
    ):
        pass


@fieldwise_init
struct NoOpBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype


comptime DestroyerFn = def(
    Pointer[UInt8, MutAnyOrigin]
) thin -> None


def make_destroyer[T: ArgumentType]() -> DestroyerFn:
    def destroy(p: Pointer[UInt8, MutAnyOrigin]) -> None:
        p.unsafe_bitcast[T]().unsafe_deinit_pointee()
        p.unsafe_bitcast[T]().unsafe_free()

    return destroy


comptime CopyFn = def(
    Pointer[UInt8, MutAnyOrigin]
) thin -> Pointer[UInt8, MutAnyOrigin]


def make_copier[T: ArgumentType]() -> CopyFn:
    def copy_it(
        src: Pointer[UInt8, MutAnyOrigin]
    ) -> Pointer[UInt8, MutAnyOrigin]:
        var dst = unsafe_alloc[T](1)
        dst.unsafe_write(src.unsafe_bitcast[T]()[])
        return dst.unsafe_bitcast[UInt8]().as_unsafe_any_origin()

    return copy_it


# BackwardFnHandle — Type-Erased Backward Handler Pointer
# A `def(...) thin` pointer to one backward handler, specialized per dtype but
# exposed behind fully erased raw pointers. Stored in BackwardFn at the
# forward site (where the handler is comptime known), so Backward.invoke calls
# ONLY the used handler instead of elaborating a jump table for every program.
#
# The signature takes raw pointers instead of `Ancestor[dtype]` / `List[UInt]`
# so the stored fn type never references `Ancestor` — the type-level recursion
# that broke GPU codegen when BackwardFn lived inside the ancestor graph.


comptime BackwardFnHandle = def(
    Pointer[UInt8, MutAnyOrigin],
    Pointer[UInt8, MutAnyOrigin],
) thin


def make_backward_fn_handle[T: BackwardFnType]() -> BackwardFnHandle:
    """Lift a comptime-known backward handler into a type-erased call pointer.

    The closure body reconstructs the typed `Ancestor[T.datatype]` and
    `List[UInt]` arguments, so handlers stay fully typed.
    """

    def handle(
        output_ptr: Pointer[UInt8, MutAnyOrigin],
        parent_ids_ptr: Pointer[UInt8, MutAnyOrigin],
    ) -> None:
        T.backward(
            output_ptr.unsafe_bitcast[Ancestor[T.datatype]]()[],
            parent_ids_ptr.unsafe_bitcast[List[UInt]]()[],
        )

    return handle


@fieldwise_init
struct BackwardFn(ImplicitlyCopyable):
    """BackwardFn — Type-Erased Container
    Type-erased container for backward operation arguments.
    Stores type-erased ptr, destroy, and copy_fn.
    """
    # NOTE: deliberately non-generic. The dtype parameter carried no storage
    # meaning — all fields (ptr/destroy/copy_fn/backward_fn) were already fully
    # type-erased — so the parameter only forced the ancestry graph to stay
    # output-dtype-homogeneous. Payload-generic factories take an explicit
    # leading `dtype: DType` comptime param instead.
    var ptr: Pointer[UInt8, MutUntrackedOrigin]  # type-erased arg
    var destroy: DestroyerFn
    var copy_fn: CopyFn
    var needs_parent_data: Bool
    var backward_fn: BackwardFnHandle

    def __init__[T: ArgumentType, h_t: BackwardFnType](
        out self, var arg: T, bw: h_t,
    ):
        var p = unsafe_alloc[T](1)
        p.unsafe_write(arg^)
        self.ptr = p.unsafe_bitcast[UInt8]()
        self.destroy = make_destroyer[T]()
        self.copy_fn = make_copier[T]()
        self.needs_parent_data = False
        self.backward_fn = make_backward_fn_handle[h_t]()

    def __deinit__(deinit self):
        self.destroy(self.ptr.as_unsafe_any_origin())  # calls T.__del__

    def __init__(out self, *, deinit move: Self):
        self.ptr = move.ptr
        self.destroy = move.destroy
        self.copy_fn = move.copy_fn
        self.needs_parent_data = move.needs_parent_data
        self.backward_fn = move.backward_fn

    def __init__(out self, *, copy: Self):
        self.destroy = copy.destroy
        self.copy_fn = copy.copy_fn
        self.ptr = self.copy_fn(
            copy.ptr.as_unsafe_any_origin()
        ).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()  # deep copy via T.__init__
        self.needs_parent_data = copy.needs_parent_data
        self.backward_fn = copy.backward_fn

    def get[T: ArgumentType](ref self) -> ref[self.ptr] T:
        return self.ptr.unsafe_bitcast[T]()[]

    @staticmethod
    def null_arg[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](bw: h_t = NoOpBackward[dtype](),) -> BackwardFn:
        return BackwardFn(NullArg(0), bw)

    @staticmethod
    def boolean_arg[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](
        is_true: Bool,
        bw: h_t = NoOpBackward[dtype](),
    ) -> BackwardFn:
        return BackwardFn(Boolean(is_true), bw)

    @staticmethod
    def scalar_arg[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](
        value: Scalar[dtype],
        bw: h_t = NoOpBackward[dtype](),
    ) -> BackwardFn:
        return BackwardFn(ScalarArg[dtype](value), bw)

    @staticmethod
    def integer_arg[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](
        value: Int,
        bw: h_t = NoOpBackward[dtype](),
    ) -> BackwardFn:
        return BackwardFn(Integer(value), bw)

    @staticmethod
    def from_intarray[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](
        array: IntArray,
        bw: h_t = NoOpBackward[dtype](),
    ) -> BackwardFn:
        return BackwardFn(IntArrayArg(array), bw)

    @staticmethod
    def from_buffer[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](
        buffer: Buffer[dtype],
        bw: h_t = NoOpBackward[dtype](),
    ) -> BackwardFn:
        return BackwardFn(BufferArg[dtype](buffer), bw)

    @staticmethod
    def from_ndbuffer[
        dtype: DType, h_t: BackwardFnType = NoOpBackward[dtype]
    ](
        ndb: NDBuffer[dtype],
        bw: h_t = NoOpBackward[dtype](),
    ) -> BackwardFn:
        return BackwardFn(NDBufferArg[dtype](ndb), bw)


@fieldwise_init
struct NullArg(ArgumentType):
    """Argument Payload Types
    NullArg: Empty payload (ops with no extra arguments)
    Boolean: Bool (used by DROPOUT-style ops)
    ScalarArg: Scalar value (used by *_SCALAR ops)
    Integer: Int value (used by axis/shape-parameterized ops)
    IntArrayArg: IntArray (used by transpose/unsqueeze axes)
    BufferArg: Buffer[dtype] (used by ops needing forward buffer data)
    NDBufferArg: NDBuffer[dtype] (used by ops needing forward output values:
                Sigmoid, Tanh, Exp)
    """
    var zero: UInt8


@fieldwise_init
struct Boolean(ArgumentType):
    var is_true: Bool


@fieldwise_init
struct ScalarArg[dtype: DType](ArgumentType):
    var value: Scalar[Self.dtype]


@fieldwise_init
struct Integer(ArgumentType):
    var value: Int


@fieldwise_init
struct IntArrayArg(ArgumentType):
    var array: IntArray


@fieldwise_init
struct BufferArg[dtype: DType](ArgumentType):
    var buffer: Buffer[Self.dtype]


@fieldwise_init
struct NDBufferArg[dtype: DType](ArgumentType):
    var ndb: NDBuffer[Self.dtype]


@fieldwise_init
struct Backward[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """Backward — Type-Erased Backward Dispatcher
    Each node's BackwardFn stores the single erased handler pointer used at
    forward time. Backward.invoke calls that pointer directly — no op_code, no
    dispatch table.
    """
    @staticmethod
    def invoke(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ) raises:
        if not output.has_ancestry():
            print("Inside Backward invoke: output ancestry is not set")
            return
        # Handler-boundary invariant: every backward handler receives a
        # contiguous, zero-offset gradbox. Debug builds fail loud here;
        # release builds erase this check entirely.
        ref incoming_gb = output.gradients()
        debug_assert(
            incoming_gb.buffer().is_contiguous()
            and incoming_gb.buffer().offset == 0,
            "Backward.invoke: incoming gradbox must be contiguous with"
            " zero offset",
        )
        ref arg = output.ancestry().backward_fn()
        var output_ptr = (
            Pointer(to=output)
            .unsafe_origin_cast[MutAnyOrigin]()
            .unsafe_bitcast[UInt8]()
        )
        var parent_ids_ptr = (
            Pointer(to=parent_ids)
            .unsafe_origin_cast[MutAnyOrigin]()
            .unsafe_bitcast[UInt8]()
        )
        arg.backward_fn(output_ptr, parent_ids_ptr)
