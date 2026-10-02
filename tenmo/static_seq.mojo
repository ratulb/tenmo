"""Static variadic mixed-dtype chain.

Path A: `Seq[*Ms: LayerTrait]` stores a heterogeneous `Tuple[*Ms]` of
concrete layers and resolves EVERYTHING at compile time — including
boundary casts. Where MixedSequential inserts its boundary hop at append
time (runtime fn pointer), Seq instantiates a tiny recursive walker
(`_run_from`) per chain shape. Each hop passes the carrier through
`_static_seq_seam`: differing dtypes get the same real grad-tracked
`to_dtype` edge as MixedSequential; identical dtypes get a copy plus an
IDENTITY edge — plain `to_dtype` would return a bare leaf copy there and
sever the graph between hops (root-caused via all-zero weight grads).
Graphs stay byte-equivalent to Option-C manual segmentation, but every
cast/call is statically dispatched (zero erasure, zero indirection).

The design-sketch carrier loop (`x = x.to_dtype[..]()`) cannot compile —
a Mojo `var` keeps one static type while the carrier changes dtype per
hop — hence the `_run_from[cur, CurDT]` recursion instead: the carrier's
dtype travels as a comptime parameter, so each level binds an exactly
typed handle.

Head/tail annotations are enforced at COMPILE time (`comptime assert`
against the computed `InDT`/`OutDT` members — stronger than
MixedSequential's runtime panic). Callers spell them explicitly:
`model[head_dt, tail_dt](x)` — see `__call__`'s docstring for why the
"no annotation needed" promise does not survive contact with this pin's
constraint solver. Device transfer is intentionally omitted: Tuple
cannot be rebuilt element-wise without pack gymnastics — static chains
are cheap to re-construct per device.

Stage-5 note: per the decision this coexists with BOTH
`net.Sequential[dtype]` and `net.MixedSequential`; nothing here
replaces them.
"""

from .tensor import Tensor
from .net import LayerTrait
from .named_parameter import NamedParameter
from .backpropagation import BackwardFn
from .cast import ToDtypeBackward
from .ndbuffer import NDBuffer
from std.memory import Pointer


def _static_seq_seam[S: DType, T: DType](x: Tensor[S]) raises -> Tensor[T]:
    """Graph-preserving dtype seam between static chain hops.

    Replica of `Tensor.to_dtype` (tensor.mojo) WITHOUT its
    `comptime if NewType != Self.dtype` guard: same-dtype `to_dtype`
    returns an independent leaf COPY, which would silently disconnect
    the chain at every same-dtype hop. Here identical dtypes instead
    get a buffer-sharing copy plus an identity `ToDtypeBackward[S, S]`
    edge — gradient math unchanged (multiply-by-1), connectivity
    preserved. Differing dtypes produce exactly the node `to_dtype`
    would have built.

    Same-dtype path uses `Pointer.unsafe_bitcast` to reinterpret the
    `NDBuffer[S]` as `NDBuffer[T]` (sound because S == T guarantees
    identical layout), then deep-copies via `NDBuffer.__init__(*, copy:)`.
    This works on both CPU (Buffer refcount bump or memcpy) and GPU
    (DeviceState.copy = GPU-side refcount bump).
    """
    var trackable = S.is_floating_point() and T.is_floating_point()
    var grad_required = x.requires_grad and trackable

    var out: Tensor[T]
    comptime if S == T:
        # Identical dtype: copy the NDBuffer (refcount bump for shared,
        # memcpy for unshared) instead of paying to_dtype's unconditional
        # allocate + per-element scalar cast.  The compiler treats
        # NDBuffer[S] and NDBuffer[T] as distinct types even when S == T,
        # so we bitcast through a pointer (same layout guaranteed by the
        # comptime guard) then deep-copy via NDBuffer.__init__(*, copy:).
        #
        # GPU: same approach — DeviceState.copy() is a GPU-side
        # refcount bump (O(1)).
        var src_ptr = Pointer(to=x.buffer)
        var dst_ptr = src_ptr.unsafe_bitcast[NDBuffer[T]]()
        var ndb_copy = NDBuffer[T](copy=dst_ptr[])
        out = Tensor[T](ndb_copy^, requires_grad=grad_required)
    else:
        var new_type_buffer = x.buffer.to_dtype[T]()
        out = Tensor[T](new_type_buffer^, requires_grad=grad_required)

    if grad_required:
        out.requires_grad_(True)
        var backwardFn = BackwardFn.null_arg[T](ToDtypeBackward[S, T]())
        out.add_ancestry_erased[S](backwardFn^, x)
    return out^


struct Seq[*Ms: LayerTrait](Copyable):
    """Statically-typed heterogeneous layer chain with auto seam casts."""

    var steps: Tuple[*Self.Ms]

    # Chain contract: head InputDType in, tail OutputDType out — enforced by
    # the comptime asserts in __call__ below; seams cast between steps.
    comptime InDT = Self.Ms[0].InputDType
    comptime OutDT = Self.Ms[len(Self.Ms) - 1].OutputDType

    def __init__(out self, steps: Tuple[*Self.Ms]):
        self.steps = steps

    def __call__[In: DType, Out: DType](
        mut self, x: Tensor[In], sync: Bool = True
    ) raises -> Tensor[Out]:
        """Annotated forward — `model[head_dtype, tail_dtype](x)`.

        Both annotations are verified at COMPILE time against the chain.
        Why explicit: computed pack members (`Self.OutDT =
        Self.Ms[len-1].OutputDType`) read fine inside `comptime if` but
        never FOLD on this pin — a declared return type of
        `Tensor[Self.OutDT]` stays symbolic at call sites and breaks
        constraint solving (`all_close`, `backward`'s
        `is_floating_point` where-clause, arithmetic overloads). An
        `Out: DType = Self.OutDT` default folds no better. Concrete
        caller-supplied parameters are the only spelling that produces a
        fully instantiated signature; the asserts keep misannotations a
        compile error (stronger than MixedSequential's runtime panic).
        """
        comptime assert (
            In == Self.InDT
        ), "Seq.__call__: input dtype annotation must equal the chain head InputDType"
        comptime assert (
            Out == Self.OutDT
        ), "Seq.__call__: output dtype annotation must equal the chain tail OutputDType"
        return self._run_from[0, In, Out](x, sync=sync)

    def _run_from[cur: Int, CurDT: DType, Out: DType](
        mut self, x: Tensor[CurDT], sync: Bool
    ) raises -> Tensor[Out]:
        """Run hops `cur..end`; carrier dtype threaded as comptime param."""
        # Unconditional seam (see _static_seq_seam): the conditional form
        # is unrepresentable — passing `x` uncast in the same-dtype branch
        # needs a conversion that never folds, even under a proven
        # comptime-if condition.
        var h = _static_seq_seam[CurDT, Self.Ms[cur].InputDType](x)
        var y = self.steps[cur](h, sync=sync)
        comptime if cur == len(Self.Ms) - 1:
            # Same reasoning as the hop seam: to_dtype here would return a
            # bare LEAF COPY and backward() would stop at it immediately
            # (root cause of the all-zero-gradient run). Route through the
            # seam so the returned tensor carries an identity edge back
            # into the live graph.
            return _static_seq_seam[Self.Ms[cur].OutputDType, Out](y)
        else:
            return self._run_from[cur + 1, Self.Ms[cur].OutputDType, Out](
                y, sync=sync
            )

    def num_parameters(self) -> Int:
        var total = 0
        comptime for i in range(len(Self.Ms)):
            total += self.steps[i].num_parameters()
        return total

    def parameters_of[D: DType](
        ref self,
    ) -> List[Pointer[Tensor[D], MutAnyOrigin]]:
        """Per-dtype parameter pointers (multi-SGD story, static filter).

        Elements are re-typed via pointer bitcast: pack-computed
        `Ms[i].OutputDType` does not fold into a conversion check
        against the concrete `D`, but the surrounding comptime if
        guarantees the types are identical, so the bitcast is sound.
        """
        var result = List[Pointer[Tensor[D], MutAnyOrigin]]()
        comptime for i in range(len(Self.Ms)):
            comptime if Self.Ms[i].OutputDType == D:
                var src = self.steps[i].parameters()
                for j in range(len(src)):
                    result.append(src[j].unsafe_bitcast[Tensor[D]]())
        return result^

    def named_parameters_of[D: DType](
        ref self, prefix: String
    ) -> List[NamedParameter[D]]:
        """Dtype-filtered named parameters with per-layer index prefixes.

        Same bitcast re-typing as parameters_of (see note there).
        """
        var result = List[NamedParameter[D]]()
        comptime for i in range(len(Self.Ms)):
            comptime if Self.Ms[i].OutputDType == D:
                var layer_prefix = prefix + String(i) + "."
                var src = self.steps[i].named_parameters(layer_prefix)
                for j in range(len(src)):
                    var q = src[j]
                    result.append(
                        NamedParameter[D](
                            q.name,
                            q.tensor_ptr.unsafe_bitcast[Tensor[D]](),
                        )
                    )
        return result^

    def zero_grad(mut self):
        comptime for i in range(len(Self.Ms)):
            self.steps[i].zero_grad()

    def train(mut self):
        comptime for i in range(len(Self.Ms)):
            self.steps[i].train()

    def eval(mut self):
        comptime for i in range(len(Self.Ms)):
            self.steps[i].eval()

    def __len__(self) -> Int:
        return len(Self.Ms)
