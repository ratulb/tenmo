from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .validators import Validator
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from std.random import shuffle, seed
from .gradbox import Gradbox
from std.sys import has_accelerator
from .ndbuffer import NDBuffer
from .shared.panic import panic
from .ancestry import Ancestor
from .kernels.shuffle_kernel import ShuffleKernel


@fieldwise_init
struct ShuffleArg(ArgumentType):
    var axis: Int
    var permutation: List[Int]

    def __init__(out self, *, copy: Self):
        self.axis = copy.axis
        self.permutation = copy.permutation.copy()


@fieldwise_init
struct ShuffleBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref bwd_fn_arg = output.ancestry().backward_fn().get[ShuffleArg]()
        var axis = bwd_fn_arg.axis
        var permutation = bwd_fn_arg.permutation.copy()
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var shape = gradbox.shape()
        var gradbox_parent: Gradbox[Self.dtype]

        comptime if has_accelerator():
            if gradbox.is_on_gpu():
                try:
                    var result = ShuffleKernel[Self.dtype].launch_scatter(
                        gradbox.buffer().layout(),
                        gradbox.buffer().device_state.value(),
                        permutation,
                        axis,
                        sync=False,
                    )
                    var result_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(result[0], result[1])
                    gradbox_parent = Gradbox[Self.dtype](
                        result_ndb^,
                    )
                except e:
                    panic("ShuffleBackward GPU scatter failed: " + String(e))
                    # Unreachable
                    gradbox_parent = Gradbox[Self.dtype].zeros(
                        shape,
                    )
                parent.update_grad(gradbox_parent^, AddTensor, None)
                parent_ids.append(parent._id)
                # Mirror the CPU leg: clear the consumed grad unless the
                # caller retains intermediates, else repeat backward()
                # double-counts the stale grad.
                gradbox.zero_grad()
                return

        # CPU path
        # parent.shape == gradients.shape, only difference is coord postions
        # along the permuted axis
        # Scatter gradients back using the original permutation
        # For each position in the output gradient, find where it came from in the input

        gradbox_parent = Gradbox[Self.dtype].zeros(shape)
        for grad_coord in shape:
            var parent_coord = grad_coord
            parent_coord[axis] = permutation[grad_coord[axis]]
            gradbox_parent[parent_coord] = gradbox[grad_coord]

        parent.update_grad(gradbox_parent^, AddTensor, None)
        parent_ids.append(parent._id)
        gradbox.zero_grad()


@fieldwise_init
struct Shuffle[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        perm: List[Int],  # permutation, length == axis length/span/spread
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        # Normalize negative axes (axis=-1 = last). Everything downstream —
        # axis_length lookup, check_permutation, the NDBuffer fast path
        # (range(axis) is empty for axis<0), the GPU kernel, and the stored
        # ShuffleArg — needs the non-negative form.
        var axis_norm = axis if axis >= 0 else rank + axis
        if axis_norm < 0 or axis_norm >= rank:
            panic(
                "Shuffle → forward: axis ",
                String(axis),
                " out of bounds for rank ",
                String(rank),
            )
        var axis_length = shape[axis_norm]
        var permutation: List[Int]

        if len(perm) > 0:
            Validator.check_permutation(perm, axis_length)
            permutation = perm.copy()
        else:
            seed()
            permutation = List[Int](capacity=axis_length)
            for i in range(axis_length):
                permutation.append(i)
            shuffle(permutation)

        var result_ndb: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    var result = ShuffleKernel[Self.dtype].launch_gather(
                        self.buffer.layout(),
                        self.buffer.device_state.value(),
                        permutation,
                        axis_norm,
                        sync=sync,
                    )
                    result_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0], result[1]
                    )
                except e:
                    panic("Shuffle → forward GPU failed: " + String(e))
                    result_ndb = NDBuffer[Self.dtype].Empty()  # unreachable
            else:
                result_ndb = self.buffer.shuffle(permutation, axis_norm)
        else:
            result_ndb = self.buffer.shuffle(permutation, axis_norm)

        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    ShuffleArg(axis_norm, permutation^),
                    ShuffleBackward[Self.dtype](),
                )
                out.add_ancestry(backwardFn^, self)

        return out^
