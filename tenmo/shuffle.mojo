from .tensor import Tensor
from .mnemonics import AddTensor
from .validators import Validator
from .backpropagation import BackwardFnArg, ShuffleArg, BACKWARD_SHUFFLE
from std.random import shuffle, seed
from .gradbox import Gradbox
from std.sys import has_accelerator
from .ndbuffer import NDBuffer
from .common_utils import panic
from .ancestry import Ancestor
from .kernels.shuffle_kernel import ShuffleGPU


@fieldwise_init
struct ShuffleBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
        sync: Bool = True,
    ):
        ref bwd_fn_arg = output.ancestry().backward_fn_arg().get[ShuffleArg]()
        var axis = bwd_fn_arg.axis
        var permutation = bwd_fn_arg.permutation.copy()
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var shape = gradbox.shape()
        var gradbox_parent: Gradbox[Self.dtype]

        comptime if has_accelerator():
            if gradbox.is_on_gpu():
                try:
                    var result_ndb = ShuffleGPU[Self.dtype].launch_scatter(
                        gradbox.buffer(), permutation, axis, sync=sync
                    )
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
        if not retain_graph:
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
        var axis_length = shape[axis]
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
                    result_ndb = ShuffleGPU[Self.dtype].launch_gather(
                        self.buffer, permutation, axis, sync=sync
                    )
                except e:
                    panic("Shuffle → forward GPU failed: " + String(e))
                    result_ndb = NDBuffer[Self.dtype].Empty()  # unreachable
            else:
                result_ndb = self.buffer.shuffle(permutation, axis)
        else:
            result_ndb = self.buffer.shuffle(permutation, axis)

        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_SHUFFLE, ShuffleArg(axis, permutation^)
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
