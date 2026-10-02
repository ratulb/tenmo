"""LayerTrait — base contract for all neural-network layers.

Extracted from net.mojo to break the 5-module import cycle:
  net → {dropout, layernorm, embedding, pooling} → net

LayerTrait is the only type those modules need to implement;
Module and Layer stay in net.mojo.
"""

from .tensor import Tensor
from .named_parameter import NamedParameter
from .gpu.device import GPU
from std.traits.deinitable import Deinitable


trait LayerTrait(Deinitable & ImplicitlyCopyable):
    """Unified layer contract for the dtype-erased Sequential container.

    Every layer declares its comptime input/output dtypes; homogeneous
    layers inherit `OutputDType = Self.InputDType`. Container-level
    capabilities (parameter listing, mode switching, device transfer)
    have defaults sized for parameterless layers — parameterized layers
    override them. Verified feature-by-feature against the pinned
    toolchain.
    """

    # Container contract — read before changing these:
    # - MixedSequential/Seq bitcast heap blobs to Tensor[InputDType] and
    #   record OutputDType as the record's io_dtype (net.mojo make_forward,
    #   append). The declaration must be the TRUE tensor dtype your __call__
    #   accepts/returns: the compiler pins the signature, but the containers
    #   trust the VALUE blindly.
    # - A boundary cast is inserted wherever tail OutputDType != next
    #   InputDType (grad-tracked float->float via ToDtypeBackward; int
    #   targets come back as leaves — see Tensor.to_dtype).
    #
    # The OutputDType default is a documentation/grep affordance, NOT a
    # safety net whose absence is a hazard. An earlier note here claimed
    # a heterogeneous layer forgetting OutputDType would "compile cleanly
    # and silently declare itself homogeneous", corrupting the seam at
    # runtime. That is false on this pin: because __call__ below is
    # written in terms of Self.InputDType/Self.OutputDType, the default
    # changes what the compiler DEMANDS of __call__, so the omission is a
    # conformance error naming both signatures, and one added declaration
    # fixes it. Verified.
    #
    # What the compiler genuinely cannot check is declared-vs-intended
    # semantics: whether a layer's declared InputDType/OutputDType match
    # what it actually computes. `scripts/check_layer_dtypes.py` lints that,
    # since the signature is pinned but the intent is not.
    comptime InputDType: DType
    comptime OutputDType: DType = Self.InputDType

    def __call__(
        mut self, x: Tensor[Self.InputDType], sync: Bool = True
    ) -> Tensor[Self.OutputDType]:
        ...

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.OutputDType], MutAnyOrigin]]:
        """Empty default — parameterless layers inherit it.

        Implementations MUST return pointers to Tensor[Self.OutputDType]:
        the erased collectors (net.mojo make_param_collector) and
        MixedSequential.parameters_of[D] assume params live in OutputDType.
        """
        return List[Pointer[Tensor[Self.OutputDType], MutAnyOrigin]]()

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.OutputDType]]:
        return List[NamedParameter[Self.OutputDType]]()

    def num_parameters(self) -> Int:
        return 0

    def zero_grad(mut self):
        for parameter in self.parameters():
            parameter[].zero_grad()

    def train(mut self):
        ...

    def eval(mut self):
        ...

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """Identity default — parameterless layers need no transfer."""
        return self

    def to_cpu(self) raises -> Self:
        """Identity default — parameterless layers need no transfer."""
        return self
