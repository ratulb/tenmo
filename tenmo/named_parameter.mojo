from .tensor import Tensor


@fieldwise_init
struct NamedParameter[dtype: DType](ImplicitlyCopyable):
    var name: String
    var tensor_ptr: Pointer[Tensor[Self.dtype], MutUntrackedOrigin]
