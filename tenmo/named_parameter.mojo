from .tensor import Tensor

@fieldwise_init
struct NamedParameter[dtype: DType](ImplicitlyCopyable & Movable):
    var name: String
    var tensor_ptr: UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]
