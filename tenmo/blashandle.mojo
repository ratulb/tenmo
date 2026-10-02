from std.ffi import OwnedDLHandle, _DLHandle
from .tensor import Tensor
from .shared.panic import panic
from .shared.mnemonics import AddTensor
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .gradbox import Gradbox
from std.memory import ArcPointer
from .ancestry import Ancestor
from .blas_ndbuffer import blas_matmul, BLASCache


@fieldwise_init
struct BlasArg[dtype: DType](ArgumentType):
    var transpose_A: Bool
    var transpose_B: Bool
    var blas: BLASHandleLite[Self.dtype]


@fieldwise_init
struct BLASMatmul2dBackward[dtype: DType](
    BackwardFnType, RegisterPassable & ImplicitlyCopyable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = output.ancestry().backward_fn().get[BlasArg[Self.dtype]]()
        var (transpose_A, transpose_B, blas) = (
            bwd_arg.transpose_A,
            bwd_arg.transpose_B,
            bwd_arg.blas,
        )
        ref grad_out = output.gradients()
        var A_ancestor = output.ancestry().get(0)
        var B_ancestor = output.ancestry().get(1)

        var A = Tensor[Self.dtype](
            A_ancestor.buffer(), requires_grad=A_ancestor.requires_grad
        )
        var B = Tensor[Self.dtype](
            B_ancestor.buffer(), requires_grad=B_ancestor.requires_grad
        )
        if A.requires_grad:
            var grad_A: Gradbox[Self.dtype]

            if not transpose_A and not transpose_B:
                # Case 1: C = A @ B
                # grad_A = grad_out @ B^T
                grad_A = blas.matmul(grad_out, B, transpose_B=True)

            elif transpose_A and not transpose_B:
                # Case 2: C = A^T @ B
                # grad_A = B @ grad_out^T  (NOT grad_out @ B^T!)
                grad_A = blas.matmul(B, grad_out, transpose_B=True)

            elif not transpose_A and transpose_B:
                # Case 3: C = A @ B^T
                # grad_A = grad_out @ B
                grad_A = blas.matmul(grad_out, B)

            else:  # both transpose_A and transpose_B
                # Case 4: C = A^T @ B^T
                # grad_A = B^T @ grad_out^T
                grad_A = blas.matmul(
                    B, grad_out, transpose_A=True, transpose_B=True
                )

            A_ancestor.update_grad(grad_A^, AddTensor, None)
            parent_ids.append(A_ancestor._id)

        if B.requires_grad:
            var grad_B: Gradbox[Self.dtype]

            if not transpose_A and not transpose_B:
                # Case 1: C = A @ B
                # grad_B = A^T @ grad_out
                grad_B = blas.matmul(A, grad_out, transpose_A=True)

            elif transpose_A and not transpose_B:
                # Case 2: C = A^T @ B
                # grad_B = A @ grad_out
                grad_B = blas.matmul(A, grad_out)

            elif not transpose_A and transpose_B:
                # Case 3: C = A @ B^T
                # grad_B = grad_out^T @ A
                grad_B = blas.matmul(grad_out, A, transpose_A=True)

            else:  # both transpose_A and transpose_B
                # Case 4: C = A^T @ B^T
                # grad_B = grad_out^T @ A^T
                grad_B = blas.matmul(
                    grad_out, A, transpose_A=True, transpose_B=True
                )

            B_ancestor.update_grad(grad_B^, AddTensor, None)
            parent_ids.append(B_ancestor._id)

        grad_out.zero_grad()


@fieldwise_init
struct BLASHandleLite[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var _handle: _DLHandle
    var _keepalive: ArcPointer[OwnedDLHandle]

    # `arc` keeps the owning `OwnedDLHandle` alive for the whole lifetime of
    # this lite, so a lite embedded in a type-erased backward payload is still
    # backed by a live library when `backward()` runs later.
    def __init__(out self, arc: ArcPointer[OwnedDLHandle]):
        self._keepalive = arc.copy()
        self._handle = self._keepalive[].borrow()

    def __init__(out self, *, copy: Self):
        self._handle = copy._handle.copy()
        self._keepalive = copy._keepalive.copy()

    @staticmethod
    def from_cache() -> Self:
        """Build a lite from the process-global BLAS handle.

        Panics if the library failed to load — a lite is only created when
        BLAS is available (no native fallback).
        """
        var arc = BLASCache.get()
        if arc == None:
            panic("BLASHandleLite: BLAS library not available")
        return BLASHandleLite[Self.dtype](arc.value().copy())

    def matmul[
        track_grad: Bool = True
    ](
        self,
        A: Tensor[Self.dtype],
        B: Tensor[Self.dtype],
        transpose_A: Bool = False,
        transpose_B: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """
        Matrix multiplication using BLAS.

        ``sync`` is a no-op sink (BLAS path is CPU-only) kept so the
        matmul ``sync`` chain stays symmetric.
        """
        # Validate inputs
        if A.rank() != 2:
            panic("A must be a 2D tensor, got rank " + String(A.rank()))
        if B.rank() != 2:
            panic("B must be a 2D tensor, got rank " + String(B.rank()))

        # Get stored dimensions
        var A_rows = A.shape()[0]
        var A_cols = A.shape()[1]
        var B_rows = B.shape()[0]
        var B_cols = B.shape()[1]

        # Compute the inner dimension after transpose for the compatibility check
        var K: Int
        if transpose_A:
            K = A_rows  # transposed A spans A_rows along its inner dim
        else:
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # transposed B spans B_cols along its inner dim
        else:
            K_from_B = B_rows
        if K != K_from_B:
            panic(
                "Matrix dimensions incompatible for matmul: "
                + "K mismatch: "
                + String(K)
                + " vs "
                + String(K_from_B)
            )

        # Compute the result: delegate to the leaf's forward-only BLAS GEMM.
        # No native fallback — a lite is only created when BLAS is available.
        var ndb = blas_matmul[Self.dtype](
            A.buffer,
            B.buffer,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            sync=sync,
        )
        var C = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                A.requires_grad or B.requires_grad
            )
            if grad_required:
                C.requires_grad_(True)
                var backwardFn = BackwardFn(
                    BlasArg[Self.dtype](transpose_A, transpose_B, self),
                    BLASMatmul2dBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True

                C.add_ancestry(backwardFn^, A, B)

        return C^

    def matmul(
        self,
        A: Gradbox[Self.dtype],
        B: Tensor[Self.dtype],
        transpose_A: Bool = False,
        transpose_B: Bool = False,
    ) -> Gradbox[Self.dtype]:
        """
        Matrix multiplication: Gradbox @ Tensor.
        FIXED: Proper dimension checks for all transpose combinations.
        """
        # Get stored dimensions
        var A_rows = A.shape()[0]
        var A_cols = A.shape()[1]
        var B_rows = B.shape()[0]
        var B_cols = B.shape()[1]

        # Compute the inner dimension after transpose for the compatibility check
        var K: Int
        if transpose_A:
            K = A_rows  # transposed A spans A_rows along its inner dim
        else:
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # transposed B spans B_cols along its inner dim
        else:
            K_from_B = B_rows
        if K != K_from_B:
            panic(
                "Gradbox @ Tensor dimension mismatch: "
                + "inner dim K="
                + String(K)
                + " vs K_from_B="
                + String(K_from_B)
                + " with transpose_A="
                + String(transpose_A)
                + " transpose_B="
                + String(transpose_B)
            )

        # Compute the result: delegate to the leaf's forward-only BLAS GEMM.
        # No native fallback — a lite is only created when BLAS is available.
        var ndb = blas_matmul[Self.dtype](
            A.buffer(), B.buffer, transpose_A=transpose_A, transpose_B=transpose_B
        )
        var C = Gradbox[Self.dtype](ndb^)

        return C^

    def matmul(
        self,
        A: Tensor[Self.dtype],
        B: Gradbox[Self.dtype],
        transpose_A: Bool = False,
        transpose_B: Bool = False,
    ) -> Gradbox[Self.dtype]:
        """
        Matrix multiplication using BLAS for gradient calculation.
        """
        # Not validating inputs since forward pass does that
        # Compute the result: delegate to the leaf's forward-only BLAS GEMM.
        # No native fallback — a lite is only created when BLAS is available.
        var ndb = blas_matmul[Self.dtype](
            A.buffer, B.buffer(), transpose_A=transpose_A, transpose_B=transpose_B
        )
        var C = Gradbox[Self.dtype](ndb^)

        return C^
