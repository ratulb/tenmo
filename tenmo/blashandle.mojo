from std.ffi import OwnedDLHandle, _DLHandle
from .tensor import Tensor
from .shapes import Shape
from .common_utils import panic
from .mnemonics import AddTensor
from .backpropagation import BlasArg, BackwardFnArg, BLAS_BACKWARD_MATMUL_2D
from std.sys.defines import get_defined_string
from .gradbox import Gradbox
from std.memory import ArcPointer
from .ancestry import Ancestor


@fieldwise_init
struct BLASMatmul2dBackward[dtype: DType](
    RegisterPassable & ImplicitlyCopyable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[BlasArg[Self.dtype]]()
        )
        var (transpose_A, transpose_B, blas) = (
            bwd_arg.transpose_A,
            bwd_arg.transpose_B,
            bwd_arg.blas,
        )
        ref grad_out = output.gradients()[]
        var A_ancestor = output.ancestry().get(0)
        var B_ancestor = output.ancestry().get(1)

        var A = Tensor[Self.dtype](
            A_ancestor.buffer(), requires_grad=A_ancestor.requires_grad
        )
        var B = Tensor[Self.dtype](
            B_ancestor.buffer(), requires_grad=B_ancestor.requires_grad
        )
        var result = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()
        # ===== GRADIENT FOR A =====
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

            result.append((A_ancestor, grad_A^, AddTensor))

        # ===== GRADIENT FOR B =====
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

            result.append((B_ancestor^, grad_B^, AddTensor))
        return result^


comptime BLAS_PATH = get_defined_string[
    "BLAS_PATH", "/lib/x86_64-linux-gnu/libopenblas.so.0"
]()

# void cblas_sgemm(
#    int order,           1. CblasRowMajor (101)
#    int transA,          2. CblasNoTrans (111) or CblasTrans (112)
#    int transB,          3. CblasNoTrans (111) or CblasTrans (112)
#    int M,               4. Rows of op(A) and C
#    int N,               5. Columns of op(B) and C
#    int K,               6. Inner dimension
#    float alpha,         7. FLOAT - scaling for A*B (CRITICAL!)
#    const float *A,      8. Pointer to A
#    int lda,             9. Leading dimension of A (AFTER alpha!)
#    const float *B,      10. Pointer to B
#    int ldb,             11. Leading dimension of B
#    float beta,          12. FLOAT - scaling for C (CRITICAL!)
#    float *C,            13. Pointer to output C
#    int ldc              14. Leading dimension of C
# );

# CBLAS constants
comptime CblasRowMajor = Int32(101)
comptime CblasColMajor = Int32(102)
comptime CblasNoTrans = Int32(111)
comptime CblasTrans = Int32(112)

# Function type aliases
comptime CBLAS_SGEMM_FN = fn(
    Int32,  # order
    Int32,  # transA
    Int32,  # transB
    Int32,  # M
    Int32,  # N
    Int32,  # K
    Float32,  # alpha
    UnsafePointer[Float32, MutAnyOrigin],  # A
    Int32,  # lda
    UnsafePointer[Float32, MutAnyOrigin],  # B
    Int32,  # ldb
    Float32,  # beta
    UnsafePointer[Float32, MutAnyOrigin],  # C
    Int32,  # ldc
) -> None

comptime CBLAS_DGEMM_FN = fn(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Float64,
    UnsafePointer[Float64, MutAnyOrigin],
    Int32,
    UnsafePointer[Float64, MutAnyOrigin],
    Int32,
    Float64,
    UnsafePointer[Float64, MutAnyOrigin],
    Int32,
) -> None


@fieldwise_init
struct BLASHandle[dtype: DType](ImplicitlyCopyable, Movable):
    var _handle_ptr: Optional[ArcPointer[OwnedDLHandle]]
    var _error_msg: String

    fn __copyinit__(out self, copy: Self):
        self._handle_ptr = copy._handle_ptr.copy()
        self._error_msg = copy._error_msg

    fn __init__(out self):
        self._error_msg = ""

        try:
            print("Using BLAS_PATH: ", BLAS_PATH)
            var handle = OwnedDLHandle(BLAS_PATH)
            self._handle_ptr = Optional(ArcPointer(handle^))

        except e:
            print("Error loading BLAS: ", e)
            self._error_msg = "Failed to load BLAS library: " + String(e)
            self._handle_ptr = None

    fn __moveinit__(out self, deinit take: Self):
        self._handle_ptr = take._handle_ptr^
        self._error_msg = take._error_msg

    fn lite_handle(self) -> BLASHandleLite[Self.dtype]:
        return BLASHandleLite[Self.dtype](self._handle_ptr.value()[].borrow())

    fn is_initialized(self) -> Bool:
        return not self._handle_ptr == None

    fn get_error(ref self) -> ref[self._error_msg] String:
        return self._error_msg

    # ========== FIXED: REQUIRED PARAMS BEFORE OPTIONAL ==========
    fn matmul_f32(
        self,
        A: UnsafePointer[Float32, MutAnyOrigin],
        B: UnsafePointer[Float32, MutAnyOrigin],
        C: UnsafePointer[Float32, MutAnyOrigin],
        M: Int,
        N: Int,
        K: Int,
        lda: Int,  # REQUIRED: Must come before optional params
        ldb: Int,  # REQUIRED: Must come before optional params
        alpha: Float32 = 1.0,  # OPTIONAL: With default
        beta: Float32 = 0.0,  # OPTIONAL: With default
        transpose_A: Bool = False,  # OPTIONAL
        transpose_B: Bool = False,  # OPTIONAL
    ):
        """
        Compute C = alpha * op(A) @ op(B) + beta * C (Float32).

        Parameter order FIX: Required parameters (lda, ldb) come before optional ones.
        """
        if not self.is_initialized():
            panic("BLAS not initialized: ")

        ref lib = self._handle_ptr.value()[]
        var sgemm_fn = lib.get_function[CBLAS_SGEMM_FN]("cblas_sgemm")

        var trans_A = CblasTrans if transpose_A else CblasNoTrans
        var trans_B = CblasTrans if transpose_B else CblasNoTrans

        sgemm_fn(
            CblasRowMajor,
            trans_A,
            trans_B,
            Int32(M),
            Int32(N),
            Int32(K),
            alpha,  # Float32
            A,
            Int32(lda),  # Int32
            B,
            Int32(ldb),  # Int32
            beta,  # Float32
            C,
            Int32(N),  # ldc
        )

    fn matmul_f64(
        self,
        A: UnsafePointer[Float64, MutAnyOrigin],
        B: UnsafePointer[Float64, MutAnyOrigin],
        C: UnsafePointer[Float64, MutAnyOrigin],
        M: Int,
        N: Int,
        K: Int,
        lda: Int,  # REQUIRED first
        ldb: Int,  # REQUIRED first
        alpha: Float64 = 1.0,  # OPTIONAL
        beta: Float64 = 0.0,  # OPTIONAL
        transpose_A: Bool = False,
        transpose_B: Bool = False,
    ):
        """Double precision version with correct parameter ordering."""
        if not self.is_initialized():
            panic("BLAS not initialized: ")

        ref lib = self._handle_ptr.value()[]
        var dgemm_fn = lib.get_function[CBLAS_DGEMM_FN]("cblas_dgemm")

        var trans_A = CblasTrans if transpose_A else CblasNoTrans
        var trans_B = CblasTrans if transpose_B else CblasNoTrans

        dgemm_fn(
            CblasRowMajor,
            trans_A,
            trans_B,
            Int32(M),
            Int32(N),
            Int32(K),
            alpha,
            A,
            Int32(lda),
            B,
            Int32(ldb),
            beta,
            C,
            Int32(N),
        )

    fn matmul[
        track_grad: Bool = True
    ](
        self,
        A: Tensor[Self.dtype],
        B: Tensor[Self.dtype],
        transpose_A: Bool = False,
        transpose_B: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Matrix multiplication using BLAS.
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

        # Compute logical dimensions after transpose
        var M: Int
        var N: Int
        var K: Int

        if transpose_A:
            M = A_cols  # Transposed A has A_cols rows
            K = A_rows  # Transposed A has A_rows cols
        else:
            M = A_rows
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # Transposed B has B_cols rows
            N = B_rows  # Transposed B has B_rows cols
        else:
            K_from_B = B_rows
            N = B_cols

        if K != K_from_B:
            panic(
                "Matrix dimensions incompatible for matmul: "
                + "K mismatch: "
                + String(K)
                + " vs "
                + String(K_from_B)
            )

        # Allocate result
        var C = Tensor[Self.dtype].zeros(M, N)

        # Get pointers
        var A_ptr = (
            A.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var B_ptr = (
            B.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var C_ptr = (
            C.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Leading dimensions (columns in stored layout)
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS with CORRECTED parameter order
        comptime if Self.dtype == DType.float32:
            self.matmul_f32(
                A_ptr.bitcast[Float32](),
                B_ptr.bitcast[Float32](),
                C_ptr.bitcast[Float32](),
                M,
                N,
                K,
                lda,  # REQUIRED: 8th param
                ldb,  # REQUIRED: 9th param
                Float32(1.0),  # alpha: 10th param (optional with default)
                Float32(0.0),  # beta: 11th param (optional with default)
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        else:  # Self.dtype == DType.float64:
            self.matmul_f64(
                A_ptr.bitcast[Float64](),
                B_ptr.bitcast[Float64](),
                C_ptr.bitcast[Float64](),
                M,
                N,
                K,
                lda,  # REQUIRED
                ldb,  # REQUIRED
                Float64(1.0),  # alpha
                Float64(0.0),  # beta
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )

        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                A.requires_grad or B.requires_grad
            )
            if grad_required:
                C.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BLAS_BACKWARD_MATMUL_2D,
                    BlasArg[Self.dtype](
                        transpose_A, transpose_B, self.lite_handle()
                    ),
                )

                C.add_ancestry(backwardFnArg^, A, B)

        return C^

    fn matmul(
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

        # Compute logical dimensions after transpose
        var M: Int
        var N: Int
        var K: Int

        if transpose_A:
            M = A_cols  # Aᵀ has A_cols rows
            K = A_rows  # Aᵀ has A_rows cols
        else:
            M = A_rows
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # Bᵀ has B_cols rows
            N = B_rows  # Bᵀ has B_rows cols
        else:
            K_from_B = B_rows
            N = B_cols

        # Dimension compatibility check
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

        # Allocate result
        var C = Gradbox[Self.dtype].zeros(Shape([M, N]))

        # Get pointers
        var A_ptr = (
            A.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var B_ptr = (
            B.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var C_ptr = (
            C.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Leading dimensions
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS
        comptime if Self.dtype == DType.float32:
            self.matmul_f32(
                A_ptr.bitcast[Float32](),
                B_ptr.bitcast[Float32](),
                C_ptr.bitcast[Float32](),
                M,
                N,
                K,
                lda,
                ldb,
                Float32(1.0),
                Float32(0.0),
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        else:  # float64
            self.matmul_f64(
                A_ptr.bitcast[Float64](),
                B_ptr.bitcast[Float64](),
                C_ptr.bitcast[Float64](),
                M,
                N,
                K,
                lda,
                ldb,
                Float64(1.0),
                Float64(0.0),
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )

        return C^

    fn matmul(
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
        var A_rows = A.shape()[0]
        var A_cols = A.shape()[1]
        var B_rows = B.shape()[0]
        var B_cols = B.shape()[1]

        # Compute logical dimensions after transpose
        var M: Int
        var N: Int
        var K: Int

        if transpose_A:
            M = A_cols  # Transposed A has A_cols rows
            K = A_rows  # Transposed A has A_rows cols
        else:
            M = A_rows
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # Transposed B has B_cols rows
            N = B_rows  # Transposed B has B_rows cols
        else:
            K_from_B = B_rows
            N = B_cols

        # Allocate result
        var C = Gradbox[Self.dtype].zeros(Shape([M, N]))

        # Get pointers
        var A_ptr = (
            A.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var B_ptr = (
            B.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var C_ptr = (
            C.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Leading dimensions (columns in stored layout)
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS with CORRECTED parameter order
        comptime if Self.dtype == DType.float32:
            self.matmul_f32(
                A_ptr.bitcast[Float32](),
                B_ptr.bitcast[Float32](),
                C_ptr.bitcast[Float32](),
                M,
                N,
                K,
                lda,  # REQUIRED: 8th param
                ldb,  # REQUIRED: 9th param
                Float32(1.0),  # alpha: 10th param (optional with default)
                Float32(0.0),  # beta: 11th param (optional with default)
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        else:  # Self.dtype == DType.float64
            self.matmul_f64(
                A_ptr.bitcast[Float64](),
                B_ptr.bitcast[Float64](),
                C_ptr.bitcast[Float64](),
                M,
                N,
                K,
                lda,  # REQUIRED
                ldb,  # REQUIRED
                Float64(1.0),  # alpha
                Float64(0.0),  # beta
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )

        return C^


@fieldwise_init
struct BLASHandleLite[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var _handle: _DLHandle

    fn __copyinit__(out self, copy: Self):
        self._handle = copy._handle.copy()

    # ========== REQUIRED PARAMS BEFORE OPTIONAL ==========
    fn matmul_f32(
        self,
        A: UnsafePointer[Float32, MutAnyOrigin],
        B: UnsafePointer[Float32, MutAnyOrigin],
        C: UnsafePointer[Float32, MutAnyOrigin],
        M: Int,
        N: Int,
        K: Int,
        lda: Int,  # REQUIRED: Must come before optional params
        ldb: Int,  # REQUIRED: Must come before optional params
        alpha: Float32 = 1.0,  # OPTIONAL: With default
        beta: Float32 = 0.0,  # OPTIONAL: With default
        transpose_A: Bool = False,  # OPTIONAL
        transpose_B: Bool = False,  # OPTIONAL
    ):
        """
        Compute C = alpha * op(A) @ op(B) + beta * C (Float32).

        Parameter order: Required parameters (lda, ldb) come before optional ones.
        """
        var sgemm_fn = self._handle.get_function[CBLAS_SGEMM_FN]("cblas_sgemm")

        var trans_A = CblasTrans if transpose_A else CblasNoTrans
        var trans_B = CblasTrans if transpose_B else CblasNoTrans

        sgemm_fn(
            CblasRowMajor,
            trans_A,
            trans_B,
            Int32(M),
            Int32(N),
            Int32(K),
            alpha,  # Float32
            A,
            Int32(lda),  # Int32
            B,
            Int32(ldb),  # Int32
            beta,  # Float32
            C,
            Int32(N),  # ldc
        )

    fn matmul_f64(
        self,
        A: UnsafePointer[Float64, MutAnyOrigin],
        B: UnsafePointer[Float64, MutAnyOrigin],
        C: UnsafePointer[Float64, MutAnyOrigin],
        M: Int,
        N: Int,
        K: Int,
        lda: Int,  # REQUIRED first
        ldb: Int,  # REQUIRED first
        alpha: Float64 = 1.0,  # OPTIONAL
        beta: Float64 = 0.0,  # OPTIONAL
        transpose_A: Bool = False,
        transpose_B: Bool = False,
    ):
        """Double precision version with correct parameter ordering."""
        var dgemm_fn = self._handle.get_function[CBLAS_DGEMM_FN]("cblas_dgemm")

        var trans_A = CblasTrans if transpose_A else CblasNoTrans
        var trans_B = CblasTrans if transpose_B else CblasNoTrans

        dgemm_fn(
            CblasRowMajor,
            trans_A,
            trans_B,
            Int32(M),
            Int32(N),
            Int32(K),
            alpha,
            A,
            Int32(lda),
            B,
            Int32(ldb),
            beta,
            C,
            Int32(N),
        )

    fn matmul[
        track_grad: Bool = True
    ](
        self,
        A: Tensor[Self.dtype],
        B: Tensor[Self.dtype],
        transpose_A: Bool = False,
        transpose_B: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Matrix multiplication using BLAS.
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

        # Compute logical dimensions after transpose
        var M: Int
        var N: Int
        var K: Int

        if transpose_A:
            M = A_cols  # Transposed A has A_cols rows
            K = A_rows  # Transposed A has A_rows cols
        else:
            M = A_rows
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # Transposed B has B_cols rows
            N = B_rows  # Transposed B has B_rows cols
        else:
            K_from_B = B_rows
            N = B_cols

        if K != K_from_B:
            panic(
                "Matrix dimensions incompatible for matmul: "
                + "K mismatch: "
                + String(K)
                + " vs "
                + String(K_from_B)
            )

        # Ensure contiguous
        # var A_contig = A if A.is_contiguous() else A.contiguous()
        # var B_contig = B if B.is_contiguous() else B.contiguous()

        # Allocate result
        var C = Tensor[Self.dtype].zeros(M, N)

        # Get pointers
        var A_ptr = (
            A.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var B_ptr = (
            B.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var C_ptr = (
            C.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Leading dimensions (columns in stored layout)
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS with CORRECTED parameter order
        comptime if Self.dtype == DType.float32:
            self.matmul_f32(
                A_ptr.bitcast[Float32](),
                B_ptr.bitcast[Float32](),
                C_ptr.bitcast[Float32](),
                M,
                N,
                K,
                lda,  # REQUIRED: 8th param
                ldb,  # REQUIRED: 9th param
                Float32(1.0),  # alpha: 10th param (optional with default)
                Float32(0.0),  # beta: 11th param (optional with default)
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        elif Self.dtype == DType.float64:
            self.matmul_f64(
                A_ptr.bitcast[Float64](),
                B_ptr.bitcast[Float64](),
                C_ptr.bitcast[Float64](),
                M,
                N,
                K,
                lda,  # REQUIRED
                ldb,  # REQUIRED
                Float64(1.0),  # alpha
                Float64(0.0),  # beta
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        else:
            panic("Unsupported dtype for BLAS: " + String(Self.dtype))

        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                A.requires_grad or B.requires_grad
            )
            if grad_required:
                C.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BLAS_BACKWARD_MATMUL_2D,
                    BlasArg[Self.dtype](transpose_A, transpose_B, self),
                )

                C.add_ancestry(backwardFnArg^, A, B)

        return C^

    fn matmul(
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

        # Compute logical dimensions after transpose
        var M: Int
        var N: Int
        var K: Int

        if transpose_A:
            M = A_cols  # Aᵀ has A_cols rows
            K = A_rows  # Aᵀ has A_rows cols
        else:
            M = A_rows
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # Bᵀ has B_cols rows
            N = B_rows  # Bᵀ has B_rows cols
        else:
            K_from_B = B_rows
            N = B_cols

        # CRITICAL: Dimension compatibility check
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

        # Allocate result
        var C = Gradbox[Self.dtype].zeros(Shape([M, N]))

        # Get pointers
        var A_ptr = (
            A.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var B_ptr = (
            B.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var C_ptr = (
            C.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Leading dimensions
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS
        comptime if Self.dtype == DType.float32:
            self.matmul_f32(
                A_ptr.bitcast[Float32](),
                B_ptr.bitcast[Float32](),
                C_ptr.bitcast[Float32](),
                M,
                N,
                K,
                lda,
                ldb,
                Float32(1.0),
                Float32(0.0),
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        else:  # float64
            self.matmul_f64(
                A_ptr.bitcast[Float64](),
                B_ptr.bitcast[Float64](),
                C_ptr.bitcast[Float64](),
                M,
                N,
                K,
                lda,
                ldb,
                Float64(1.0),
                Float64(0.0),
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )

        return C^

    fn matmul(
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
        var A_rows = A.shape()[0]
        var A_cols = A.shape()[1]
        var B_rows = B.shape()[0]
        var B_cols = B.shape()[1]

        # Compute logical dimensions after transpose
        var M: Int
        var N: Int
        var K: Int

        if transpose_A:
            M = A_cols  # Transposed A has A_cols rows
            K = A_rows  # Transposed A has A_rows cols
        else:
            M = A_rows
            K = A_cols

        var K_from_B: Int
        if transpose_B:
            K_from_B = B_cols  # Transposed B has B_cols rows
            N = B_rows  # Transposed B has B_rows cols
        else:
            K_from_B = B_rows
            N = B_cols

        # Allocate result
        var C = Gradbox[Self.dtype].zeros(Shape([M, N]))

        # Get pointers
        var A_ptr = (
            A.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var B_ptr = (
            B.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var C_ptr = (
            C.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

        # Leading dimensions (columns in stored layout)
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS with CORRECTED parameter order
        comptime if Self.dtype == DType.float32:
            self.matmul_f32(
                A_ptr.bitcast[Float32](),
                B_ptr.bitcast[Float32](),
                C_ptr.bitcast[Float32](),
                M,
                N,
                K,
                lda,  # REQUIRED: 8th param
                ldb,  # REQUIRED: 9th param
                Float32(1.0),  # alpha: 10th param (optional with default)
                Float32(0.0),  # beta: 11th param (optional with default)
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )
        else:  # Self.dtype == DType.float64
            self.matmul_f64(
                A_ptr.bitcast[Float64](),
                B_ptr.bitcast[Float64](),
                C_ptr.bitcast[Float64](),
                M,
                N,
                K,
                lda,  # REQUIRED
                ldb,  # REQUIRED
                Float64(1.0),  # alpha
                Float64(0.0),  # beta
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )

        return C^


fn main() raises:
    test_blas_case_4_Atranspose_Btranspose()


# ============================================================================
# COMPREHENSIVE BLAS MATMUL TESTS
# ============================================================================
from std.testing import assert_true


fn test_blas_case_4_Atranspose_Btranspose() raises:
    """Test Case 4: C = A^T @ B^T."""
    print("\n" + "=" * 80)
    print("TEST CASE 4: C = A^T @ B^T")
    print("=" * 80)

    comptime dtype = DType.float32

    # Dimensions: A(3,2)^T @ B(4,3)^T → A^T(2,3) @ B^T(3,4) → C(2,4)
    var A = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )  # 3x2

    var B = Tensor[dtype].d2(
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
        ],
        requires_grad=True,
    )  # 4x3

    # === NATIVE MOJO ===
    print("\n1. NATIVE MOJO:")
    var A_T = A.transpose()  # 2x3
    var B_T = B.transpose()  # 3x4
    var C_native = A_T.matmul(B_T)  # (2x3) @ (3x4) = 2x4
    var loss_native = C_native.sum()
    loss_native.backward()

    print("Forward result shape:", C_native.shape())
    print("\ngrad_A native:")
    A.grad().print()
    print("\ngrad_B native:")
    B.grad().print()

    var native_A_grad = A.grad().copy()
    var native_B_grad = B.grad().copy()

    # Reset gradients
    A.zero_grad()
    B.zero_grad()

    # === BLAS ===
    print("\n2. BLAS:")
    var blas = BLASHandle[dtype]()
    var C_blas = blas.matmul(A, B, transpose_A=True, transpose_B=True)
    var loss_blas = C_blas.sum()
    loss_blas.backward()

    print("Forward result shape:", C_blas.shape())
    print("\ngrad_A BLAS:")
    A.grad().print()
    print("\ngrad_B BLAS:")
    B.grad().print()

    # === VALIDATION ===
    print("\n3. VALIDATION:")
    assert_true(C_native.all_close(C_blas), "Forward results differ!")

    var blas_A_grad = A.grad().copy()
    var blas_B_grad = B.grad().copy()

    assert_true(native_A_grad.all_close(blas_A_grad), "grad_A differs!")
    assert_true(native_B_grad.all_close(blas_B_grad), "grad_B differs!")
    _ = blas
    print("✓ All checks passed!")
