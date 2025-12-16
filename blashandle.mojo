from sys.ffi import OwnedDLHandle
from tenmo import Tensor
from shapes import Shape
from common_utils import panic
from walkback import Matmul2dBackward
from sys.param_env import env_get_string


alias BLAS_PATH = env_get_string[
    "BLAS_PATH", "/home/ritu91b/tenmo/.pixi/envs/default/lib/libopenblas.so.0"
]()

# void cblas_sgemm(
#    int order,          // 1. CblasRowMajor (101)
#    int transA,         // 2. CblasNoTrans (111) or CblasTrans (112)
#    int transB,         // 3. CblasNoTrans (111) or CblasTrans (112)
#    int M,              // 4. Rows of op(A) and C
#    int N,              // 5. Columns of op(B) and C
#    int K,              // 6. Inner dimension
#    float alpha,        // 7. FLOAT - scaling for A*B (CRITICAL!)
#    const float *A,     // 8. Pointer to A
#    int lda,            // 9. Leading dimension of A (AFTER alpha!)
#    const float *B,     // 10. Pointer to B
#    int ldb,            // 11. Leading dimension of B
#    float beta,         // 12. FLOAT - scaling for C (CRITICAL!)
#    float *C,           // 13. Pointer to output C
#    int ldc             // 14. Leading dimension of C
# );

# CBLAS constants
alias CblasRowMajor = Int32(101)
alias CblasColMajor = Int32(102)
alias CblasNoTrans = Int32(111)
alias CblasTrans = Int32(112)

# Function type aliases
alias CBLAS_SGEMM_FN = fn (
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

alias CBLAS_DGEMM_FN = fn (
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
struct BLASHandle[dtype: DType](ImplicitlyCopyable & Movable):
    var _handle_ptr: UnsafePointer[OwnedDLHandle, MutAnyOrigin]
    var _initialized: Bool
    var _error_msg: String

    fn __init__(out self):
        self._handle_ptr = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()
        self._initialized = False
        self._error_msg = ""

        try:
            self._handle_ptr = alloc[OwnedDLHandle](1)
            var handle = OwnedDLHandle(BLAS_PATH)
            self._handle_ptr.init_pointee_move(handle^)
            self._initialized = True
        except e:
            self._error_msg = "Failed to load BLAS library: " + e.__str__()
            if self._handle_ptr:
                self._handle_ptr.free()
                self._handle_ptr = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()

    fn __moveinit__(out self, deinit other: Self):
        self._handle_ptr = other._handle_ptr
        self._initialized = other._initialized
        self._error_msg = other._error_msg

    fn __del__(deinit self):
        if self._initialized and self._handle_ptr:
            self._handle_ptr.destroy_pointee()
            self._handle_ptr.free()

    fn is_initialized(self) -> Bool:
        return self._initialized

    fn get_error(self) -> String:
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
        if not self._initialized:
            panic("BLAS not initialized: " + self._error_msg)

        ref lib = self._handle_ptr[]
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
        if not self._initialized:
            panic("BLAS not initialized: " + self._error_msg)

        ref lib = self._handle_ptr[]
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

    fn matmul(
        self,
        mut A: Tensor[Self.dtype],
        mut B: Tensor[Self.dtype],
        transpose_A: Bool = False,
        transpose_B: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) raises -> Tensor[Self.dtype]:
        """
        Matrix multiplication using BLAS.
        """
        # Validate inputs
        if A.rank() != 2:
            raise Error("A must be a 2D tensor, got rank " + A.rank().__str__())
        if B.rank() != 2:
            raise Error("B must be a 2D tensor, got rank " + B.rank().__str__())

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
            raise Error(
                "Matrix dimensions incompatible for matmul: "
                + "K mismatch: "
                + K.__str__()
                + " vs "
                + K_from_B.__str__()
            )

        # Ensure contiguous
        # var A_contig = A if A.is_contiguous() else A.contiguous()
        # var B_contig = B if B.is_contiguous() else B.contiguous()

        # Allocate result
        var C = Tensor[Self.dtype].zeros(M, N)

        # Get pointers
        var A_ptr = A.buffer.data_buffer().data
        var B_ptr = B.buffer.data_buffer().data
        var C_ptr = C.buffer.data_buffer().data

        # Leading dimensions (columns in stored layout)
        var lda = A.shape()[1]
        var ldb = B.shape()[1]

        # Call BLAS with CORRECTED parameter order
        @parameter
        if Self.dtype == DType.float32:
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
            panic("Unsupported dtype for BLAS: " + Self.dtype.__str__())

        # Gradient tracking
        var grad_required = requires_grad.or_else(
            A.requires_grad or B.requires_grad
        )
        if grad_required:
            C.requires_grad_(True)
            var backward_fn = Matmul2dBackward[Self.dtype]().into_backward_fn()
            C.backwardFn = Optional(backward_fn^)
            C.add_ancestry(A, B)

        return C^


fn main() raises:
    pass
