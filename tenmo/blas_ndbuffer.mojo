"""Forward-only OpenBLAS GEMM on NDBuffer.

Imports only ``ndbuffer`` and ``tenmo.shared`` — never
``tensor``/``matmul``/``gradbox``/``blashandle``. This keeps the module
acyclic with respect to the tensor core, so ``matmul.mojo`` (and
``blashandle.mojo``) can import it without closing a dependency cycle.

This module owns the *raw* BLAS FFI (the ``dlopen``ed handle cache, the CBLAS
constants and the ``cblas_sgemm``/``cblas_dgemm`` call) and exposes two
forward-only operations:

- ``blas_gemm`` — raw GEMM on raw pointers.
- ``blas_matmul`` — GEMM on two contiguous, offset-0, CPU ``NDBuffer``
  operands (used by ``matmul.mojo``'s 2D matmul and the explicit
  ``BLASHandleLite`` wrapper in ``blashandle.mojo``). Returns a fresh result
  ``NDBuffer``.

Neither operation touches the autograd/ancestor graph — there is no
``requires_grad`` here. Gradient *computation* re-uses these forward GEMMs
from the caller.

Enabled at compile time with ``-D BLAS=<anything non-empty>``
(``BLAS_PATH`` overrides the default override path). Native matmul stays the
default when the flag is absent.
"""

from std.ffi import OwnedDLHandle, _DLCallable, _Global
from std.sys.defines import get_defined_string
from std.memory import ArcPointer

from .ndbuffer import NDBuffer
from .shared.shapes import Shape
from .shared.panic import panic


# Comptime configuration
comptime BLAS_PATH = get_defined_string[
    "BLAS_PATH", "/lib/x86_64-linux-gnu/libopenblas.so.0"
]()
comptime BLAS_ENABLED = get_defined_string["BLAS", ""]()


# CBLAS constants ---------------------------------------------------------
comptime CblasRowMajor = Int32(101)
comptime CblasNoTrans = Int32(111)
comptime CblasTrans = Int32(112)


# C-ABI signatures
comptime CBLAS_SGEMM_FN = def(
    Int32,  # order
    Int32,  # transA
    Int32,  # transB
    Int32,  # M
    Int32,  # N
    Int32,  # K
    Float32,  # alpha
    Pointer[Float32, MutAnyOrigin],  # A
    Int32,  # lda
    Pointer[Float32, MutAnyOrigin],  # B
    Int32,  # ldb
    Float32,  # beta
    Pointer[Float32, MutAnyOrigin],  # C
    Int32,  # ldc
) thin -> None

comptime CBLAS_DGEMM_FN = def(
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Float64,
    Pointer[Float64, MutAnyOrigin],
    Int32,
    Pointer[Float64, MutAnyOrigin],
    Int32,
    Float64,
    Pointer[Float64, MutAnyOrigin],
    Int32,
) thin -> None


def _init_blas_handle() -> ArcPointer[OwnedDLHandle]:
    """Process-global handle cache
    The owning `ArcPointer[OwnedDLHandle]` lives in a `std.ffi._Global` slot so
    `dlopen` happens once per process and the library stays alive for the whole
    run. `ArcPointer` is `Movable` (required by `_Global`), whereas a raw
    `BLASHandleLite` is not — hence we cache the ArcPointer, not a lite.
    """
    # `_Global.init_fn` must be non-raising; failure is reported via an empty
    # (uninitialized) raw handle, which `get()`/`is_available()` detect via
    # `borrow().__bool__()` (an empty `_DLHandle` is falsy).
    try:
        var lib = OwnedDLHandle(BLAS_PATH)
        return ArcPointer[OwnedDLHandle](lib^)
    except:
        print("Failed BLAS initialization from BLAS_PATH=", BLAS_PATH)
        return ArcPointer[OwnedDLHandle](
            OwnedDLHandle(unsafe_uninitialized=True)
        )


struct BLASCache(RegisterPassable):
    """Accessors over the process-global OpenBLAS handle.

    `is_enabled()` reflects the opt-in `-D BLAS` comptime flag and gates only
    the *automatic* routing in `Tensor.matmul`; `get()`/`is_available()` are
    flag-independent — the explicit `BLASHandleLite` API also loads the library
    if it exists (pre-refactor behaviour).
    """

    @staticmethod
    @always_inline
    def is_enabled() -> Bool:
        return BLAS_ENABLED != ""

    @staticmethod
    def get() -> Optional[ArcPointer[OwnedDLHandle]]:
        """Return the shared owning handle, or None if the library is unloadable."""
        try:
            comptime _CACHE = _Global["_TENMO_BLAS_HANDLE", _init_blas_handle]
            var ptr = _CACHE.get_or_create_ptr()
            # get_or_create_ptr() returns `Pointer[ArcPointer[OwnedDLHandle]]`;
            # first `[]` derefs the pointer, second `[]` derefs the ArcPointer.
            # An empty raw handle means dlopen failed earlier.
            if not ptr[][].borrow().__bool__():
                return None
            return Optional(ptr[])
        except e:
            return None

    @staticmethod
    def is_available() -> Bool:
        """True iff the library successfully loaded (flag-independent)."""
        return BLASCache.get() != None


# Raw GEMM
def blas_gemm[
    dtype: DType
](
    A: Pointer[Scalar[dtype], MutAnyOrigin],
    B: Pointer[Scalar[dtype], MutAnyOrigin],
    C: Pointer[Scalar[dtype], MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
    lda: Int,
    ldb: Int,
    ldc: Int,
    transpose_A: Bool = False,
    transpose_B: Bool = False,
    alpha: Scalar[dtype] = 1.0,
    beta: Scalar[dtype] = 0.0,
    sync: Bool = False,
):
    """Raw OpenBLAS GEMM: ``C = alpha * op(A) @ op(B) + beta * C`` (row-major).

    ``M``/``N``/``K`` and ``lda``/``ldb``/``ldc`` are the *logical* dims and
    leading dims after applying the optional operand transposes. CPU-only,
    contiguous, offset-0 storage is assumed (enforced upstream).
    ``sync`` is a no-op sink (CPU is always synchronous) kept so the
    matmul ``sync`` chain stays symmetric.
    """
    var arc_opt = BLASCache.get()
    if not arc_opt:
        panic("blas_gemm: BLAS not initialized")
    var arc = arc_opt.value()
    ref lib = arc[]

    var trans_A = CblasTrans if transpose_A else CblasNoTrans
    var trans_B = CblasTrans if transpose_B else CblasNoTrans

    comptime if dtype == DType.float32:
        var sgemm_fn: _DLCallable[CBLAS_SGEMM_FN, origin_of(lib)]
        try:
            sgemm_fn = lib.get_function[CBLAS_SGEMM_FN]("cblas_sgemm")
        except e:
            print(e)
            panic("blas_gemm: failed to load cblas_sgemm")
            return
        _ = sgemm_fn(
            CblasRowMajor,
            trans_A,
            trans_B,
            Int32(M),
            Int32(N),
            Int32(K),
            Float32(alpha),
            A.unsafe_bitcast[Float32](),
            Int32(lda),
            B.unsafe_bitcast[Float32](),
            Int32(ldb),
            Float32(beta),
            C.unsafe_bitcast[Float32](),
            Int32(ldc),
        )
    elif dtype == DType.float64:
        var dgemm_fn: _DLCallable[CBLAS_DGEMM_FN, origin_of(lib)]
        try:
            dgemm_fn = lib.get_function[CBLAS_DGEMM_FN]("cblas_dgemm")
        except e:
            print(e)
            panic("blas_gemm: failed to load cblas_dgemm")
            return
        _ = dgemm_fn(
            CblasRowMajor,
            trans_A,
            trans_B,
            Int32(M),
            Int32(N),
            Int32(K),
            Float64(alpha),
            A.unsafe_bitcast[Float64](),
            Int32(lda),
            B.unsafe_bitcast[Float64](),
            Int32(ldb),
            Float64(beta),
            C.unsafe_bitcast[Float64](),
            Int32(ldc),
        )
    else:
        panic("blas_gemm: unsupported dtype, must be float32 or float64")


# NDBuffer forward-only GEMM
def blas_matmul[
    dtype: DType
](
    A: NDBuffer[dtype],
    B: NDBuffer[dtype],
    transpose_A: Bool = False,
    transpose_B: Bool = False,
    sync: Bool = False,
) -> NDBuffer[dtype]:
    """Forward-only 2D GEMM on NDBuffers: ``C = A @ B`` (with optional
    operand transposes). Returns a fresh contiguous CPU result NDBuffer.

    No gradient/ancestor machinery — this is a pure forward GEMM; the caller
    (e.g. ``matmul.mojo``) owns autograd registration and gradient math.
    ``sync`` is a no-op sink (CPU-only path panics on GPU operands) kept
    so the matmul ``sync`` chain stays symmetric.

    Caller must have checked eligibility via ``BLASCache.is_available()`` and
    operands being contiguous/offset-0/CPU; we re-verify defensively.
    """
    var ok_dtype = False
    comptime if dtype == DType.float32:
        ok_dtype = True
    elif dtype == DType.float64:
        ok_dtype = True
    if not ok_dtype:
        panic("blas_matmul: unsupported dtype, must be float32 or float64")

    if not BLASCache.is_available():
        panic("blas_matmul: BLAS not available")

    var a = A
    var b = B

    var A_shape = a.shape
    var B_shape = b.shape
    if A_shape.rank() != 2 or B_shape.rank() != 2:
        panic("blas_matmul: operands must be rank-2")

    if not a.is_contiguous() or not b.is_contiguous():
        panic("blas_matmul: operands must be contiguous")
    if a.offset != 0 or b.offset != 0:
        panic("blas_matmul: operands must have zero offset")
    if a.is_on_gpu() or b.is_on_gpu():
        panic("blas_matmul: operands must be on CPU")

    var A_rows = A_shape[0]
    var A_cols = A_shape[1]
    var B_rows = B_shape[0]
    var B_cols = B_shape[1]

    var M: Int
    var N: Int
    var K: Int
    if transpose_A:
        M = A_cols
        K = A_rows
    else:
        M = A_rows
        K = A_cols

    var K_from_B: Int
    if transpose_B:
        K_from_B = B_cols
        N = B_rows
    else:
        K_from_B = B_rows
        N = B_cols

    if K != K_from_B:
        panic(
            "blas_matmul: inner dim mismatch: "
            + String(K)
            + " vs "
            + String(K_from_B)
        )

    var C = NDBuffer[dtype](Shape(M, N))

    blas_gemm[dtype](
        a.buffer.unsafe_ptr(),
        b.buffer.unsafe_ptr(),
        C.buffer.unsafe_ptr(),
        M,
        N,
        K,
        A_cols,
        B_cols,
        N,
        transpose_A,
        transpose_B,
        sync=sync,
    )

    return C^
