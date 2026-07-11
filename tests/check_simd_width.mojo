from std.sys import simd_width_of, has_accelerator
from std.gpu import thread_idx, block_idx, block_dim

comptime if has_accelerator():
    from std.gpu.host import DeviceContext

# =============================================================================
# Verify what simd_width_of[dtype]() returns on CPU vs GPU compilation targets.
#
# On CPU: returns the SIMD vector width for the host (e.g. 8 for AVX2 float32).
# On GPU: returns the SIMD width for the GPU target.
#
# This affects:
#   - CHUNK_SIZE = simd_vectors_per_thread * simd_width
#   - In GPU kernels: A.load[width=simd_width](i) loads that many elements/thread
#   - launch_config block count calculations
# =============================================================================


# ── Host-side print of compile-time SIMD widths ─────────────────────────


def print_cpu_simd_widths():
    print("── CPU / Host Compile-Time SIMD Widths ──")
    print("has_accelerator():   ", has_accelerator())
    print()

    print("DType.float32:       ", simd_width_of[DType.float32]())
    print("DType.float64:       ", simd_width_of[DType.float64]())
    print("DType.float16:       ", simd_width_of[DType.float16]())
    print("DType.bfloat16:      ", simd_width_of[DType.bfloat16]())
    print("DType.int32:         ", simd_width_of[DType.int32]())
    print("DType.int64:         ", simd_width_of[DType.int64]())
    print("DType.int8:          ", simd_width_of[DType.int8]())
    print("DType.uint8:         ", simd_width_of[DType.uint8]())
    print()

    # Compute derived values used in kernel launch_config
    print()
    print("── Derived values ──")

    comptime sw_f32 = simd_width_of[DType.float32]()
    print(
        "float32: simd_width=",
        sw_f32,
        ", CHUNK_SIZE(2*sw²)=",
        2 * sw_f32 * sw_f32,
    )

    comptime sw_f64 = simd_width_of[DType.float64]()
    print(
        "float64: simd_width=",
        sw_f64,
        ", CHUNK_SIZE(2*sw²)=",
        2 * sw_f64 * sw_f64,
    )

    comptime sw_f16 = simd_width_of[DType.float16]()
    print(
        "float16: simd_width=",
        sw_f16,
        ", CHUNK_SIZE(2*sw²)=",
        2 * sw_f16 * sw_f16,
    )

    comptime sw_bf16 = simd_width_of[DType.bfloat16]()
    print(
        "bfloat16: simd_width=",
        sw_bf16,
        ", CHUNK_SIZE(2*sw²)=",
        2 * sw_bf16 * sw_bf16,
    )

    comptime sw_i32 = simd_width_of[DType.int32]()
    print(
        "int32: simd_width=",
        sw_i32,
        ", CHUNK_SIZE(2*sw²)=",
        2 * sw_i32 * sw_i32,
    )


# ── GPU kernel to print SIMD widths from inside the kernel ──────────────
# This compiles only when has_accelerator() is true.


def check_simd_width_kernel():
    if thread_idx.x == 0 and block_idx.x == 0:
        comptime sw_f32 = simd_width_of[DType.float32]()
        comptime sw_f64 = simd_width_of[DType.float64]()
        comptime sw_f16 = simd_width_of[DType.float16]()
        comptime sw_bf16 = simd_width_of[DType.bfloat16]()
        print("[GPU KERNEL] f32 simd_width:", sw_f32)
        print("[GPU KERNEL] f64 simd_width:", sw_f64)
        print("[GPU KERNEL] f16 simd_width:", sw_f16)
        print("[GPU KERNEL] bf16 simd_width:", sw_bf16)


def launch_gpu_check() raises:
    comptime if has_accelerator():
        print("── GPU Kernel Check ──")
        var device_context = DeviceContext()
        var compiled = device_context.compile_function[
            check_simd_width_kernel,
        ]()
        device_context.enqueue_function(
            compiled,
            grid_dim=1,
            block_dim=1,
        )
        device_context.synchronize()
        print()


def main() raises:
    # Part 1: CPU-side SIMD widths
    print_cpu_simd_widths()

    # Part 2: GPU-side SIMD widths (only if GPU is available)
    launch_gpu_check()
