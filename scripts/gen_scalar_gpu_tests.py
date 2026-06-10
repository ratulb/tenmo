"""Generate tests/test_scalar_gpu.mojo — comprehensive GPU tests for scalar ops.

Covers both out-of-place (scalar_ops) and in-place (inplace_scalar_ops) paths
across contiguous, strided, transposed, permuted, and sliced layouts.

Usage:  python3 scripts/gen_scalar_gpu_tests.py
Output: tests/test_scalar_gpu.mojo
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(REPO, "tests", "test_scalar_gpu.mojo")

OOP_OPS = ["Add", "Subtract", "ReverseSubtract", "Multiply", "Divide",
           "ReverseDivide", "MAX", "MIN"]
IP_OPS  = ["Add", "Subtract", "ReverseSubtract", "Multiply", "Divide",
           "MAX", "MIN"]
SCALAR_MAP = {
    "Add":             ("5.0",     "Scalar[dtype](5.0)"),
    "Subtract":        ("3.0",     "Scalar[dtype](3.0)"),
    "ReverseSubtract": ("10.0",    "Scalar[dtype](10.0)"),
    "Multiply":        ("2.0",     "Scalar[dtype](2.0)"),
    "Divide":          ("4.0",     "Scalar[dtype](4.0)"),
    "ReverseDivide":   ("20.0",    "Scalar[dtype](20.0)"),
    "MAX":             ("5.0",     "Scalar[dtype](5.0)"),
    "MIN":             ("8.0",     "Scalar[dtype](8.0)"),
}

I = "    "  # one indent level

def section(lines, title):
    lines.append("")
    lines.append("# " + "=" * 77)
    lines.append(f"# {title}")
    lines.append("# " + "=" * 77)
    lines.append("")

def make_oop_test(name, op, shape_expr, scalar_str, dtype="DType.float32",
                  atol="1e-5", size_comment=""):
    lines = []
    if size_comment:
        lines.append(f"  # {size_comment}")
    lines.append(f"def {name}() raises:")
    lines.append(f"{I}comptime if has_accelerator():")
    lines.append(f"{I * 2}comptime dtype = {dtype}")
    lines.append(f"{I * 2}var gpu = GPU()")
    lines.append(f"{I * 2}var a = NDBuffer[dtype].{shape_expr}")
    lines.append(f"{I * 2}var expected = a.scalar_ops[{op}]({scalar_str})")
    lines.append(f"{I * 2}var result = a.to_gpu(gpu).scalar_ops[{op}]({scalar_str}, sync=True)")
    lines.append(f"{I * 2}assert_true(result.to_cpu().all_close[atol={atol}](expected))")
    return lines

def make_ip_test(name, op, shape_expr, scalar_str, dtype="DType.float32",
                 atol="1e-5", size_comment=""):
    lines = []
    if size_comment:
        lines.append(f"  # {size_comment}")
    lines.append(f"def {name}() raises:")
    lines.append(f"{I}comptime if has_accelerator():")
    lines.append(f"{I * 2}comptime dtype = {dtype}")
    lines.append(f"{I * 2}var gpu = GPU()")
    lines.append(f"{I * 2}var a = NDBuffer[dtype].{shape_expr}")
    lines.append(f"{I * 2}var expected = a.copy()")
    lines.append(f"{I * 2}expected.inplace_scalar_ops[{op}]({scalar_str})")
    lines.append(f"{I * 2}var a_gpu = a.to_gpu(gpu)")
    lines.append(f"{I * 2}a_gpu.inplace_scalar_ops[{op}]({scalar_str}, sync=True)")
    lines.append(f"{I * 2}assert_true(a_gpu.to_cpu().all_close[atol={atol}](expected))")
    return lines

def oop_strided_block(lines, name, op, ss, base_shape, reshape_to,
                      view_expr, expected_expr, dtype, atol):
    lines.append("")
    lines.append(f"def {name}() raises:")
    lines.append(f"{I}comptime if has_accelerator():")
    lines.append(f"{I * 2}comptime dtype = {dtype}")
    lines.append(f"{I * 2}var gpu = GPU()")
    lines.append(f"{I * 2}var a_base = NDBuffer[dtype].{base_shape}")
    lines.append(f"{I * 2}var a = {view_expr}")
    lines.append(f"{I * 2}var expected = {expected_expr}.scalar_ops[{op}]({ss})")
    lines.append(f"{I * 2}var result = a.to_gpu(gpu).scalar_ops[{op}]({ss}, sync=True)")
    lines.append(f"{I * 2}assert_true(result.to_cpu().all_close[atol={atol}](expected))")

def ip_strided_block(lines, name, op, ss, base_shape, reshape_to,
                     view_expr, expected_expr, dtype, atol):
    lines.append("")
    lines.append(f"def {name}() raises:")
    lines.append(f"{I}comptime if has_accelerator():")
    lines.append(f"{I * 2}comptime dtype = {dtype}")
    lines.append(f"{I * 2}var gpu = GPU()")
    lines.append(f"{I * 2}var a_base = NDBuffer[dtype].{base_shape}")
    lines.append(f"{I * 2}var a = {view_expr}")
    lines.append(f"{I * 2}var expected = a.contiguous()")
    lines.append(f"{I * 2}expected.inplace_scalar_ops[{op}]({ss})")
    lines.append(f"{I * 2}var a_gpu = a.to_gpu(gpu)")
    lines.append(f"{I * 2}a_gpu.inplace_scalar_ops[{op}]({ss}, sync=True)")
    lines.append(f"{I * 2}assert_true(a_gpu.to_cpu().all_close[atol={atol}](expected))")

def gen():
    all_lines = []
    all_lines.append("from tenmo import NDBuffer, Shape, IntArray")
    all_lines.append("from std.testing import assert_true, TestSuite")
    all_lines.append("from std.sys import has_accelerator")
    all_lines.append("from tenmo.device import GPU")
    all_lines.append("from tenmo.mnemonics import (")
    all_lines.append("    Add, Subtract, ReverseSubtract, Multiply, Divide, ReverseDivide, MAX, MIN, POW,")
    all_lines.append(")")
    all_lines.append("")

    # ===== A. OUT-OF-PLACE — CONTIGUOUS =====
    section(all_lines, "A. Out-of-place scalar_ops — contiguous input (flat SIMD kernel)")

    dims = [
        ("1d", "arange(1, 9)",         "1D, size=8"),
        ("2d", "arange(1, 25).reshape(Shape(5, 5))", "2D (5,5)"),
        ("3d", "arange(1, 61).reshape(Shape(3, 4, 5))", "3D (3,4,5)"),
        ("4d", "arange(1, 121).reshape(Shape(3, 2, 4, 5))", "4D (3,2,4,5)"),
    ]
    for dim_label, shape_expr, desc in dims:
        for op in OOP_OPS:
            sv, ss = SCALAR_MAP[op]
            name = f"test_oop_{dim_label}_{op.lower()}_gpu_scalar"
            all_lines += make_oop_test(name, op, shape_expr, ss, size_comment=desc)

    section(all_lines, "A5. Out-of-place contiguous — float64")
    for op in OOP_OPS:
        sv, ss = SCALAR_MAP[op]
        ss_f64 = ss.replace("[dtype]", "[DType.float64]").replace("Scalar[dtype]", "Scalar[DType.float64]")
        name = f"test_oop_1d_{op.lower()}_f64_gpu_scalar"
        all_lines += make_oop_test(name, op, "arange(1, 9)", ss_f64,
                                   dtype="DType.float64", size_comment="1D f64, size=8")

    # ===== B. OUT-OF-PLACE — STRIDED =====
    section(all_lines, "B. Out-of-place scalar_ops — strided input (strided kernel)")

    for op in OOP_OPS:
        sv, ss = SCALAR_MAP[op]
        oop_strided_block(all_lines,
            f"test_oop_2d_{op.lower()}_transposed_gpu_scalar",
            op, ss,
            "arange(1, 25).reshape(Shape(6, 4))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in OOP_OPS:
        sv, ss = SCALAR_MAP[op]
        oop_strided_block(all_lines,
            f"test_oop_3d_{op.lower()}_transposed_gpu_scalar",
            op, ss,
            "arange(1, 61).reshape(Shape(3, 4, 5))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in OOP_OPS:
        sv, ss = SCALAR_MAP[op]
        oop_strided_block(all_lines,
            f"test_oop_4d_{op.lower()}_transposed_gpu_scalar",
            op, ss,
            "arange(1, 121).reshape(Shape(3, 2, 4, 5))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in OOP_OPS:
        sv, ss = SCALAR_MAP[op]
        oop_strided_block(all_lines,
            f"test_oop_3d_{op.lower()}_permuted_gpu_scalar",
            op, ss,
            "arange(1, 61).reshape(Shape(3, 4, 5))", "",
            "a_base.permute(IntArray(2, 0, 1))",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in OOP_OPS:
        sv, ss = SCALAR_MAP[op]
        oop_strided_block(all_lines,
            f"test_oop_2d_{op.lower()}_sliced_gpu_scalar",
            op, ss,
            "arange(1, 41).reshape(Shape(5, 8))", "",
            "a_base[1:4, 2:6]",
            "a.contiguous()", "DType.float32", "1e-5")

    # ===== C. POW =====
    section(all_lines, "C. POW — dedicated dtype-specific kernel")

    pow_examples = [
        ("2",   "Scalar[dtype](2)"),
        ("3",   "Scalar[dtype](3)"),
        ("0_5", "Scalar[dtype](0.5)"),
        ("0",   "Scalar[dtype](0)"),
        ("neg1","Scalar[dtype](-1)"),
    ]
    for suffix, exponent_str in pow_examples:
        for dtype, dt_label in [("DType.float32", "f32"), ("DType.float64", "f64")]:
            ss = exponent_str.replace("[dtype]", f"[{dtype}]")
            name = f"test_oop_1d_pow_{suffix}_{dt_label}_gpu_scalar"
            all_lines += make_oop_test(name, "POW", "arange(1, 9)", ss,
                                       dtype=dtype, atol="1e-4",
                                       size_comment=f"pow {exponent_str}")

    # ===== D. INPLACE — CONTIGUOUS =====
    section(all_lines, "D. In-place inplace_scalar_ops — contiguous input (flat SIMD kernel)")

    for dim_label, shape_expr, desc in dims:
        for op in IP_OPS:
            sv, ss = SCALAR_MAP[op]
            name = f"test_ip_{dim_label}_{op.lower()}_gpu_scalar"
            all_lines += make_ip_test(name, op, shape_expr, ss, size_comment=desc)

    section(all_lines, "D5. In-place contiguous — float64")
    for op in IP_OPS:
        sv, ss = SCALAR_MAP[op]
        ss_f64 = ss.replace("[dtype]", "[DType.float64]").replace("Scalar[dtype]", "Scalar[DType.float64]")
        name = f"test_ip_1d_{op.lower()}_f64_gpu_scalar"
        all_lines += make_ip_test(name, op, "arange(1, 9)", ss_f64,
                                  dtype="DType.float64", size_comment="1D f64, size=8")

    # ===== E. INPLACE — STRIDED =====
    section(all_lines, "E. In-place inplace_scalar_ops — strided input (strided kernel)")

    for op in IP_OPS:
        sv, ss = SCALAR_MAP[op]
        ip_strided_block(all_lines,
            f"test_ip_2d_{op.lower()}_transposed_gpu_scalar",
            op, ss,
            "arange(1, 25).reshape(Shape(6, 4))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in IP_OPS:
        sv, ss = SCALAR_MAP[op]
        ip_strided_block(all_lines,
            f"test_ip_3d_{op.lower()}_transposed_gpu_scalar",
            op, ss,
            "arange(1, 61).reshape(Shape(3, 4, 5))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in IP_OPS:
        sv, ss = SCALAR_MAP[op]
        ip_strided_block(all_lines,
            f"test_ip_4d_{op.lower()}_transposed_gpu_scalar",
            op, ss,
            "arange(1, 121).reshape(Shape(3, 2, 4, 5))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in IP_OPS:
        sv, ss = SCALAR_MAP[op]
        ip_strided_block(all_lines,
            f"test_ip_3d_{op.lower()}_permuted_gpu_scalar",
            op, ss,
            "arange(1, 61).reshape(Shape(3, 4, 5))", "",
            "a_base.permute(IntArray(2, 0, 1))",
            "a.contiguous()", "DType.float32", "1e-5")

    for op in IP_OPS:
        sv, ss = SCALAR_MAP[op]
        ip_strided_block(all_lines,
            f"test_ip_2d_{op.lower()}_sliced_gpu_scalar",
            op, ss,
            "arange(1, 41).reshape(Shape(5, 8))", "",
            "a_base[1:4, 2:6]",
            "a.contiguous()", "DType.float32", "1e-5")

    # ===== F. EDGE CASES — OUT-OF-PLACE =====
    section(all_lines, "F. Edge cases — out-of-place")

    # F1-F7 use make_oop_test helper
    edge_oop = [
        ("tail7", "arange(1, 8)", "tail size 7"),
        ("1elem", "arange(1, 2)", "single element"),
    ]
    for suffix, shape, desc in edge_oop:
        for op in OOP_OPS:
            sv, ss = SCALAR_MAP[op]
            name = f"test_oop_{suffix}_{op.lower()}_gpu_scalar"
            all_lines += make_oop_test(name, op, shape, ss, size_comment=desc)

    # Medium/big sizes
    for sz_label, sz_val in [("100", "101"), ("1000", "1001"), ("7777", "7778")]:
        for op in ["Add", "Multiply"]:
            sv, ss = SCALAR_MAP[op]
            name = f"test_oop_size{sz_label}_{op.lower()}_gpu_scalar"
            all_lines += make_oop_test(name, op, f"arange(1, {sz_val})", ss,
                                       size_comment=f"{sz_label} elements")

    # Identity
    for op, scalar_val in [("Add", "Scalar[dtype](0)"), ("Multiply", "Scalar[dtype](1.0)")]:
        name = f"test_oop_identity_{op.lower()}_gpu_scalar"
        all_lines += make_oop_test(name, op, "arange(1, 25).reshape(Shape(5, 5))",
                                   scalar_val, size_comment="identity op")

    # Negative
    for op in ["Add", "Multiply", "MAX", "MIN"]:
        sv, ss = SCALAR_MAP[op]
        neg_s = (ss.replace("Scalar[dtype](5.0)", "Scalar[dtype](-3.0)")
                   .replace("Scalar[dtype](8.0)", "Scalar[dtype](-3.0)")
                   .replace("Scalar[dtype](2.0)", "Scalar[dtype](-2.0)")
                   .replace("Scalar[dtype](4.0)", "Scalar[dtype](-4.0)")
                   .replace("Scalar[dtype](10.0)", "Scalar[dtype](-10.0)")
                   .replace("Scalar[dtype](20.0)", "Scalar[dtype](-20.0)"))
        name = f"test_oop_neg_{op.lower()}_gpu_scalar"
        all_lines += make_oop_test(name, op, "arange(1, 9)", neg_s,
                                   size_comment="negative scalar")

    # f64 strided 2D transposed for OOP
    section(all_lines, "F8. f64 strided — 2D transposed out-of-place")
    for op in ["Add", "Multiply"]:
        sv, ss = SCALAR_MAP[op]
        ss_f64 = ss.replace("[dtype]", "[DType.float64]").replace("Scalar[dtype]", "Scalar[DType.float64]")
        oop_strided_block(all_lines,
            f"test_oop_2d_{op.lower()}_transposed_f64_gpu_scalar",
            op, ss_f64,
            "arange(1, 25).reshape(Shape(6, 4))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float64", "1e-10")

    # ===== G. EDGE CASES — INPLACE =====
    section(all_lines, "G. Edge cases — in-place")

    for suffix, shape, desc in edge_oop:
        for op in IP_OPS:
            sv, ss = SCALAR_MAP[op]
            name = f"test_ip_{suffix}_{op.lower()}_gpu_scalar"
            all_lines += make_ip_test(name, op, shape, ss, size_comment=desc)

    for sz_label, sz_val in [("100", "101"), ("1000", "1001"), ("7777", "7778")]:
        for op in ["Add", "Multiply"]:
            sv, ss = SCALAR_MAP[op]
            name = f"test_ip_size{sz_label}_{op.lower()}_gpu_scalar"
            all_lines += make_ip_test(name, op, f"arange(1, {sz_val})", ss,
                                      size_comment=f"{sz_label} elements")

    for op, scalar_val in [("Add", "Scalar[dtype](0)"), ("Multiply", "Scalar[dtype](1.0)")]:
        name = f"test_ip_identity_{op.lower()}_gpu_scalar"
        all_lines += make_ip_test(name, op, "arange(1, 25).reshape(Shape(5, 5))",
                                  scalar_val, size_comment="identity op")

    for op in ["Add", "Multiply", "MAX", "MIN"]:
        sv, ss = SCALAR_MAP[op]
        neg_s = (ss.replace("Scalar[dtype](5.0)", "Scalar[dtype](-3.0)")
                   .replace("Scalar[dtype](8.0)", "Scalar[dtype](-3.0)")
                   .replace("Scalar[dtype](2.0)", "Scalar[dtype](-2.0)")
                   .replace("Scalar[dtype](4.0)", "Scalar[dtype](-4.0)")
                   .replace("Scalar[dtype](10.0)", "Scalar[dtype](-10.0)")
                   .replace("Scalar[dtype](20.0)", "Scalar[dtype](-20.0)"))
        name = f"test_ip_neg_{op.lower()}_gpu_scalar"
        all_lines += make_ip_test(name, op, "arange(1, 9)", neg_s,
                                  size_comment="negative scalar")

    section(all_lines, "G8. f64 strided — 2D transposed in-place")
    for op in ["Add", "Multiply"]:
        sv, ss = SCALAR_MAP[op]
        ss_f64 = ss.replace("[dtype]", "[DType.float64]").replace("Scalar[dtype]", "Scalar[DType.float64]")
        ip_strided_block(all_lines,
            f"test_ip_2d_{op.lower()}_transposed_f64_gpu_scalar",
            op, ss_f64,
            "arange(1, 25).reshape(Shape(6, 4))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float64", "1e-10")

    # ===== H. f64 3D transposed =====
    section(all_lines, "H. f64 strided 3D transposed")
    for op in ["Add", "Multiply"]:
        sv, ss = SCALAR_MAP[op]
        ss_f64 = ss.replace("[dtype]", "[DType.float64]").replace("Scalar[dtype]", "Scalar[DType.float64]")
        oop_strided_block(all_lines,
            f"test_oop_3d_{op.lower()}_transposed_f64_gpu_scalar",
            op, ss_f64,
            "arange(1, 61).reshape(Shape(3, 4, 5))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float64", "1e-10")
        ip_strided_block(all_lines,
            f"test_ip_3d_{op.lower()}_transposed_f64_gpu_scalar",
            op, ss_f64,
            "arange(1, 61).reshape(Shape(3, 4, 5))", "",
            "a_base.transpose()",
            "a.contiguous()", "DType.float64", "1e-10")

    # ── Main ──
    all_lines.append("")
    all_lines.append("")
    all_lines.append("def main() raises:")
    all_lines.append(f"{I}TestSuite.discover_tests[__functions_in_module()]().run()")

    content = "\n".join(all_lines) + "\n"
    with open(OUTPUT, "w") as f:
        f.write(content)
    print(f"Generated {OUTPUT} ({sum(1 for l in all_lines if l.startswith('def test_'))} tests, {len(all_lines)} lines)")

if __name__ == "__main__":
    gen()
