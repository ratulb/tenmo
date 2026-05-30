"""Generate tests/test_gpu_all.mojo by extracting all GPU-guarded test functions.

Scans every tests/test_*.mojo file, extracts:
- GPU-guarded test functions (body starts with `comptime if has_accelerator():`)
- Per-file helper functions (non-test fn/def) used by those tests
- Deduplicated imports
- Global `comptime dtype = DType.float32` (all source files use float32)

Groups output by source file so helpers are available to the test functions
that depend on them.
"""

import argparse
import glob
import os
import re
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO, "tests")
OUTPUT = os.path.join(TESTS_DIR, "test_gpu_all.mojo")

NO_GPU = {
    "test_ANCESTRY.mojo", "test_autograd.mojo", "test_broadcaster.mojo",
    "test_buffers.mojo", "test_builders.mojo", "test_cholesky.mojo",
    "test_clamp.mojo", "test_dataloader.mojo", "test_detach.mojo",
    "test_embedding_fwd.mojo", "test_fill.mojo", "test_initialization.mojo",
    "test_inter_matmul.mojo", "test_matmul.mojo", "test_net.mojo",
    "test_numpy_interop.mojo", "test_pad.mojo", "test_pooling.mojo",
    "test_repeat.mojo", "test_slice.mojo", "test_strides.mojo",
    "test_tail.mojo", "test_tensor.mojo", "test_tile.mojo",
    "test_topk.mojo",     "test_view_slice.mojo",
    "test_views.mojo", "test_where.mojo",
}

ALLOWED_MODULES = {"tenmo.", "std.", "python."}
ALLOWED_MODULES_EXACT = {"tenmo", "std"}
BLOCKED_MODULES = {"tensors", "layers", "bpe", "python"}


def _module_ok(imp: str) -> bool:
    m = re.match(r"^(from|import)\s+(\S+)", imp)
    if not m:
        return False
    module = m.group(2).strip()
    module = module.split("#")[0].strip()
    if module in ALLOWED_MODULES_EXACT:
        return True
    if any(module.startswith(p) for p in ALLOWED_MODULES):
        return True
    if module in BLOCKED_MODULES:
        return False
    if "/" in module or "\\" in module:
        return False
    return False


# ── Collect imports ──────────────────────────────────────────────────────────


def collect_imports(source: str) -> list:
    imports = []
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            imp_lines = [stripped]
            if "(" in stripped and ")" not in stripped:
                j = i + 1
                while j < len(lines):
                    ls = lines[j].strip()
                    imp_lines.append(ls)
                    if ")" in ls:
                        break
                    j += 1
                i = j
            imp = " ".join(imp_lines)
            imp = re.sub(r"\s*#.*$", "", imp).strip()
            if imp and _module_ok(imp):
                imports.append(imp)
        i += 1
    return imports


# ── Helper function extraction ──────────────────────────────────────────────


def _is_test_func(line: str) -> bool:
    return bool(re.match(r"^(?:fn|def)\s+test_\w+", line))


def _is_main(line: str) -> bool:
    return bool(re.match(r"^(?:fn|def)\s+main\b", line))


def _collect_function(lines: list, start: int) -> tuple:
    """Collect a complete function def including multi-line signature and body.

    Handles multi-line generic param blocks [dtype, ...], multi-line paren args,
    return type annotations, and decorator lines.

    Returns (func_text, end_idx_exclusive).
    """
    line = lines[start]
    indent = len(line) - len(line.lstrip())
    func_lines = [line]

    # Phase 1: collect the full signature.
    # Track generic `[...]` and argument `(...)` depth.
    # Signature ends when both depths are 0 and we hit a line ending with `:`.
    gen_depth = line.count("[") - line.count("]")
    paren_depth = line.count("(") - line.count(")")
    i = start + 1

    # If the def line already ends with `:` at depth 0, the signature is single-line.
    sig_done = line.rstrip().endswith(":") and gen_depth == 0 and paren_depth == 0
    if not sig_done:
        while i < len(lines):
            sig_line = lines[i]
            func_lines.append(sig_line)

            gen_depth += sig_line.count("[") - sig_line.count("]")
            paren_depth += sig_line.count("(") - sig_line.count(")")

            if gen_depth == 0 and paren_depth == 0 and sig_line.rstrip().endswith(":"):
                i += 1
                break
            i += 1

    # Phase 2: collect the body (lines indented deeper than `indent`).
    while i < len(lines):
        nxt = lines[i]
        if nxt.strip() == "":
            func_lines.append(nxt)
            i += 1
            continue
        leading = len(nxt) - len(nxt.lstrip())
        if leading <= indent:
            break
        func_lines.append(nxt)
        i += 1

    return "\n".join(func_lines), i


def has_gpu_guard(first_body_line: str) -> bool:
    return first_body_line.strip().startswith("comptime if has_accelerator()")


def extract_helpers(source: str, file_imports: list) -> list:
    """Return [(func_name, func_text), ...] for all non-test, non-main fn/def.

    Collects helpers that appear anywhere in the source file (before, between,
    or after test functions). Includes @decorator lines if present.
    Skips functions whose names conflict with the file's own imports
    (import takes precedence).
    """
    # Collect names imported by this file
    imported_names = set()
    for imp in file_imports:
        if "import " in imp:
            after = imp.split("import ", 1)[-1]
            # Handle "from X import A, B, C" and "import A, B, C"
            names = re.findall(r'\b\w+\b', after)
            for n in names:
                if n not in ("from",):
                    imported_names.add(n)

    lines = source.split("\n")
    helpers = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip test functions and main
        if _is_test_func(line) or _is_main(line):
            i += 1
            continue

        # Check for @decorator before fn/def
        if line.strip().startswith("@"):
            if i + 1 < len(lines) and re.match(r"^(?:fn|def)\s+", lines[i + 1]):
                decorator = lines[i]
                name = _get_function_name(lines[i + 1])
                text, end = _collect_function(lines, i + 1)
                if name not in imported_names:
                    helpers.append((name, decorator + "\n" + text))
                i = end
                continue
            i += 1
            continue

        if re.match(r"^(?:fn|def)\s+", line):
            name = _get_function_name(line)
            text, end = _collect_function(lines, i)
            if name not in imported_names:
                helpers.append((name, text))
            i = end
        else:
            i += 1

    return helpers


def extract_aliases(source: str) -> list:
    """Return list of module-level alias/comptime lines.

    Only captures lines before the first test function.
    Lines like `comptime tol = Float32(1e-4)` or `alias dtype = DType.float32`.
    Excludes `comptime dtype` lines (already provided globally).
    """
    lines = source.split("\n")
    first_test = _find_first_test_idx(lines)
    aliases = []
    for i, line in enumerate(lines[:first_test]):
        stripped = line.strip()
        # Match `comptime <name> = ...` or `alias <name> = ...`
        # But NOT `comptime if` or `comptime else` or `comptime assert`
        m = re.match(r"^\s*(?:comptime|alias)\s+(\w+)\s*=", stripped)
        if m:
            name = m.group(1)
            if name not in ("dtype", "tol"):
                aliases.append(line.rstrip())
    return aliases


# ── GPU test function extraction ────────────────────────────────────────────


def extract_gpu_functions(source: str):
    """Yield (func_name, func_text) for every GPU-guarded test function."""
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)(?:fn|def)\s+(test_\w+)\(\)\sraises:", line)
        if not m:
            i += 1
            continue
        indent = m.group(1)
        name = m.group(2)
        base_indent = len(indent)

        # Find first non-empty, non-comment, non-docstring body line
        first_body = None
        for j in range(i + 1, len(lines)):
            stripped = lines[j].strip()
            if stripped == "" or stripped.startswith("#") or stripped.startswith('"""'):
                continue
            first_body = lines[j]
            break

        if first_body is None or not has_gpu_guard(first_body):
            i += 1
            continue

        text, end = _collect_function(lines, i)
        i = end
        yield name, text


# ── File structure helpers ──────────────────────────────────────────────────


def _find_first_test_idx(lines: list) -> int:
    for i, line in enumerate(lines):
        if re.match(r"^(fn|def)\s+test_\w+", line):
            return i
    return len(lines)


def _find_last_test_idx(lines: list) -> int:
    idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^(fn|def)\s+test_\w+", line):
            idx = i
    return idx


def _get_function_name(line: str) -> str:
    m = re.match(r"^(?:fn|def)\s+(\w+)", line)
    return m.group(1) if m else ""


# ── Standard preamble (always included) ─────────────────────────────────────

# Helpers to always skip (problematic cross-file dependencies)
SKIP_HELPERS = {"run_all_minmax_tests", "run_all_matrix_vector_tests", "run_all_minmax_tests"}

STD_PREAMBLE = [
    "from std.testing import assert_true, assert_false, assert_equal, assert_almost_equal, TestSuite",
    "from std.sys import has_accelerator",
    "",
    "comptime dtype = DType.float32",
    "comptime tol = Float32(1e-4)",
]


# ── Import dedup ─────────────────────────────────────────────────────────


# Names built into Mojo — importing them from std.math is redundant.
# Only bare names (not aliased with `as`) are filtered.
BUILTIN_NAMES = {"abs", "max", "min", "round", "pow"}


def merge_imports(imports: OrderedDict) -> list:
    """Merge duplicate 'from X import ...' lines by module.

    Strips built-in names (e.g. bare `abs`, `max`) from `std.math` imports
    since those are available in Mojo without import.

    Input: OrderedDict of import strings (keys are the import lines).
    Output: list of deduplicated import strings.
    """
    from_groups = OrderedDict()
    bare_seen = OrderedDict()

    for imp in imports:
        imp_stripped = imp.strip()
        if imp_stripped.startswith("from "):
            m = re.match(r"^from\s+(\S+)\s+import\s+(.+)$", imp_stripped)
            if m:
                module = m.group(1)
                names_part = m.group(2).strip()
                names_part = names_part.strip("()")
                names = [n.strip().strip("()") for n in names_part.split(",") if n.strip().strip("()")]
                has_star = "*" in names
                if module not in from_groups:
                    from_groups[module] = {"has_star": False, "names": set()}
                if has_star:
                    from_groups[module]["has_star"] = True
                else:
                    for n in names:
                        if n == "*":
                            continue
                        # Strip bare built-in names (keep aliased like "abs as scalar_abs")
                        if module.startswith("std.math") and n in BUILTIN_NAMES:
                            continue
                        from_groups[module]["names"].add(n)
        elif imp_stripped.startswith("import "):
            bare_seen[imp_stripped] = None

    result = []
    for module, info in from_groups.items():
        if info["has_star"]:
            result.append(f"from {module} import *")
        elif info["names"]:
            sorted_names = sorted(info["names"])
            result.append(f"from {module} import {', '.join(sorted_names)}")

    result.extend(bare_seen.keys())
    return result


# ── Chunked output writer ─────────────────────────────────────────────────


def write_chunk(
    output_path: str,
    chunk_label: str,
    chunk_file_data: OrderedDict,
    chunk_imports: OrderedDict,
):
    """Write a single chunk file with imports + subset of file_data."""
    std_norm = set()
    for imp in STD_PREAMBLE:
        if imp:
            std_norm.add(re.sub(r"\s+", " ", imp))

    with open(output_path, "w") as f:
        f.write('"""Auto-generated GPU test suite — %s.\n' % chunk_label)
        f.write("Generated by scripts/generate_gpu_test_suite.py\n")
        total_funcs = sum(len(d["functions"]) for d in chunk_file_data.values())
        f.write(f"Contains {total_funcs} GPU-guarded test functions from {len(chunk_file_data)} files.\n")
        f.write('"""\n\n')

        for line in STD_PREAMBLE:
            f.write(line + "\n")
        f.write("\n")

        # Parse STD_PREAMBLE names per module (e.g. std.testing → {assert_true, ...})
        preamble_names = {}
        for line in STD_PREAMBLE:
            if line.startswith("from "):
                m = re.match(r"^from\s+(\S+)\s+import\s+(.+)$", line)
                if m:
                    mod = m.group(1)
                    names = [n.strip() for n in m.group(2).split(",")]
                    preamble_names[mod] = set(n.strip() for n in names)

        # Collect struct names in this chunk (still unprefixed — can conflict with imports)
        chunk_struct_names = set()
        for data in chunk_file_data.values():
            for sname, _ in data.get("structs", []):
                chunk_struct_names.add(sname)

        # Write imports, merging by module and skipping names already in
        # preamble or names that conflict with local struct defs
        merged = merge_imports(chunk_imports)
        for imp in merged:
            m = re.match(r"^from\s+(\S+)\s+import\s+(.+)$", imp)
            if m:
                module = m.group(1)
                names_part = m.group(2).strip()
                if module in preamble_names:
                    # Only emit names not already provided by STD_PREAMBLE
                    names = [n.strip() for n in names_part.split(",")]
                    remaining = [n for n in names if n not in preamble_names[module]]
                    if remaining:
                        f.write(f"from {module} import {', '.join(remaining)}\n")
                    continue
            if "import " in imp and " as " not in imp:
                after = imp.split("import ", 1)[-1]
                imported_names = re.findall(r'\b\w+\b', after)
                if any(n in chunk_struct_names for n in imported_names):
                    continue
            f.write(f"{imp}\n")

        f.write("\n\n")

        dup_names = set()
        seen_helpers = set()
        seen_aliases = set()
        dup_count = 0
        for bname, data in chunk_file_data.items():
            f.write(f"# === From {bname} ===\n\n")

            if data["aliases"]:
                first = True
                for a in data["aliases"]:
                    if a not in seen_aliases:
                        seen_aliases.add(a)
                        f.write(a + "\n")
                        first = False
                if not first:
                    f.write("\n")

            for hname, htext in data["helpers"]:
                if hname in SKIP_HELPERS:
                    continue
                if hname not in seen_helpers:
                    seen_helpers.add(hname)
                    f.write(htext)
                    f.write("\n\n")

            for func_name, func_text in data["functions"]:
                if func_name in dup_names:
                    dup_count += 1
                    continue
                dup_names.add(func_name)
                f.write(func_text)
                f.write("\n\n")

        f.write("def main() raises:\n")
        f.write("    TestSuite.discover_tests[__functions_in_module()]().run()\n")

    print(f"Wrote {output_path}")
    print(f"  Source files: {len(chunk_file_data)}")
    print(f"  Functions:    {total_funcs}")
    print(f"  Imports:      {len(chunk_imports)}")
    if dup_count:
        print(f"  Duplicates:   {dup_count}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated GPU test suite")
    parser.add_argument(
        "--chunks", type=int, default=1,
        help="Split output into N files (default: 1 = monolithic)",
    )
    args = parser.parse_args()
    num_chunks = max(1, args.chunks)

    file_data = OrderedDict()

    pattern = os.path.join(TESTS_DIR, "test_*.mojo")
    for filepath in sorted(glob.glob(pattern)):
        basename = os.path.basename(filepath)
        if basename in NO_GPU:
            continue
        if "_all" in basename:
            continue

        with open(filepath) as f:
            source = f.read()

        imports = collect_imports(source)
        aliases = extract_aliases(source)
        helpers = extract_helpers(source, imports)
        funcs = list(extract_gpu_functions(source))

        if not funcs:
            continue

        file_data[basename] = {
            "imports": imports,
            "aliases": aliases,
            "helpers": helpers,
            "functions": funcs,
        }

    # Collect all helper names for import conflict detection
    all_helper_names = set()
    for data in file_data.values():
        for hname, _ in data["helpers"]:
            all_helper_names.add(hname)

    # ── Distribute across chunks (balanced by test count) ─────────────────
    # Greedy bin-packing: sort files descending by test count, assign each
    # to the chunk with the fewest tests so far.

    if num_chunks > 1:
        file_counts = sorted(
            [(f, len(d["functions"])) for f, d in file_data.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        bins = [[] for _ in range(num_chunks)]
        bin_counts = [0] * num_chunks

        for fname, count in file_counts:
            best = min(range(num_chunks), key=lambda i: bin_counts[i])
            bins[best].append(fname)
            bin_counts[best] += count

        for chunk_id in range(1, num_chunks + 1):
            chunk_files = bins[chunk_id - 1]
            chunk_data = OrderedDict((f, file_data[f]) for f in chunk_files)
            chunk_imports = OrderedDict()
            for data in chunk_data.values():
                for imp in data["imports"]:
                    chunk_imports[imp] = None
            output_path = os.path.join(
                TESTS_DIR, f"test_gpu_all_{chunk_id}.mojo"
            )
            write_chunk(
                output_path=output_path,
                chunk_label=f"chunk {chunk_id}/{num_chunks}",
                chunk_file_data=chunk_data,
                chunk_imports=chunk_imports,
            )

        for i, count in enumerate(bin_counts):
            print(f"  Chunk {i+1}: {count} tests, {len(bins[i])} files")
    else:
        all_imports = OrderedDict()
        for data in file_data.values():
            for imp in data["imports"]:
                all_imports[imp] = None
        write_chunk(
            output_path=OUTPUT,
            chunk_label="monolithic",
            chunk_file_data=file_data,
            chunk_imports=all_imports,
        )


if __name__ == "__main__":
    main()
