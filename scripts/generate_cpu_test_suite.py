"""Generate tests/test_cpu_all.mojo by extracting all non-GPU-guarded test functions.

Scans every tests/test_*.mojo file, extracts:
- CPU test functions (body does NOT start with `comptime if has_accelerator():`)
- Per-file helper functions (non-test fn/def) used by those tests
- Per-file struct definitions needed by those tests
- Per-file comptime aliases (renamed with file prefix to avoid collisions)
- Deduplicated imports
- Global `comptime dtype = DType.float32` (all source files use float32)

Groups output by source file so helpers are available to the test functions
that depend on them. Renames test functions with a file prefix to avoid
name collisions across files. Also renames test function calls within
helper function bodies.
"""

import argparse
import glob
import os
import re
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO, "tests")
OUTPUT = os.path.join(TESTS_DIR, "test_cpu_all.mojo")

# Files to skip entirely
SKIP_FILES = {
    "test_gpu_all.mojo",
    "test_mnist.mojo",       # old import pattern, 0 tests
    "test_synthetic_mnist.mojo",  # 0 tests
    "test_relu.mojo",        # from tenmo.relu import ReLU causes ambiguity
                             # when tenmo.net is loaded via other imports
    "test_data.mojo",        # from bpe import BasicTokenizer (blocked module)
                             # + Int(ptr) pre-existing bug in 9+ locations
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


# ── Helper, struct, alias extraction ─────────────────────────────────────────


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

    gen_depth = line.count("[") - line.count("]")
    paren_depth = line.count("(") - line.count(")")
    i = start + 1

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
    """Return [(func_name, func_text), ...] for all non-test, non-main fn/def."""
    imported_names = set()
    for imp in file_imports:
        if "import " in imp:
            after = imp.split("import ", 1)[-1]
            names = re.findall(r'\b\w+\b', after)
            for n in names:
                if n not in ("from",):
                    imported_names.add(n)

    lines = source.split("\n")
    helpers = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if _is_test_func(line) or _is_main(line):
            i += 1
            continue

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


def extract_structs(source: str, file_imports: list) -> list:
    """Return [(struct_name, struct_text), ...] for all struct definitions.

    Also handles @decorator lines before struct.
    Skips structs whose names conflict with imports.
    """
    imported_names = set()
    for imp in file_imports:
        if "import " in imp:
            after = imp.split("import ", 1)[-1]
            names = re.findall(r'\b\w+\b', after)
            for n in names:
                if n not in ("from",):
                    imported_names.add(n)

    lines = source.split("\n")
    structs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("@"):
            if i + 1 < len(lines) and re.match(r"^(?:struct|fn|def)\s+", lines[i + 1]):
                decorator = lines[i]
                name = _get_function_name(lines[i + 1])
                text, end = _collect_function(lines, i + 1)
                if re.match(r"^struct\s+", lines[i + 1]) and name not in imported_names:
                    structs.append((name, decorator + "\n" + text))
                i = end
                continue
            i += 1
            continue

        if re.match(r"^struct\s+", line):
            name = _get_function_name(line)
            text, end = _collect_function(lines, i)
            if name not in imported_names:
                structs.append((name, text))
            i = end
        else:
            i += 1

    return structs


def extract_aliases(source: str) -> list:
    """Return [(line_text, alias_name), ...] for ALL module-level alias/comptime lines.

    Only matches lines with zero indentation (module-level), excluding comptime
    declarations inside function bodies. Excludes `comptime dtype` and `comptime tol`
    lines (provided globally).
    """
    lines = source.split("\n")
    aliases = []
    for line in lines:
        if line and line[0] == " ":
            continue  # skip indented lines (inside function bodies)
        stripped = line.strip()
        m = re.match(r"^(?:comptime|alias)\s+(\w+)\s*=", stripped)
        if m:
            name = m.group(1)
            if name not in ("dtype", "tol"):
                aliases.append((line.rstrip(), name))
    return aliases


# ── CPU test function extraction ────────────────────────────────────────────


def extract_cpu_functions(source: str):
    """Yield (func_name, func_text) for every non-GPU-guarded test function.

    Skips functions whose first body line is `comptime if has_accelerator():`
    (those belong in test_gpu_all.mojo).
    """
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)(?:fn|def)\s+(test_\w+)", line)
        if not m:
            i += 1
            continue
        name = m.group(2)

        first_body = None
        for j in range(i + 1, len(lines)):
            stripped = lines[j].strip()
            if stripped == "" or stripped.startswith("#") or stripped.startswith('"""'):
                continue
            first_body = lines[j]
            break

        if first_body is not None and has_gpu_guard(first_body):
            _, end = _collect_function(lines, i)
            i = end
            continue

        text, end = _collect_function(lines, i)
        i = end
        yield name, text


# ── Renaming logic ──────────────────────────────────────────────────────────


def _file_prefix(basename: str) -> str:
    """Extract the file prefix for renaming.

    test_tensors.mojo -> tensors
    test_gpu.mojo -> gpu
    test_blas.mojo -> blas
    """
    name = basename
    if name.startswith("test_"):
        name = name[5:]
    if name.endswith(".mojo"):
        name = name[:-5]
    return name


def rename_test_function(func_name: str, func_text: str, file_prefix: str, used_names: set) -> tuple:
    """Rename a test function to avoid collisions across files.

    Returns (new_name, new_text, updated used_names set).
    """
    base = func_name.removeprefix("test_")
    new_name = f"test_{file_prefix}_{base}"

    if new_name in used_names:
        suffix = 2
        while f"{new_name}_{suffix}" in used_names:
            suffix += 1
        new_name = f"{new_name}_{suffix}"

    used_names.add(new_name)

    lines = func_text.split("\n")
    lines[0] = lines[0].replace(func_name, new_name, 1)
    new_text = "\n".join(lines)

    return new_name, new_text, used_names


def rename_alias(alias_line: str, alias_name: str, file_prefix: str) -> str:
    """Prefix a comptime alias with the file name to avoid collisions.

    comptime SMALL_SIZE = 7  ->  comptime buffers_SMALL_SIZE = 7
    """
    new_name = f"{file_prefix}_{alias_name}"
    return alias_line.replace(alias_name, new_name, 1)


def rename_alias_usages_in_text(text: str, old_name: str, new_name: str) -> str:
    """Replace all occurrences of old_name as a whole word in the given text.

    Used to update function bodies that reference aliases by their unqualified name.
    """
    return re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, text)


# ── File structure helpers ──────────────────────────────────────────────────


def _get_function_name(line: str) -> str:
    m = re.match(r"^(?:fn|def|struct)\s+(\w+)", line)
    return m.group(1) if m else ""


# ── Standard preamble (always included) ─────────────────────────────────────

SKIP_HELPERS = {
    "run_all_minmax_tests",
    "run_all_matrix_vector_tests",
    "skip_helper_if_no_device",
}

STD_PREAMBLE = [
    "from std.testing import assert_true, assert_false, assert_equal, assert_almost_equal, TestSuite",
    "from std.sys import has_accelerator",
    "",
    "comptime dtype = DType.float32",
    "comptime tol = Float32(1e-4)",
]


# ── Chunked output writer ─────────────────────────────────────────────────


def write_chunk(
    output_path: str,
    chunk_label: str,
    chunk_file_data: OrderedDict,
    all_imports: OrderedDict,
    all_helper_names: set,
    old_to_new: dict,
    _rename_test_calls,
    total_source_funcs: int,
):
    """Write a single chunk file with imports + subset of file_data."""
    std_norm = set()
    for imp in STD_PREAMBLE:
        if imp:
            std_norm.add(re.sub(r"\s+", " ", imp))

    with open(output_path, "w") as f:
        f.write('"""Auto-generated CPU test suite — %s.\n' % chunk_label)
        f.write("Generated by scripts/generate_cpu_test_suite.py\n")
        f.write(f"Contains test functions from {len(chunk_file_data)} files.\n")
        f.write('"""\n\n')

        for line in STD_PREAMBLE:
            f.write(line + "\n")
        f.write("\n")

        # Write imports, skipping those that conflict with helper/struct names
        for imp in all_imports:
            normalized = re.sub(r"\s+", " ", imp)
            if normalized in std_norm:
                continue
            if "import " in imp and " as " not in imp:
                after = imp.split("import ", 1)[-1]
                imported_names = re.findall(r'\b\w+\b', after)
                if any(n in all_helper_names for n in imported_names):
                    continue
            f.write(f"{imp}\n")

        f.write("\n\n")

        # Reset used_names for actual writing (re-run rename to get new_name)
        used_names = set()
        dup_renamed = 0
        total_written = 0
        for bname, data in chunk_file_data.items():
            prefix = _file_prefix(bname)

            # Build per-file combined regex for alias renames
            alias_rename_re = None
            alias_rename_map = {}
            if data["aliases"]:
                for _, alias_name in data["aliases"]:
                    alias_rename_map[alias_name] = f"{prefix}_{alias_name}"
                sorted_aliases = sorted(alias_rename_map.keys(), key=len, reverse=True)
                alias_rename_re = re.compile(
                    '|'.join(r'\b' + re.escape(old) + r'\b' for old in sorted_aliases)
                )
            def _apply_alias_renames(text: str) -> str:
                if alias_rename_re is None:
                    return text
                def _replacer(m):
                    return alias_rename_map[m.group(0)]
                return alias_rename_re.sub(_replacer, text)

            f.write(f"# === From {bname} ===\n\n")

            # Write per-file aliases (prefixed with file name)
            if data["aliases"]:
                for alias_line, alias_name in data["aliases"]:
                    renamed = rename_alias(alias_line, alias_name, prefix)
                    f.write(renamed + "\n")
                f.write("\n")

            # Write struct definitions (with alias usage renamed)
            for sname, stext in data["structs"]:
                f.write(_apply_alias_renames(stext))
                f.write("\n\n")

            # Write helpers (with test function calls + alias usage renamed)
            for hname, htext in data["helpers"]:
                if hname in SKIP_HELPERS:
                    continue
                modified = _rename_test_calls(_apply_alias_renames(htext))
                f.write(modified)
                f.write("\n\n")

            # Write test functions (renamed + alias usage + test call sites renamed)
            for func_name, func_text in data["functions"]:
                renamed_text = _apply_alias_renames(func_text)
                new_name, new_text, used_names = rename_test_function(
                    func_name, renamed_text, prefix, used_names
                )
                # Rename test function calls in body (skip line 0 — already
                # handled by rename_test_function)
                lines = new_text.split("\n")
                for idx in range(1, len(lines)):
                    lines[idx] = _rename_test_calls(lines[idx])
                new_text = "\n".join(lines)
                if new_name != func_name:
                    dup_renamed += 1
                f.write(new_text)
                f.write("\n\n")
                total_written += 1

        f.write("def main() raises:\n")
        f.write("    TestSuite.discover_tests[__functions_in_module()]().run()\n")

    print(f"Wrote {output_path}")
    print(f"  Source files:        {len(chunk_file_data)}")
    print(f"  Functions written:   {total_written}")
    print(f"  Functions renamed:   {dup_renamed}")
    print(f"  Total funcs across all files: {total_source_funcs}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated CPU test suite")
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
        if basename in SKIP_FILES:
            continue
        if basename.endswith("_all.mojo"):
            continue

        with open(filepath) as f:
            source = f.read()

        imports = collect_imports(source)
        aliases = extract_aliases(source)
        helpers = extract_helpers(source, imports)
        structs = extract_structs(source, imports)
        funcs = list(extract_cpu_functions(source))

        if not funcs:
            continue

        file_data[basename] = {
            "imports": imports,
            "aliases": aliases,
            "helpers": helpers,
            "structs": structs,
            "functions": funcs,
        }

    # Collect all helper/struct names for import conflict detection
    all_helper_names = set()
    for data in file_data.values():
        for hname, _ in data["helpers"]:
            all_helper_names.add(hname)
        for sname, _ in data["structs"]:
            all_helper_names.add(sname)

    # Build old→new name mapping for test functions (used to fix helper/test call sites)
    old_to_new = {}
    used_names = set()
    for bname, data in file_data.items():
        prefix = _file_prefix(bname)
        for func_name, func_text in data["functions"]:
            new_name, _, used_names = rename_test_function(
                func_name, func_text, prefix, used_names
            )
            old_to_new[func_name] = new_name

    # Pre-compile combined regex for test function renames
    _old_to_new_re = None
    if old_to_new:
        # Sort by descending length so longer names match before shorter prefixes
        sorted_names = sorted(old_to_new.keys(), key=len, reverse=True)
        _old_to_new_re = re.compile(
            '|'.join(r'\b' + re.escape(old) + r'\b' for old in sorted_names)
        )
    def _rename_test_calls(text: str) -> str:
        if _old_to_new_re is None:
            return text
        def _replacer(m):
            return old_to_new[m.group(0)]
        return _old_to_new_re.sub(_replacer, text)

    # Deduplicate imports across files
    all_imports = OrderedDict()
    for data in file_data.values():
        for imp in data["imports"]:
            all_imports[imp] = None

    total_source_funcs = sum(len(d["functions"]) for d in file_data.values())

    # ── Distribute across chunks ──────────────────────────────────────────
    sorted_files = sorted(file_data.keys())

    if num_chunks > 1:
        for chunk_id in range(1, num_chunks + 1):
            chunk_files = [
                f for i, f in enumerate(sorted_files)
                if i % num_chunks == chunk_id - 1
            ]
            chunk_data = OrderedDict((f, file_data[f]) for f in chunk_files)
            output_path = os.path.join(
                TESTS_DIR, f"test_cpu_all_{chunk_id}.mojo"
            )
            write_chunk(
                output_path=output_path,
                chunk_label=f"chunk {chunk_id}/{num_chunks}",
                chunk_file_data=chunk_data,
                all_imports=all_imports,
                all_helper_names=all_helper_names,
                old_to_new=old_to_new,
                _rename_test_calls=_rename_test_calls,
                total_source_funcs=total_source_funcs,
            )
    else:
        write_chunk(
            output_path=OUTPUT,
            chunk_label="monolithic",
            chunk_file_data=file_data,
            all_imports=all_imports,
            all_helper_names=all_helper_names,
            old_to_new=old_to_new,
            _rename_test_calls=_rename_test_calls,
            total_source_funcs=total_source_funcs,
        )


if __name__ == "__main__":
    main()
