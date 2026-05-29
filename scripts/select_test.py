#!/usr/bin/env python3
"""Extract a single test function into an isolated runnable .mojo file.

Usage:
    python3 scripts/select_test.py tests/test_tensors.mojo test_2d_1d_add_backward

Output: /tmp/test_2d_1d_add_backward.mojo (or set with -o)
"""

import re
import sys
import argparse


# Names that are always available in Mojo or are Tenmo builtins — never need
# extracting from the source file.
_BUILTINS = frozenset({
    'print', 'len', 'range', 'int', 'float', 'str', 'bool',
    'abs', 'min', 'max', 'sum', 'type', 'list', 'dict', 'set',
    'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
    'hasattr', 'getattr', 'setattr', 'isinstance', 'issubclass',
    'super', 'open', 'next', 'iter', 'all', 'any', 'repr',
    'input', 'eval', 'exec', 'compile', 'object', 'id',
    'callable', 'delattr', 'dir', 'divmod', 'format',
    'globals', 'hex', 'locals', 'memoryview', 'oct', 'ord',
    'pow', 'property', 'round', 'slice', 'staticmethod', 'vars',
    '__functions_in_module',
    'has_accelerator', 'ComptimeAcceleratorSymbol',
    'assert_true', 'assert_false', 'assert_raises', 'TestSuite',
    'UnsafePointer', 'Pointer', 'Optional',
    'String', 'Variety', 'Bool',
    'AddressSpace',
    'SIMD', 'DType', 'Float32', 'Float64',
})


def _extract_text(lines: list[str], start: int, end: int) -> str:
    return '\n'.join(lines[start:end])


def _find_definitions(lines: list[str]):
    """Scan module-level definitions.

    Returns:
        functions: dict[str, (start, end)] — def/fn at indent 0
        structs:   dict[str, (start, end)] — struct at indent 0
        def_spans: set[int] — all line indices covered by any definition
    """
    functions: dict[str, tuple[int, int]] = {}
    structs: dict[str, tuple[int, int]] = {}
    def_spans: set[int] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_fn = re.match(r'^(?:def|fn)\s+(\w+)', line)
        m_st = re.match(r'^struct\s+(\w+)', line)

        if m_fn or m_st:
            name = (m_fn or m_st).group(1)
            indent = len(line) - len(line.lstrip())
            start = i
            # Scan backwards for decorator lines (@...) at same indent
            while start > 0 and lines[start - 1].strip().startswith('@'):
                d_indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
                if d_indent > indent:
                    break
                start -= 1
            i += 1
            while i < len(lines):
                ln = lines[i]
                if ln.strip() == '':
                    i += 1
                    continue
                curr_indent = len(ln) - len(ln.lstrip())
                if curr_indent <= indent and ln.strip():
                    # Multi-line signature continuation — not the end
                    stripped = ln.strip()
                    if (re.match(r'^[\]\)]', stripped)
                            or stripped.startswith('->')
                            or stripped.startswith('raises')):
                        i += 1
                        continue
                    break
                i += 1
            for j in range(start, i):
                def_spans.add(j)
            target = functions if m_fn else structs
            target[name] = (start, i)
        else:
            i += 1

    return functions, structs, def_spans


def _get_called_names(text: str) -> set[str]:
    """Return names of functions/structs called in the given body text."""
    text = re.sub(r'#.*', '', text)
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)

    names: set[str] = set()
    for m in re.finditer(r'(?<!\w)([a-zA-Z_]\w*)\s*\(', text):
        n = m.group(1)
        if n not in _BUILTINS and not n.startswith('__'):
            names.add(n)
    return names


def _get_dep_closure(fn_name: str, functions: dict, lines: list[str]) -> set[str]:
    """Return names of all helpers transitively called by fn_name."""
    result: set[str] = set()
    stack = [fn_name]
    while stack:
        cur = stack.pop()
        if cur in result:
            continue
        result.add(cur)
        if cur not in functions:
            continue
        start, end = functions[cur]
        body = _extract_text(lines, start, end)
        for name in _get_called_names(body):
            if name in functions and name not in result:
                stack.append(name)
    return result


def _get_global_lines(lines: list[str], def_spans: set[int]) -> list[str]:
    """Return indent-0 lines that are not imports and not inside definitions."""
    result: list[str] = []
    in_string: str | None = None  # '"""' or "'''" when inside triple-quoted string
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Track triple-quoted strings spanning multiple lines
        if in_string:
            if in_string in stripped:
                in_string = None
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            delim = stripped[:3]
            rest = stripped[3:]
            if delim not in rest:
                in_string = delim
            continue

        if stripped.startswith('from ') or stripped.startswith('import '):
            continue
        if i in def_spans:
            continue
        if line.startswith(' '):
            continue
        if stripped.startswith('#'):
            continue

        result.append(line.rstrip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract a single test function into an isolated runnable .mojo file')
    parser.add_argument('source_file', help='Path to a Tenmo test .mojo file')
    parser.add_argument('test_fn_name',
                        help='Exact name of the test function (e.g. test_add)')
    parser.add_argument('-o', '--output',
                        help='Output path (default: /tmp/<test_fn_name>.mojo)')
    args = parser.parse_args()

    with open(args.source_file) as f:
        content = f.read()
    lines = content.split('\n')

    # --- 1. Imports (deduplicated, preserving order) ---
    imports: list[str] = []
    seen_imports: set[str] = set()
    for l in lines:
        sl = l.strip()
        if sl.startswith('from ') or sl.startswith('import '):
            normalized = sl.replace('(', '').replace(')', '').replace(',', '').strip()
            key = re.sub(r'\s+', ' ', normalized)
            if key not in seen_imports:
                seen_imports.add(key)
                imports.append(l.rstrip())

    # --- 2. Definitions ---
    functions, structs, def_spans = _find_definitions(lines)

    if args.test_fn_name not in functions:
        print(f"Error: function '{args.test_fn_name}' not found in {args.source_file}",
              file=sys.stderr)
        print("Available test functions:", file=sys.stderr)
        for name in sorted(functions):
            if name.startswith('test_') or name.startswith('_'):
                print(f"  {name}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Transitive helper dependency closure ---
    dep_names = _get_dep_closure(args.test_fn_name, functions, lines)
    dep_names.discard(args.test_fn_name)

    # --- 4. Helper function texts in definition order ---
    helper_texts: list[str] = []
    for name in sorted(functions, key=lambda n: functions[n][0]):
        if name in dep_names and name != 'main':
            s, e = functions[name]
            helper_texts.append(_extract_text(lines, s, e))

    # --- 5. Structs referenced by test or its deps ---
    # Collect all unique names called by the test + helpers
    ref_names: set[str] = set()
    test_start, test_end = functions[args.test_fn_name]
    ref_names.update(_get_called_names(_extract_text(lines, test_start, test_end)))
    for dep in dep_names:
        if dep in functions:
            s, e = functions[dep]
            ref_names.update(_get_called_names(_extract_text(lines, s, e)))

    struct_texts: list[str] = []
    for name in sorted(structs, key=lambda n: structs[n][0]):
        if name in ref_names:
            s, e = structs[name]
            struct_texts.append(_extract_text(lines, s, e))

    # --- 6. Module-level globals (comptime, constants) ---
    global_lines = _get_global_lines(lines, def_spans)

    # --- 7. Write output ---
    output_path = args.output or f'/tmp/{args.test_fn_name}.mojo'
    with open(output_path, 'w') as f:
        for imp in imports:
            f.write(imp + '\n')
        if imports:
            f.write('\n')

        for g in global_lines:
            f.write(g + '\n')
        if global_lines:
            f.write('\n')

        for s in struct_texts:
            f.write(s + '\n\n')

        for h in helper_texts:
            f.write(h + '\n\n')

        test_text = _extract_text(lines, test_start, test_end)
        f.write(test_text + '\n\n')

        f.write('def main() raises:\n')
        f.write('    TestSuite.discover_tests[__functions_in_module()]().run()\n')
        f.write('    print("All tests passed ✓")\n')

    print(output_path)


if __name__ == '__main__':
    main()
