#!/usr/bin/env python3
"""Strip CPU-only test functions from Mojo test files, keeping GPU tests + helpers.

Usage: python3 scripts/strip_cpu_tests.py
Output: tests/gpu/<name>.mojo for each GPU-containing test file.
"""

import os
import re
import shutil

TESTS_DIR = "tests"
GPU_DIR = os.path.join(TESTS_DIR, "gpu")


def has_gpu_code(text: str) -> bool:
    """Check if a function body contains GPU-related code."""
    return "has_accelerator()" in text


def is_test_def(line: str) -> bool:
    """Check if a line starts a test function at module level."""
    stripped = line.lstrip()
    return stripped.startswith("def test_") or stripped.startswith("fn test_")


def extract_test_functions(content: str):
    """Split file content into blocks: [preamble, func1, func2, ...].

    The preamble is everything before the first test function.
    Each func is a complete function definition including its body.
    Returns list of (is_test, text) tuples.
    """
    lines = content.split("\n")
    blocks = []
    current_block = []
    in_test = False

    for line in lines:
        if not in_test and is_test_def(line):
            # End of preamble, start of first test function
            if current_block:
                blocks.append((False, "\n".join(current_block)))
                current_block = []
            in_test = True
            current_block.append(line)
        elif in_test:
            if is_test_def(line):
                # Previous test function ended, new one starts
                blocks.append((True, "\n".join(current_block)))
                current_block = [line]
            elif line.strip() == "":
                current_block.append(line)
            elif line[0] != " " and line[0] != "\t" and line[0] != "":
                # Non-indented line that's not a test def — could be
                # a helper def/fn/struct/comment at module level.
                # End current test function.
                if current_block:
                    blocks.append((True, "\n".join(current_block)))
                    current_block = []
                in_test = False
                current_block.append(line)
            else:
                current_block.append(line)
        else:
            current_block.append(line)

    if current_block:
        blocks.append((in_test, "\n".join(current_block)))

    return blocks


def process_file(src_path: str) -> str | None:
    """Process a single test file. Returns output content or None if no GPU tests."""
    with open(src_path) as f:
        content = f.read()

    blocks = extract_test_functions(content)

    # Check if any test function has GPU code
    has_gpu = any(
        has_gpu_code(text) for is_test, text in blocks if is_test
    )
    if not has_gpu:
        return None

    out_lines = []
    for is_test, text in blocks:
        if is_test and not has_gpu_code(text):
            continue  # skip CPU-only test function
        out_lines.append(text)

    return "\n".join(out_lines)


def main():
    os.makedirs(GPU_DIR, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(TESTS_DIR)):
        if not fname.endswith(".mojo"):
            continue
        if fname.startswith("test_gpu_all_") or fname.startswith("test_cpu_all_"):
            continue

        src = os.path.join(TESTS_DIR, fname)
        result = process_file(src)
        if result is not None:
            dst = os.path.join(GPU_DIR, fname)
            with open(dst, "w") as f:
                f.write(result)
            kept = result.count("def test_") + result.count("fn test_")
            total = (
                open(src).read().count("def test_") + open(src).read().count("fn test_")
            )
            removed = total - kept
            print(f"{fname}: {removed:2d} CPU tests removed, {kept:2d} GPU tests kept → tests/gpu/")
            count += 1
        else:
            print(f"{fname}: no GPU tests, skipped")

    print(f"\nDone. {count} files written to {GPU_DIR}/")


if __name__ == "__main__":
    main()
