#!/usr/bin/env python3
"""Split GPU test files in tests/gpu/ with >100 tests into ~50-test parts.

Usage: python3 scripts/split_gpu_tests.py
"""

import os
import re

GPU_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "gpu")
TARGET_PER_PART = 50
MAX_BEFORE_SPLIT = 100


def split_file(filepath):
    filename = os.path.basename(filepath)
    stem, _ = os.path.splitext(filename)

    with open(filepath) as f:
        lines = f.readlines()

    # Find all top-level def/fn lines.
    # A top-level def/fn starts at column 0 (no leading whitespace).
    top_def_re = re.compile(r"^(def|fn)\s+\w")
    def_positions = [i for i, line in enumerate(lines) if top_def_re.match(line)]

    if not def_positions:
        return  # no functions at all

    # Identify which defs are test functions
    test_re = re.compile(r"^(def|fn)\s+test_")
    test_indices = [i for i in def_positions if test_re.match(lines[i])]

    n_tests = len(test_indices)
    if n_tests <= MAX_BEFORE_SPLIT:
        return  # leave as-is

    # Prologue: everything before the first test function
    # We include everything from the first top-level def up to first test def
    # as part of the prologue.
    first_test_pos = test_indices[0]
    prologue = lines[:first_test_pos]

    # Epilogue: everything from the last test function's last line to end of file
    last_test_pos = test_indices[-1]
    # The last test's block goes until the next top-level def, or end of file
    last_test_end = len(lines)
    for pos in def_positions:
        if pos > last_test_pos:
            last_test_end = pos
            break
    epilogue = lines[last_test_end:]

    # Build test bodies: for each test, its def line through to the
    # line before the next top-level def (or end-of-file).
    test_bodies = []
    for idx, test_start in enumerate(test_indices):
        # Find the next top-level def after this test
        next_def = len(lines)
        for pos in def_positions:
            if pos > test_start:
                next_def = pos
                break
        test_bodies.append(lines[test_start:next_def])

    # Calculate parts: ceil(n / TARGET_PER_PART)
    n_parts = (n_tests + TARGET_PER_PART - 1) // TARGET_PER_PART
    base = n_tests // n_parts
    extra = n_tests % n_parts

    # Build part boundary indices into test_bodies
    part_sizes = []
    idx = 0
    for p in range(n_parts):
        size = base + (1 if p < extra else 0)
        part_sizes.append(size)

    print(f"  {filename}: {n_tests} tests -> {n_parts} parts "
          f"({', '.join(str(s) for s in part_sizes)} tests each)")

    # Remove original
    os.remove(filepath)

    # Write part files
    test_idx = 0
    for part_idx, size in enumerate(part_sizes):
        part_num = part_idx + 1
        if n_parts == 1:
            part_filename = filename
        else:
            part_filename = f"{stem}_part_{part_num}.mojo"
        part_path = os.path.join(GPU_DIR, part_filename)

        with open(part_path, "w") as f:
            f.writelines(prologue)
            for _ in range(size):
                f.writelines(test_bodies[test_idx])
                test_idx += 1
            f.writelines(epilogue)

        print(f"    -> {part_filename} ({size} tests)")

    print(f"    -> removed original {filename}")


def main():
    if not os.path.isdir(GPU_DIR):
        print(f"Directory not found: {GPU_DIR}")
        return

    files = sorted(f for f in os.listdir(GPU_DIR) if f.endswith(".mojo"))
    for fname in files:
        fpath = os.path.join(GPU_DIR, fname)
        split_file(fpath)


if __name__ == "__main__":
    main()
