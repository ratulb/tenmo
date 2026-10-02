"""Test runner that isolates each test file in a subprocess to avoid JIT SIGILL."""
from __future__ import annotations

import subprocess
import sys
import os
import glob

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "python")


def run_test_file(path: str) -> tuple[int, str]:
    """Run a single test file in a subprocess. Returns (returncode, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-v", "--tb=short"],
        capture_output=True, timeout=120,
    )
    output = result.stdout.decode() + result.stderr.decode()
    return result.returncode, output


def main():
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*.mojo")))
    # Not needed — we're running Python test files
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*.py")))

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    errors = 0

    for f in test_files:
        name = os.path.basename(f)
        rc, output = run_test_file(f)
        # Parse results from output
        lines = output.strip().split("\n")
        summary = ""
        for line in lines:
            if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                summary = line.strip()
                break
            elif line.startswith("=") and ("passed" in line or "failed" in line or "error" in line):
                summary = line.strip()
                break

        if rc == 0:
            print(f"  PASS  {name}  {summary}")
            passed += 1
        elif rc == 5:
            print(f"  SKIP  {name}  (no tests collected)")
            skipped += 1
        else:
            print(f"  FAIL  {name}  rc={rc}")
            # Print last few lines of output for debugging
            for line in lines[-5:]:
                print(f"        {line}")
            errors += 1
        total += 1

    print(f"\n{'='*60}")
    print(f"Files: {total}  Passed: {passed}  Failed: {errors}  Skipped: {skipped}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
