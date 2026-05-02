---
name: update-test-file
description: Given a .mojo file with test functions, fix imports, add main fn, consolidate test names in main, run it, then move to tests/ and update execute.sh. Use when a mojo test file needs to be wired up end to end.
---

## What I do

I take a given `.mojo` file containing test functions and wire it up completely
for the Tenmo project. I follow these steps in order — no skipping.

### Step 1 — Read the file
Read the given file in full. Identify:
- All `fn test_*() raises:` function names
- All imports already present at the top
- Whether `fn main() raises:` exists

### Step 2 — Fix imports
Read a reference test file (e.g. `tests/test_batched_matmul.mojo` or any
similar test that uses the same Tenmo types) to understand what imports are
needed. Then add any missing imports to the top of the given file. Common
imports for Tenmo tests:
```mojo
from testing import assert_true
from sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
```
Only add what is actually needed by the functions in the file. Don't add imports for DType etc. They available by default.

### Step 3 — Add or fix main
If `fn main() raises:` does not exist, add it at the bottom of the file.
Inside main, call every `fn test_*() raises:` in the file in the order they
appear. End main with:
```mojo
    print("All <functionality> tests passed ✓")
```
If `fn main() raises:` already exists, check that every `fn test_*` in the
file is invoked inside it. Add any missing calls. Do not add duplicate calls.

### Step 4 — Run the file
Execute: `./run_c.sh <given_filename>`
Read the output carefully.
- If there are compilation errors, fix them and re-run. Repeat until clean.
- If there are assertion errors, do NOT silently fix the assertions — pause
  and report the failure to me with the full error output.
- If all tests pass, proceed to Step 5.

### Step 5 — Check for existing test file
Check whether a file with the same name already exists in `tests/`.

- If NO: move the file to `tests/` using `mv <file> tests/<file>`. Done.

- If YES:
  1. Read both files in full.
  2. Compare all `fn test_*` names between the two files.
  3. Check for name conflicts — any `fn test_*` that exists in **both** files
     with the same name.

  **If NO name conflicts:**
  - Copy any import that the new file uses that is missing in the target file in tests/
  - Copy all functions from the new file into the target file in `tests/`
    (append before the closing of the file, after the last existing function
    but before `fn main()`).
  - Add all new test function names into the existing `fn main()` in the
    target file, after the last existing call.
  - Delete the new file — it has been fully merged.
  - Re-run `./run_c.sh tests/<filename>` to confirm the merged file is clean.

  **If there ARE name conflicts:**
  - Do NOT touch either file.
  - Pause and report to me:
    - Which function names conflict
    - The full signature of the conflicting function in each file
    - Ask me how to proceed before doing anything further

### Step 6 — Update execute.sh
Read `execute.sh` in full. It contains sections for:
- Individual test entries (a case statement or similar)
- GPU test list
- "all tests" list
- Help/usage output listing available tests

Extract the test name from the filename (e.g. `test_var_std.mojo` → `var_std`).

Make the following additions:
1. Add `var_std` as an individual runnable test entry
2. Add it to the GPU tests section if the file contains any
   `comptime if has_accelerator():` blocks
3. Add it to the "all tests" list
4. Add it to the help/usage output

Show me the diff of changes to `execute.sh` before writing. Ask for
confirmation before applying.

### Step 7 — Report
Print a summary:
- File location: `tests/<filename>`
- Tests found: list all `fn test_*` names
- execute.sh: what was added
- Any warnings or things I should know

## What I do NOT do
- I do not silently fix failing assertions — I report them to you
- I do not overwrite an existing file in `tests/` without asking
- I do not apply execute.sh changes without showing the diff first
- I do not guess at imports — I read a reference file

## When to use me
Use when:
- You receive a `.mojo` file with test functions that needs to be wired up
- A file has missing imports, no main, or test functions not invoked in main
- You want to add a new test file to the Tenmo test suite end-to-end

Do not use for:
- Fixing test logic or assertion values
- Creating new test functions from scratch
