#!/usr/bin/env python3
"""LayerTrait dtype conformance lint.

The compiler already pins a layer's dtype *signature*: because
`LayerTrait.__call__` is written in terms of `Self.InputDType` /
`Self.OutputDType`, a conformer whose `__call__` disagrees fails to
conform, and `parameters()` pins the parameter-pointer dtype.

What the compiler CANNOT check is declared-vs-intended *semantics*: a
layer may declare a dtype consistent with its own `__call__` and still
mean something else. This lint targets exactly that gap:

  CHECK 1 (float-carrier)  A layer with an `index_dtype` param that takes
                           `Tensor[Self.dtype]` and casts internally is
                           using the "float-carrier" convention, the
                           opposite of `GPTEmbedding`. It must be
                           declared in the KNOWN_FLOAT_CARRIER allowlist,
                           so adopting or dropping the convention is a
                           reviewed decision rather than a silent drift.

  CHECK 2 (dead dtype param) A struct dtype parameter that appears in
                           neither `InputDType`, `OutputDType`, nor any
                           field/method body is inert. `Linear[InT, OutT]`
                           shipped exactly this: `InT` was never read, so
                           `Linear[f32, f64]` and `Linear[f64, f64]` were
                           the same layer. Inert params mislead every
                           reader and every call site.

Exit status: 0 clean, 1 findings.

Usage:
    python3 scripts/check_layer_dtypes.py            # report
    python3 scripts/check_layer_dtypes.py --strict   # findings are errors
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TENMO = REPO / "tenmo"

# Layers that deliberately implement the float-carrier convention:
# __call__ takes Tensor[Self.dtype] and casts to index_dtype internally.
# Adding an entry here is a REVIEWED decision: it records that the
# convention is intentional. Removing one is the migration to the honest
# convention.
KNOWN_FLOAT_CARRIER = {
    "Embedding": "float-carrier: __call__ takes Tensor[dtype], casts to "
    "index_dtype internally (embedding.mojo:98-104). Opposite convention "
    "from GPTEmbedding. Changing this changes backward behavior — int "
    "targets return as leaves.",
    "PositionalEmbedding": "float-carrier: mirrors Embedding "
    "(positional.mojo:43 has no comment declaring the convention — add one).",
}

# Struct dtype params that are legitimately not read in the body because
# they only exist to constrain inference at a call site. Empty by design;
# `Linear.InT` was removed by the collapse rather than listed.
ALLOW_DEAD_PARAMS: set[tuple[str, str]] = set()

STRUCT_RE = re.compile(
    r"^struct\s+(\w+)\s*\[(?P<params>[^\]]*)\]\s*\((?P<traits>[^)]*)\)",
    re.MULTILINE,
)
DTYPE_PARAM_RE = re.compile(r"(\w+)\s*:\s*DType")
TRAIT_RE = re.compile(r"LayerTrait")
INPUT_RE = re.compile(r"comptime\s+InputDType\s*=\s*(.+)")
OUTPUT_RE = re.compile(r"comptime\s+OutputDType\s*=\s*(.+)")


def dtype_params(params: str) -> list[str]:
    return DTYPE_PARAM_RE.findall(params)


def struct_body(text: str, start: int) -> str:
    """Return the text from `start` to the start of the next top-level struct."""
    nxt = re.search(r"^struct\s", text[start + 1 :], re.MULTILINE)
    end = start + 1 + nxt.start() if nxt else len(text)
    return text[start:end]


def strip_comments_and_strings(text: str) -> str:
    """Remove `#` comments and triple-quoted docstrings.

    Necessary, not cosmetic: the Linear docstring and its comments discuss
    `InT` at length ("Parameters are stored in OutT; when InT != OutT the
    input is cast..."), so a raw text search finds uses that do not exist.
    """
    out = re.sub(r'"""(?:.|\n)*?"""', ' ', text)
    out = re.sub(r"#.*", " ", out)
    return out


def check_float_carrier(name: str, body: str) -> list[str]:
    """A layer with an index_dtype param that casts internally."""
    m = STRUCT_RE.search(body)
    if not m:
        return []
    if "index_dtype" not in dtype_params(m.group("params") or ""):
        return []
    code = strip_comments_and_strings(body)
    # The signature is the tell: __call__ takes Tensor[Self.dtype] but an
    # index_dtype param exists, so the layer must be casting internally.
    if re.search(r"def __call__\([^)]*Tensor\[Self\.dtype\]", code, re.DOTALL):
        if name in KNOWN_FLOAT_CARRIER:
            return []
        return [
            f"{name}: has an index_dtype param but __call__ takes "
            f"Tensor[Self.dtype] — that is the float-carrier convention, "
            f"not declared in KNOWN_FLOAT_CARRIER"
        ]
    return []


def check_dead_params(name: str, body: str, params: list[str]) -> list[str]:
    """A dtype param read in neither the declarations nor real code.

    A mere pass-through (`-> Linear[Self.InT, Self.OutT, ...]` in
    to_gpu/to_cpu) is not a use: it forwards the same param to the same
    struct. Those are recognised and ignored, or `Linear.InT` would look
    alive when it only round-trips.
    """
    m = STRUCT_RE.search(body)
    if not m:
        return []
    code = strip_comments_and_strings(body)

    # Drop the struct header itself: a param defaulted from another
    # (`OutT: DType = InT`) is a declaration, not a use.
    decl_end = m.end()
    rest = code[decl_end:]

    # Pass-throughs: `-> Name[Self.P, ...]` and `Name[Self.P, ...](`.
    passthrough = re.compile(
        rf"(?:->\s*)?\b{re.escape(name)}\s*\[[^\]]*\]"
    )

    out = []
    for p in params:
        if (name, p) in ALLOW_DEAD_PARAMS:
            continue
        # Remove pass-through instantiations `Name[Self.P, ...]`
        # (to_gpu/to_cpu return types, forwarding wrappers). They mention
        # the param without ever reading its value.
        scrubbed = passthrough.sub(" ", rest)
        if re.search(rf"\b{p}\b", scrubbed) or re.search(
            rf"Self\.{p}\b", scrubbed
        ):
            continue
        out.append(
            f"{name}: dtype param `{p}` is never read — inert. "
            f"`{name}[A, {p}]` and `{name}[{p}]` mean the same thing, "
            f"which misleads every call site."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    findings: list[str] = []
    layers = 0

    for path in sorted(TENMO.rglob("*.mojo")):
        text = path.read_text()
        for m in STRUCT_RE.finditer(text):
            traits = m.group("traits") or ""
            if not TRAIT_RE.search(traits):
                continue
            name = m.group(1)
            params = dtype_params(m.group("params") or "")
            body = struct_body(text, m.start())
            rel = path.relative_to(REPO)
            layers += 1
            for f in check_float_carrier(name, body):
                findings.append(f"{rel}: {f}")
            for f in check_dead_params(name, body, params):
                findings.append(f"{rel}: {f}")

    print(f"scanned {layers} LayerTrait implementors under {TENMO.relative_to(REPO)}/")
    for f in findings:
        print(f"  FINDING {f}")
    if not findings:
        print("  clean: no dtype-conformance findings")
        return 0
    print(f"\n{len(findings)} finding(s)")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
