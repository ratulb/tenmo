"""Pointer identity helper (moved from tenmo/common_utils.mojo).

Layer-0 leaf: stdlib only.
"""


@always_inline("nodebug")
def id[type: AnyType, //](t: type) -> Int:
    """The object's address as an integer (identity, not equality)."""
    return Int(Pointer(to=t).as_imm())