"""Fatal error reporting for the whole library.

Layer-0 leaf: only depends on `shared.ansi` and the stdlib.
"""

from std.os import abort
from .ansi import RED, RESET


@always_inline("nodebug")
def panic(*s: String):
    var message = String(capacity_bytes=len(s))
    if len(s) > 0:
        var start = String(s[0])
        message += start.strip()
        for i in range(1, len(s)):
            var next_part = String(s[i])
            message += " " + next_part.strip()
    abort(RED + message + String(RESET))
