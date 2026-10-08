"""Fatal error reporting for the whole library.

Only depends on `shared.ansi` and the stdlib.
"""

from std.os import abort
from std.reflection import call_location
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
    # Call-site location: panic is always-inline, so inline_count=1 is
    # the panic invocation site (file:line). Parameter-expression
    # callsites may misreport (stdlib limitation) — still strictly more
    # informative than a bare message. Pair with
    # MODULAR_DEBUG=stack-trace-on-error for the full stack.
    var loc = call_location[inline_count=1]()
    var where = (
        String(loc.file_name()) + ":" + String(loc.line()) + " "
    )
    abort(RED + where + message + String(RESET))
