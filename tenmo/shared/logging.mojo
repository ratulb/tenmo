"""Logging helpers — `log_debug` / `log_info` / `log_warning`.

Layer-0 leaf: only depends on `shared.ansi` and the stdlib.
"""

from std.sys.defines import get_defined_string
from std.logger import Level, Logger
from .ansi import RED, BLUE, YELLOW, RESET

comptime LOG_LEVEL = get_defined_string["LOGGING_LEVEL", "INFO"]()
comptime log = Logger[Level._from_str(LOG_LEVEL)]()


@always_inline("nodebug")
def log_debug(msg: String, color: String = RED):
    log.debug(color + msg + String(RESET))


@always_inline("nodebug")
def log_info(msg: String, color: String = BLUE):
    log.info(color + msg + String(RESET))


@always_inline("nodebug")
def log_warning(msg: String, color: String = YELLOW):
    log.warning(color + msg + String(RESET))
