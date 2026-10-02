"""ANSI color codes for terminal output.

Layer-0 leaf: no tenmo imports.
"""

comptime RED: String = "\033[31m"
comptime CYAN: String = "\033[36m"
comptime MAGENTA: String = "\033[35m"
comptime BLUE: String = "\033[34m"  # Standard blue
comptime YELLOW: String = "\033[33m"  # Standard yellow
comptime RESET: String = "\033[0m"

# Bright variants (more vibrant)
comptime BRIGHT_BLUE: String = "\033[94m"
