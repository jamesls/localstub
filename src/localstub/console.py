"""Rich console utilities for lstub CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Custom theme with colors for traffic direction
theme = Theme({
    "inbound": "blue",
    "outbound": "green",
    "lstub": "cyan bold",
    "closed": "yellow",
    "error": "red bold",
})

console = Console(theme=theme)
