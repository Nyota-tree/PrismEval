"""
Terminal output helpers for readable, consistent CLI display.
"""

from typing import Any, Optional


def format_section_header(title: str, width: int = 60, char: str = "=") -> str:
    """
    Format a section title with a line above and below.

    Args:
        title: Section title text.
        width: Total line width.
        char: Character used for the line.

    Returns:
        A multi-line string suitable for print().
    """
    line = char * width
    return f"\n{line}\n{title}\n{line}"


def format_metric_line(label: str, value: Any, width: int = 12) -> str:
    """
    Format a single metric line with aligned label and value.

    Args:
        label: Metric name or description.
        value: Metric value (will be stringified).
        width: Minimum width for the value field.

    Returns:
        A single line string, e.g. "  Average score:    82.50"
    """
    value_str = str(value)
    return f"  {label}: {value_str:>{width}}"


def format_progress(current: int, total: int, prefix: str = "") -> str:
    """Format a progress string like '  [3/10] ...'."""
    return f"  [{current}/{total}] {prefix}".strip()
