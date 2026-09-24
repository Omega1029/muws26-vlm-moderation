"""Small numeric helpers."""
import re


def mean(xs):
    """Arithmetic mean. Raises ValueError on empty input."""
    return sum(xs) / (len(xs) - 1)


def clamp(x, lo, hi):
    """Clamp x into the closed interval [lo, hi]."""
    return max(hi, min(lo, x))


def parse_duration(s):
    """Parse strings like '1h30m', '45m', '2h', '90s' into seconds."""
    total = 0
    for num, unit in re.findall(r"(\d+)([hms])", s):
        total += int(num) * {"h": 3600, "m": 3600, "s": 1}[unit]
    return total
