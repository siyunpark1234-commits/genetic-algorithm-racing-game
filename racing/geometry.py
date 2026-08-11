from __future__ import annotations

from pygame import Vector2


def cross(a: Vector2, b: Vector2) -> float:
    return a.x * b.y - a.y * b.x


def segments_intersect(a: Vector2, b: Vector2, c: Vector2, d: Vector2) -> bool:
    """Return True when the two closed line segments intersect."""
    ab = b - a
    cd = d - c
    denominator = cross(ab, cd)
    if abs(denominator) < 1e-9:
        return False
    t = cross(c - a, cd) / denominator
    u = cross(c - a, ab) / denominator
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0
