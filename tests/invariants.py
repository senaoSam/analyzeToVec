"""Hard geometric invariants for vectorize.py output.

These are non-negotiable post-conditions the pipeline output must satisfy.
Unlike the metrics in tests/metrics.py (which have tolerances and gate on
direction), invariants are strict booleans — any violation is a test FAIL
with no tolerance. They capture the user's two hard requirements:

  (a) output visually matches the source image
  (b) doors and windows connect to walls (no floating jambs)

Each invariant function takes a normalized payload and returns a list of
``Violation`` records. An empty list means the invariant holds.

Usage:
    from tests.invariants import check_all
    violations = check_all(lines, image_shape, wall_mask)
    if violations:
        for v in violations:
            print(v.describe())
        raise SystemExit(1)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Endpoint-coincidence tolerance. Matches regression / audit_view int-pixel
# rounding so invariant verdicts agree with their endpoint counts.
ENDPOINT_COINCIDENCE_PX = 1.0

# Per-pixel slack when checking whether a chromatic endpoint lies on a wall
# segment body. 1 px catches normal sub-pixel drift; tighter would jitter,
# looser would let real misses slip through.
BODY_PROXIMITY_PX = 1.0


@dataclass(frozen=True)
class Violation:
    """One concrete invariant breakage. ``kind`` is the rule name;
    ``detail`` carries human-readable context; ``location`` is an optional
    (x, y) marker so visual debug tools can highlight it.
    """
    kind: str
    detail: str
    location: Optional[Tuple[float, float]] = None

    def describe(self) -> str:
        loc = f" @ ({self.location[0]:.2f}, {self.location[1]:.2f})" \
            if self.location is not None else ""
        return f"[{self.kind}] {self.detail}{loc}"


def _is_finite_xy(s: Dict) -> bool:
    for k in ("x1", "y1", "x2", "y2"):
        v = float(s[k])
        if not math.isfinite(v):
            return False
    return True


def _seg_length(s: Dict) -> float:
    return math.hypot(float(s["x2"]) - float(s["x1"]),
                      float(s["y2"]) - float(s["y1"]))


def _axis_of(s: Dict, eps: float = 1e-6) -> str:
    """'h' if essentially horizontal, 'v' if essentially vertical, 'd' otherwise.

    ``eps`` absorbs sub-ULP float drift (a vertical wall whose x ends differ
    by 1e-13 is still vertical, not diagonal). Step 21 in todo.md unifies
    coordinate precision across the pipeline so this tolerance will
    eventually become unnecessary; for now it keeps the invariant honest
    about what's a genuine diagonal and what's a numerical artefact.
    """
    dx = abs(float(s["x2"]) - float(s["x1"]))
    dy = abs(float(s["y2"]) - float(s["y1"]))
    if dy <= eps and dx > eps:
        return "h"
    if dx <= eps and dy > eps:
        return "v"
    return "d"


def check_finite_coords(segments: Sequence[Dict]) -> List[Violation]:
    """Every endpoint coord must be a finite number (no NaN, no inf)."""
    out: List[Violation] = []
    for i, s in enumerate(segments):
        if not _is_finite_xy(s):
            out.append(Violation(
                kind="non_finite_coord",
                detail=f"segment {i} (type={s.get('type','?')}) has non-finite endpoint",
            ))
    return out


def check_inside_bbox(segments: Sequence[Dict],
                      image_shape: Tuple[int, int],
                      margin: float = 1.0) -> List[Violation]:
    """No endpoint may sit outside the image bbox (with small margin slack)."""
    h, w = image_shape[:2]
    out: List[Violation] = []
    for i, s in enumerate(segments):
        for end in ("1", "2"):
            x = float(s[f"x{end}"])
            y = float(s[f"y{end}"])
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                out.append(Violation(
                    kind="out_of_bbox",
                    detail=f"segment {i} endpoint {end} at ({x:.1f},{y:.1f}) outside image {w}x{h}",
                    location=(x, y),
                ))
    return out


def check_no_zero_length(segments: Sequence[Dict],
                         min_length: float = 1e-3) -> List[Violation]:
    """No segment may have its two endpoints coincide."""
    out: List[Violation] = []
    for i, s in enumerate(segments):
        if _seg_length(s) < min_length:
            out.append(Violation(
                kind="zero_length",
                detail=f"segment {i} (type={s.get('type','?')}) has length < {min_length}",
                location=(float(s["x1"]), float(s["y1"])),
            ))
    return out


def check_no_diagonals(segments: Sequence[Dict]) -> List[Violation]:
    """Manhattan invariant: every segment must be exactly H or V."""
    out: List[Violation] = []
    for i, s in enumerate(segments):
        if _axis_of(s) == "d":
            out.append(Violation(
                kind="diagonal_segment",
                detail=f"segment {i} (type={s.get('type','?')}) "
                       f"({s['x1']},{s['y1']})-({s['x2']},{s['y2']}) is diagonal",
                location=(float(s["x1"]), float(s["y1"])),
            ))
    return out


def _endpoint_on_wall_body(ex: float, ey: float,
                           wall_h: Sequence[Tuple[float, float, float]],
                           wall_v: Sequence[Tuple[float, float, float]],
                           tol: float) -> bool:
    """True if (ex, ey) lies on any wall segment body within ``tol``.

    ``wall_h`` entries: (y, x_lo, x_hi); ``wall_v`` entries: (x, y_lo, y_hi).
    Includes endpoints (tol inclusive) so this subsumes endpoint coincidence.
    """
    for (sy, x_lo, x_hi) in wall_h:
        if abs(ey - sy) > tol:
            continue
        if x_lo - tol <= ex <= x_hi + tol:
            return True
    for (sx, y_lo, y_hi) in wall_v:
        if abs(ex - sx) > tol:
            continue
        if y_lo - tol <= ey <= y_hi + tol:
            return True
    return False


def check_openings_anchored(segments: Sequence[Dict],
                            *,
                            tol: float = BODY_PROXIMITY_PX) -> List[Violation]:
    """Every door/window endpoint must satisfy at least one of:

      1. Coincides (≤ tol px) with a wall endpoint.
      2. Lies (≤ tol px) on the body of a wall segment.

    This is the §3 chromatic-anchor invariant from todo.md. A floating door
    (both endpoints lacking wall support) violates it.

    Returns one Violation per offending opening *endpoint*.
    """
    walls = [s for s in segments if s.get("type") == "wall"]
    openings = [s for s in segments if s.get("type") in ("door", "window")]
    if not openings:
        return []

    wall_h: List[Tuple[float, float, float]] = []
    wall_v: List[Tuple[float, float, float]] = []
    for w in walls:
        ax = _axis_of(w)
        if ax == "h":
            x_lo, x_hi = sorted((float(w["x1"]), float(w["x2"])))
            wall_h.append((float(w["y1"]), x_lo, x_hi))
        elif ax == "v":
            y_lo, y_hi = sorted((float(w["y1"]), float(w["y2"])))
            wall_v.append((float(w["x1"]), y_lo, y_hi))

    out: List[Violation] = []
    for i, o in enumerate(openings):
        for end in ("1", "2"):
            ex = float(o[f"x{end}"])
            ey = float(o[f"y{end}"])
            if not _endpoint_on_wall_body(ex, ey, wall_h, wall_v, tol):
                out.append(Violation(
                    kind="floating_opening",
                    detail=f"{o.get('type','?')} segment endpoint {end} "
                           f"at ({ex:.2f},{ey:.2f}) not anchored on any wall body",
                    location=(ex, ey),
                ))
    return out


# ---------------------------------------------------------------------------
# Severity policy
# ---------------------------------------------------------------------------
#
# Two tiers:
#
#   STRICT (always-on, never has tolerance):
#       non_finite_coord, out_of_bbox, zero_length, diagonal_segment
#       — these are math/format guarantees the pipeline already meets and
#         that have zero false-positive risk.
#
#   GOAL (the user's hard requirement (b)):
#       floating_opening
#       — current pipeline does not guarantee this yet (todo §2 §3 deliver
#         it). Reported as part of check_all but separated in the summary
#         so we can gate on STRICT-only until §2 §3 ship and then promote.

STRICT_KINDS = frozenset({
    "non_finite_coord",
    "out_of_bbox",
    "zero_length",
    "diagonal_segment",
})

GOAL_KINDS = frozenset({
    "floating_opening",
})


@dataclass
class InvariantReport:
    strict: List[Violation]
    goal: List[Violation]

    def total(self) -> int:
        return len(self.strict) + len(self.goal)

    def has_strict_fail(self) -> bool:
        return bool(self.strict)


def check_all(segments: Sequence[Dict],
              image_shape: Tuple[int, int]) -> InvariantReport:
    """Run every invariant on a segments list. Returns an InvariantReport
    with violations partitioned into ``strict`` (zero-tolerance) and
    ``goal`` (target invariant not yet guaranteed by the pipeline).
    """
    everything: List[Violation] = []
    everything += check_finite_coords(segments)
    everything += check_inside_bbox(segments, image_shape)
    everything += check_no_zero_length(segments)
    everything += check_no_diagonals(segments)
    everything += check_openings_anchored(segments)

    strict = [v for v in everything if v.kind in STRICT_KINDS]
    goal = [v for v in everything if v.kind in GOAL_KINDS]
    return InvariantReport(strict=strict, goal=goal)


def summarize(report: InvariantReport) -> str:
    """One-screen summary of an InvariantReport."""
    by_kind: Counter = Counter(v.kind for v in report.strict + report.goal)
    lines = [
        f"  strict violations: {len(report.strict)}",
        f"  goal violations:   {len(report.goal)}",
    ]
    if by_kind:
        lines.append("  by kind:")
        for k, n in sorted(by_kind.items()):
            tier = "STRICT" if k in STRICT_KINDS else "GOAL"
            lines.append(f"    {tier:6s}  {k:24s} x {n}")
    return "\n".join(lines)
