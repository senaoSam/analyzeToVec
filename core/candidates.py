"""Candidate-generation infrastructure for the step-4 refactor.

Three gates that the 5 candidate-based passes (insert_missing_connectors,
brute_force_ray_extend, extend_trunk_to_loose, mask_gated_l_extend,
t_snap_with_extension) consult before proposing a repair:

  1. SpatialGate  — uniform-grid bucket index. Endpoint and axis-line
                    queries return only nearby candidates so the O(N²) pair
                    scan in each old pass becomes O(N · neighbours).

  2. SemanticGate — table of which (type_a, type_b, operation) repairs are
                    legal. Wall-to-wall snap is legal, opening-to-opening
                    is not, etc. Filters candidate kinds before scoring.

  3. evidence_gate / mask_support_along — checks how much of a proposed
                    segment's path lies on a binary mask (or evidence map).
                    Bucketed into high / maybe / reject tiers.

None of these gates mutate state. They are filters that the candidate
generators consult before submitting proposals to ``scoring.compute_score``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 4.2 SpatialGate — uniform-grid bucket index
# ---------------------------------------------------------------------------

def _axis_of(seg: Dict) -> str:
    dx = abs(seg["x2"] - seg["x1"])
    dy = abs(seg["y2"] - seg["y1"])
    if dy < 1e-6 and dx > 1e-6:
        return "h"
    if dx < 1e-6 and dy > 1e-6:
        return "v"
    return "d"


@dataclass
class HAxisRecord:
    idx: int        # segment index in the source list
    x0: float       # min x along the body
    x1: float       # max x along the body
    y: float        # axis line


@dataclass
class VAxisRecord:
    idx: int
    y0: float
    y1: float
    x: float


class SpatialGate:
    """2D bucket grid plus axis-aware indexes.

    bucket_size is set per-pipeline-run to::

        max(L_EXTEND_TOL_PX, GAP_CLOSE_TOL_PX) * scale

    so the same number of buckets fits each query radius regardless of
    image dimensions. The endpoint grid uses (bucket_x, bucket_y) cells.
    The axis indexes bucket only along the perpendicular axis (H by y, V
    by x) since lookups are always "find lines whose axis is within radius
    of this query coordinate".
    """

    def __init__(self, segments: Sequence[Dict], *, bucket_size: float):
        if bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")
        self.bucket = float(bucket_size)
        self._segments: List[Dict] = list(segments)
        self._endpoint_grid: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
        self._h_axis: Dict[int, List[HAxisRecord]] = {}
        self._v_axis: Dict[int, List[VAxisRecord]] = {}
        for i, s in enumerate(self._segments):
            self._insert(i, s)

    # bucket helpers ---------------------------------------------------------

    def _b(self, v: float) -> int:
        return int(np.floor(v / self.bucket))

    # insertion --------------------------------------------------------------

    def _insert(self, i: int, s: Dict) -> None:
        for end in ("1", "2"):
            x = float(s["x" + end])
            y = float(s["y" + end])
            cell = (self._b(x), self._b(y))
            self._endpoint_grid.setdefault(cell, []).append((i, end))

        ax = _axis_of(s)
        if ax == "h":
            x0, x1 = sorted((float(s["x1"]), float(s["x2"])))
            y = float(s["y1"])
            self._h_axis.setdefault(self._b(y), []).append(
                HAxisRecord(idx=i, x0=x0, x1=x1, y=y))
        elif ax == "v":
            y0, y1 = sorted((float(s["y1"]), float(s["y2"])))
            x = float(s["x1"])
            self._v_axis.setdefault(self._b(x), []).append(
                VAxisRecord(idx=i, y0=y0, y1=y1, x=x))

    # queries ----------------------------------------------------------------

    def endpoints_near(self, x: float, y: float,
                       radius: float) -> List[Tuple[int, str]]:
        """Endpoints (seg_idx, "1"|"2") within radius of (x, y).

        Includes the queried point itself if it happens to be an endpoint
        — callers filter that themselves.
        """
        bx, by = self._b(x), self._b(y)
        rb = int(np.ceil(radius / self.bucket))
        r2 = radius * radius
        out: List[Tuple[int, str]] = []
        for dx in range(-rb, rb + 1):
            for dy in range(-rb, rb + 1):
                for (idx, end) in self._endpoint_grid.get((bx + dx, by + dy), ()):
                    px = float(self._segments[idx]["x" + end])
                    py = float(self._segments[idx]["y" + end])
                    if (px - x) * (px - x) + (py - y) * (py - y) <= r2:
                        out.append((idx, end))
        return out

    def h_segments_near_y(self, y: float, perp_radius: float,
                          x_min: Optional[float] = None,
                          x_max: Optional[float] = None
                          ) -> List[HAxisRecord]:
        """H segments whose axis line is within perp_radius of y. If x_min
        / x_max are given, also require the body interval to overlap
        [x_min, x_max].
        """
        by = self._b(y)
        rb = int(np.ceil(perp_radius / self.bucket))
        out: List[HAxisRecord] = []
        for dy in range(-rb, rb + 1):
            for rec in self._h_axis.get(by + dy, ()):
                if abs(rec.y - y) > perp_radius:
                    continue
                if x_min is not None and rec.x1 < x_min:
                    continue
                if x_max is not None and rec.x0 > x_max:
                    continue
                out.append(rec)
        return out

    def v_segments_near_x(self, x: float, perp_radius: float,
                          y_min: Optional[float] = None,
                          y_max: Optional[float] = None
                          ) -> List[VAxisRecord]:
        bx = self._b(x)
        rb = int(np.ceil(perp_radius / self.bucket))
        out: List[VAxisRecord] = []
        for dx in range(-rb, rb + 1):
            for rec in self._v_axis.get(bx + dx, ()):
                if abs(rec.x - x) > perp_radius:
                    continue
                if y_min is not None and rec.y1 < y_min:
                    continue
                if y_max is not None and rec.y0 > y_max:
                    continue
                out.append(rec)
        return out

    def loose_endpoints(self) -> List[Tuple[float, float]]:
        """All endpoints whose 1-px-rounded coordinate occurs exactly once
        across the full segment list. Convenience for passes that operate
        on free endpoints.
        """
        from .geom_utils import node_degree
        cnt = node_degree(self._segments)
        return [pt for pt, c in cnt.items() if c == 1]


# ---------------------------------------------------------------------------
# 4.3 SemanticGate — legal repair combinations
# ---------------------------------------------------------------------------

# Operations recognized by the gate. Each candidate generator passes its op
# name; SemanticGate decides whether a given (type_a, type_b) is permitted.
#
#   gap_close   — bridge two endpoints with a synthetic segment along an axis
#   t_snap      — pull an endpoint onto the body of an orthogonal segment
#   l_extend    — stretch an endpoint to meet a perpendicular partner's line
#   attach      — door / window endpoint onto a wall (or vice versa)
#
# Walls anchor everything; door / window never anchor a wall. Door↔window
# never repair against each other — they share no architectural relation
# beyond both being "openings", and bridging them produces garbage geometry.

_OP_RULES: Dict[str, Dict[Tuple[str, str], bool]] = {
    "gap_close": {
        ("wall", "wall"): True,
        # opening-to-opening explicitly disallowed; any (opening, *) likewise.
    },
    "t_snap": {
        ("wall", "wall"): True,
        # An opening endpoint snapping onto a wall body is handled by
        # "attach"; "t_snap" is reserved for wall-on-wall corner closure.
    },
    "l_extend": {
        ("wall", "wall"): True,
    },
    "attach": {
        ("door", "wall"): True,
        ("window", "wall"): True,
        ("wall", "door"): True,
        ("wall", "window"): True,
    },
}


def is_legal(op: str, type_a: str, type_b: str) -> bool:
    rules = _OP_RULES.get(op)
    if not rules:
        return False
    return rules.get((type_a, type_b), False)


# ---------------------------------------------------------------------------
# 4.4 EvidenceGate — mask support along a candidate path
# ---------------------------------------------------------------------------

# Tier thresholds. Tighter than "all-or-nothing" so a candidate that runs
# through partial mask coverage (e.g. across a doorway) is still
# representable for the scoring step instead of being silently rejected.
EVIDENCE_HIGH_THRESHOLD = 0.75
EVIDENCE_MAYBE_THRESHOLD = 0.45
EVIDENCE_GAP_CONNECTOR_MIN = 0.30   # spec: gap connectors below this rejected


def mask_support_along(mask: Optional[np.ndarray],
                       x1: float, y1: float, x2: float, y2: float,
                       samples_per_px: float = 1.0) -> float:
    """Fraction of sampled pixels along the segment that lie on the mask.

    Returns 1.0 if mask is None (caller treats as "evidence not available,
    don't filter"). Returns 0.0 if the segment has zero length.
    """
    if mask is None:
        return 1.0
    L = float(np.hypot(x2 - x1, y2 - y1))
    if L < 1.0:
        return 0.0
    n = max(2, int(round(L * samples_per_px)))
    xs = np.linspace(x1, x2, n).round().astype(int)
    ys = np.linspace(y1, y2, n).round().astype(int)
    h, w = mask.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return float((mask[ys, xs] > 0).sum()) / float(n)


def evidence_tier(support: float) -> str:
    """Map a 0.0–1.0 support value to high / maybe / reject."""
    if support >= EVIDENCE_HIGH_THRESHOLD:
        return "high"
    if support >= EVIDENCE_MAYBE_THRESHOLD:
        return "maybe"
    return "reject"


def gap_connector_passes(support: float) -> bool:
    """True iff a synthetic gap-closing connector should even be scored.

    Per todo.md spec: support < 0.30 → don't bother building a candidate.
    """
    return support >= EVIDENCE_GAP_CONNECTOR_MIN


# ---------------------------------------------------------------------------
# Candidate type
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A proposed repair. ``op`` names the kind for SemanticGate; ``add``
    and ``remove`` describe the geometric edit; ``mutate`` describes
    endpoint relocations on existing segments.

    A candidate is consumed by the score-and-apply loop:

        1. cheap gates accept (else skipped)
        2. compute_score(before)
        3. apply (a copy), compute_score(after)
        4. delta > MIN_ACCEPT_DELTA → keep the apply
    """
    op: str
    add: List[Dict]
    remove: List[int] = None        # source-list indices to drop
    mutate: List[Tuple[int, str, float, float]] = None  # (idx, end, new_x, new_y)
    meta: Dict = None               # debug info — kind, support tier, …

    def __post_init__(self):
        if self.remove is None:
            self.remove = []
        if self.mutate is None:
            self.mutate = []
        if self.meta is None:
            self.meta = {}


def apply_candidate(segments: List[Dict], cand: Candidate) -> List[Dict]:
    """Return a new list of segments with the candidate applied.

    Mutations on existing segments are written into deep copies so the
    input list is not aliased. Removed indices are dropped. Added segments
    are appended.
    """
    out: List[Dict] = []
    remove_set: Set[int] = set(cand.remove)
    # Build per-index mutation map.
    mut: Dict[int, List[Tuple[str, float, float]]] = {}
    for (idx, end, x, y) in cand.mutate:
        mut.setdefault(idx, []).append((end, x, y))
    for i, s in enumerate(segments):
        if i in remove_set:
            continue
        if i in mut:
            ns = dict(s)
            for (end, x, y) in mut[i]:
                ns["x" + end] = float(x)
                ns["y" + end] = float(y)
            out.append(ns)
        else:
            out.append(s)
    for new_seg in cand.add:
        out.append(dict(new_seg))
    return out
