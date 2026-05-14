"""Score function for the candidate-based geometry pipeline (step 4).

Replaces the heuristic-soup "carry-and-pray" model of the 25-pass pipeline
with: every repair is a candidate, every candidate gets a delta score, and
only candidates with positive delta are accepted.

Two layers of terms (per todo.md):

    Primary (4, tunable weights):
        wall_evidence         — walls land on real wall pixels (higher is better)
        opening_evidence      — door/window segments land on chromatic pixels
        free_endpoint         — count of degree-1 endpoints (lower is better)
        invalid_crossing      — segment pairs crossing at interior points (lower is better)

    Derived (5, fixed coefficients; no separate tuning knobs):
        phantom               — wall pixels NOT supported by evidence (penalty)
        duplicate             — overlapping near-parallel same-type pairs (penalty)
        junction              — degree-3+ nodes (bonus, helps watertightness)
        opening_attachment    — chromatic endpoints connected to wall endpoints (bonus)
        manhattan_consistency — fraction of axis-aligned segments (bonus)

The total weights count is 4 (primary). Derived coefficients are baked in.
This keeps the tuning surface small enough to reason about and large enough
to express the trade-offs ablation showed were real.

This module is read-only with respect to the pipeline: nothing here mutates
segments. Callers pass a segments list (+ masks + scale) and get back a
PipelineScore with .terms and .total. Candidate evaluation does

    base   = compute_score(state)
    trial  = apply_candidate(state, candidate)
    after  = compute_score(trial)
    delta  = after.total - base.total      # accept if > MIN_ACCEPT_DELTA

with the breakdown in (after.terms[k] - base.terms[k]) available for audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# Primary-term weights (the 4 knobs).
PRIMARY_WEIGHTS: Dict[str, float] = {
    "wall_evidence":         1.0,
    "opening_evidence":      1.0,
    "free_endpoint":         1.0,
    "invalid_crossing":      5.0,
}

# Derived-term coefficients (no separate tuning surface).
DERIVED_COEFFS: Dict[str, float] = {
    "phantom":               -2.0,
    "duplicate":             -0.5,
    "junction":               0.5,
    "pseudo_junction":        0.5,
    "opening_attachment":     1.0,
    "manhattan_consistency":  0.5,
}

# Evidence thresholds used by phantom / opening_evidence checks.
PHANTOM_EVIDENCE_THRESHOLD = 0.5     # wall pixel below this → counted phantom

# Sampling density for evidence integrals.
EVIDENCE_SAMPLES_PER_PX = 1.0

# Tolerance for "is this endpoint coincident with another point" comparisons.
ENDPOINT_COINCIDENCE_PX = 1.0


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class PipelineScore:
    """Holds the per-term scores and the weighted total.

    ``terms`` carries the *signed* contribution of each term — negative for
    penalties — so delta auditing just subtracts dicts.
    """
    terms: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    def delta(self, other: "PipelineScore") -> Dict[str, float]:
        keys = set(self.terms) | set(other.terms)
        return {k: self.terms.get(k, 0.0) - other.terms.get(k, 0.0)
                for k in sorted(keys)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_along(seg: Dict, n_samples: int) -> np.ndarray:
    """Return Nx2 array of (x, y) sample points along the segment."""
    xs = np.linspace(seg["x1"], seg["x2"], n_samples)
    ys = np.linspace(seg["y1"], seg["y2"], n_samples)
    return np.stack([xs, ys], axis=1)


def _seg_len(seg: Dict) -> float:
    return float(np.hypot(seg["x2"] - seg["x1"], seg["y2"] - seg["y1"]))


def _evidence_integral(segments: Sequence[Dict],
                       evidence_map: np.ndarray,
                       samples_per_px: float = EVIDENCE_SAMPLES_PER_PX
                       ) -> float:
    """Average evidence value along the given segments.

    Returns 0.0 if no segments or no evidence map. Range: 0.0 — 1.0 for a
    map already in [0, 1].
    """
    if evidence_map is None or len(segments) == 0:
        return 0.0
    h, w = evidence_map.shape[:2]
    total_sum = 0.0
    total_count = 0
    for seg in segments:
        L = _seg_len(seg)
        if L < 1.0:
            continue
        n = max(2, int(round(L * samples_per_px)))
        pts = _sample_along(seg, n).round().astype(int)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        vals = evidence_map[ys, xs]
        total_sum += float(vals.sum())
        total_count += n
    return total_sum / max(total_count, 1)


def _segments_axis(seg: Dict) -> str:
    dx = abs(seg["x2"] - seg["x1"])
    dy = abs(seg["y2"] - seg["y1"])
    if dy < 1e-6 and dx > 1e-6:
        return "h"
    if dx < 1e-6 and dy > 1e-6:
        return "v"
    return "d"


def _segments_intersect_interior(a: Dict, b: Dict, tol: float = 1.0
                                  ) -> bool:
    """True iff segment A and segment B cross strictly inside both, i.e. the
    crossing point is at least ``tol`` px away from any of the four endpoints.

    Cheap special-case implementation for axis-aligned segments — the
    Manhattan pipeline only outputs H and V. Diagonals are reported as
    "no intersection" since we don't expect them at this stage and a general
    O(1) parametric formula would just add code with no current callers.
    """
    aa = _segments_axis(a)
    ba = _segments_axis(b)
    if aa == "d" or ba == "d":
        return False
    if aa == ba:
        return False  # parallel; no crossing
    # One is H, one is V. Find the V and the H.
    h_seg = a if aa == "h" else b
    v_seg = b if aa == "h" else a
    hy = h_seg["y1"]                  # H runs along this y
    vx = v_seg["x1"]                  # V runs along this x
    hx0, hx1 = sorted((h_seg["x1"], h_seg["x2"]))
    vy0, vy1 = sorted((v_seg["y1"], v_seg["y2"]))
    # Crossing point must lie strictly inside both segments' interior.
    if not (hx0 + tol <= vx <= hx1 - tol):
        return False
    if not (vy0 + tol <= hy <= vy1 - tol):
        return False
    return True


def _endpoint_iter(segments: Sequence[Dict]):
    for s in segments:
        yield (s["x1"], s["y1"])
        yield (s["x2"], s["y2"])


def _node_degree_counter(segments: Sequence[Dict]) -> Counter:
    """Count occurrences of each endpoint after 1-px rounding."""
    from geom_utils import node_degree
    return node_degree(segments)


# ---------------------------------------------------------------------------
# Primary terms
# ---------------------------------------------------------------------------

def wall_evidence_integral(walls: Sequence[Dict],
                           evidence_map: Optional[np.ndarray]) -> float:
    return _evidence_integral(walls, evidence_map) if evidence_map is not None else 0.0


def opening_evidence_integral(openings: Sequence[Dict],
                              door_mask: Optional[np.ndarray],
                              window_mask: Optional[np.ndarray]) -> float:
    if door_mask is None and window_mask is None:
        return 0.0
    h, w = (door_mask if door_mask is not None else window_mask).shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)
    if door_mask is not None:
        combined[door_mask > 0] = 1.0
    if window_mask is not None:
        combined[window_mask > 0] = 1.0
    return _evidence_integral(openings, combined)


def free_endpoint_count(segments: Sequence[Dict]) -> int:
    nodes = _node_degree_counter(segments)
    return sum(1 for v in nodes.values() if v == 1)


def invalid_crossing_count(segments: Sequence[Dict]) -> int:
    """Pairs of segments that cross at a strictly-interior point of both.

    O(N²). N is ~100 for our test images → ~10k pair tests, sub-millisecond.
    If this becomes hot under candidate sweeps we can swap in a spatial gate.
    """
    n = len(segments)
    if n < 2:
        return 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _segments_intersect_interior(segments[i], segments[j]):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Derived terms
# ---------------------------------------------------------------------------

def phantom_penalty(walls: Sequence[Dict],
                    evidence_map: Optional[np.ndarray],
                    threshold: float = PHANTOM_EVIDENCE_THRESHOLD) -> float:
    """Fraction of wall sample pixels with evidence below threshold.

    A wall segment running through white space (synthetic connector invented
    by ``insert_missing_connectors``) returns ~1.0 here. A wall sitting on
    real ink returns ~0.0.
    """
    if evidence_map is None or len(walls) == 0:
        return 0.0
    h, w = evidence_map.shape[:2]
    off = 0
    total = 0
    for seg in walls:
        L = _seg_len(seg)
        if L < 1.0:
            continue
        n = max(2, int(round(L * EVIDENCE_SAMPLES_PER_PX)))
        pts = _sample_along(seg, n).round().astype(int)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        vals = evidence_map[ys, xs]
        off += int((vals < threshold).sum())
        total += n
    return off / max(total, 1)


def duplicate_penalty(segments: Sequence[Dict],
                      perp_tol: float = 2.0) -> int:
    """Count pairs of same-type, near-parallel segments whose bodies overlap.

    Slightly looser than ``cluster_parallel_duplicates``'s merge criterion:
    we are flagging redundancy, not deciding to remove. Only same-type pairs
    count since cross-type "parallel" segments (door next to wall) are
    expected.
    """
    by_type: Dict[str, List[Dict]] = {}
    for s in segments:
        by_type.setdefault(s.get("type", ""), []).append(s)
    n_overlap = 0
    for type_name, segs in by_type.items():
        for i in range(len(segs)):
            ai = _segments_axis(segs[i])
            if ai == "d":
                continue
            for j in range(i + 1, len(segs)):
                aj = _segments_axis(segs[j])
                if aj != ai:
                    continue
                a, b = segs[i], segs[j]
                if ai == "h":
                    if abs(a["y1"] - b["y1"]) > perp_tol:
                        continue
                    ax0, ax1 = sorted((a["x1"], a["x2"]))
                    bx0, bx1 = sorted((b["x1"], b["x2"]))
                    overlap = min(ax1, bx1) - max(ax0, bx0)
                else:  # 'v'
                    if abs(a["x1"] - b["x1"]) > perp_tol:
                        continue
                    ay0, ay1 = sorted((a["y1"], a["y2"]))
                    by0, by1 = sorted((b["y1"], b["y2"]))
                    overlap = min(ay1, by1) - max(ay0, by0)
                if overlap > 0:
                    n_overlap += 1
    return n_overlap


def junction_count(segments: Sequence[Dict],
                   *,
                   wall_mask: Optional[np.ndarray] = None,
                   ) -> float:
    """Number of physical T-junctions (degree-3+ node clusters).

    Legacy (``wall_mask=None``): counts every distinct rounded-coord
    node with degree >= 3 separately. This over-counts the
    "thick-wall ridge artefact": when skeletonisation produces two
    parallel centerlines for one thick wall, each ridge has its own
    apparent T-junction where it meets an orthogonal trunk; legacy
    counts both, so collapsing the two ridges into one merged
    centerline (which preserves the same physical T) appears to
    *lose* one junction and the score penalises the merge.

    Thick-wall-aware (``wall_mask`` provided): cluster junction nodes
    that lie within local wall half-thickness of each other (sampled
    from the distance transform). Each cluster counts as one physical
    junction. Two ridges' T-nodes at (x, y0) and (x, y1) with
    ``|y1 - y0| <= half_thickness`` collapse into one physical
    junction; merging the ridges leaves the count unchanged.

    Half-thickness sampling: ``2 * dt[y, x]`` at the junction point
    gives the full local wall thickness; we use ``0.5 * max(t_i, t_j)``
    as the per-pair clustering radius so a thin partition with a thick
    wall's T-node nearby doesn't drag the partition's apparent junction
    into the thick wall's cluster.
    """
    nodes = _node_degree_counter(segments)
    pts = [pt for pt, d in nodes.items() if d >= 3]
    if not pts:
        return 0.0
    if wall_mask is None or wall_mask.size == 0:
        return float(len(pts))

    import cv2
    dt = cv2.distanceTransform((wall_mask > 0).astype(np.uint8),
                               cv2.DIST_L2, 3)
    h, w = dt.shape
    thicknesses: List[float] = []
    for (x, y) in pts:
        xi = int(min(max(x, 0), w - 1))
        yi = int(min(max(y, 0), h - 1))
        # 2 * dt = full local wall thickness at that point. If the
        # junction sits exactly off-mask (dt = 0), thickness is 0 and
        # the junction won't cluster with anything -- which is correct
        # for free-floating phantom junctions.
        thicknesses.append(2.0 * float(dt[yi, xi]))

    # Union-find by intra-thickness Euclidean proximity.
    n = len(pts)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            d2 = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
            tol = 0.5 * max(thicknesses[i], thicknesses[j])
            if tol > 0 and d2 <= tol * tol:
                _union(i, j)

    roots = set(_find(k) for k in range(n))
    return float(len(roots))


def pseudo_junction_count(segments: Sequence[Dict],
                          tol: float = 1.0) -> int:
    """Number of degree-1 endpoints that sit strictly on a *wall* segment's
    body interior.

    These are *deferred* T-junctions: the host wall is not (yet) split at
    the point, so the endpoint shows up as degree-1 in the node graph
    even though geometrically a T exists. Downstream passes like
    ``manhattan_t_project`` and ``fuse_close_endpoints`` materialise
    them into real degree-3 nodes.

    Only wall bodies count as valid hosts — doors and windows attach to
    walls but never anchor each other (matches SemanticGate's "attach"
    legality table). Without this filter, the score would reward a door
    endpoint snapping onto another door's body, which is geometric
    coincidence, not a real T-junction.

    The score rewards this state proportionally so that "snap loose
    endpoint onto orthogonal *wall* trunk's body" candidates earn a
    positive delta on their own — replacing the compound-macro approach
    that proved index-unsafe in the apply_candidate / batch-accept loop.
    """
    if not segments:
        return 0
    deg = _node_degree_counter(segments)
    loose_pts = [pt for pt, d in deg.items() if d == 1]
    if not loose_pts:
        return 0
    count = 0
    # Only walls qualify as junction hosts.
    h_walls: List[Tuple[float, float, float]] = []
    v_walls: List[Tuple[float, float, float]] = []
    for s in segments:
        if s.get("type") != "wall":
            continue
        ax = _segments_axis(s)
        if ax == "h":
            x0, x1 = sorted((float(s["x1"]), float(s["x2"])))
            h_walls.append((float(s["y1"]), x0, x1))
        elif ax == "v":
            y0, y1 = sorted((float(s["y1"]), float(s["y2"])))
            v_walls.append((float(s["x1"]), y0, y1))
    if not h_walls and not v_walls:
        return 0
    for (px, py) in loose_pts:
        hit = False
        for (sy, x0, x1) in h_walls:
            if abs(py - sy) > tol:
                continue
            if x0 + tol < px < x1 - tol:
                hit = True
                break
        if hit:
            count += 1
            continue
        for (sx, y0, y1) in v_walls:
            if abs(px - sx) > tol:
                continue
            if y0 + tol < py < y1 - tol:
                count += 1
                break
    return count


def opening_attachment_ratio(walls: Sequence[Dict],
                             openings: Sequence[Dict],
                             tol: float = ENDPOINT_COINCIDENCE_PX) -> float:
    """Fraction of opening endpoints that coincide with a wall endpoint.

    Properly-attached door / window jambs touch the wall they pierce. An
    isolated opening (no wall on either side) returns 0.0.
    """
    if not openings:
        return 0.0
    wall_pts = list(_endpoint_iter(walls))
    if not wall_pts:
        return 0.0
    wall_arr = np.array(wall_pts, dtype=np.float64)
    matched = 0
    total = 0
    for o in openings:
        for ex, ey in ((o["x1"], o["y1"]), (o["x2"], o["y2"])):
            total += 1
            dx = wall_arr[:, 0] - ex
            dy = wall_arr[:, 1] - ey
            d = float(np.min(np.hypot(dx, dy)))
            if d <= tol:
                matched += 1
    return matched / max(total, 1)


def manhattan_consistency(segments: Sequence[Dict],
                          tol_deg: float = 5.0) -> float:
    """Fraction of segments that are within tol_deg of an axis."""
    if not segments:
        return 1.0
    aligned = 0
    for s in segments:
        dx = abs(s["x2"] - s["x1"])
        dy = abs(s["y2"] - s["y1"])
        if dx < 1e-6 and dy < 1e-6:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if min(ang, 90.0 - ang) < tol_deg:
            aligned += 1
    return aligned / len(segments)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def compute_score(segments: Sequence[Dict],
                  *,
                  wall_evidence: Optional[np.ndarray] = None,
                  door_mask: Optional[np.ndarray] = None,
                  window_mask: Optional[np.ndarray] = None,
                  wall_mask: Optional[np.ndarray] = None,
                  primary_weights: Optional[Dict[str, float]] = None,
                  ) -> PipelineScore:
    """Compute the full pipeline score for ``segments``.

    Pass the wall *evidence map* (continuous 0.0—1.0, from
    ``vectorize.compute_wall_evidence``) for sharper wall_evidence and
    phantom signals. If only the binary wall mask is available, pass it as
    ``wall_evidence`` — the integral degenerates to "fraction on mask".

    ``wall_mask`` (separately from ``wall_evidence``) is the *binary* wall
    mask used for thick-wall-aware ``junction_count`` clustering. Pass it
    only when you want skeleton-ridge T-junctions on the same physical
    thick wall to count as one physical junction instead of two; pass
    ``None`` to use the legacy per-node count.
    """
    walls = [s for s in segments if s.get("type") == "wall"]
    openings = [s for s in segments if s.get("type") in ("door", "window")]

    terms: Dict[str, float] = {}

    # Primary terms.
    terms["wall_evidence"] = wall_evidence_integral(walls, wall_evidence)
    terms["opening_evidence"] = opening_evidence_integral(openings, door_mask, window_mask)
    terms["free_endpoint"] = -float(free_endpoint_count(segments))
    terms["invalid_crossing"] = -float(invalid_crossing_count(segments))

    # Derived terms (each signed so total contribution is +term*coef with
    # bonus coefficients positive and penalty coefficients negative).
    terms["phantom"] = phantom_penalty(walls, wall_evidence)
    terms["duplicate"] = float(duplicate_penalty(segments))
    terms["junction"] = junction_count(segments, wall_mask=wall_mask)
    terms["pseudo_junction"] = float(pseudo_junction_count(segments))
    terms["opening_attachment"] = opening_attachment_ratio(walls, openings)
    terms["manhattan_consistency"] = manhattan_consistency(segments)

    weights = dict(PRIMARY_WEIGHTS)
    if primary_weights:
        weights.update(primary_weights)

    total = 0.0
    for k, v in terms.items():
        if k in weights:
            total += weights[k] * v
        else:
            total += DERIVED_COEFFS[k] * v

    return PipelineScore(terms=terms, total=total)


def format_score(score: PipelineScore) -> str:
    """Human-readable one-block dump for verbose logging."""
    lines = [f"  total = {score.total:+.4f}"]
    for k in ("wall_evidence", "opening_evidence",
              "free_endpoint", "invalid_crossing",
              "phantom", "duplicate", "junction", "pseudo_junction",
              "opening_attachment", "manhattan_consistency"):
        v = score.terms.get(k, 0.0)
        w = PRIMARY_WEIGHTS.get(k, DERIVED_COEFFS.get(k, 1.0))
        lines.append(f"    {k:24s}  raw={v:+.4f}  weighted={w*v:+.4f}")
    return "\n".join(lines)
