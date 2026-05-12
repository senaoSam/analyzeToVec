"""Candidate generators for the step-4 architecture.

Each generator is a *function* that, given the current pipeline state and
the mask/evidence side channels, returns a list of ``candidates.Candidate``
proposals. Generators never mutate state — they only propose. The caller
runs the score-and-accept loop on whatever they return.

The aim of consolidating generation in this module is to avoid the 19-pass
heuristic-soup re-incarnation that todo.md warns against. Each generator
covers a *class* of repair (not a single defect pattern), and the score
+ gates decide between proposals.

Currently implemented:

  proximal_bridge_candidates
      For pairs of wall endpoints within a configurable radius that are
      not yet connected, propose either an axis-bridge (1 new wall) or
      an L-bridge (2 new walls forming a corner). Mask support on every
      new segment must clear ``min_support``. Subsumes both the
      "two-loose-endpoints-share-an-axis" case (which the older
      ``insert_missing_connectors`` handled) and the "two-degree-2-corners
      that need an L-shaped bridge" case (which nothing in the prior
      pipeline addressed — observed as a wrong corner on sg2's top-right).

Future generators (step 5 territory) can fold the existing
``mask_gated_l_extend`` and the per-pass scaffolds of
``brute_force_ray_extend`` / ``extend_trunk_to_loose`` /
``t_snap_with_extension`` into the same module, once the candidate stream
is rich enough to let the score model do the work.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import candidates as C
from canonical_line import compute_local_thickness


def _axis_of(seg: Dict) -> str:
    """Return "h", "v" or "d" for the segment's axis (strict equality)."""
    dx = abs(float(seg["x2"]) - float(seg["x1"]))
    dy = abs(float(seg["y2"]) - float(seg["y1"]))
    if dy < 1e-6 and dx > 1e-6:
        return "h"
    if dx < 1e-6 and dy > 1e-6:
        return "v"
    return "d"


# ---------------------------------------------------------------------------
# Constants (callers pass scale-aware values per pipeline run)
# ---------------------------------------------------------------------------

# Default minimum mask support for any new segment in a bridge.
BRIDGE_MIN_SUPPORT = 0.85

# Default colinear tolerance: |dx| or |dy| below this counts as "shared
# axis", which routes the proposal to the axis-bridge (single new segment)
# case instead of the L-bridge (two-segment corner).
BRIDGE_COLINEAR_TOL_PX = 2.5

# Maximum stroke-width ratio between two endpoints' host wall segments for
# a bridge proposal to be even built. A 4 px wall and a 12 px wall almost
# never belong to the same logical wall — bridging them creates a phantom
# segment. The check is a no-op when either segment's local_thickness is
# unknown (returns ``-1`` from ``compute_local_thickness``).
BRIDGE_MAX_THICKNESS_RATIO = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_key(x: float, y: float, quantize: int = 1) -> tuple:
    return (int(round(x / quantize)) * quantize,
            int(round(y / quantize)) * quantize)


def _node_degree(segments: Sequence[Dict]) -> Counter:
    cnt: Counter = Counter()
    for s in segments:
        cnt[_round_key(s["x1"], s["y1"])] += 1
        cnt[_round_key(s["x2"], s["y2"])] += 1
    return cnt


# ---------------------------------------------------------------------------
# proximal_bridge_candidates
# ---------------------------------------------------------------------------

def proximal_bridge_candidates(segments: List[Dict],
                                wall_mask: Optional[np.ndarray],
                                *,
                                max_radius: float,
                                min_support: float = BRIDGE_MIN_SUPPORT,
                                colinear_tol: float = BRIDGE_COLINEAR_TOL_PX,
                                max_thickness_ratio: float = BRIDGE_MAX_THICKNESS_RATIO,
                                ) -> List[C.Candidate]:
    """Propose bridges between nearby pairs of wall endpoints.

    For every (endpoint_a, endpoint_b) pair on **different** wall segments
    that sit within ``max_radius`` of each other and are not already
    coincident, enumerate the geometrically valid axis-aligned bridges
    connecting them, gate by mask support, and return the survivors as
    ``Candidate`` objects.

    Bridge enumeration cases:

      - **axis-bridge** (1 new wall, optional safe endpoint-mutation):
        if |dx| ≤ colinear_tol XOR |dy| ≤ colinear_tol — the endpoints
        share a near-common axis; a single new H or V segment joins them.
        The candidate **also** mutates each source endpoint onto the
        bridge axis when that mutation is along the owning segment's
        free direction (i.e. mutating x of an H endpoint extends the H
        body, but mutating x of a V endpoint would break Manhattan and
        is forbidden). This is the case ``insert_missing_connectors``
        used to handle via direct mutate; folded in here.

      - **L-bridge** (2 new walls, no mutate): if both |dx| > colinear_tol
        and |dy| > colinear_tol — neither axis aligns. Two corner
        orientations are proposed (corner at (b.x, a.y) and corner at
        (a.x, b.y)); each yields one H + one V new segment. The scorer
        picks between them. junction-aware merge folds the new bridge
        into the host trunks downstream, giving the same final pixel
        set as the older mask_gated_l_extend mutate.

    The gates:

      - **wall_mask gate**: each proposed segment is sampled against the
        wall mask; if any falls below ``min_support`` fraction on-mask,
        the proposal is dropped before scoring.

      - **safe-mutate gate**: a mutation is emitted only when the
        endpoint's owning segment axis is perpendicular to the bridge
        direction (so the mutation extends the segment along its free
        axis, preserving H-/V-strictness).

      - **chromatic avoidance** is *implicit* via the wall_mask.

    Everything else (invalid crossings, phantom, junction count,
    free_endpoint count) is judged by ``scoring.compute_score`` when
    the caller runs the score-and-accept loop on the returned list.
    """
    if wall_mask is None or not segments:
        return []

    # Per-segment local thickness, sampled from the wall mask distance
    # transform once up front. Used downstream to reject bridge proposals
    # between endpoints whose host walls have incompatible stroke widths
    # (per step 4.9.6 — a single bridge across a thin-to-thick boundary
    # is almost always a phantom).
    dt = cv2.distanceTransform((wall_mask > 0).astype(np.uint8),
                               cv2.DIST_L2, 3)
    seg_thickness: Dict[int, float] = {}
    for i, s in enumerate(segments):
        if s.get("type") != "wall":
            continue
        seg_thickness[i] = compute_local_thickness(s, dt)

    # All wall endpoints with owning segment + the segment's axis. The
    # axis tells us which mutations are "safe" (axis-preserving).
    wall_endpoints: List[Dict] = []
    for i, s in enumerate(segments):
        if s.get("type") != "wall":
            continue
        ax = _axis_of(s)
        for end in ("1", "2"):
            wall_endpoints.append({
                "seg": i,
                "end": end,
                "x": float(s[f"x{end}"]),
                "y": float(s[f"y{end}"]),
                "seg_axis": ax,    # "h" / "v" / "d"
            })

    if len(wall_endpoints) < 2:
        return []

    r2 = max_radius * max_radius
    cands: List[C.Candidate] = []

    for ia in range(len(wall_endpoints)):
        a = wall_endpoints[ia]
        for ib in range(ia + 1, len(wall_endpoints)):
            b = wall_endpoints[ib]

            # Cannot bridge a segment to itself.
            if a["seg"] == b["seg"]:
                continue

            dxs = b["x"] - a["x"]
            dys = b["y"] - a["y"]
            d2 = dxs * dxs + dys * dys
            if d2 > r2:
                continue
            # Already coincident (or near-coincident) → already a
            # junction or "connected enough"; skip.
            if d2 < 1.0:
                continue

            # Step 4.9.6: stroke-width compatibility. Reject pairs whose
            # host wall segments have very different local thickness —
            # a thin-wall endpoint should not be bridged to a thick-wall
            # endpoint, because the resulting bridge would have to be one
            # uniform width, contradicting at least one of the inputs.
            # ``compute_local_thickness`` returns -1 for segments where
            # sampling failed (very short / off-mask); we let those
            # through unchecked rather than over-rejecting on weak data.
            ta = seg_thickness.get(a["seg"], -1.0)
            tb = seg_thickness.get(b["seg"], -1.0)
            if ta > 0 and tb > 0:
                ratio = max(ta, tb) / max(min(ta, tb), 1e-6)
                if ratio > max_thickness_ratio:
                    continue
            dx = abs(dxs)
            dy = abs(dys)

            # Enumerate bridge proposals as
            #   (segs_to_add, mutations, kind)
            # where mutations is a list of (seg_idx, end, new_x, new_y).
            bridges: List[Tuple[List[Dict], List, str]] = []

            if dy <= colinear_tol and dx > colinear_tol:
                # Axis-bridge along x (shared horizontal axis at y=cy):
                # a single new H wall. Bridge runs in x; the H-direction
                # of an endpoint's owning segment is "h", so an H-axis
                # endpoint can safely have its y mutated to cy (along V
                # free direction? no — wait: bridge runs horizontal at
                # y=cy. To mutate an endpoint *onto* the bridge, we need
                # to change that endpoint's y to cy. Changing y is along
                # the free direction of a V segment (V's body extends in
                # y). So only V endpoints can be safely y-mutated.
                cy = 0.5 * (a["y"] + b["y"])
                mutates = []
                if a["seg_axis"] == "v" and abs(a["y"] - cy) > 1e-6:
                    mutates.append((a["seg"], a["end"], a["x"], cy))
                if b["seg_axis"] == "v" and abs(b["y"] - cy) > 1e-6:
                    mutates.append((b["seg"], b["end"], b["x"], cy))
                bridges.append((
                    [{"type": "wall",
                      "x1": min(a["x"], b["x"]), "y1": cy,
                      "x2": max(a["x"], b["x"]), "y2": cy}],
                    mutates,
                    "axis_h"
                ))
            elif dx <= colinear_tol and dy > colinear_tol:
                # Axis-bridge along y (shared vertical axis at x=cx):
                # a single new V wall. Mutating x is along the free
                # direction of an H segment.
                cx = 0.5 * (a["x"] + b["x"])
                mutates = []
                if a["seg_axis"] == "h" and abs(a["x"] - cx) > 1e-6:
                    mutates.append((a["seg"], a["end"], cx, a["y"]))
                if b["seg_axis"] == "h" and abs(b["x"] - cx) > 1e-6:
                    mutates.append((b["seg"], b["end"], cx, b["y"]))
                bridges.append((
                    [{"type": "wall",
                      "x1": cx, "y1": min(a["y"], b["y"]),
                      "x2": cx, "y2": max(a["y"], b["y"])}],
                    mutates,
                    "axis_v"
                ))
            elif dx > colinear_tol and dy > colinear_tol:
                # L-bridge: two orientations. No mutate (would risk
                # orphaning the source corner; junction-aware merge of
                # the bridge into the existing trunks handles topology).
                bridges.append((
                    [{"type": "wall",
                      "x1": a["x"], "y1": a["y"],
                      "x2": b["x"], "y2": a["y"]},
                     {"type": "wall",
                      "x1": b["x"], "y1": a["y"],
                      "x2": b["x"], "y2": b["y"]}],
                    [],
                    "l_h_first"
                ))
                bridges.append((
                    [{"type": "wall",
                      "x1": a["x"], "y1": a["y"],
                      "x2": a["x"], "y2": b["y"]},
                     {"type": "wall",
                      "x1": a["x"], "y1": b["y"],
                      "x2": b["x"], "y2": b["y"]}],
                    [],
                    "l_v_first"
                ))

            for new_segs, mutates, kind in bridges:
                supports: List[float] = []
                ok = True
                for ns in new_segs:
                    sup = C.mask_support_along(
                        wall_mask, ns["x1"], ns["y1"], ns["x2"], ns["y2"])
                    supports.append(sup)
                    if sup < min_support:
                        ok = False
                        break
                if not ok:
                    continue

                total_len = sum(
                    float(np.hypot(ns["x2"] - ns["x1"], ns["y2"] - ns["y1"]))
                    for ns in new_segs)
                cands.append(C.Candidate(
                    op="bridge",
                    add=new_segs,
                    mutate=list(mutates),
                    meta={
                        "kind": kind,
                        "supports": supports,
                        "min_support_observed": min(supports),
                        "total_len": total_len,
                        "pair_endpoints": ((a["x"], a["y"]),
                                           (b["x"], b["y"])),
                        "pair_seg_ends": ((a["seg"], a["end"]),
                                          (b["seg"], b["end"])),
                    },
                ))

    return cands


# ---------------------------------------------------------------------------
# collinear_merge_candidates  (step 6, phase 1)
# ---------------------------------------------------------------------------
#
# Replaces the family of merge passes (``merge_collinear`` x2,
# ``cluster_parallel_duplicates``, ``manhattan_ultimate_merge``) with a single
# generator. Phase 1 covers the strict "exact-same-line + touching-or-
# overlapping" case (manhattan_ultimate_merge semantics); phases 2 / 3 will
# add the gap-tolerant collinear case and the perp-tolerant parallel-duplicate
# case respectively. Junction-awareness from the legacy pass is preserved:
# a candidate is *not* generated when the would-be interior point of the
# merged span coincides with an endpoint of any segment outside the pair.

_MERGE_QUANTIZE = 100  # 0.01-px grid for robust endpoint equality


def _qk(x: float, y: float) -> Tuple[int, int]:
    return (int(round(x * _MERGE_QUANTIZE)), int(round(y * _MERGE_QUANTIZE)))


def collinear_merge_candidates(segments: List[Dict],
                                *,
                                perp_tol: float = 0.0,
                                gap_tol: float = 0.0,
                                junction_aware: bool = True,
                                ) -> List[C.Candidate]:
    """Propose merge candidates for pairs of same-type, same-axis segments.

    For phase 1 the default tolerances reproduce the legacy
    ``manhattan_ultimate_merge``:

      - ``perp_tol = 0``: only segments with bit-identical line coordinate
        are paired (after the 0.01-px quantisation that the legacy uses
        for float robustness)
      - ``gap_tol = 0``: bodies must touch or overlap (``hi_a >= lo_b``)
      - ``junction_aware = True``: refuse to merge when the would-be
        interior coordinate is shared by an endpoint of any segment
        outside the bucket (preserves explicit T-junctions)

    Each emitted candidate carries ``remove=[i, j]`` and
    ``add=[merged_seg]``; the score-and-accept loop then decides whether
    to take it. The merged segment's line coordinate is a length-weighted
    mean (matches ``cluster_parallel_duplicates``); for the strict
    ``perp_tol=0`` case this is equivalent to either input's line.

    Phases 2 / 3 will widen the gates by passing ``perp_tol > 0`` and
    ``gap_tol > 0``; the generator itself does not need to change.
    """
    if not segments:
        return []

    # Bucket axis-aligned segments by (type, axis, quantised line coord).
    # With ``perp_tol > 0`` the same segment may land in two adjacent
    # buckets (the quantisation grid is 0.01 px so the over-bucketing
    # only happens at the exact tol boundary); we de-dup pairs at the
    # end via a seen-set keyed by (min(i,j), max(i,j)).
    h_buckets: Dict[Tuple[str, int], List[int]] = {}
    v_buckets: Dict[Tuple[str, int], List[int]] = {}

    def _line_keys(line: float) -> List[int]:
        """Quantised line coordinate(s) for bucketing.

        With perp_tol=0 returns one key (the rounded coord). With perp_tol>0
        we also insert the segment into neighbouring quantised buckets so
        a pair whose line coords differ by <= perp_tol still meets in at
        least one bucket.
        """
        base = int(round(line * _MERGE_QUANTIZE))
        if perp_tol <= 0:
            return [base]
        span = int(np.ceil(perp_tol * _MERGE_QUANTIZE))
        return list(range(base - span, base + span + 1))

    for i, s in enumerate(segments):
        ax = _axis_of(s)
        if ax == "h":
            for k in _line_keys(s["y1"]):
                h_buckets.setdefault((s["type"], k), []).append(i)
        elif ax == "v":
            for k in _line_keys(s["x1"]):
                v_buckets.setdefault((s["type"], k), []).append(i)

    # Endpoint counter for junction-aware filter (global, across all
    # segments — an external endpoint at the would-be interior is a
    # T-junction the merge would destroy).
    endpoint_count: Dict[Tuple[int, int], int] = {}
    for s in segments:
        for end in ("1", "2"):
            k = _qk(s[f"x{end}"], s[f"y{end}"])
            endpoint_count[k] = endpoint_count.get(k, 0) + 1

    cands: List[C.Candidate] = []
    seen_pairs: set = set()

    def _emit(i: int, j: int, axis: str) -> None:
        if i == j:
            return
        key = (min(i, j), max(i, j))
        if key in seen_pairs:
            return
        a, b = segments[i], segments[j]
        # Same type guaranteed by bucket key, axis guaranteed by bucket choice.
        if axis == "h":
            a_lo, a_hi = sorted((a["x1"], a["x2"]))
            b_lo, b_hi = sorted((b["x1"], b["x2"]))
            a_line, b_line = a["y1"], b["y1"]
        else:
            a_lo, a_hi = sorted((a["y1"], a["y2"]))
            b_lo, b_hi = sorted((b["y1"], b["y2"]))
            a_line, b_line = a["x1"], b["x1"]

        if abs(a_line - b_line) > perp_tol + 1e-9:
            return  # filter spurious cross-bucket pairs

        # Touching/overlapping check (in along-axis dimension).
        lo, hi = min(a_lo, b_lo), max(a_hi, b_hi)
        overlap_lo, overlap_hi = max(a_lo, b_lo), min(a_hi, b_hi)
        gap = overlap_lo - overlap_hi  # positive when there is a real gap
        if gap > gap_tol + 1e-9:
            return  # bodies neither touch nor overlap within tol

        # Junction-aware: if the merged span has an internal coordinate
        # that is also an external endpoint, refuse. Internal coords here
        # are any endpoint of either input that becomes strictly interior
        # of the union (i.e. is not lo or hi of the merged span).
        if junction_aware:
            cand_endpoints = [(a["x1"], a["y1"]), (a["x2"], a["y2"]),
                              (b["x1"], b["y1"]), (b["x2"], b["y2"])]
            own_keys = {_qk(x, y) for (x, y) in cand_endpoints}
            for (ex, ey) in cand_endpoints:
                # Is this endpoint strictly interior to the merged span?
                along = ex if axis == "h" else ey
                if abs(along - lo) < 1e-6 or abs(along - hi) < 1e-6:
                    continue  # it's a span boundary, not interior
                # Check if any segment OUTSIDE this pair also has an
                # endpoint here. The endpoint_count includes both pair
                # contributions; subtract them to get the external count.
                k = _qk(ex, ey)
                own_contrib = sum(1 for (cx, cy) in cand_endpoints
                                  if _qk(cx, cy) == k)
                external = endpoint_count.get(k, 0) - own_contrib
                if external > 0:
                    return  # merging would bury an external T-junction
        seen_pairs.add(key)

        # Length-weighted line coordinate.
        len_a = a_hi - a_lo
        len_b = b_hi - b_lo
        wsum = max(len_a + len_b, 1e-9)
        canon_line = (len_a * a_line + len_b * b_line) / wsum

        if axis == "h":
            merged = {"type": a["type"], "x1": lo, "y1": canon_line,
                      "x2": hi, "y2": canon_line}
        else:
            merged = {"type": a["type"], "x1": canon_line, "y1": lo,
                      "x2": canon_line, "y2": hi}

        cands.append(C.Candidate(
            op="merge",
            add=[merged],
            remove=[i, j],
            mutate=[],
            meta={
                "axis": axis,
                "merged_len": hi - lo,
                "gap": max(0.0, gap),
                "perp_dist": abs(a_line - b_line),
            },
        ))

    for (_type, _key), idxs in h_buckets.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            for j in idxs:
                if j <= i:
                    continue
                _emit(i, j, "h")
    for (_type, _key), idxs in v_buckets.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            for j in idxs:
                if j <= i:
                    continue
                _emit(i, j, "v")

    return cands


# ---------------------------------------------------------------------------
# parallel_merge_candidates  (step 6, phase 3)
# ---------------------------------------------------------------------------
#
# Subsumes ``cluster_parallel_duplicates``. Generates merge proposals for
# pairs of same-type same-axis segments that fall into either of the two
# legacy cases:
#
#   Case 1 (near-collinear touching): perp distance <= ``touch_perp_tol``
#          AND along-axis bodies touch or overlap (overlap >= 0). Catches
#          single-wall segments that the skeleton broke into a couple of
#          pieces with a tiny perpendicular drift.
#
#   Case 2 (thick-wall parallel duplicate): perp distance <= ``perp_tol``
#          (wider, ~13 px) AND along-axis body overlap >= ``min_overlap``
#          x shorter segment length. Catches the parallel-ridges case
#          where a thick wall mask skeletonised into two side-by-side
#          centerlines.
#
# Pair-based (not cluster-based) by design: the score-and-accept loop's
# fixed-point iteration handles transitive merges (A+B, then (AB)+C in
# the next iteration), and pair candidates preserve the partial-accept
# semantics that all-or-nothing cluster candidates lose.
#
# Pipeline position: runs AFTER canonicalize_offsets and the manhattan
# snaps, so coordinates are settled. This is the post-canonical
# environment that phase 1 worked in — score deltas are reliable here,
# unlike phase 2's pre-canonical experiment.

PARALLEL_TOUCH_PERP_TOL_PX = 12.0   # cluster_parallel_duplicates' inner TOUCH_PERP_TOL
PARALLEL_MIN_OVERLAP_RATIO = 0.5    # cluster_parallel_duplicates' >= 0.5 * shorter


def parallel_merge_candidates(segments: List[Dict],
                               *,
                               perp_tol: float,
                               touch_perp_tol: float = PARALLEL_TOUCH_PERP_TOL_PX,
                               min_overlap_ratio: float = PARALLEL_MIN_OVERLAP_RATIO,
                               ) -> List[C.Candidate]:
    """Pair-based parallel-merge proposals, mirroring legacy
    ``cluster_parallel_duplicates`` semantics."""
    if not segments:
        return []

    h_groups: Dict[str, List[int]] = {}
    v_groups: Dict[str, List[int]] = {}
    for i, s in enumerate(segments):
        ax = _axis_of(s)
        if ax == "h":
            h_groups.setdefault(s["type"], []).append(i)
        elif ax == "v":
            v_groups.setdefault(s["type"], []).append(i)

    cands: List[C.Candidate] = []

    def _emit_pair(i: int, j: int, axis: str) -> None:
        a, b = segments[i], segments[j]
        if axis == "h":
            a_lo, a_hi = sorted((a["x1"], a["x2"]))
            b_lo, b_hi = sorted((b["x1"], b["x2"]))
            a_line, b_line = a["y1"], b["y1"]
        else:
            a_lo, a_hi = sorted((a["y1"], a["y2"]))
            b_lo, b_hi = sorted((b["y1"], b["y2"]))
            a_line, b_line = a["x1"], b["x1"]

        dline = abs(a_line - b_line)
        if dline > perp_tol + 1e-9:
            return

        len_a = a_hi - a_lo
        len_b = b_hi - b_lo
        overlap_lo, overlap_hi = max(a_lo, b_lo), min(a_hi, b_hi)
        overlap = overlap_hi - overlap_lo  # >= 0 if bodies touch/overlap
        shorter = min(len_a, len_b)

        # Case 1: near-collinear touching/overlapping
        case_1 = (dline <= touch_perp_tol + 1e-9) and (overlap >= -1e-6)
        # Case 2: thick-wall parallel duplicate
        case_2 = (shorter > 0) and (overlap > 0) and \
                 (overlap >= min_overlap_ratio * shorter - 1e-9)

        if not (case_1 or case_2):
            return

        lo = min(a_lo, b_lo)
        hi = max(a_hi, b_hi)
        wsum = max(len_a + len_b, 1e-9)
        canon_line = (len_a * a_line + len_b * b_line) / wsum

        if axis == "h":
            merged = {"type": a["type"], "x1": lo, "y1": canon_line,
                      "x2": hi, "y2": canon_line}
        else:
            merged = {"type": a["type"], "x1": canon_line, "y1": lo,
                      "x2": canon_line, "y2": hi}

        cands.append(C.Candidate(
            op="merge",
            add=[merged],
            remove=[i, j],
            mutate=[],
            meta={
                "axis": axis,
                "merged_len": hi - lo,
                "perp_dist": dline,
                "overlap": max(0.0, overlap),
                "case": 1 if case_1 else 2,
            },
        ))

    for t, idxs in h_groups.items():
        for i in idxs:
            for j in idxs:
                if j <= i:
                    continue
                _emit_pair(i, j, "h")
    for t, idxs in v_groups.items():
        for i in idxs:
            for j in idxs:
                if j <= i:
                    continue
                _emit_pair(i, j, "v")

    return cands
