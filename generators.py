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


# Wall-priority table — keep in sync with vectorize.TYPE_PRIORITY. Walls
# anchor every snap; chromatic endpoints snap onto walls, never the other
# direction. Duplicated here to keep this module free of upward imports.
_TYPE_PRIORITY: Dict[str, int] = {"wall": 0, "window": 1, "door": 1}


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


# ---------------------------------------------------------------------------
# Step 7 phase 1: endpoint_fuse_candidates  (replaces fuse_close_endpoints)
# ---------------------------------------------------------------------------
#
# Legacy ``fuse_close_endpoints`` is a 1D-cluster-per-axis pass: collect every
# endpoint's x (and every y), sort, walk to build clusters where consecutive
# values are within the pairwise min thickness-aware tol, then rewrite each
# endpoint's coord to the cluster's wall-priority mean. x and y clusters are
# independent (mutating x doesn't touch y).
#
# This generator emits one ``Candidate`` per 1D cluster that has at least one
# member needing to move. Each candidate is a mutate-only batch (``add=[]``,
# ``remove=[]``); the mutates relocate every cluster member to the canonical
# coord along that axis dimension. The accept loop is fixed-point with
# ``skip_score=True`` — the geometric gate (within thickness-aware tol +
# wall-priority anchor) is the safety net, and the score signals for
# sub-pixel coord canonicalisation are too weak to gate reliably (most
# fuses move endpoints < 2 px, smaller than the integer-count granularity
# of free_endpoint / junction / pseudo_junction).
#
# x and y axes are independent: a candidate carries the ``axis_dim`` it
# operates on, so accept loop can interleave the two without breaking
# bit-identical legacy semantics. Cluster definition matches legacy
# ``_cluster`` in vectorize.fuse_close_endpoints exactly (sort + walk +
# min-of-neighbour-tol join + wall-priority mean canonical).

def endpoint_fuse_candidates(segments: List[Dict],
                              seg_tols: Sequence[float],
                              ) -> List[C.Candidate]:
    """Emit one ``mutate``-only candidate per axis-direction 1D cluster.

    Args:
        segments: list of segment dicts.
        seg_tols: per-segment thickness-aware tol (length == len(segments)).
            For an endpoint owned by ``segments[i]``, this is the tol it
            contributes when clustering. Two endpoints with tols ``ta``,
            ``tb`` join into the same cluster when their sort-adjacent
            coord delta is <= ``min(ta, tb)`` — the strict legacy rule.
    """
    if not segments:
        return []
    if len(seg_tols) != len(segments):
        raise ValueError("seg_tols must have same length as segments")

    cands: List[C.Candidate] = []

    for axis_dim in ("x", "y"):
        # items: (coord, seg_idx, end, type, tol)
        items: List[Tuple[float, int, str, str, float]] = []
        for i, s in enumerate(segments):
            tol = float(seg_tols[i])
            for end in ("1", "2"):
                items.append((float(s[f"{axis_dim}{end}"]),
                              i, end, s.get("type", ""), tol))
        if len(items) < 2:
            continue
        items.sort(key=lambda it: it[0])

        clusters: List[List[Tuple[float, int, str, str, float]]] = [[items[0]]]
        for it in items[1:]:
            prev = clusters[-1][-1]
            join_tol = min(it[4], prev[4])
            if it[0] - prev[0] <= join_tol:
                clusters[-1].append(it)
            else:
                clusters.append([it])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            prios = [_TYPE_PRIORITY.get(c[3], 99) for c in cluster]
            top = min(prios)
            anchor_vals = [c[0] for c, p in zip(cluster, prios) if p == top]
            canonical = float(np.mean(anchor_vals))

            mutates: List[Tuple[int, str, float, float]] = []
            touched_endpoints: List[Tuple[int, str]] = []
            for (coord, seg_idx, end, _t, _tol) in cluster:
                if abs(coord - canonical) <= 1e-12:
                    continue
                seg = segments[seg_idx]
                if axis_dim == "x":
                    new_x = canonical
                    new_y = float(seg[f"y{end}"])
                else:
                    new_x = float(seg[f"x{end}"])
                    new_y = canonical
                mutates.append((seg_idx, end, new_x, new_y))
                touched_endpoints.append((seg_idx, end))

            if not mutates:
                continue

            cands.append(C.Candidate(
                op="fuse",
                add=[],
                remove=[],
                mutate=mutates,
                meta={
                    "axis_dim": axis_dim,
                    "canonical": canonical,
                    "cluster_size": len(cluster),
                    "moved_count": len(mutates),
                    "spread": cluster[-1][0] - cluster[0][0],
                    "touched_endpoints": touched_endpoints,
                },
            ))

    return cands


# ---------------------------------------------------------------------------
# Step 7 phase 2: t_project_candidates  (replaces manhattan_t_project)
# ---------------------------------------------------------------------------
#
# Legacy ``manhattan_t_project``: for every endpoint (i, end), find the closest
# orthogonal trunk j whose body covers the endpoint's along-axis coord within
# trunk j's thickness-aware tol, then rewrite the endpoint to that trunk's
# exact projection. Each endpoint gets ONE projection at most (the closest
# valid trunk). Wall-priority: walls never project onto chromatic trunks.
#
# Translation: one Candidate per (i, end) with a valid trunk match, carrying
# a single-endpoint mutate. The accept loop tracks ``used_endpoints`` so each
# (i, end) is mutated at most once — mirrors legacy single-pass semantics
# without needing a fresh regenerate per iteration. The trunk projection is
# computed from the original geometry inside the generator and baked into
# the mutate's ``(new_x, new_y)``, so subsequent accepts can't be skewed
# by partial mutations (the new_x / new_y is fixed at generation time).
#
# No fixed-point loop here. A second pass would only kick in if an already-
# projected endpoint became near a *different* trunk after its initial
# projection — legacy explicitly doesn't do that, and step-4.9.7 ablation
# already confirmed the existing ``manhattan_intersection_snap`` pass (which
# did a similar two-pass thing) was a pure no-op. So single-shot via
# ``used_endpoints`` matches.

def t_project_candidates(segments: List[Dict],
                          seg_tols: Sequence[float],
                          ) -> List[C.Candidate]:
    """Emit one mutate-only candidate per endpoint with a valid T-projection.

    Each endpoint scans every strictly-orthogonal segment, gates on:
      - axis orthogonality (H endpoint -> V trunk, and vice versa)
      - wall-priority (priority(self) >= priority(trunk); walls don't snap
        onto chromatic trunks)
      - along-axis containment: the endpoint's along-trunk coord lies in
        ``[lo - trunk_tol, hi + trunk_tol]`` of the trunk's body
      - perpendicular distance to the trunk's line <= trunk_tol

    Of all gates-passing trunks, the closest (smallest perpendicular
    distance) wins; tie-broken by trunk index. Each candidate carries
    ``meta["endpoint_key"] = (seg_idx, end)`` so the accept loop can
    enforce "at most one projection per endpoint".
    """
    if not segments:
        return []
    if len(seg_tols) != len(segments):
        raise ValueError("seg_tols must have same length as segments")

    n = len(segments)
    axes = [_axis_of(s) for s in segments]
    cands: List[C.Candidate] = []

    for i in range(n):
        seg = segments[i]
        my_axis = axes[i]
        if my_axis not in ("h", "v"):
            continue
        my_prio = _TYPE_PRIORITY.get(seg.get("type", ""), 99)

        for end in ("1", "2"):
            ex = float(seg[f"x{end}"])
            ey = float(seg[f"y{end}"])
            best_d = float("inf")
            best_target: Optional[Tuple[float, float, int]] = None

            for j in range(n):
                if j == i:
                    continue
                trunk_axis = axes[j]
                if trunk_axis == my_axis:
                    continue
                if trunk_axis not in ("h", "v"):
                    continue
                trunk = segments[j]
                trunk_prio = _TYPE_PRIORITY.get(trunk.get("type", ""), 99)
                # Wall-priority: forbid projecting a higher-priority (lower
                # number, e.g. wall=0) endpoint onto a lower-priority
                # (chromatic) trunk.
                if my_prio < trunk_prio:
                    continue
                trunk_tol = float(seg_tols[j])
                if trunk_axis == "v":
                    line_x = float(trunk["x1"])
                    lo, hi = sorted((float(trunk["y1"]), float(trunk["y2"])))
                    if not (lo - trunk_tol <= ey <= hi + trunk_tol):
                        continue
                    d = abs(ex - line_x)
                    if d <= trunk_tol and d < best_d:
                        best_d = d
                        best_target = (line_x, ey, j)
                else:  # trunk is horizontal
                    line_y = float(trunk["y1"])
                    lo, hi = sorted((float(trunk["x1"]), float(trunk["x2"])))
                    if not (lo - trunk_tol <= ex <= hi + trunk_tol):
                        continue
                    d = abs(ey - line_y)
                    if d <= trunk_tol and d < best_d:
                        best_d = d
                        best_target = (ex, line_y, j)

            if best_target is None:
                continue
            new_x, new_y, trunk_idx = best_target
            # No-op projection: same coord already.
            if abs(new_x - ex) <= 1e-12 and abs(new_y - ey) <= 1e-12:
                continue
            cands.append(C.Candidate(
                op="t_project",
                add=[],
                remove=[],
                mutate=[(i, end, new_x, new_y)],
                meta={
                    "endpoint_key": (i, end),
                    "trunk_idx": trunk_idx,
                    "projection_dist": best_d,
                    "self_axis": my_axis,
                },
            ))

    return cands


# ---------------------------------------------------------------------------
# Step 7 phase 5: endpoint_cluster_2d_candidates  (replaces snap_endpoints)
# ---------------------------------------------------------------------------
#
# Legacy ``snap_endpoints`` builds a graph where every endpoint is a node and
# an edge joins two nodes within Euclidean distance ``tol`` (2D circular tol,
# *not* rectangular). Connected components are then rewritten: every member
# endpoint moves to the component's wall-priority anchor mean.
#
# This is fundamentally different from ``endpoint_fuse_candidates``'s 1D
# rectangular-tol clustering: in 2D, ``dist=sqrt(dx^2 + dy^2) <= tol`` is
# strictly tighter than ``|dx| <= tol AND |dy| <= tol`` -- two endpoints at
# (0, 0) and (4, 4) have 2D dist 5.66 but 1D max dist 4. With tol=5, the 1D
# version would fuse them, the 2D version wouldn't. Phase 5 attempt 1 tried
# reusing the 1D fuse and failed regression hard (wall IOU drop ~18 % on
# both source and sg2) because pre-canonical chromatic endpoints near walls
# get incorrectly pulled across corners by the 1D tol.
#
# The generator emits one Candidate per non-singleton connected component;
# each candidate is a mutate-only batch that moves every component member
# to the canonical (anchor mean rounded to 4 decimals). Legacy rounds *every*
# output endpoint to 4 decimals including singletons -- handled outside the
# generator by the accept-loop wrapper.

def endpoint_cluster_2d_candidates(segments: List[Dict],
                                    tol: float,
                                    ) -> List[C.Candidate]:
    """Build a 2D-circular-tol endpoint cluster graph and emit one candidate
    per non-singleton component.

    Mirrors ``vectorize.snap_endpoints`` exactly:
      * graph nodes = (seg_idx, end), in iteration order so node indices
        match legacy ``k_a = 2*i``, ``k_b = 2*i + 1``
      * edges = pairs within Euclidean tol (O(N^2); fine for N <= 250)
      * for each component, anchor = top-priority members' coord mean,
        non-anchor members snap toward the anchor mean
      * canonical coords rounded to 4 decimals at candidate-build time so
        ``apply_candidate`` writes the same rounded values legacy writes

    The candidate's ``meta["component_size"]`` lets the accept loop sort
    by largest cluster first (bigger fuses earlier = fewer late-stage
    micro-adjustments).
    """
    if not segments:
        return []

    pts: List[Tuple[int, str, float, float]] = []
    for i, s in enumerate(segments):
        pts.append((i, "1", float(s["x1"]), float(s["y1"])))
        pts.append((i, "2", float(s["x2"]), float(s["y2"])))
    n = len(pts)
    if n < 2:
        return []

    # Union-find over endpoint pairs within tol. Avoids the optional
    # networkx dependency that would otherwise be needed only here; for
    # n <= 250 (our worst case) the O(N^2) edge scan is microseconds.
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

    tol2 = tol * tol
    for i in range(n):
        xi, yi = pts[i][2], pts[i][3]
        for j in range(i + 1, n):
            dx = pts[j][2] - xi
            dy = pts[j][3] - yi
            if dx * dx + dy * dy <= tol2:
                _union(i, j)

    comps: Dict[int, List[int]] = {}
    for k in range(n):
        comps.setdefault(_find(k), []).append(k)

    cands: List[C.Candidate] = []
    for root, members in comps.items():
        if len(members) < 2:
            continue
        # Wall-priority anchor mean (matches legacy exactly).
        prios = [_TYPE_PRIORITY.get(segments[pts[k][0]].get("type", ""), 99)
                 for k in members]
        top = min(prios)
        anchor_xs = [pts[k][2] for k, p in zip(members, prios) if p == top]
        anchor_ys = [pts[k][3] for k, p in zip(members, prios) if p == top]
        cx = round(float(np.mean(anchor_xs)), 4)
        cy = round(float(np.mean(anchor_ys)), 4)

        mutates: List[Tuple[int, str, float, float]] = []
        for k in members:
            seg_idx, end, x, y = pts[k]
            if abs(x - cx) <= 1e-12 and abs(y - cy) <= 1e-12:
                continue
            mutates.append((seg_idx, end, cx, cy))
        if not mutates:
            continue
        cands.append(C.Candidate(
            op="cluster_2d",
            add=[],
            remove=[],
            mutate=mutates,
            meta={
                "component_size": len(members),
                "moved_count": len(mutates),
                "canonical": (cx, cy),
            },
        ))

    return cands


# ---------------------------------------------------------------------------
# Step 8: cluster_collinear_merge_candidates  (replaces early merge_collinear)
# ---------------------------------------------------------------------------
#
# Legacy ``merge_collinear`` is structurally a cluster-component merge:
#   1. bucket by (type, axis)
#   2. within each bucket, 1D-cluster by perp coord within ``perp_tol`` (walk
#      sorted, consecutive-within-tol joins; this is connected-component
#      clustering on 1D sorted perp coords -- same shape as the 1D fuse
#      generator's clustering)
#   3. within each perp-cluster, sort by along-axis lo; sweep merging while
#      ``next.lo - cur.hi <= gap_tol``; when the sweep fails the gap test,
#      flush the current group and start a new one
#   4. each "along-group" (the result of the sweep) becomes one output
#      segment with canon_line = length-weighted mean of *own segment
#      length* of every member
#
# The phase-1 ``collinear_merge_candidates`` (pair-based) cannot reproduce
# this bit-identical: when chains of 3+ segments fold transitively, phase 1's
# fixed-point loop uses the *union* along-axis length as the weight after the
# first merge (because the merged seg's ``len_a = a_hi - a_lo`` is now the
# union), whereas legacy keeps each original segment's own length in the
# weighted mean. The difference is sub-pixel but real -- step 6 phase 2
# observed it as drift, reverted.
#
# This generator emits one Candidate per along-group, ``remove`` listing every
# member's index and ``add`` carrying one new merged segment. The accept loop
# applies them as atomic batches; clusters are disjoint by construction (the
# bucket + 1D-perp-cluster + along-sweep partition is a partition) so order
# of accept doesn't matter. Score gate is bypassed (``skip_score=True``) --
# the geometric gate is identical to legacy and is the entire safety net.

def cluster_collinear_merge_candidates(segments: List[Dict],
                                        *,
                                        perp_tol: float,
                                        gap_tol: float,
                                        ) -> List[C.Candidate]:
    """1:1 port of legacy ``vectorize.merge_collinear``.

    Bucket by (type, axis); 1D-cluster by perp coord within ``perp_tol``;
    inside each perp-cluster sort by along-axis lo and sweep-merge while
    ``next.lo - cur.hi <= gap_tol``. Each along-group with >= 2 members
    becomes one ``Candidate(op="merge", remove=[...], add=[merged_seg])``.

    The merged segment's line coordinate is the length-weighted mean of
    every original member's ``own`` along-axis length -- *not* the
    running union's length, which is what pair-based fixed-point merging
    drifts into. Diagonals pass through (no candidate emitted).
    """
    if not segments:
        return []

    h_buckets: Dict[str, List[int]] = {}
    v_buckets: Dict[str, List[int]] = {}
    for i, s in enumerate(segments):
        ax = _axis_of(s)
        if ax == "h":
            h_buckets.setdefault(s.get("type", ""), []).append(i)
        elif ax == "v":
            v_buckets.setdefault(s.get("type", ""), []).append(i)

    cands: List[C.Candidate] = []

    def _process(idxs: List[int], axis: str, type_name: str) -> None:
        if len(idxs) < 2:
            return
        # items[k] = (orig_idx, lo, hi, line, own_length)
        items: List[Tuple[int, float, float, float, float]] = []
        for i in idxs:
            seg = segments[i]
            if axis == "h":
                lo, hi = sorted((float(seg["x1"]), float(seg["x2"])))
                line = float(seg["y1"])
            else:
                lo, hi = sorted((float(seg["y1"]), float(seg["y2"])))
                line = float(seg["x1"])
            items.append((i, lo, hi, line, hi - lo))

        # 1D perp-cluster (matches legacy: walk sorted by line, join when
        # delta <= perp_tol).
        order = sorted(range(len(items)), key=lambda k: items[k][3])
        perp_clusters: List[List[int]] = []
        for k in order:
            if perp_clusters and items[k][3] - items[perp_clusters[-1][-1]][3] <= perp_tol:
                perp_clusters[-1].append(k)
            else:
                perp_clusters.append([k])

        for perp_cluster in perp_clusters:
            # Sweep-and-merge by along-axis lo within gap_tol.
            perp_cluster.sort(key=lambda k: items[k][1])
            cur_members: List[int] = [perp_cluster[0]]
            cur_lo = items[perp_cluster[0]][1]
            cur_hi = items[perp_cluster[0]][2]
            for k in perp_cluster[1:]:
                lo, hi = items[k][1], items[k][2]
                if lo - cur_hi <= gap_tol:
                    cur_members.append(k)
                    cur_hi = max(cur_hi, hi)
                else:
                    _emit(cur_members, cur_lo, cur_hi, axis, type_name, items)
                    cur_members = [k]
                    cur_lo, cur_hi = lo, hi
            _emit(cur_members, cur_lo, cur_hi, axis, type_name, items)

    def _emit(member_ks: List[int],
              lo: float, hi: float,
              axis: str, type_name: str,
              items: List[Tuple[int, float, float, float, float]]) -> None:
        # Emit a candidate for EVERY along-group, including singletons. The
        # wrapper relies on the candidates emerging in legacy's iteration
        # order to assemble the output list in that order; if singletons
        # were skipped here they'd be left in their original positions and
        # the resulting segment ordering would break downstream order-
        # sensitive passes (``t_junction_snap`` / ``truncate_overshoots``
        # both read mutated state from earlier iterations and so depend
        # on segment-list ordering).
        if not member_ks:
            return
        # Length-weighted line from each member's OWN length (not union).
        # IMPORTANT: must use Python's ``sum()`` builtin (NOT a manual
        # ``+=`` accumulator) to stay bit-identical with legacy
        # ``vectorize._length_weighted``. From Python 3.12 onward ``sum()``
        # uses Neumaier compensated summation on float iterables, which
        # produces a ULP-level different result from a naive ``+=`` loop
        # for clusters with 4+ members. Legacy uses ``sum(...)`` so we
        # must too.
        total_w = sum(max(items[k][4], 1e-6) for k in member_ks)
        weighted = sum(items[k][3] * max(items[k][4], 1e-6)
                       for k in member_ks)
        canon_line = weighted / total_w

        if axis == "h":
            merged_seg = {"type": type_name,
                          "x1": lo, "y1": canon_line,
                          "x2": hi, "y2": canon_line}
        else:
            merged_seg = {"type": type_name,
                          "x1": canon_line, "y1": lo,
                          "x2": canon_line, "y2": hi}

        remove_idxs = [items[k][0] for k in member_ks]
        cands.append(C.Candidate(
            op="merge",
            add=[merged_seg],
            remove=remove_idxs,
            mutate=[],
            meta={
                "axis": axis,
                "type": type_name,
                "cluster_size": len(member_ks),
                "merged_len": hi - lo,
                "canon_line": canon_line,
            },
        ))

    for t, idxs in h_buckets.items():
        _process(idxs, "h", t)
    for t, idxs in v_buckets.items():
        _process(idxs, "v", t)

    return cands


# ---------------------------------------------------------------------------
# Step 9 phase 1: axis_align_candidates  (replaces axis_align_segments)
# ---------------------------------------------------------------------------
#
# Legacy ``axis_align_segments`` walks the segment list and, for each near-
# axis segment (within ``tol_deg`` of an axis), snaps both endpoints onto
# the axis by setting the line coord to the mean of both endpoints. Zero-
# length segments are pruned.
#
# Each segment's transformation is independent of every other -- no
# inter-segment dependency. So we emit:
#   * a ``prune`` Candidate (remove=[i]) for each zero-length segment
#   * an ``axis_align`` Candidate (mutate=[(i,"1",..),(i,"2",..)]) for each
#     near-H segment with non-trivial drift (and likewise for near-V)
#   * no candidate for segments that are already exactly axis-aligned or
#     that fall outside the tol cone (diagonals): they pass through.
#
# The wrapper batches all candidates and applies them in one pass while
# preserving legacy's input-order iteration (the per-seg dict rewrite means
# order-of-application doesn't matter for the segment's content, but the
# output list must still be in legacy order so downstream order-sensitive
# passes see the same sequence).

def axis_align_candidates(segments: List[Dict],
                           tol_deg: float,
                           ) -> List[C.Candidate]:
    """Emit per-segment axis-snap candidates.

    Mirrors ``vectorize.axis_align_segments`` exactly: prune zero-length
    inputs; for each remaining segment whose angle is within ``tol_deg``
    of an axis, snap onto that axis by averaging both endpoints' line
    coord. Diagonals beyond the tol cone pass through with no candidate.
    """
    if not segments:
        return []
    rad = float(np.deg2rad(tol_deg))
    cands: List[C.Candidate] = []
    for i, seg in enumerate(segments):
        x1 = float(seg["x1"]); y1 = float(seg["y1"])
        x2 = float(seg["x2"]); y2 = float(seg["y2"])
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            # Legacy ``continue``s on zero-length: emit a prune candidate.
            cands.append(C.Candidate(
                op="prune",
                add=[],
                remove=[i],
                mutate=[],
                meta={"reason": "zero_length"},
            ))
            continue
        ang = float(np.arctan2(dy, dx))
        # Near-horizontal: snap y1, y2 -> mean
        if abs(ang) < rad or abs(abs(ang) - np.pi) < rad:
            ymid = 0.5 * (y1 + y2)
            if abs(y1 - ymid) > 1e-12 or abs(y2 - ymid) > 1e-12:
                cands.append(C.Candidate(
                    op="axis_align",
                    add=[],
                    remove=[],
                    mutate=[(i, "1", x1, ymid),
                            (i, "2", x2, ymid)],
                    meta={"axis": "h", "drift": max(abs(y1 - ymid),
                                                    abs(y2 - ymid))},
                ))
        elif abs(abs(ang) - np.pi / 2) < rad:
            xmid = 0.5 * (x1 + x2)
            if abs(x1 - xmid) > 1e-12 or abs(x2 - xmid) > 1e-12:
                cands.append(C.Candidate(
                    op="axis_align",
                    add=[],
                    remove=[],
                    mutate=[(i, "1", xmid, y1),
                            (i, "2", xmid, y2)],
                    meta={"axis": "v", "drift": max(abs(x1 - xmid),
                                                    abs(x2 - xmid))},
                ))
        # else: diagonal beyond tol cone -- pass through, no candidate.
    return cands


# ---------------------------------------------------------------------------
# Step 9 phase 2: truncate_overshoot_candidates  (replaces truncate_overshoots)
# ---------------------------------------------------------------------------
#
# Legacy ``truncate_overshoots``: for each axis-aligned segment's endpoint,
# scan every orthogonal segment. If the endpoint is within ``tol`` of the
# trunk's line AND the trunk's body covers the endpoint's along-axis coord
# (within tol) AND the segment's OTHER endpoint is on the opposite side of
# the trunk's line (i.e. the segment crosses through the trunk by a small
# margin), pull the endpoint back to the trunk's line. Only the
# perpendicular coord is mutated; the segment's own line coord stays.
#
# The "crossing" check ``(ex - line_x) * (far_x - line_x) < 0`` is the
# distinguishing gate vs ``t_project_candidates``: t_project pulls
# endpoints onto a trunk regardless of side; truncate_overshoots only
# fires when the segment genuinely passes through the trunk by a few px.
# This is a different semantic that t_project's generator deliberately
# doesn't capture.
#
# Each endpoint takes AT MOST one mutation (matches legacy's iterative
# fallback by accepting the first valid trunk under the closest-wins
# tie-break). ``used_endpoints`` set enforces this in the wrapper.

def truncate_overshoot_candidates(segments: List[Dict],
                                   tol: float,
                                   ) -> List[C.Candidate]:
    """Emit one mutate-only candidate per endpoint that overshoots an
    orthogonal trunk within ``tol`` and crosses through it.

    Mirrors legacy ``vectorize.truncate_overshoots`` bit-identically by
    simulating the order-dependent mutation cascade in a local copy:

      * legacy iterates ``(i, end)`` then inner ``j``, mutating ``segs[i]``
        in place; subsequent ``(i', end')`` iterations read the mutated
        state when scanning trunks
      * our simulation runs the same nested loop on a local ``segs_sim``;
        when an iteration mutates ``segs_sim[i]`` later iterations see
        the new coords (because we sample the trunk's coord from
        ``segs_sim[j]`` LIVE inside the inner loop)
      * after the full simulation each mutated endpoint's *final*
        coord is captured as one ``Candidate``; applying these to the
        original input via ``apply_candidate`` reproduces the legacy
        final state exactly

    Single-pass version (without live mutation) drifted on sg2 by dN +1
    / dFree -1 / wall IOU 0.99; the cascade matters.
    """
    if not segments:
        return []
    n = len(segments)
    segs_sim = [dict(s) for s in segments]
    axes = [_axis_of(s) for s in segs_sim]
    final_mutates: List[Tuple[int, str, float, float]] = []

    for i in range(n):
        if axes[i] not in ("h", "v"):
            continue
        my_axis = axes[i]
        for end in ("1", "2"):
            ex0 = float(segs_sim[i][f"x{end}"])
            ey0 = float(segs_sim[i][f"y{end}"])
            far_end = "2" if end == "1" else "1"
            # legacy reads far_x / far_y once at the top of the (i, end)
            # iteration -- it's the OTHER endpoint's coord, which this
            # inner j-loop never mutates, so a single read is correct.
            far_x = float(segs_sim[i][f"x{far_end}"])
            far_y = float(segs_sim[i][f"y{far_end}"])
            ex = ex0
            ey = ey0
            mutated = False
            for j in range(n):
                if j == i:
                    continue
                if axes[j] == my_axis or axes[j] not in ("h", "v"):
                    continue
                other = segs_sim[j]   # LIVE: reflects prior mutations
                if axes[j] == "v":
                    line_x = float(other["x1"])
                    olo, ohi = sorted((float(other["y1"]),
                                       float(other["y2"])))
                    if not (abs(ex - line_x) <= tol and
                            (olo - tol) <= ey <= (ohi + tol)):
                        continue
                    if (ex - line_x) * (far_x - line_x) >= 0:
                        continue
                    segs_sim[i][f"x{end}"] = line_x
                    ex = line_x
                    mutated = True
                else:  # axes[j] == "h"
                    line_y = float(other["y1"])
                    olo, ohi = sorted((float(other["x1"]),
                                       float(other["x2"])))
                    if not (abs(ey - line_y) <= tol and
                            (olo - tol) <= ex <= (ohi + tol)):
                        continue
                    if (ey - line_y) * (far_y - line_y) >= 0:
                        continue
                    segs_sim[i][f"y{end}"] = line_y
                    ey = line_y
                    mutated = True
            if mutated and (abs(ex - ex0) > 1e-12 or
                            abs(ey - ey0) > 1e-12):
                final_mutates.append((i, end, ex, ey))

    cands: List[C.Candidate] = []
    for (i, end, new_x, new_y) in final_mutates:
        cands.append(C.Candidate(
            op="truncate",
            add=[],
            remove=[],
            mutate=[(i, end, new_x, new_y)],
            meta={"endpoint_key": (i, end)},
        ))
    return cands


# ---------------------------------------------------------------------------
# Step 9 phase 3: canonicalize_offset_candidates  (replaces canonicalize_offsets)
# ---------------------------------------------------------------------------
#
# Legacy ``canonicalize_offsets``: per (type, axis) bucket, 1D-cluster
# segments by their perp offset (y for H, x for V) within an adaptive
# thickness-aware tol; per cluster, the offset of every member is
# rewritten to the length-weighted MEDIAN of the cluster.
#
# Two key differences vs ``endpoint_fuse_candidates``:
#   1. operates on SEGMENT perp offsets (not endpoint coords). For an H
#      segment both y1 and y2 carry the same offset; mutate both to
#      preserve the H invariant
#   2. canonical is length-weighted MEDIAN (the offset value whose
#      cumulative length first crosses 50% of cluster total), not mean.
#      Robust against short-stub outliers next to a long wall
#
# The ``attach_thickness=True`` side-channel (which writes a
# ``local_thickness`` field onto every output segment for downstream
# thickness-aware-tol consumers) is handled by the wrapper -- the
# candidate API doesn't add fields, only mutates existing ones.

def canonicalize_offset_candidates(segments: List[Dict],
                                    *,
                                    thicknesses: Sequence[float],
                                    abs_min: float = 2.0,
                                    abs_max: float = 6.0,
                                    thickness_frac: float = 0.25,
                                    fallback_tol: float = 3.0,
                                    ) -> List[C.Candidate]:
    """Emit one mutate-only candidate per offset cluster (>=2 members).

    Args:
        segments: full segment list.
        thicknesses: per-segment local thickness (length must match
            ``segments``). Caller computes from the wall-mask distance
            transform once and reuses for tol computation.
        abs_min, abs_max: clamp range for the adaptive perp-cluster tol.
        thickness_frac: ``tol = thickness_frac * median(group thickness)``,
            clamped.
        fallback_tol: applied when no group thickness is computable.
    """
    if not segments:
        return []
    if len(thicknesses) != len(segments):
        raise ValueError("thicknesses must align with segments")

    # Bucket by (type, axis).
    by_key: Dict[Tuple[str, str], List[int]] = {}
    for i, seg in enumerate(segments):
        ax = _axis_of(seg)
        if ax in ("h", "v"):
            by_key.setdefault((seg.get("type", ""), ax), []).append(i)

    cands: List[C.Candidate] = []

    for (_type, ax), idxs in by_key.items():
        if len(idxs) < 2:
            continue
        # items[k] = (orig_idx, offset, length, thickness)
        items: List[Tuple[int, float, float, float]] = []
        for i in idxs:
            seg = segments[i]
            if ax == "h":
                off = float(seg["y1"])
                length = abs(float(seg["x2"]) - float(seg["x1"]))
            else:
                off = float(seg["x1"])
                length = abs(float(seg["y2"]) - float(seg["y1"]))
            items.append((i, off, length, float(thicknesses[i])))

        # Adaptive tol from in-group thicknesses.
        valid_th = [it[3] for it in items if it[3] > 0]
        if valid_th:
            med = float(np.median(valid_th))
            tol = max(abs_min, min(abs_max, thickness_frac * med))
        else:
            tol = max(abs_min, min(abs_max, fallback_tol))

        items.sort(key=lambda it: it[1])
        clusters: List[List[Tuple[int, float, float, float]]] = [[items[0]]]
        for it in items[1:]:
            if it[1] - clusters[-1][-1][1] <= tol:
                clusters[-1].append(it)
            else:
                clusters.append([it])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            # Length-weighted median (legacy ``_length_weighted_median``):
            # sort by offset, walk cumulative length, return the offset
            # whose cumulative weight first crosses 50% of total.
            pairs = sorted(((c[1], max(c[2], 1e-6)) for c in cluster),
                           key=lambda p: p[0])
            total = sum(L for _, L in pairs)
            half = 0.5 * total
            cum = 0.0
            canonical = pairs[-1][0]
            for off, L in pairs:
                cum += L
                if cum >= half:
                    canonical = off
                    break

            mutates: List[Tuple[int, str, float, float]] = []
            for (i, off, _len, _th) in cluster:
                if abs(off - canonical) <= 1e-9:
                    continue
                seg = segments[i]
                if ax == "h":
                    # Mutate both endpoints' y to canonical; keep x's.
                    mutates.append((i, "1", float(seg["x1"]), canonical))
                    mutates.append((i, "2", float(seg["x2"]), canonical))
                else:
                    mutates.append((i, "1", canonical, float(seg["y1"])))
                    mutates.append((i, "2", canonical, float(seg["y2"])))
            if not mutates:
                continue
            cands.append(C.Candidate(
                op="canonical_offset",
                add=[],
                remove=[],
                mutate=mutates,
                meta={
                    "axis": ax,
                    "type": _type,
                    "canonical": canonical,
                    "cluster_size": len(cluster),
                    "tol_used": tol,
                },
            ))

    return cands


# ---------------------------------------------------------------------------
# Step 10 phase 1: manhattan_force_axis_candidates  (replaces manhattan_force_axis)
# ---------------------------------------------------------------------------
#
# Legacy ``manhattan_force_axis`` is the brutal "Manhattanization" step:
# every segment is force-classified as H or V based on which dimension is
# longer, then the off-axis coord is collapsed to the midpoint. There is
# no tol / gate; even genuinely diagonal segments are forced. Zero-length
# inputs are pruned.
#
# Per-segment independent transformation. Translation to candidates:
#   * ``prune`` candidate for zero-length inputs
#   * ``force_axis`` candidate (mutate both endpoints) for each segment
#     with non-trivial perpendicular drift (i.e. not already strictly
#     axis-aligned)
# Already-axis-aligned segments get no candidate (pass through).

def manhattan_force_axis_candidates(segments: List[Dict]) -> List[C.Candidate]:
    """Emit per-segment Manhattan-force candidates.

    Mirrors ``vectorize.manhattan_force_axis`` exactly:
      * zero-length: prune candidate (legacy ``continue``s)
      * |dx| >= |dy|: H -- mutate y1, y2 -> mean(y1, y2); x's unchanged
      * else: V -- mutate x1, x2 -> mean(x1, x2); y's unchanged
      * if already perfectly axis-aligned, no candidate emitted
    """
    if not segments:
        return []
    cands: List[C.Candidate] = []
    for i, seg in enumerate(segments):
        x1 = float(seg["x1"]); y1 = float(seg["y1"])
        x2 = float(seg["x2"]); y2 = float(seg["y2"])
        if x1 == x2 and y1 == y2:
            cands.append(C.Candidate(
                op="prune",
                add=[],
                remove=[i],
                mutate=[],
                meta={"reason": "zero_length"},
            ))
            continue
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx >= dy:
            ymid = 0.5 * (y1 + y2)
            if y1 != ymid or y2 != ymid:
                cands.append(C.Candidate(
                    op="force_axis",
                    add=[],
                    remove=[],
                    mutate=[(i, "1", x1, ymid),
                            (i, "2", x2, ymid)],
                    meta={"axis": "h", "drift": max(abs(y1 - ymid),
                                                    abs(y2 - ymid))},
                ))
        else:
            xmid = 0.5 * (x1 + x2)
            if x1 != xmid or x2 != xmid:
                cands.append(C.Candidate(
                    op="force_axis",
                    add=[],
                    remove=[],
                    mutate=[(i, "1", xmid, y1),
                            (i, "2", xmid, y2)],
                    meta={"axis": "v", "drift": max(abs(x1 - xmid),
                                                    abs(x2 - xmid))},
                ))
    return cands
