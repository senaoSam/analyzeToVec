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
