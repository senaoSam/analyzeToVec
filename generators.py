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
from typing import Dict, List, Optional, Sequence

import numpy as np

import candidates as C


# ---------------------------------------------------------------------------
# Constants (callers pass scale-aware values per pipeline run)
# ---------------------------------------------------------------------------

# Default minimum mask support for any new segment in a bridge.
BRIDGE_MIN_SUPPORT = 0.85

# Default colinear tolerance: |dx| or |dy| below this counts as "shared
# axis", which routes the proposal to the axis-bridge (single new segment)
# case instead of the L-bridge (two-segment corner).
BRIDGE_COLINEAR_TOL_PX = 2.5


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
                                ) -> List[C.Candidate]:
    """Propose bridges between nearby pairs of wall endpoints.

    For every (endpoint_a, endpoint_b) pair on **different** wall segments
    that sit within ``max_radius`` of each other and are not already
    coincident, enumerate the geometrically valid axis-aligned bridges
    connecting them, gate by mask support, and return the survivors as
    ``Candidate`` objects (with ``add=`` populated; no ``mutate`` /
    ``remove``).

    Bridge enumeration cases:

      - **axis-bridge** (1 new wall): if |dx| ≤ colinear_tol XOR
        |dy| ≤ colinear_tol — the endpoints share a near-common axis;
        a single new H or V segment joins them.

      - **L-bridge** (2 new walls): if both |dx| > colinear_tol and
        |dy| > colinear_tol — neither axis aligns. Two corner orientations
        are proposed (corner at (b.x, a.y) and corner at (a.x, b.y));
        each yields one H + one V new segment. The scorer picks between
        them via the same delta-evaluation as every other candidate.

    The gates:

      - **wall_mask gate**: each proposed segment is sampled against the
        wall mask; if any falls below ``min_support`` fraction on-mask,
        the proposal is dropped before scoring. This is the same evidence
        gate the older mask-based passes use, generalised here.

      - **chromatic avoidance** is *implicit* via the wall_mask: per
        ``segment_colors``, the wall mask has door / window pixels
        suppressed, so a bridge across an opening reports near-zero
        support and is rejected.

    The mask gate is the only generator-side reject; everything else
    (invalid crossings, phantom, junction count, free_endpoint count) is
    judged by ``scoring.compute_score`` when the caller runs the
    score-and-accept loop on the returned list.
    """
    if wall_mask is None or not segments:
        return []

    # All wall endpoints with their owning (seg_idx, end).
    wall_endpoints: List[Dict] = []
    for i, s in enumerate(segments):
        if s.get("type") != "wall":
            continue
        for end in ("1", "2"):
            wall_endpoints.append({
                "seg": i,
                "end": end,
                "x": float(s[f"x{end}"]),
                "y": float(s[f"y{end}"]),
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
            dx = abs(dxs)
            dy = abs(dys)

            # Enumerate bridge proposals.
            bridges: List[tuple] = []  # (segs_to_add, kind)

            if dy <= colinear_tol and dx > colinear_tol:
                # Axis-bridge along y (shared horizontal axis).
                cy = 0.5 * (a["y"] + b["y"])
                bridges.append((
                    [{"type": "wall",
                      "x1": min(a["x"], b["x"]), "y1": cy,
                      "x2": max(a["x"], b["x"]), "y2": cy}],
                    "axis_h"
                ))
            elif dx <= colinear_tol and dy > colinear_tol:
                # Axis-bridge along x (shared vertical axis).
                cx = 0.5 * (a["x"] + b["x"])
                bridges.append((
                    [{"type": "wall",
                      "x1": cx, "y1": min(a["y"], b["y"]),
                      "x2": cx, "y2": max(a["y"], b["y"])}],
                    "axis_v"
                ))
            elif dx > colinear_tol and dy > colinear_tol:
                # L-bridge: two orientations.
                # Orientation 1: corner at (b.x, a.y) → H then V.
                bridges.append((
                    [{"type": "wall",
                      "x1": a["x"], "y1": a["y"],
                      "x2": b["x"], "y2": a["y"]},
                     {"type": "wall",
                      "x1": b["x"], "y1": a["y"],
                      "x2": b["x"], "y2": b["y"]}],
                    "l_h_first"
                ))
                # Orientation 2: corner at (a.x, b.y) → V then H.
                bridges.append((
                    [{"type": "wall",
                      "x1": a["x"], "y1": a["y"],
                      "x2": a["x"], "y2": b["y"]},
                     {"type": "wall",
                      "x1": a["x"], "y1": b["y"],
                      "x2": b["x"], "y2": b["y"]}],
                    "l_v_first"
                ))

            for new_segs, kind in bridges:
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
                    mutate=[],
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
