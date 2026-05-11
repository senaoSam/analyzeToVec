"""Step 4.9: canonical line clustering + local thickness.

Per (type, axis) bucket, cluster segments whose perpendicular offsets fall
within an adaptive thickness-aware tolerance, then rewrite each segment's
perp coordinate to the cluster's length-weighted mean.

Unlike :func:`cluster_parallel_duplicates`, this does NOT require body
overlap. Its purpose is to canonicalise the 1-3 px y / x drift that
skeletonisation routinely introduces between segments that should lie on
the same logical line. Body-overlap is the wrong gate for that case: two
collinear-but-disjoint pieces of one physical wall are exactly the
pattern we want to fold here, and they are exactly what the existing
overlap-gated pass leaves alone.

Designed to run after :func:`manhattan_force_axis` (so every segment is
strict H or V) and before :func:`manhattan_intersection_snap` (so the
downstream T / L snap sees a single canonical offset per logical wall
line, which is what makes the snap stable across thick-wall corners).

Adaptive tolerance: ``clamp(thickness_frac * median_local_thickness,
abs_min, abs_max)``. The clamp is the physical safety net: never tighter
than ``abs_min`` (so the 1-3 px drift case always clusters) and never
wider than ``abs_max`` (so genuinely distinct adjacent walls a few pixels
apart never collapse). Local thickness comes from the distance transform
of the wall mask sampled along each segment's body.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _axis(seg: Dict) -> str:
    if seg["y1"] == seg["y2"] and seg["x1"] != seg["x2"]:
        return "h"
    if seg["x1"] == seg["x2"] and seg["y1"] != seg["y2"]:
        return "v"
    return "d"


def _length(seg: Dict) -> float:
    if _axis(seg) == "h":
        return abs(seg["x2"] - seg["x1"])
    return abs(seg["y2"] - seg["y1"])


def compute_local_thickness(seg: Dict,
                            dist_transform: Optional[np.ndarray]) -> float:
    """Median ``2 * dt`` sampled along ``seg``'s body. Distance-transform
    values at or below 0.5 (background / junction interior) are dropped
    before the median so they don't pull thick walls down at junction
    holes. Returns ``-1.0`` when sampling is impossible or too sparse.
    """
    if dist_transform is None or dist_transform.size == 0:
        return -1.0
    h, w = dist_transform.shape
    x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
    seg_len = max(abs(x2 - x1), abs(y2 - y1))
    if seg_len <= 0:
        return -1.0
    n = max(3, int(round(seg_len)))
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    samples = dist_transform[yi, xi]
    interior = samples[samples > 0.5]
    if interior.size < max(3, n // 4):
        return -1.0
    return float(2.0 * np.median(interior))


def _adaptive_tol(thicknesses: List[float],
                  *,
                  abs_min: float,
                  abs_max: float,
                  thickness_frac: float,
                  fallback: float) -> float:
    valid = [t for t in thicknesses if t > 0]
    if not valid:
        return max(abs_min, min(abs_max, fallback))
    med = float(np.median(valid))
    return max(abs_min, min(abs_max, thickness_frac * med))


def _length_weighted_median(offsets: List[float], lengths: List[float]) -> float:
    """Length-weighted median: the offset value whose cumulative length
    first crosses 50 % of the cluster's total length. Robust against
    short-stub outliers — a long wall surrounded by short jamb-end fragments
    has the long wall's offset returned, with zero drift. Mean would have
    been pulled toward the stubs by their non-zero (if small) weight."""
    pairs = sorted(zip(offsets, lengths), key=lambda p: p[0])
    total = sum(max(L, 1e-6) for _, L in pairs)
    half = 0.5 * total
    cum = 0.0
    for off, L in pairs:
        cum += max(L, 1e-6)
        if cum >= half:
            return off
    return pairs[-1][0]


def canonicalize_offsets(segments: List[Dict],
                         wall_mask: Optional[np.ndarray] = None,
                         *,
                         abs_min: float = 2.0,
                         abs_max: float = 6.0,
                         thickness_frac: float = 0.25,
                         fallback_tol: float = 3.0,
                         attach_thickness: bool = False
                         ) -> List[Dict]:
    """Cluster offsets per (type, axis); rewrite each member's perp
    coordinate to the cluster's length-weighted mean.

    Args:
        segments: list of segment dicts; non-axis-aligned entries pass
            through untouched.
        wall_mask: binary wall mask (HxW). When ``None``, every group
            falls back to ``fallback_tol`` for its cluster width.
        abs_min, abs_max: clamp range for the adaptive tolerance.
        thickness_frac: ``tol = thickness_frac * median(thickness)``.
        fallback_tol: applied when no thickness is computable in a group.
        attach_thickness: when True, each returned segment carries a
            ``local_thickness`` field (``-1.0`` when unknown). Off by
            default so the JSON payload is unchanged.
    """
    if not segments:
        return list(segments)

    if wall_mask is not None and wall_mask.size > 0:
        dt = cv2.distanceTransform((wall_mask > 0).astype(np.uint8),
                                   cv2.DIST_L2, 3)
    else:
        dt = None

    thicknesses = [compute_local_thickness(s, dt) for s in segments]

    by_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, seg in enumerate(segments):
        ax = _axis(seg)
        if ax in ("h", "v"):
            by_key[(seg["type"], ax)].append(i)

    new_offsets: Dict[int, float] = {}

    for (_t, ax), idxs in by_key.items():
        if len(idxs) < 2:
            continue
        items: List[Dict] = []
        for i in idxs:
            seg = segments[i]
            off = seg["y1"] if ax == "h" else seg["x1"]
            items.append({"i": i, "off": off,
                          "len": _length(seg),
                          "th": thicknesses[i]})

        group_th = [it["th"] for it in items]
        tol = _adaptive_tol(group_th,
                            abs_min=abs_min, abs_max=abs_max,
                            thickness_frac=thickness_frac,
                            fallback=fallback_tol)

        items.sort(key=lambda it: it["off"])
        clusters: List[List[Dict]] = [[items[0]]]
        for it in items[1:]:
            if it["off"] - clusters[-1][-1]["off"] <= tol:
                clusters[-1].append(it)
            else:
                clusters.append([it])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical = _length_weighted_median(
                [c["off"] for c in cluster],
                [c["len"] for c in cluster],
            )
            for c in cluster:
                if abs(c["off"] - canonical) > 1e-9:
                    new_offsets[c["i"]] = canonical

    out: List[Dict] = []
    for i, seg in enumerate(segments):
        new = dict(seg)
        if attach_thickness:
            new["local_thickness"] = thicknesses[i]
        if i in new_offsets:
            ax = _axis(seg)
            if ax == "h":
                new["y1"] = new["y2"] = new_offsets[i]
            elif ax == "v":
                new["x1"] = new["x2"] = new_offsets[i]
        out.append(new)
    return out
