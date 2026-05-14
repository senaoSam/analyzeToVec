"""Quality metrics for vectorize.py output.

The single ``compute_metrics`` entry point returns a flat dict of named
quantities, intended to drive the metric-based regression gate (step 18 in
todo.md). Each metric has a documented direction (higher-better / lower-
better / target-value), and the regression layer applies per-metric
tolerances on top.

Categories:

  - Absolute fidelity to the source image:
      wall_iou_vs_source       (higher = better; bounded [0, 1])
      door_iou_vs_source       (higher = better; bounded [0, 1])
      window_iou_vs_source     (higher = better; bounded [0, 1])

  - Geometric health (independent of baseline):
      floating_openings        (lower = better; target 0)
      free_endpoints           (lower = better)
      diagonal_count           (must equal 0; invariant-enforced)
      phantom_wall_frac        (lower = better; fraction of wall length
                                landing on low-evidence pixels)

  - Inventory (monitor, not gate):
      segment_count, segment_count_by_type
      total_length_by_type
      endpoint_degree_histogram

  - Relative-to-baseline IOU (mirrors the existing 3-thickness scheme
    so the existing regression report stays usable, but now flows
    through the same dict):
      iou_baseline.<type>.<thin|normal|loose>

The module reuses primitives from regression.py rather than duplicating
them, so any rasterizer change stays in one place.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Allow ``from tests.metrics import compute_metrics`` and direct script use.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools import regression as R  # noqa: E402
from core import scoring as S  # noqa: E402
from tests.invariants import (  # noqa: E402
    _axis_of,
    check_openings_anchored,
    check_no_diagonals,
)


# Evidence threshold matching scoring.PHANTOM_EVIDENCE_THRESHOLD so the
# metric and the score's phantom term agree on what counts as "low".
PHANTOM_EVIDENCE_THR = 0.5


def _segments_by_type(segments: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"wall": [], "door": [], "window": []}
    for s in segments:
        t = s.get("type", "")
        if t in out:
            out[t].append(s)
    return out


def _iou_vs_source_mask(segments: Sequence[Dict],
                        source_mask: np.ndarray,
                        stroke_thickness: int) -> float:
    """IOU of (rasterized segments at ``stroke_thickness``) against the
    raw source mask. Measures absolute fidelity to the input image.

    ``stroke_thickness`` should be roughly the local stroke width so the
    rasterized segment line covers the same pixel band the source mask
    occupies. Too thin and the segment misses on-mask pixels (false
    drop); too thick and segments puff into off-mask area.
    """
    if source_mask is None or source_mask.size == 0:
        return 0.0
    h, w = source_mask.shape[:2]
    # Use the regression rasterizer for consistency.
    seg_mask = R.rasterize_lines(list(segments), (h, w),
                                 type_filter=None,
                                 thickness=stroke_thickness)
    return R.compute_iou(seg_mask, source_mask)


def _wall_iou_vs_source(walls: Sequence[Dict],
                        wall_mask: np.ndarray,
                        thickness: int) -> float:
    h, w = wall_mask.shape[:2]
    seg = np.zeros((h, w), dtype=np.uint8)
    for s in walls:
        x1, y1 = int(round(float(s["x1"]))), int(round(float(s["y1"])))
        x2, y2 = int(round(float(s["x2"]))), int(round(float(s["y2"])))
        cv2.line(seg, (x1, y1), (x2, y2), 255, thickness)
    return R.compute_iou(seg, wall_mask)


def _phantom_wall_frac(walls: Sequence[Dict],
                       wall_evidence: Optional[np.ndarray]) -> float:
    """Fraction of wall *length* whose rasterized samples land in regions
    where ``wall_evidence`` is below ``PHANTOM_EVIDENCE_THR``. 0 = no
    phantom pixels; 1 = entire wall length sits on white space.

    Mirrors scoring.phantom_penalty's signal but normalised to a fraction
    rather than a raw count, so it's directly comparable across image
    sizes.
    """
    if wall_evidence is None or not walls:
        return 0.0
    h, w = wall_evidence.shape[:2]
    n_low = 0
    n_total = 0
    for s in walls:
        x1, y1 = float(s["x1"]), float(s["y1"])
        x2, y2 = float(s["x2"]), float(s["y2"])
        length = math.hypot(x2 - x1, y2 - y1)
        n_samples = max(2, int(round(length)))
        ts = np.linspace(0.0, 1.0, n_samples)
        xs = (x1 + (x2 - x1) * ts).astype(np.int32)
        ys = (y1 + (y2 - y1) * ts).astype(np.int32)
        # Clip into bounds.
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not valid.any():
            continue
        vals = wall_evidence[ys[valid], xs[valid]]
        n_low += int((vals < PHANTOM_EVIDENCE_THR).sum())
        n_total += int(valid.sum())
    if n_total == 0:
        return 0.0
    return n_low / n_total


def _endpoint_degree_histogram(segments: Sequence[Dict]) -> Dict[int, int]:
    """Degree histogram on int-pixel-rounded endpoints (matches
    regression.compute_graph_metrics so the two never disagree).
    """
    from core.geom_utils import node_degree
    hist: Counter = Counter(node_degree(segments).values())
    return {int(k): int(v) for k, v in sorted(hist.items())}


def compute_metrics(segments: Sequence[Dict],
                    *,
                    image_shape: Tuple[int, int],
                    masks: Optional[Dict[str, np.ndarray]] = None,
                    wall_evidence: Optional[np.ndarray] = None,
                    stroke_width: Optional[float] = None,
                    ) -> Dict:
    """Compute the full metric dict for one segments list.

    Args:
        segments: pipeline output (list of ``{type, x1, y1, x2, y2}`` dicts).
        image_shape: (h, w) of the source image.
        masks: optional ``{wall, door, window}`` masks from
            ``vectorize.segment_colors``. Required for fidelity metrics.
        wall_evidence: optional continuous 0-1 wall evidence map from
            ``vectorize.compute_wall_evidence``. Required for phantom_wall_frac.
        stroke_width: optional pre-computed stroke width. If absent and
            ``masks['wall']`` is provided, it's derived via
            ``regression.estimate_stroke_width``.

    Returns:
        Flat dict; structure documented at module top.
    """
    by_type = _segments_by_type(segments)
    walls = by_type["wall"]
    h, w = image_shape[:2]

    if stroke_width is None and masks is not None and "wall" in masks:
        stroke_width = R.estimate_stroke_width(masks["wall"])
    if stroke_width is None:
        stroke_width = max(2.0, min(h, w) / 400.0)
    stroke_int = max(1, int(round(stroke_width)))

    # Fidelity vs source masks.
    fidelity: Dict[str, float] = {}
    if masks is not None:
        if "wall" in masks and masks["wall"] is not None:
            fidelity["wall_iou_vs_source"] = _wall_iou_vs_source(
                walls, masks["wall"], stroke_int)
        if "door" in masks and masks["door"] is not None:
            fidelity["door_iou_vs_source"] = _iou_vs_source_mask(
                by_type["door"], masks["door"], stroke_int)
        if "window" in masks and masks["window"] is not None:
            fidelity["window_iou_vs_source"] = _iou_vs_source_mask(
                by_type["window"], masks["window"], stroke_int)

    # Geometric health.
    floating = check_openings_anchored(segments)
    diagonals = check_no_diagonals(segments)
    graph = R.compute_graph_metrics(list(segments))
    phantom_frac = _phantom_wall_frac(walls, wall_evidence)
    deg_hist = _endpoint_degree_histogram(segments)
    duplicate_pairs = S.duplicate_penalty(list(segments))

    return {
        # Absolute fidelity (higher = better, bounded [0, 1])
        **{k: round(float(v), 6) for k, v in fidelity.items()},

        # Geometric health
        "floating_openings": len(floating),
        "free_endpoints": int(graph["free_endpoints"]),
        "diagonal_count": len(diagonals),
        "phantom_wall_frac": round(phantom_frac, 6),
        "duplicate_pairs": int(duplicate_pairs),

        # Inventory
        "segment_count": int(graph["num_segments"]),
        "segment_count_by_type": {k: int(v)
                                  for k, v in graph["count_by_type"].items()},
        "total_length_by_type": {k: float(v)
                                 for k, v in graph["total_length_by_type"].items()},
        "endpoint_degree_histogram": deg_hist,

        # Context for downstream tools.
        "stroke_width": round(float(stroke_width), 4),
        "image_shape": [int(h), int(w)],
    }


# ---------------------------------------------------------------------------
# Direction / tolerance schema for the regression layer
# ---------------------------------------------------------------------------
#
# Each metric is one of:
#   "higher_better"     (current >= baseline - tol)
#   "lower_better"      (current <= baseline + tol)
#   "must_equal"        (current == target; tol ignored)
#   "monitor"           (recorded but never gates)
#
# The regression layer reads this table once and applies the rules
# uniformly. Adding a new metric = add a row here.

METRIC_RULES: Dict[str, Dict] = {
    # Fidelity vs source image. Per-type IOU drop tolerances mirror the
    # existing NORMAL_IOU_DROP_MAX in regression.py so behaviour doesn't
    # shift suddenly when this layer takes over.
    "wall_iou_vs_source":   {"direction": "higher_better", "tol": 0.010},
    "door_iou_vs_source":   {"direction": "higher_better", "tol": 0.020},
    "window_iou_vs_source": {"direction": "higher_better", "tol": 0.020},

    # Geometric health. Goal invariants — once §2/§3 ship these will
    # become "must_equal: 0" via the invariant layer instead.
    "floating_openings":    {"direction": "lower_better", "tol": 0},
    "free_endpoints":       {"direction": "lower_better", "tol": 2},
    "diagonal_count":       {"direction": "must_equal",   "target": 0},
    "phantom_wall_frac":    {"direction": "lower_better", "tol": 0.020},
    # IOU is invariant under duplication (same pixels covered twice still
    # cover the same pixels), so duplicates need their own signal.
    # Tol = 5 absorbs slight noise; doubling walls adds 60+ which is well
    # above the bar.
    "duplicate_pairs":      {"direction": "lower_better", "tol": 5},

    # Inventory — change ratios already tracked by regression.compare_case.
    # Recorded here but not directly gated to avoid double-counting.
    "segment_count":            {"direction": "monitor"},
    "segment_count_by_type":    {"direction": "monitor"},
    "total_length_by_type":     {"direction": "monitor"},
    "endpoint_degree_histogram": {"direction": "monitor"},
    "stroke_width":             {"direction": "monitor"},
    "image_shape":              {"direction": "monitor"},
}


def compare_metrics(baseline: Dict, current: Dict
                    ) -> Tuple[List[str], List[str]]:
    """Compare current metrics to baseline. Returns (fails, warns).

    Per the contract in todo.md §1:
      - higher_better metric: regression when ``current < baseline - tol``
      - lower_better metric: regression when ``current > baseline + tol``
      - must_equal metric: regression when ``current != target``
      - monitor: never reported here (the dump in the regression report
        is enough)
    """
    fails: List[str] = []
    warns: List[str] = []
    for name, rule in METRIC_RULES.items():
        direction = rule["direction"]
        if direction == "monitor":
            continue
        if name not in current:
            continue
        cur = current[name]
        if direction == "must_equal":
            target = rule["target"]
            if cur != target:
                fails.append(f"{name}: {cur} != target {target}")
            continue
        if name not in baseline:
            # New metric introduced — baseline ungated.
            continue
        base = baseline[name]
        tol = rule.get("tol", 0)
        if direction == "higher_better":
            if cur < base - tol:
                fails.append(f"{name}: {cur:.4f} < baseline {base:.4f} - tol {tol}")
        elif direction == "lower_better":
            if cur > base + tol:
                fails.append(f"{name}: {cur} > baseline {base} + tol {tol}")
    return fails, warns


# ---------------------------------------------------------------------------
# Top-level v18 hook for regression.py
# ---------------------------------------------------------------------------

def compute_v18_report(current_lines: Sequence[Dict],
                       baseline_lines: Sequence[Dict],
                       bgr: np.ndarray,
                       ) -> Dict:
    """Compute the full step-18 layer for one case.

    Both baseline and current metrics are computed against the same source
    masks (derived from ``bgr`` once) so the comparison is fair: any IOU
    delta reflects pipeline output differences, not mask differences.

    Returns:
        ``{
            "current_metrics":  {...},
            "baseline_metrics": {...},
            "metric_fails":     [str, ...],
            "metric_warns":     [str, ...],
            "invariants": {
                "strict_count": int,
                "goal_count":   int,
                "strict":       [str, ...],   # one per violation
                "goal":         [str, ...],
            },
            "invariant_fails": [str, ...],    # strict only — these always FAIL
        }``
    """
    # Lazy imports keep this module importable without the full pipeline
    # stack at definition time.
    import vectorize as V
    from tests.invariants import check_all

    masks = V.segment_colors(bgr)
    wall_evidence = V.compute_wall_evidence(bgr)
    image_shape = bgr.shape[:2]
    stroke_width = R.estimate_stroke_width(masks.get("wall"))

    common_kw = dict(image_shape=image_shape,
                     masks=masks,
                     wall_evidence=wall_evidence,
                     stroke_width=stroke_width)
    cur = compute_metrics(list(current_lines), **common_kw)
    base = compute_metrics(list(baseline_lines), **common_kw)
    metric_fails, metric_warns = compare_metrics(base, cur)

    inv = check_all(list(current_lines), image_shape)
    return {
        "current_metrics": cur,
        "baseline_metrics": base,
        "metric_fails": metric_fails,
        "metric_warns": metric_warns,
        "invariants": {
            "strict_count": len(inv.strict),
            "goal_count": len(inv.goal),
            "strict": [v.describe() for v in inv.strict],
            "goal": [v.describe() for v in inv.goal],
        },
        "invariant_fails": [v.describe() for v in inv.strict],
    }
