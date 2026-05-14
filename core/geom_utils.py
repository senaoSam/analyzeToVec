"""Shared geometry primitives — endpoint keying, degree counting, free
endpoint enumeration. Consolidates the half-dozen homegrown ``int(round)``
and ``_qk`` quantisers that previously lived inside vectorize.py /
generators.py / scoring.py / regression.py / audit_view.py / candidates.py
/ tests/metrics.py.

Why this module exists:

  Topology decisions (degree, free-endpoint detection, junction
  recognition) all need to bucket coordinates so float drift doesn't
  fragment what is logically one node. Different parts of the pipeline
  used different precisions ad-hoc — 0.01-px in vectorize._qkey and
  generators._qk, 1-px integer rounding everywhere else. Step 21 Phase 1
  unifies the *function* without changing per-site precision: every
  call site now invokes ``endpoint_key(x, y, precision)``, and the
  precision each site picks today exactly matches what it used before.
  Phase 2 (later) can collapse to a single precision once we've measured
  what changes.

Public surface:

  endpoint_key(x, y, precision=0)
      Quantise (x, y) to an integer tuple suitable as a dict key.
      precision=0 → behaves like ``int(round(x))``, the prior 1-px usage.
      precision>0 → multiplies coord by 10**precision before rounding,
      so two endpoints differ in their key iff they differ by ≥
      10**(-precision) px.

  node_degree(segments, precision=0)
      Counter mapping endpoint_key -> count. Same precision semantics.

  free_endpoints(segments, precision=0)
      List of (x, y, type, seg_idx, end) records for every degree-1
      endpoint. ``x``/``y`` are the original float coords, not the
      quantised key, so callers can mutate the underlying segment
      without re-quantisation drift.

  endpoint_keys_for_segment(seg, precision=0)
      Returns the two endpoint keys for one segment. Light shortcut
      used by sites that iterate segments end-by-end.

No side effects, no I/O.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple


# Two precisions in active use today. Named so call sites are readable.
PRECISION_INT = 0           # int(round(x)) — 1-px buckets
PRECISION_CENTI = 2         # int(round(x * 100)) — 0.01-px buckets


def endpoint_key(x: float, y: float, precision: int = PRECISION_INT
                 ) -> Tuple[int, int]:
    """Bucket (x, y) into an integer tuple at ``precision`` decimal places.

    ``precision=0`` keys two endpoints to the same bucket iff their
    coords agree to the nearest integer (the legacy 1-px behaviour
    used in regression, scoring, audit_view, candidates, and most of
    generators / vectorize). ``precision=2`` keys at 0.01-px (matches
    the legacy ``_qkey`` in vectorize and ``_qk`` in generators'
    collinear_merge code path).
    """
    if precision == 0:
        return (int(round(float(x))), int(round(float(y))))
    factor = 10 ** precision
    return (int(round(float(x) * factor)),
            int(round(float(y) * factor)))


def endpoint_keys_for_segment(seg: Dict[str, Any],
                              precision: int = PRECISION_INT
                              ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Both endpoint keys for one segment, as ``(key1, key2)``."""
    return (
        endpoint_key(seg["x1"], seg["y1"], precision),
        endpoint_key(seg["x2"], seg["y2"], precision),
    )


def node_degree(segments: Sequence[Dict[str, Any]],
                precision: int = PRECISION_INT
                ) -> Counter:
    """Counter of endpoint_key -> number of segment endpoints at that key."""
    cnt: Counter = Counter()
    for s in segments:
        k1, k2 = endpoint_keys_for_segment(s, precision)
        cnt[k1] += 1
        cnt[k2] += 1
    return cnt


def free_endpoints(segments: Sequence[Dict[str, Any]],
                   precision: int = PRECISION_INT,
                   ) -> List[Dict[str, Any]]:
    """Return the degree-1 endpoints as a list of dicts.

    Each entry: ``{"x": float, "y": float, "type": str, "seg_idx": int,
    "end": "1"|"2"}``. ``x`` and ``y`` are the un-quantised original
    coords — callers that mutate the segment list directly need them
    to match the float values stored in ``segments[seg_idx]``.
    """
    deg = node_degree(segments, precision)
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(segments):
        for end in ("1", "2"):
            x = float(s[f"x{end}"])
            y = float(s[f"y{end}"])
            if deg[endpoint_key(x, y, precision)] == 1:
                out.append({"x": x, "y": y,
                            "type": s.get("type", "?"),
                            "seg_idx": i, "end": end})
    return out
