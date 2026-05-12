"""Audit recorder for candidate accept / reject decisions.

Step 5 ranking model needs training data: for each candidate the pipeline
considers, what were its features, what did score say, and did the
gate accept it. This module is that capture layer.

Default-off design: callers must explicitly pass an ``AuditRecorder``
into the score-using wrappers; without one nothing is recorded and the
pipeline has zero overhead. Pipeline entry ``vectorize_bgr`` exposes an
``audit_path`` parameter that, when set, creates a recorder, threads it
into the wrappers, and dumps the events as JSON at the end.

Event schema (one record per candidate evaluated, accepted or rejected):

  op:           candidate's op string ("merge", "fuse", "bridge", ...)
  accepted:    True/False
  delta:        trial_score.total - base_score.total (the gating signal)
  delta_terms: per-term contribution to delta (which terms moved the
                score, used to attribute the decision to specific signals)
  meta:         candidate's meta dict (geometric features, position, ...)
  reason:       short string explaining why this event was recorded
                (e.g. "score gate", "skip_score", "used_endpoint")

Output is a flat JSON list so downstream consumers can scan with any
JSON tool (line-by-line jq, pandas read_json, etc).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AuditEvent:
    """One candidate decision event."""
    op: str
    accepted: bool
    delta: float
    delta_terms: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    position: Optional[Any] = None  # (x, y) in image coords, or None

    def to_jsonable(self) -> Dict[str, Any]:
        out = asdict(self)
        # meta may contain tuples and numpy types; coerce to JSON-safe.
        out["meta"] = _coerce(self.meta)
        out["delta_terms"] = {k: float(v) for k, v in self.delta_terms.items()}
        out["position"] = _coerce(self.position)
        return out


def candidate_position(cand: Any) -> Optional[tuple]:
    """Compute a representative (x, y) for a Candidate for audit overlays.

    Strategy:
      * ``add``-only / mixed: midpoint of the first added segment's endpoints
      * ``mutate``-only: mean of every mutate's (new_x, new_y) target
      * ``remove``-only (e.g. ``prune`` of zero-length): None (nothing
        geometric to plot)
    """
    add = getattr(cand, "add", None)
    mutate = getattr(cand, "mutate", None)
    if add:
        s = add[0]
        return (0.5 * (float(s["x1"]) + float(s["x2"])),
                0.5 * (float(s["y1"]) + float(s["y2"])))
    if mutate:
        xs = [float(m[2]) for m in mutate]
        ys = [float(m[3]) for m in mutate]
        if xs:
            return (sum(xs) / len(xs), sum(ys) / len(ys))
    return None


def _coerce(obj: Any) -> Any:
    """Make ``obj`` JSON-serialisable. Handles tuples, lists, dicts,
    numpy scalars (best-effort), strings, and numbers. Unknown types
    fall back to repr()."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    # numpy scalar / numpy array best-effort
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return repr(obj)


class AuditRecorder:
    """Append-only log of candidate decisions for the current pipeline
    invocation. Not thread-safe (the pipeline is single-threaded)."""

    def __init__(self) -> None:
        self.events: List[AuditEvent] = []

    def record(self,
               op: str,
               accepted: bool,
               delta: float,
               delta_terms: Optional[Dict[str, float]] = None,
               meta: Optional[Dict[str, Any]] = None,
               reason: str = "",
               position: Optional[tuple] = None) -> None:
        self.events.append(AuditEvent(
            op=op,
            accepted=accepted,
            delta=float(delta),
            delta_terms=dict(delta_terms) if delta_terms else {},
            meta=dict(meta) if meta else {},
            reason=reason,
            position=position,
        ))

    def dump_json(self, path: str) -> None:
        data = [e.to_jsonable() for e in self.events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        return len(self.events)

    def summary(self) -> Dict[str, Any]:
        """Quick aggregate counts for verbose logging / regression debug."""
        by_op: Dict[str, Dict[str, int]] = {}
        for e in self.events:
            bucket = by_op.setdefault(e.op, {"accepted": 0, "rejected": 0})
            bucket["accepted" if e.accepted else "rejected"] += 1
        return {
            "total": len(self.events),
            "by_op": by_op,
        }
