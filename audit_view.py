"""Audit viewer + stats tool.

Reads an audit JSON dumped by ``vectorize_bgr(audit_path=...)`` and:

  1. Renders an overlay PNG showing accepted (green) / rejected (red)
     candidate decisions on top of the input image. Useful for
     eyeballing where score is systematically rejecting things.

  2. Prints a stats summary: by op, accept/reject counts plus the
     top score terms that drove rejections. This is the data we need
     to decide whether a term needs fixing (whole-population bias) or
     individual decisions are just edge cases.

CLI:
    py -3 audit_view.py overlay AUDIT_JSON IMAGE_PATH OUTPUT_PNG
    py -3 audit_view.py stats   AUDIT_JSON

The overlay format intentionally stays sparse so spatial structure is
visible: small filled circle per event, color by accepted state, size
scaled by ``|delta|``. Op type is *not* color-coded -- inspect the JSON
or the stats output for op breakdown.

The stats output is designed to surface "term needs work" patterns: if
op X's rejected events have ``invalid_crossing`` consistently negative,
that's where to look for a term fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


# Colors in BGR (OpenCV).
COLOR_ACCEPT = (0, 200, 0)        # green
COLOR_REJECT = (0, 0, 220)        # red
COLOR_SKIPPED = (180, 180, 0)     # teal-ish for used_endpoint / used_mutation


def _load_audit(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_overlay(events: List[Dict[str, Any]],
                   image_path: str,
                   output_path: str,
                   *,
                   accepted_radius_scale: float = 4.0,
                   rejected_radius_scale: float = 4.0,
                   accepted_min_radius: int = 3,
                   rejected_min_radius: int = 3) -> None:
    """Draw circle markers at each event's position onto the source image.

    Radius = ``min_radius + scale * |delta|`` (clamped). Larger |delta|
    -> bigger marker. Accepted = green, rejected (score_gate) = red,
    skipped (used_endpoint / used_mutation) = teal.
    """
    import cv2
    import numpy as np

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h, w = bgr.shape[:2]

    # Mute the background so markers pop.
    canvas = cv2.addWeighted(bgr, 0.55,
                             np.full_like(bgr, 255), 0.45, 0)

    n_drawn = 0
    n_no_position = 0
    for e in events:
        pos = e.get("position")
        if pos is None or not isinstance(pos, (list, tuple)) or len(pos) != 2:
            n_no_position += 1
            continue
        x, y = pos
        if not (0 <= x < w and 0 <= y < h):
            continue

        delta = abs(float(e.get("delta", 0.0)))
        reason = e.get("reason", "")
        accepted = e.get("accepted", False)
        if accepted:
            color = COLOR_ACCEPT
            r = max(accepted_min_radius,
                    int(round(accepted_min_radius + accepted_radius_scale * delta)))
        elif reason in ("used_endpoint", "used_mutation"):
            color = COLOR_SKIPPED
            r = accepted_min_radius
        else:
            color = COLOR_REJECT
            r = max(rejected_min_radius,
                    int(round(rejected_min_radius + rejected_radius_scale * delta)))
        r = min(r, 14)
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   r, color, -1, lineType=cv2.LINE_AA)
        n_drawn += 1

    cv2.imwrite(output_path, canvas)
    print(f"  drew {n_drawn} markers ({n_no_position} events without position)")
    print(f"  overlay: {output_path}")


def _term_summary(events: List[Dict[str, Any]], op: str
                  ) -> Dict[str, Dict[str, float]]:
    """Aggregate per-term contributions across the events of one op.

    Returns ``{term: {"mean": mean_signed_delta, "n_neg": count,
    "n_pos": count, "biggest_neg": value, "biggest_pos": value}}``.
    """
    by_term: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        if e.get("op") != op:
            continue
        dt = e.get("delta_terms") or {}
        for k, v in dt.items():
            try:
                by_term[k].append(float(v))
            except (TypeError, ValueError):
                continue
    out: Dict[str, Dict[str, float]] = {}
    for term, vals in by_term.items():
        if not vals:
            continue
        n_pos = sum(1 for x in vals if x > 0)
        n_neg = sum(1 for x in vals if x < 0)
        out[term] = {
            "mean": sum(vals) / len(vals),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "biggest_neg": min(vals),
            "biggest_pos": max(vals),
        }
    return out


def print_stats(events: List[Dict[str, Any]]) -> None:
    """Print accept/reject breakdown by op + dominant terms for rejected.

    Designed to surface "term needs work" patterns: if op X's rejected
    events show a term with consistently negative mean delta, that
    term is systematically blocking op X.
    """
    by_op: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_op[e.get("op", "?")].append(e)

    print(f"=== Audit summary ({len(events)} events) ===\n")
    print(f"{'op':<22} {'accepted':>9} {'rejected':>9} {'skipped':>9}  "
          f"{'reject rate':>11}")
    print("-" * 70)
    total = Counter()
    for op in sorted(by_op):
        es = by_op[op]
        n_acc = sum(1 for e in es if e.get("accepted"))
        n_rej = sum(1 for e in es if (not e.get("accepted")
                                       and e.get("reason") not in
                                       ("used_endpoint", "used_mutation")))
        n_skip = sum(1 for e in es if e.get("reason") in
                     ("used_endpoint", "used_mutation"))
        total["accepted"] += n_acc
        total["rejected"] += n_rej
        total["skipped"] += n_skip
        total_decided = n_acc + n_rej
        rate = (n_rej / total_decided) if total_decided > 0 else 0.0
        print(f"{op:<22} {n_acc:>9d} {n_rej:>9d} {n_skip:>9d}  {rate:>10.1%}")
    print("-" * 70)
    print(f"{'TOTAL':<22} {total['accepted']:>9d} {total['rejected']:>9d} "
          f"{total['skipped']:>9d}\n")

    # Per-op: for rejected events, which terms drove the rejection?
    # Sort terms by mean signed delta (most-negative first = "this term
    # is the dominant rejection driver for this op").
    for op in sorted(by_op):
        rejected = [e for e in by_op[op]
                    if not e.get("accepted")
                    and e.get("reason") not in ("used_endpoint", "used_mutation")]
        if not rejected:
            continue
        summary = _term_summary(rejected, op)
        if not summary:
            continue
        # Rank terms by mean signed delta (lowest first = strongest
        # systematic reject driver). Filter trivial movers.
        ranked = sorted(summary.items(),
                        key=lambda kv: kv[1]["mean"])
        ranked = [(t, s) for t, s in ranked if abs(s["mean"]) > 0.01]
        if not ranked:
            continue
        print(f"--- op '{op}' rejected events ({len(rejected)}) — dominant terms:")
        print(f"{'  term':<24} {'mean delta':>12} {'n_neg':>6} {'n_pos':>6} "
              f"{'worst':>9}")
        for term, s in ranked[:5]:
            arrow = "v" if s["mean"] < 0 else "^"
            print(f"  {arrow} {term:<22} {s['mean']:>+12.4f} "
                  f"{s['n_neg']:>6d} {s['n_pos']:>6d} {s['biggest_neg']:>+9.3f}")
        print()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_overlay = sub.add_parser("overlay",
                                help="render colored-dot overlay PNG")
    p_overlay.add_argument("audit_json")
    p_overlay.add_argument("image_path")
    p_overlay.add_argument("output_png")

    p_stats = sub.add_parser("stats",
                              help="print accept/reject + term-driver summary")
    p_stats.add_argument("audit_json")

    args = p.parse_args(argv)

    events = _load_audit(args.audit_json)

    if args.cmd == "overlay":
        render_overlay(events, args.image_path, args.output_png)
    elif args.cmd == "stats":
        print_stats(events)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
