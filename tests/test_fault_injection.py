"""Fault-injection sanity check for step 18 metric gate.

The contract from todo.md §1:

    Done when: ... 某個刻意製造的「拆牆」測試會在 metric-based 上 FAIL、
    bit-identical 上也 FAIL(雙重防護驗證)。

This script applies four synthetic faults to a known-good pipeline output
and verifies each triggers at least one v18 metric FAIL or invariant
STRICT violation. If any fault slips past the gate it means the gate is
too loose — investigate before relying on it for real refactors.

Faults exercised:
    1. drop 20 % of wall segments        — must drop wall_iou_vs_source
    2. detach every door from walls      — must raise floating_openings
    3. insert one diagonal segment       — must trip diagonal invariant
    4. duplicate every wall segment      — must move segment_count and
                                             change endpoint degrees
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from typing import Dict, List

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tests.metrics import compute_v18_report  # noqa: E402


def _load_case(case: str):
    with open(os.path.join(REPO, "tests", "cases", case, "manifest.json")) as f:
        mani = json.load(f)
    img_path = os.path.normpath(os.path.join(
        REPO, "tests", "cases", case, mani["input_image"]))
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    with open(os.path.join(REPO, "tests", "baseline", case, "lines.json")) as f:
        lines = json.load(f)["lines"]
    return bgr, lines


def fault_drop_walls(lines: List[Dict], frac: float = 0.20) -> List[Dict]:
    walls = [s for s in lines if s.get("type") == "wall"]
    others = [s for s in lines if s.get("type") != "wall"]
    keep_n = max(1, int(round(len(walls) * (1.0 - frac))))
    return others + walls[:keep_n]


def fault_detach_doors(lines: List[Dict], offset: float = 50.0) -> List[Dict]:
    out: List[Dict] = []
    for s in lines:
        if s.get("type") == "door":
            ns = dict(s)
            ns["x1"] = float(s["x1"]) + offset
            ns["y1"] = float(s["y1"]) + offset
            ns["x2"] = float(s["x2"]) + offset
            ns["y2"] = float(s["y2"]) + offset
            out.append(ns)
        else:
            out.append(s)
    return out


def fault_inject_diagonal(lines: List[Dict]) -> List[Dict]:
    out = deepcopy(lines)
    if not out:
        return out
    s = out[0]
    s["x1"] = float(s["x1"])
    s["y1"] = float(s["y1"])
    s["x2"] = float(s["x1"]) + 10.0
    s["y2"] = float(s["y1"]) + 10.0     # forces dx=dy=10 → diagonal
    return out


def fault_duplicate_walls(lines: List[Dict]) -> List[Dict]:
    out = list(lines)
    for s in lines:
        if s.get("type") == "wall":
            out.append(dict(s))
    return out


def _assert_gate_catches(case: str, bgr, baseline_lines, faulted_lines,
                         expected_kind: str) -> None:
    """Run v18 layer and require either a metric_fail OR a strict
    invariant violation. ``expected_kind`` is a short label printed on
    PASS so the report is readable.
    """
    r = compute_v18_report(faulted_lines, baseline_lines, bgr)
    caught_metric = bool(r["metric_fails"])
    caught_invariant = bool(r["invariant_fails"])
    caught = caught_metric or caught_invariant
    tag = "OK   " if caught else "MISS!"
    print(f"  [{tag}] {expected_kind}")
    if caught_metric:
        for m in r["metric_fails"][:3]:
            print(f"           metric_fail: {m}")
    if caught_invariant:
        for m in r["invariant_fails"][:3]:
            print(f"           invariant:   {m}")
    if not caught:
        print(f"           current_metrics: {r['current_metrics']}")
        raise SystemExit(f"FAULT-INJECTION FAILED for {case}: "
                         f"v18 layer did not catch '{expected_kind}'")


def main() -> None:
    cases = ["source", "sg2"]
    for case in cases:
        print(f"\n=== {case} ===")
        bgr, baseline_lines = _load_case(case)

        # 0. Sanity: identical input must NOT fail the gate.
        r = compute_v18_report(baseline_lines, baseline_lines, bgr)
        assert not r["metric_fails"], f"baseline vs baseline failed: {r['metric_fails']}"
        assert not r["invariant_fails"], f"baseline invariants failed: {r['invariant_fails']}"
        print("  [OK   ] no-op (baseline vs baseline)")

        _assert_gate_catches(case, bgr, baseline_lines,
                             fault_drop_walls(baseline_lines),
                             "drop 20% walls")
        _assert_gate_catches(case, bgr, baseline_lines,
                             fault_detach_doors(baseline_lines),
                             "detach all doors")
        _assert_gate_catches(case, bgr, baseline_lines,
                             fault_inject_diagonal(baseline_lines),
                             "inject diagonal segment")
        _assert_gate_catches(case, bgr, baseline_lines,
                             fault_duplicate_walls(baseline_lines),
                             "duplicate every wall")

    print("\nALL FAULT INJECTIONS CAUGHT.")


if __name__ == "__main__":
    main()
