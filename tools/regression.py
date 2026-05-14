"""Regression harness for vectorize.py.

Compares current pipeline output against a frozen baseline across 3 reference
images. Detects regressions via three-thickness rasterized IOU, symmetric
distance metrics, and graph-level counts. Status is FAIL / WARNING / PASS.

Usage:
  python regression.py                  # run all cases, exit 1 on FAIL
  python regression.py --case source    # only run one case
  python regression.py --update-baseline [--case X | --all] [--yes]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow ``py -3 tools/regression.py`` from the repo root — make sure the
# project root is on sys.path so ``import vectorize`` / ``from core ...``
# / ``from tests.metrics ...`` resolve regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import vectorize as V  # noqa: E402

# Step 18: metric + invariant layer. Imported lazily inside ``run_one_case``
# to avoid forcing tests/ on the sys.path for callers that don't need it.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGRESSION_FORMAT_VERSION = 1
RASTERIZER_VERSION = 1

TYPES = ("wall", "window", "door")
THICKNESS_LEVELS = ("thin", "normal", "loose")

# IOU drop limits, indexed by [level][type]. FAIL = drop exceeds normal/loose.
NORMAL_IOU_DROP_MAX = {"wall": 0.010, "window": 0.020, "door": 0.020}
LOOSE_IOU_DROP_MAX = {"wall": 0.004, "window": 0.008, "door": 0.008}
THIN_IOU_DROP_WARN = {"wall": 0.030, "window": 0.050, "door": 0.050}

FREE_ENDPOINTS_INCREASE_MAX = 5
NUM_SEGMENTS_CHANGE_RATIO_MAX = 0.15
TOTAL_LENGTH_CHANGE_RATIO_MAX = 0.10

# distance thresholds are computed per-case from baseline stroke_width:
#   p95_distance_max  = max(2.0, 0.30 * stroke_width)
#   mean_distance_max = max(1.0, 0.12 * stroke_width)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = _REPO_ROOT
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
CASES_DIR = os.path.join(TESTS_DIR, "cases")
BASELINE_DIR = os.path.join(TESTS_DIR, "baseline")
CURRENT_DIR = os.path.join(TESTS_DIR, "current")


# ---------------------------------------------------------------------------
# 0.2 Core helpers
# ---------------------------------------------------------------------------

def normalize_lines(lines: List[Dict]) -> List[Tuple]:
    """Round coords to 1 px, put endpoints in canonical (min,max) order,
    sort by (type, x1, y1, x2, y2). Ignore non-geometric fields.
    """
    from core.geom_utils import endpoint_keys_for_segment
    norm: List[Tuple] = []
    for s in lines:
        t = s.get("type", "")
        a, b = endpoint_keys_for_segment(s)
        # canonical endpoint order: smaller (x, y) first
        if b < a:
            a, b = b, a
        norm.append((t, a[0], a[1], b[0], b[1]))
    norm.sort()
    return norm


def hash_lines(lines: List[Dict]) -> str:
    """SHA-1 over normalized line list."""
    norm = normalize_lines(lines)
    payload = json.dumps(norm, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def estimate_stroke_width(wall_mask: np.ndarray) -> float:
    """2.0 * median(distanceTransform[wall>0]), clamp [2.0, 80.0].

    Fallback: max(2, round(min(w,h)/400)) when mask is empty.
    """
    if wall_mask is None or wall_mask.size == 0 or not (wall_mask > 0).any():
        h, w = (wall_mask.shape if wall_mask is not None and wall_mask.size else (0, 0))
        return float(max(2.0, round(min(h, w) / 400.0))) if h and w else 2.0
    bin_mask = (wall_mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
    vals = dt[bin_mask > 0]
    sw = 2.0 * float(np.median(vals))
    return float(min(80.0, max(2.0, sw)))


def thickness_for_level(level: str, stroke_width: float) -> int:
    """Map (level, stroke_width) -> integer pixel thickness for rasterize."""
    if level == "thin":
        return int(max(1, round(0.2 * stroke_width)))
    if level == "normal":
        return int(max(2, round(0.4 * stroke_width)))
    if level == "loose":
        return int(max(3, round(0.8 * stroke_width)))
    raise ValueError(f"unknown thickness level: {level}")


def rasterize_lines(lines: List[Dict], shape: Tuple[int, int], type_filter: Optional[str],
                    thickness: int) -> np.ndarray:
    """Render lines of `type_filter` (or all if None) into a binary uint8 mask."""
    h, w = shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    if thickness < 1:
        thickness = 1
    for s in lines:
        if type_filter is not None and s.get("type") != type_filter:
            continue
        p1 = (int(round(float(s["x1"]))), int(round(float(s["y1"]))))
        p2 = (int(round(float(s["x2"]))), int(round(float(s["y2"]))))
        cv2.line(canvas, p1, p2, 255, thickness=thickness, lineType=cv2.LINE_8)
    return canvas


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return inter / union


def _one_dir_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    """For foreground pixels of A, distance to nearest foreground pixel of B.
    Returns (mean, p95). 0.0 if either mask is empty.
    """
    a_fg = mask_a > 0
    b_fg = mask_b > 0
    if not a_fg.any():
        return 0.0, 0.0
    if not b_fg.any():
        # Anything in A is infinitely far from B; cap with diagonal length.
        h, w = mask_a.shape[:2]
        big = float(np.hypot(h, w))
        return big, big
    inv_b = (~b_fg).astype(np.uint8)
    dt = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
    vals = dt[a_fg]
    return float(vals.mean()), float(np.percentile(vals, 95))


def compute_distance_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, float]:
    """Symmetric Hausdorff-like distance (max of the two directions)."""
    ma, pa = _one_dir_distance(mask_a, mask_b)
    mb, pb = _one_dir_distance(mask_b, mask_a)
    return {"mean": max(ma, mb), "p95": max(pa, pb)}


def compute_graph_metrics(lines: List[Dict]) -> Dict:
    """num_segments / count_by_type / free_endpoints / num_short_segments / total_length_by_type."""
    from collections import Counter
    from core.geom_utils import endpoint_keys_for_segment
    count_by_type: Counter = Counter()
    total_length_by_type: Dict[str, float] = {}
    nodes: Counter = Counter()
    n_short = 0
    for s in lines:
        t = s.get("type", "")
        count_by_type[t] += 1
        k1, k2 = endpoint_keys_for_segment(s)
        nodes[k1] += 1
        nodes[k2] += 1
        length = float(np.hypot(k2[0] - k1[0], k2[1] - k1[1]))
        total_length_by_type[t] = total_length_by_type.get(t, 0.0) + length
        if length < 10.0:
            n_short += 1
    deg_hist: Counter = Counter(nodes.values())
    return {
        "num_segments": len(lines),
        "count_by_type": dict(count_by_type),
        "free_endpoints": int(deg_hist.get(1, 0)),
        "num_short_segments": n_short,
        "total_length_by_type": {k: round(v, 3) for k, v in total_length_by_type.items()},
    }


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def discover_cases(only: Optional[str] = None) -> List[str]:
    """Return case names. With ``only`` set, return just that case (and honour
    its skip flag only if --include-skipped is implicit via explicit naming).
    Without ``only``, drop any case whose manifest has ``"skip": true``.
    """
    if not os.path.isdir(CASES_DIR):
        return []
    names = sorted(d for d in os.listdir(CASES_DIR)
                   if os.path.isdir(os.path.join(CASES_DIR, d)))
    if only:
        if only not in names:
            raise SystemExit(f"unknown case: {only!r} (have: {names})")
        return [only]
    # Auto-discovery drops cases manifest-marked skip:true.
    out: List[str] = []
    for n in names:
        try:
            mf = load_manifest(n)
        except SystemExit:
            continue
        if mf.get("skip"):
            print(f"[SKIP] {n}: {mf.get('skip_reason', 'manifest skip=true')}")
            continue
        out.append(n)
    return out


def load_manifest(case: str) -> Dict:
    path = os.path.join(CASES_DIR, case, "manifest.json")
    if not os.path.isfile(path):
        raise SystemExit(f"missing manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_input_image(case: str, manifest: Dict) -> str:
    rel = manifest.get("input_image")
    if not rel:
        raise SystemExit(f"manifest for {case} missing input_image")
    base = os.path.join(CASES_DIR, case)
    return os.path.normpath(os.path.join(base, rel))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_pure(image_path: str) -> Tuple[List[Dict], Tuple[int, int], np.ndarray, np.ndarray]:
    """cv2.imread -> vectorize_bgr(bgr) -> (lines, shape, wall_mask, bgr).

    Returns the BGR image alongside the pipeline output so callers can
    feed it back into step-18 metric computation without re-reading the
    file. Wall mask is kept as a separate return for back-compat with
    existing callers that only need stroke-width estimation.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image: {image_path}")
    result = V.vectorize_bgr(bgr, verbose=False)
    masks = V.segment_colors(bgr)
    return result["lines"], bgr.shape[:2], masks.get("wall"), bgr


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def baseline_paths(case: str) -> Dict[str, str]:
    base = os.path.join(BASELINE_DIR, case)
    return {
        "dir": base,
        "lines": os.path.join(base, "lines.json"),
        "metrics": os.path.join(base, "metrics.json"),
        "hash": os.path.join(base, "hash.txt"),
        "masks_dir": os.path.join(base, "masks"),
    }


def current_paths(case: str) -> Dict[str, str]:
    base = os.path.join(CURRENT_DIR, case)
    return {
        "dir": base,
        "lines": os.path.join(base, "lines.json"),
        "metrics": os.path.join(base, "metrics.json"),
        "hash": os.path.join(base, "hash.txt"),
        "report": os.path.join(base, "report.json"),
        "overlay": os.path.join(base, "diff_overlay.png"),
        "masks_dir": os.path.join(base, "masks"),
    }


def baseline_exists(case: str) -> bool:
    p = baseline_paths(case)
    return os.path.isfile(p["lines"]) and os.path.isfile(p["metrics"])


def write_masks(masks_dir: str, lines: List[Dict], shape: Tuple[int, int],
                stroke_width: float) -> Dict[str, Dict[str, str]]:
    """Render & save thin/normal/loose masks for {wall, window, door}.

    Returns dict[type][level] -> filename.
    """
    os.makedirs(masks_dir, exist_ok=True)
    out: Dict[str, Dict[str, str]] = {}
    for type_name in TYPES:
        out[type_name] = {}
        for level in THICKNESS_LEVELS:
            thk = thickness_for_level(level, stroke_width)
            mask = rasterize_lines(lines, shape, type_name, thk)
            fname = f"{type_name}_{level}.png"
            cv2.imwrite(os.path.join(masks_dir, fname), mask)
            out[type_name][level] = fname
    return out


def write_baseline(case: str, lines: List[Dict], shape: Tuple[int, int],
                   stroke_width: float, source: str) -> None:
    p = baseline_paths(case)
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["lines"], "w", encoding="utf-8") as f:
        json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)
    h = hash_lines(lines)
    with open(p["hash"], "w", encoding="utf-8") as f:
        f.write(h + "\n")
    masks_idx = write_masks(p["masks_dir"], lines, shape, stroke_width)
    graph = compute_graph_metrics(lines)
    metrics = {
        "regression_format_version": REGRESSION_FORMAT_VERSION,
        "rasterizer_version": RASTERIZER_VERSION,
        "stroke_width": stroke_width,
        "image_shape": [shape[0], shape[1]],
        "hash": h,
        "graph": graph,
        "thickness_px": {lvl: thickness_for_level(lvl, stroke_width)
                         for lvl in THICKNESS_LEVELS},
        "masks": masks_idx,
        "baseline_updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "baseline_updated_by": source,
    }
    with open(p["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_baseline(case: str) -> Tuple[List[Dict], Dict, Dict[str, Dict[str, np.ndarray]]]:
    p = baseline_paths(case)
    with open(p["lines"], "r", encoding="utf-8") as f:
        lines = json.load(f)["lines"]
    with open(p["metrics"], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    masks: Dict[str, Dict[str, np.ndarray]] = {}
    for type_name in TYPES:
        masks[type_name] = {}
        for level in THICKNESS_LEVELS:
            fn = os.path.join(p["masks_dir"], f"{type_name}_{level}.png")
            if os.path.isfile(fn):
                masks[type_name][level] = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            else:
                masks[type_name][level] = None
    return lines, metrics, masks


# ---------------------------------------------------------------------------
# Comparison + status synthesis
# ---------------------------------------------------------------------------

def _ratio_change(new: float, old: float) -> float:
    """abs(new-old)/max(old, eps). Returns 0.0 when old==new==0."""
    if old == 0 and new == 0:
        return 0.0
    denom = max(abs(old), 1e-6)
    return abs(new - old) / denom


def compare_case(case: str, current_lines: List[Dict],
                 shape: Tuple[int, int]) -> Dict:
    """Run all comparisons against this case's baseline. Returns a report dict."""
    base_lines, base_metrics, base_masks = load_baseline(case)
    stroke_width = float(base_metrics["stroke_width"])

    base_graph = base_metrics["graph"]
    cur_graph = compute_graph_metrics(current_lines)
    cur_hash = hash_lines(current_lines)
    base_hash = base_metrics.get("hash") or ""

    # Per-type IOU + distance across thin/normal/loose.
    per_type: Dict[str, Dict] = {}
    fail_reasons: List[str] = []
    warn_reasons: List[str] = []
    for type_name in TYPES:
        per_type[type_name] = {"iou": {}, "iou_drop": {}, "distance": {}}
        for level in THICKNESS_LEVELS:
            thk = thickness_for_level(level, stroke_width)
            cur_mask = rasterize_lines(current_lines, shape, type_name, thk)
            base_mask = base_masks[type_name].get(level)
            if base_mask is None:
                continue
            # Baseline mask is uint8; treat as binary.
            iou_v = compute_iou(cur_mask, base_mask)
            base_iou = 1.0  # baseline vs itself
            drop = base_iou - iou_v
            per_type[type_name]["iou"][level] = round(iou_v, 6)
            per_type[type_name]["iou_drop"][level] = round(drop, 6)

            if level == "normal":
                dist = compute_distance_metrics(cur_mask, base_mask)
                per_type[type_name]["distance"]["normal"] = {
                    "mean": round(dist["mean"], 4),
                    "p95": round(dist["p95"], 4),
                }
                mean_max = max(1.0, 0.12 * stroke_width)
                p95_max = max(2.0, 0.30 * stroke_width)
                if dist["mean"] > mean_max:
                    fail_reasons.append(
                        f"{type_name}: normal mean distance {dist['mean']:.2f} > {mean_max:.2f}")
                if dist["p95"] > p95_max:
                    fail_reasons.append(
                        f"{type_name}: normal p95 distance {dist['p95']:.2f} > {p95_max:.2f}")

            if level == "normal" and drop > NORMAL_IOU_DROP_MAX[type_name]:
                fail_reasons.append(
                    f"{type_name}: normal IOU drop {drop:.4f} > {NORMAL_IOU_DROP_MAX[type_name]:.4f}")
            elif level == "loose" and drop > LOOSE_IOU_DROP_MAX[type_name]:
                fail_reasons.append(
                    f"{type_name}: loose IOU drop {drop:.4f} > {LOOSE_IOU_DROP_MAX[type_name]:.4f}")
            elif level == "thin" and drop > THIN_IOU_DROP_WARN[type_name]:
                warn_reasons.append(
                    f"{type_name}: thin IOU drop {drop:.4f} > {THIN_IOU_DROP_WARN[type_name]:.4f}")

    # Graph-level checks.
    free_delta = cur_graph["free_endpoints"] - base_graph["free_endpoints"]
    if free_delta > FREE_ENDPOINTS_INCREASE_MAX:
        fail_reasons.append(
            f"free_endpoints +{free_delta} > {FREE_ENDPOINTS_INCREASE_MAX}")

    seg_ratio = _ratio_change(cur_graph["num_segments"], base_graph["num_segments"])
    if seg_ratio > NUM_SEGMENTS_CHANGE_RATIO_MAX:
        fail_reasons.append(
            f"num_segments change ratio {seg_ratio:.3f} > {NUM_SEGMENTS_CHANGE_RATIO_MAX:.3f}")

    # Total length: per-type total or summed across all types?
    # Spec just says total_length_change_ratio — use overall sum across types.
    base_total_len = sum(base_graph.get("total_length_by_type", {}).values())
    cur_total_len = sum(cur_graph["total_length_by_type"].values())
    len_ratio = _ratio_change(cur_total_len, base_total_len)
    if len_ratio > TOTAL_LENGTH_CHANGE_RATIO_MAX:
        fail_reasons.append(
            f"total_length change ratio {len_ratio:.3f} > {TOTAL_LENGTH_CHANGE_RATIO_MAX:.3f}")

    # Soft warnings
    if cur_graph["num_short_segments"] > base_graph.get("num_short_segments", 0):
        warn_reasons.append(
            f"num_short_segments grew {base_graph.get('num_short_segments',0)} -> "
            f"{cur_graph['num_short_segments']}")

    hash_changed = cur_hash != base_hash
    # If hash changed but no fail/warn reasons accumulated -> WARNING (per spec).
    if hash_changed and not fail_reasons and not warn_reasons:
        warn_reasons.append("hash changed but all IOUs within thresholds")

    if fail_reasons:
        status = "FAIL"
    elif warn_reasons:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "case": case,
        "status": status,
        "stroke_width": stroke_width,
        "hash": {"baseline": base_hash, "current": cur_hash, "changed": hash_changed},
        "graph": {
            "baseline": base_graph,
            "current": cur_graph,
            "free_endpoints_delta": free_delta,
            "num_segments_change_ratio": round(seg_ratio, 4),
            "total_length_change_ratio": round(len_ratio, 4),
        },
        "per_type": per_type,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "thresholds": {
            "normal_iou_drop_max": NORMAL_IOU_DROP_MAX,
            "loose_iou_drop_max": LOOSE_IOU_DROP_MAX,
            "thin_iou_drop_warn": THIN_IOU_DROP_WARN,
            "free_endpoints_increase_max": FREE_ENDPOINTS_INCREASE_MAX,
            "num_segments_change_ratio_max": NUM_SEGMENTS_CHANGE_RATIO_MAX,
            "total_length_change_ratio_max": TOTAL_LENGTH_CHANGE_RATIO_MAX,
            "mean_distance_max": round(max(1.0, 0.12 * stroke_width), 3),
            "p95_distance_max": round(max(2.0, 0.30 * stroke_width), 3),
        },
    }


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def write_overlay(overlay_path: str, current_lines: List[Dict],
                  baseline_lines: List[Dict], shape: Tuple[int, int],
                  stroke_width: float) -> None:
    """3-color diff using normal-thickness combined mask of all types.

    baseline-only -> red (BGR (0,0,255))
    current-only  -> green (BGR (0,255,0))
    overlap       -> black
    background    -> white
    """
    thk = thickness_for_level("normal", stroke_width)
    h, w = shape[:2]
    base_all = np.zeros((h, w), dtype=np.uint8)
    cur_all = np.zeros((h, w), dtype=np.uint8)
    for t in TYPES:
        base_all = cv2.bitwise_or(base_all, rasterize_lines(baseline_lines, shape, t, thk))
        cur_all = cv2.bitwise_or(cur_all, rasterize_lines(current_lines, shape, t, thk))
    b = base_all > 0
    c = cur_all > 0
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)  # white bg
    overlay[b & c] = (0, 0, 0)        # overlap -> black
    overlay[b & ~c] = (0, 0, 255)     # baseline-only -> red
    overlay[~b & c] = (0, 255, 0)     # current-only -> green
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    cv2.imwrite(overlay_path, overlay)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_report(report: Dict) -> None:
    s = report["status"]
    case = report["case"]
    marker = {"PASS": "PASS", "WARNING": "WARN", "FAIL": "FAIL",
              "NO_BASELINE": "NEW "}.get(s, s)
    print(f"\n[{marker}] {case}")
    g = report["graph"]
    bg, cg = g.get("baseline", {}) or {}, g["current"]
    print(f"  hash:    {report['hash']['baseline'][:10] or '<none>':10s} -> "
          f"{report['hash']['current'][:10]}"
          f" {'(changed)' if report['hash']['changed'] else '(same)'}")
    if bg:
        print(f"  segs:    {bg['num_segments']:4d} -> {cg['num_segments']:4d}  "
              f"(ratio {g['num_segments_change_ratio']:+.3f})")
        print(f"  free:    {bg['free_endpoints']:4d} -> {cg['free_endpoints']:4d}  "
              f"(delta {g['free_endpoints_delta']:+d})")
        print(f"  length:  ratio {g['total_length_change_ratio']:+.3f}")
    else:
        print(f"  segs:    (new) {cg['num_segments']}")
        print(f"  free:    (new) {cg['free_endpoints']}")
    print(f"  stroke_width (baseline): {report['stroke_width']:.2f}")
    for t in TYPES:
        pt = report["per_type"].get(t, {})
        iou = pt.get("iou", {})
        if not iou:
            continue
        print(f"  {t:6s}  thin={iou.get('thin', float('nan')):.4f}  "
              f"normal={iou.get('normal', float('nan')):.4f}  "
              f"loose={iou.get('loose', float('nan')):.4f}")
    v18 = report.get("v18")
    if v18:
        cm = v18["current_metrics"]
        bm = v18["baseline_metrics"]
        print(f"  v18 wall_iou_vs_source:   "
              f"{bm.get('wall_iou_vs_source', float('nan')):.4f} -> "
              f"{cm.get('wall_iou_vs_source', float('nan')):.4f}")
        print(f"  v18 door_iou_vs_source:   "
              f"{bm.get('door_iou_vs_source', float('nan')):.4f} -> "
              f"{cm.get('door_iou_vs_source', float('nan')):.4f}")
        print(f"  v18 window_iou_vs_source: "
              f"{bm.get('window_iou_vs_source', float('nan')):.4f} -> "
              f"{cm.get('window_iou_vs_source', float('nan')):.4f}")
        print(f"  v18 phantom_wall_frac:    "
              f"{bm.get('phantom_wall_frac', float('nan')):.4f} -> "
              f"{cm.get('phantom_wall_frac', float('nan')):.4f}")
        print(f"  v18 invariants: strict={v18['invariants']['strict_count']}  "
              f"goal={v18['invariants']['goal_count']}")
        print(f"  v18 floating_openings:    "
              f"{bm.get('floating_openings', 0)} -> "
              f"{cm.get('floating_openings', 0)}")
    if report["warn_reasons"]:
        print("  WARN:")
        for r in report["warn_reasons"]:
            print(f"    - {r}")
    if report["fail_reasons"]:
        print("  FAIL:")
        for r in report["fail_reasons"]:
            print(f"    - {r}")


# ---------------------------------------------------------------------------
# --update-baseline interactive gate
# ---------------------------------------------------------------------------

def _prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return ""


def confirm_update(case: str, status: str, exists: bool, *, force_yes: bool) -> Tuple[bool, str]:
    """Returns (do_update, source_tag)."""
    if force_yes:
        return True, "manual_yes_flag"
    if not exists:
        token = _prompt(f"Type CREATE {case} to confirm baseline creation: ").strip()
        if token == f"CREATE {case}":
            return True, "interactive_create"
        return False, "declined_create"
    if status == "FAIL":
        token = _prompt(f"Type UPDATE {case} to update baseline anyway: ").strip()
        if token == f"UPDATE {case}":
            return True, "interactive_fail_override"
        return False, "declined_fail"
    # PASS / WARNING
    ans = _prompt(f"Update baseline for {case}? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        return True, "interactive_yes"
    return False, "declined"


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_one_case(case: str) -> Tuple[Dict, List[Dict], Tuple[int, int], float]:
    """Run pipeline + write current outputs + (if baseline exists) compare.

    Returns (report_or_skeleton, lines, shape, stroke_width).
    """
    manifest = load_manifest(case)
    image_path = resolve_input_image(case, manifest)
    if not os.path.isfile(image_path):
        raise SystemExit(f"image not found for case {case!r}: {image_path}")

    lines, shape, wall_mask, bgr = run_pipeline_pure(image_path)
    stroke_width: float
    if baseline_exists(case):
        base_metrics_path = baseline_paths(case)["metrics"]
        with open(base_metrics_path, "r", encoding="utf-8") as f:
            stroke_width = float(json.load(f)["stroke_width"])
    else:
        stroke_width = estimate_stroke_width(wall_mask)

    cp = current_paths(case)
    os.makedirs(cp["dir"], exist_ok=True)
    with open(cp["lines"], "w", encoding="utf-8") as f:
        json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)
    cur_hash = hash_lines(lines)
    with open(cp["hash"], "w", encoding="utf-8") as f:
        f.write(cur_hash + "\n")
    write_masks(cp["masks_dir"], lines, shape, stroke_width)
    graph = compute_graph_metrics(lines)
    with open(cp["metrics"], "w", encoding="utf-8") as f:
        json.dump({
            "regression_format_version": REGRESSION_FORMAT_VERSION,
            "rasterizer_version": RASTERIZER_VERSION,
            "stroke_width": stroke_width,
            "image_shape": [shape[0], shape[1]],
            "hash": cur_hash,
            "graph": graph,
            "thickness_px": {lvl: thickness_for_level(lvl, stroke_width)
                             for lvl in THICKNESS_LEVELS},
        }, f, ensure_ascii=False, indent=2)

    if not baseline_exists(case):
        skeleton = {
            "case": case,
            "status": "NO_BASELINE",
            "stroke_width": stroke_width,
            "hash": {"baseline": "", "current": cur_hash, "changed": True},
            "graph": {
                "baseline": {},
                "current": graph,
                "free_endpoints_delta": 0,
                "num_segments_change_ratio": 0.0,
                "total_length_change_ratio": 0.0,
            },
            "per_type": {},
            "fail_reasons": [],
            "warn_reasons": [],
        }
        with open(cp["report"], "w", encoding="utf-8") as f:
            json.dump(skeleton, f, ensure_ascii=False, indent=2)
        return skeleton, lines, shape, stroke_width

    report = compare_case(case, lines, shape)

    # Step 18: layer the metric + invariant report on top. STRICT invariant
    # violations are unconditional FAIL (no tolerance). GOAL invariant
    # violations are reported but do not fail the gate until §2/§3 ship —
    # they would all fire right now and hide real regressions.
    # Per-metric regressions append to fail_reasons via the rules in
    # tests/metrics.METRIC_RULES.
    from tests.metrics import compute_v18_report  # lazy import (avoids cost when this module is imported by ablation etc)
    base_lines, _, _ = load_baseline(case)
    v18 = compute_v18_report(lines, base_lines, bgr)
    report["v18"] = v18
    if v18["invariant_fails"]:
        report["fail_reasons"].extend(
            f"invariant: {msg}" for msg in v18["invariant_fails"])
    if v18["metric_fails"]:
        report["fail_reasons"].extend(
            f"metric: {msg}" for msg in v18["metric_fails"])
    if v18["invariants"]["goal_count"] > 0:
        report["warn_reasons"].append(
            f"v18 goal-invariants: {v18['invariants']['goal_count']} "
            f"floating_opening(s) — gated once §2/§3 ship")
    # Recompute final status after our additions.
    if report["fail_reasons"]:
        report["status"] = "FAIL"
    elif report["warn_reasons"] and report["status"] == "PASS":
        report["status"] = "WARNING"

    with open(cp["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_overlay(cp["overlay"], lines, base_lines, shape, stroke_width)
    return report, lines, shape, stroke_width


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="only run this case", default=None)
    ap.add_argument("--update-baseline", action="store_true",
                    help="after running, prompt to update baseline(s)")
    ap.add_argument("--yes", action="store_true",
                    help="bypass all interactive prompts (logged in metrics)")
    ap.add_argument("--all", action="store_true",
                    help="when updating, require UPDATE ALL token to update every case")
    args = ap.parse_args()

    cases = discover_cases(args.case)
    if not cases:
        raise SystemExit("no cases found under tests/cases/")

    # If --all, require global token (unless --yes).
    if args.update_baseline and args.all and not args.yes:
        tok = _prompt("Type UPDATE ALL to confirm updating every case: ").strip()
        if tok != "UPDATE ALL":
            print("aborted: UPDATE ALL not entered")
            sys.exit(1)

    # Determine missing-baseline situation up front for the "no --update-baseline" path.
    any_missing = any(not baseline_exists(c) for c in cases)
    if any_missing and not args.update_baseline:
        missing = [c for c in cases if not baseline_exists(c)]
        print(f"ERROR: no baseline for case(s): {missing}", file=sys.stderr)
        print("Run with --update-baseline (you will be prompted to confirm).",
              file=sys.stderr)
        sys.exit(1)

    any_fail = False
    any_warn = False
    for case in cases:
        existed_before = baseline_exists(case)
        report, lines, shape, stroke_width = run_one_case(case)
        print_report(report)
        cp = current_paths(case)
        if report["status"] != "NO_BASELINE":
            print(f"  overlay: {cp['overlay']}")
        else:
            print("  status:  NO_BASELINE (use --update-baseline to create one)")

        if report["status"] == "FAIL":
            any_fail = True
        elif report["status"] == "WARNING":
            any_warn = True

        if args.update_baseline:
            status_for_prompt = "PASS" if report["status"] == "NO_BASELINE" else report["status"]
            do_update, source = confirm_update(
                case, status_for_prompt, existed_before,
                force_yes=args.yes or args.all,
            )
            if do_update:
                write_baseline(case, lines, shape, stroke_width, source=source)
                print(f"  baseline updated ({source})")
            else:
                print(f"  baseline NOT updated ({source})")

    print()
    if any_fail:
        print("RESULT: FAIL")
        sys.exit(1)
    if any_warn:
        print("RESULT: WARNING")
        sys.exit(0)
    print("RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
