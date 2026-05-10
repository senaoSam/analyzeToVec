"""Pass-ablation harness for vectorize.py.

Runs the full pipeline once as baseline, then re-runs it 22 times — once per
pass, with that single pass replaced by a no-op. Reports for each ablation:

  - segment_count delta vs baseline
  - free_endpoints (degree=1) delta vs baseline
  - diagonal_count
  - wall_mask_support (avg fraction of output wall pixels that lie on the
    HSV wall mask -- detects phantom-wall invention)
  - IOU vs baseline (rasterized line image, dilated to thickness 2)

Across all 3 reference images. Output: ablation_report.csv + console summary.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

import vectorize as V


REF_IMAGES = [
    "source.png",
    "sg2.png",
    "Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png",
]


PASSES: List[Tuple[str, str]] = [
    # (id, human-readable description)
    ("axis_align",            "Step 4a: axis_align_segments"),
    ("snap_colinear",         "Step 4b: snap_colinear_coords (pass 1)"),
    ("merge_collin_1",        "Step 4c: merge_collinear (pass 1, pre-T/L)"),
    ("t_junction_snap",       "Step 4d: t_junction_snap"),
    ("extend_to_intersect",   "Step 4d: extend_to_intersect"),
    ("truncate_overshoots",   "Step 4d: truncate_overshoots"),
    ("merge_collin_2",        "Step 4d: merge_collinear (pass 2, post T/L)"),
    ("snap_endpoints_1",      "Step 4: snap_endpoints (pass 1, post T/L)"),
    ("prune_tails",           "Step 4e: prune_tails"),
    ("snap_endpoints_2",      "Step 4e: snap_endpoints (pass 2, post-prune)"),
    ("manhattan_force_axis",  "Step 4f-a: manhattan_force_axis"),
    ("manhattan_isect_snap",  "Step 4f-b: manhattan_intersection_snap (x2)"),
    ("manhattan_t_project",   "Step 4f-c: manhattan_t_project"),
    ("manhattan_merge_1",     "Step 4f-d: manhattan_ultimate_merge (post-Manhattan)"),
    ("cluster_parallel",      "Step 4g: cluster_parallel_duplicates"),
    ("grid_snap_1",           "Step 4g: grid_snap_endpoints (pass 1)"),
    ("force_l_corner",        "Step 4g: force_l_corner_closure"),
    ("grid_snap_2",           "Step 4g: grid_snap_endpoints (pass 2)"),
    ("force_close_free_l_1",  "Step 4h: force_close_free_l_corners (pass 1)"),
    ("t_snap_with_extension", "Step 4h: t_snap_with_extension"),
    ("force_close_free_l_2",  "Step 4h: force_close_free_l_corners (pass 2)"),
    ("final_polish_tails",    "Step 4h: final_polish_short_tails"),
    ("brute_force_ray",       "Step 4i: brute_force_ray_extend"),
    ("extend_trunk_to_loose", "Step 4i: extend_trunk_to_loose"),
    ("mask_gated_l_extend",   "Step 4i: mask_gated_l_extend"),
    ("insert_connectors",     "Step 4i: insert_missing_connectors"),
    ("fuse_close_endpoints",  "Step 4i: fuse_close_endpoints"),
]


def _scaled_tols(h: int, w: int) -> Dict[str, float]:
    s = max(1.0, max(h, w) / V.REFERENCE_DIM_PX)
    return {
        "scale": s,
        "snap": V.SNAP_TOLERANCE_PX * s,
        "colinear": V.COLINEAR_TOL_PX * s,
        "merge_perp": V.MERGE_PERP_TOL_PX * s,
        "merge_gap": V.MERGE_GAP_TOL_PX * s,
        "t_snap": V.T_SNAP_TOL_PX * s,
        "l_extend": V.L_EXTEND_TOL_PX * s,
        "tail_prune": V.TAIL_PRUNE_LEN_PX * s,
        "manhattan": V.MANHATTAN_SNAP_TOL_PX * s,
        "parallel_merge": V.PARALLEL_MERGE_TOL_PX * s,
        "grid_snap": V.GRID_SNAP_TOL_PX * s,
        "gap_close": V.GAP_CLOSE_TOL_PX * s,
        "gap_final": V.GAP_FINAL_PRUNE_PX * s,
        "ray_ext": V.RAY_EXT_TOL_PX * s,
        "ray_fuse": V.RAY_EXT_FUSE_PX * s,
        "colinear_loose": V.COLINEAR_LOOSE_TOL_PX * s,
        "connector_max": V.CONNECTOR_MAX_LEN_PX * s,
        "trunk_perp": V.TRUNK_EXTEND_PERP_PX * s,
        "trunk_gap": V.TRUNK_EXTEND_GAP_PX * s,
        "l_ext_asym": V.L_EXT_ASYM_PX * s,
    }


def run_pipeline(bgr: np.ndarray, disabled: str = "") -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Run the full pipeline. If `disabled` matches a pass id, that pass is no-op'd.

    Returns (final segments list, masks).
    """
    h, w = bgr.shape[:2]
    t = _scaled_tols(h, w)

    masks = V.segment_colors(bgr)

    typed_segments: List[Dict] = []
    for label, mask in masks.items():
        skel = V.skeletonize_mask(mask)
        branches = V.extract_branches(skel)
        segs = V.branches_to_segments(branches)
        for x1, y1, x2, y2 in segs:
            typed_segments.append({"type": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    s = typed_segments

    # 4a
    if disabled != "axis_align":
        s = V.axis_align_segments(s, V.AXIS_SNAP_DEG)
    if disabled != "snap_colinear":
        s = V.snap_colinear_coords(s, t["colinear"])
    # 4c (1)
    if disabled != "merge_collin_1":
        s = V.merge_collinear(s, t["merge_perp"], t["merge_gap"])
    # 4d
    if disabled != "t_junction_snap":
        s = V.t_junction_snap(s, t["t_snap"])
    if disabled != "extend_to_intersect":
        s = V.extend_to_intersect(s, t["l_extend"])
    if disabled != "truncate_overshoots":
        s = V.truncate_overshoots(s, t["l_extend"])
    if disabled != "merge_collin_2":
        s = V.merge_collinear(s, t["merge_perp"], t["merge_gap"])
    if disabled != "snap_endpoints_1":
        s = V.snap_endpoints(s, t["snap"])
    # 4e
    if disabled != "prune_tails":
        s = V.prune_tails(s, t["tail_prune"])
    if disabled != "snap_endpoints_2":
        s = V.snap_endpoints(s, t["snap"])
    # 4f
    if disabled != "manhattan_force_axis":
        s = V.manhattan_force_axis(s)
    if disabled != "manhattan_isect_snap":
        s = V.manhattan_intersection_snap(s, t["manhattan"])
        s = V.manhattan_intersection_snap(s, t["manhattan"])
    if disabled != "manhattan_t_project":
        s = V.manhattan_t_project(s, t["manhattan"])
    if disabled != "manhattan_merge_1":
        s = V.manhattan_ultimate_merge(s)
    # 4g
    if disabled != "cluster_parallel":
        s = V.cluster_parallel_duplicates(s, t["parallel_merge"])
    if disabled != "grid_snap_1":
        s = V.grid_snap_endpoints(s, t["grid_snap"])
    if disabled != "force_l_corner":
        s = V.force_l_corner_closure(s, t["grid_snap"])
    if disabled != "grid_snap_2":
        s = V.grid_snap_endpoints(s, t["grid_snap"])
    s = V.manhattan_ultimate_merge(s)
    # 4h
    if disabled != "force_close_free_l_1":
        s = V.force_close_free_l_corners(s, t["gap_close"])
    if disabled != "t_snap_with_extension":
        s = V.t_snap_with_extension(s, t["gap_close"], masks=masks)
    if disabled != "force_close_free_l_2":
        s = V.force_close_free_l_corners(s, t["gap_close"])
    s = V.manhattan_ultimate_merge(s)
    if disabled != "final_polish_tails":
        s = V.final_polish_short_tails(s, t["gap_final"])
    s = V.manhattan_ultimate_merge(s)
    # 4i (in-place)
    if disabled != "brute_force_ray":
        V.brute_force_ray_extend(s, t["ray_ext"], V.RAY_EXT_LOOSE_PX)
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "extend_trunk_to_loose":
        V.extend_trunk_to_loose(s, t["trunk_perp"], t["trunk_gap"], masks=masks)
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "mask_gated_l_extend":
        s = V.mask_gated_l_extend(s, max_gap=t["l_ext_asym"], masks=masks)
    if disabled != "insert_connectors":
        V.insert_missing_connectors(s, t["colinear_loose"], t["connector_max"],
                                    wall_mask=masks.get("wall"))
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "fuse_close_endpoints":
        V.fuse_close_endpoints(s, t["ray_fuse"])
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    s = V.manhattan_ultimate_merge(s)
    return s, masks


def rasterize(segments: List[Dict], shape: Tuple[int, int], thickness: int = 2) -> np.ndarray:
    h, w = shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for seg in segments:
        p1 = (int(round(seg["x1"])), int(round(seg["y1"])))
        p2 = (int(round(seg["x2"])), int(round(seg["y2"])))
        cv2.line(canvas, p1, p2, 255, thickness=thickness)
    return canvas


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = (a > 0)
    b_b = (b > 0)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def metrics(segments: List[Dict], wall_mask: np.ndarray, shape: Tuple[int, int]) -> Dict:
    from collections import Counter
    nodes: Counter = Counter()
    n_diag = 0
    for s in segments:
        nodes[(s["x1"], s["y1"])] += 1
        nodes[(s["x2"], s["y2"])] += 1
        if V._classify_axis(s) == "d":
            n_diag += 1
    deg_hist = Counter(nodes.values())
    free_eps = deg_hist.get(1, 0)

    # wall_mask_support: for each WALL output segment, sample along it and
    # measure fraction of pixels that fall on the wall mask. Phantom walls
    # (segments invented across white space) score low here.
    walls = [s for s in segments if s["type"] == "wall"]
    if walls and wall_mask is not None:
        h, w = wall_mask.shape
        total_pts = 0
        on_pts = 0
        for s in walls:
            length = max(1, int(round(np.hypot(s["x2"] - s["x1"], s["y2"] - s["y1"]))))
            xs = np.linspace(s["x1"], s["x2"], length).round().astype(int)
            ys = np.linspace(s["y1"], s["y2"], length).round().astype(int)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            on_pts += int((wall_mask[ys, xs] > 0).sum())
            total_pts += int(length)
        wall_support = on_pts / total_pts if total_pts else 0.0
    else:
        wall_support = 0.0

    return {
        "n": len(segments),
        "free": free_eps,
        "diag": n_diag,
        "wall_support": wall_support,
        "n_wall": sum(1 for s in segments if s["type"] == "wall"),
        "n_window": sum(1 for s in segments if s["type"] == "window"),
        "n_door": sum(1 for s in segments if s["type"] == "door"),
    }


def main() -> None:
    out_csv = os.path.join(os.path.dirname(__file__), "output", "ablation_report.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    per_image_baseline = {}

    print("=" * 92)
    print("ABLATION HARNESS — disable one pass at a time, measure delta vs baseline")
    print("=" * 92)

    # Load all images once.
    bgrs = {}
    for name in REF_IMAGES:
        path = V._resolve_src_path(name)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  !! could not read {path}, skipping")
            continue
        bgrs[name] = bgr

    # Baseline.
    print("\n--- BASELINE ---")
    baselines = {}
    for name, bgr in bgrs.items():
        t0 = time.time()
        segs, masks = run_pipeline(bgr, disabled="")
        dt = time.time() - t0
        m = metrics(segs, masks.get("wall"), bgr.shape[:2])
        raster = rasterize(segs, bgr.shape[:2])
        baselines[name] = {"segs": segs, "masks": masks, "raster": raster, "m": m}
        per_image_baseline[name] = m
        print(f"  {name:50s}  n={m['n']:4d}  free={m['free']:4d}  diag={m['diag']:3d}  "
              f"wall_sup={m['wall_support']:.3f}  ({dt:.1f}s)")

    # Ablation passes.
    print("\n--- ABLATIONS ---")
    print(f"{'PASS':<28} {'IMAGE':<48} {'dN':>6} {'dFree':>6} {'Diag':>5} "
          f"{'WS':>6} {'IOU':>6}")
    print("-" * 110)

    for pass_id, pass_desc in PASSES:
        for name, bgr in bgrs.items():
            try:
                segs, masks = run_pipeline(bgr, disabled=pass_id)
            except Exception as e:
                print(f"  !! {pass_id} on {name} -> {type(e).__name__}: {e}")
                continue
            m = metrics(segs, masks.get("wall"), bgr.shape[:2])
            base = baselines[name]
            raster = rasterize(segs, bgr.shape[:2])
            iou_v = iou(raster, base["raster"])
            dn = m["n"] - base["m"]["n"]
            dfree = m["free"] - base["m"]["free"]
            print(f"{pass_id:<28} {name:<48} {dn:>+6d} {dfree:>+6d} {m['diag']:>5d} "
                  f"{m['wall_support']:>6.3f} {iou_v:>6.3f}")
            rows.append({
                "pass": pass_id,
                "pass_desc": pass_desc,
                "image": name,
                "baseline_n": base["m"]["n"],
                "ablated_n": m["n"],
                "delta_n": dn,
                "baseline_free": base["m"]["free"],
                "ablated_free": m["free"],
                "delta_free": dfree,
                "diag": m["diag"],
                "wall_support": round(m["wall_support"], 4),
                "baseline_wall_support": round(base["m"]["wall_support"], 4),
                "iou_vs_baseline": round(iou_v, 4),
                "n_wall": m["n_wall"],
                "n_window": m["n_window"],
                "n_door": m["n_door"],
            })
        print("-" * 110)

    # Write CSV.
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {out_csv}")

    # Summary: rank passes by max |delta_n| across images and by min IOU.
    print("\n=== PASS IMPACT SUMMARY ===")
    by_pass: Dict[str, List[Dict]] = {}
    for r in rows:
        by_pass.setdefault(r["pass"], []).append(r)

    summary = []
    for pid, rs in by_pass.items():
        max_dn_abs = max(abs(r["delta_n"]) for r in rs)
        max_dfree_abs = max(abs(r["delta_free"]) for r in rs)
        min_iou = min(r["iou_vs_baseline"] for r in rs)
        sum_wall_drop = sum(max(0.0, r["baseline_wall_support"] - r["wall_support"]) for r in rs)
        summary.append({
            "pass": pid,
            "max_|dN|": max_dn_abs,
            "max_|dFree|": max_dfree_abs,
            "min_IOU": min_iou,
            "wall_support_drop_sum": round(sum_wall_drop, 4),
        })
    # Sort by smallest impact first (=most disposable)
    summary.sort(key=lambda r: (r["min_IOU"], -r["max_|dN|"]), reverse=True)
    print(f"\n{'PASS':<28} {'max|dN|':>8} {'max|dFree|':>11} {'min IOU':>9} {'WS drop':>9}")
    print("-" * 70)
    for s in summary:
        marker = ""
        if s["min_IOU"] > 0.999 and s["max_|dN|"] == 0 and s["max_|dFree|"] == 0:
            marker = "  <-- NO-OP across all 3 images"
        elif s["min_IOU"] > 0.99 and s["max_|dN|"] <= 2 and s["max_|dFree|"] <= 2:
            marker = "  <-- minimal impact"
        print(f"{s['pass']:<28} {s['max_|dN|']:>8} {s['max_|dFree|']:>11} "
              f"{s['min_IOU']:>9.4f} {s['wall_support_drop_sum']:>9.4f}{marker}")


if __name__ == "__main__":
    main()
