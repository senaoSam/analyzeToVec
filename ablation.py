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
    # (id, human-readable description). Mirrors the current vectorize_bgr
    # geometric-optimisation pipeline (step 7 phase 5 state). Notes:
    #   - ``manhattan_intersection_snap`` removed step 4.9.7 (pure NO-OP
    #     post canonical_line + thickness-aware manhattan_t_project)
    #   - ``cluster_parallel`` is the step 6 phase 3 candidate-based
    #     ``_accept_parallel_merge_candidates(skip_score=True)``
    #   - ``merge_final`` is the step 6 phase 1 candidate-based
    #     ``_accept_merge_candidates(perp_tol=0, gap_tol=0,
    #     junction_aware=True)``
    #   - step 7 phases 1-5 migrated all 5 snap passes to candidate-based:
    #     snap_colinear / snap_endpoints_1 -> endpoint_fuse / cluster_2d,
    #     manhattan_t_project / grid_snap_2 -> t_project_candidates,
    #     fuse_close_endpoints -> endpoint_fuse_candidates. Ablation labels
    #     keep the legacy IDs so historical CSVs are still comparable.
    ("axis_align",            "axis_align_segments"),
    ("snap_colinear",         "_accept_fuse_candidates (step 7 phase 4)"),
    ("merge_collin_1",        "merge_collinear (pre-T/L)"),
    ("t_junction_snap",       "t_junction_snap"),
    ("truncate_overshoots",   "truncate_overshoots"),
    ("merge_collin_2",        "merge_collinear (post T/L)"),
    ("snap_endpoints_1",      "_accept_2d_cluster_candidates (step 7 phase 5)"),
    ("manhattan_force_axis",  "manhattan_force_axis"),
    ("canonical_line",        "canonicalize_offsets (step 4.9.2)"),
    ("manhattan_t_project",   "_accept_t_project_candidates (step 7 phase 2)"),
    ("cluster_parallel",      "_accept_parallel_merge_candidates (step 6 phase 3)"),
    ("grid_snap_2",           "_accept_t_project_candidates (step 7 phase 3)"),
    ("t_snap_with_extension", "t_snap_with_extension (candidate)"),
    ("brute_force_ray",       "brute_force_ray_extend (candidate)"),
    ("insert_connectors",     "insert_missing_connectors (candidate)"),
    ("proximal_bridge",       "_accept_bridge_candidates (step 4.7)"),
    ("fuse_close_endpoints",  "_accept_fuse_candidates (step 7 phase 1)"),
    ("merge_final",           "_accept_merge_candidates (step 6 phase 1)"),
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

    # Mirror of current vectorize_bgr's geometric-optimisation pipeline.
    if disabled != "axis_align":
        s = V.axis_align_segments(s, V.AXIS_SNAP_DEG)
    if disabled != "snap_colinear":
        # Step 7 phase 4: candidate-based 1D fuse (fixed colinear_tol).
        s = V._accept_fuse_candidates(s, fallback_tol=t["colinear"],
                                       masks=None)
    if disabled != "merge_collin_1":
        s = V.merge_collinear(s, t["merge_perp"], t["merge_gap"])
    if disabled != "t_junction_snap":
        s = V.t_junction_snap(s, t["t_snap"])
    if disabled != "truncate_overshoots":
        s = V.truncate_overshoots(s, t["l_extend"])
    if disabled != "merge_collin_2":
        s = V.merge_collinear(s, t["merge_perp"], t["merge_gap"])
    if disabled != "snap_endpoints_1":
        # Step 7 phase 5: candidate-based 2D NetworkX-style cluster.
        s = V._accept_2d_cluster_candidates(s, tol=t["snap"])
    if disabled != "manhattan_force_axis":
        s = V.manhattan_force_axis(s)
    if disabled != "canonical_line":
        s = V.canonicalize_offsets(s, wall_mask=masks.get("wall"),
                                   attach_thickness=True)
    # manhattan_intersection_snap call removed step 4.9.7 (pure NO-OP)
    if disabled != "manhattan_t_project":
        # Step 7 phase 2: candidate-based thickness-aware T-project.
        s = V._accept_t_project_candidates(
            s, fallback_tol=t["manhattan"], masks=masks)
    if disabled != "cluster_parallel":
        # Step 6 phase 3: candidate-based parallel merge (skip_score=True).
        s = V._accept_parallel_merge_candidates(
            s, perp_tol=t["parallel_merge"], skip_score=True,
            wall_evidence=None,
            door_mask=masks.get("door"),
            window_mask=masks.get("window"))
    if disabled != "grid_snap_2":
        # Step 7 phase 3: same generator as manhattan_t_project, fixed tol.
        s = V._accept_t_project_candidates(
            s, fallback_tol=t["grid_snap"], masks=None)
    if disabled != "t_snap_with_extension":
        s = V.t_snap_with_extension(s, t["gap_close"], masks=masks)
    if disabled != "brute_force_ray":
        V.brute_force_ray_extend(s, t["ray_ext"], V.RAY_EXT_LOOSE_PX)
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "insert_connectors":
        V.insert_missing_connectors(s, t["colinear_loose"], t["connector_max"],
                                    wall_mask=masks.get("wall"))
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "proximal_bridge":
        s = V._accept_bridge_candidates(s,
                                         max_radius=t["l_ext_asym"],
                                         wall_mask=masks.get("wall"),
                                         wall_evidence=None,
                                         door_mask=masks.get("door"),
                                         window_mask=masks.get("window"))
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "fuse_close_endpoints":
        # Step 7 phase 1: candidate-based thickness-aware 1D fuse.
        s = V._accept_fuse_candidates(s, fallback_tol=t["ray_fuse"],
                                       masks=masks)
    s = [seg for seg in s if (seg["x1"], seg["y1"]) != (seg["x2"], seg["y2"])]
    if disabled != "merge_final":
        # Step 6 phase 1: candidate-based final merge.
        s = V._accept_merge_candidates(
            s, perp_tol=0.0, gap_tol=0.0, junction_aware=True,
            wall_evidence=None,
            door_mask=masks.get("door"),
            window_mask=masks.get("window"))
    # Strip pipeline-internal fields so segments serialise back to canonical
    # {type, x1, y1, x2, y2} like vectorize_bgr's tail does.
    for seg in s:
        for k in V._INTERNAL_SEG_FIELDS:
            seg.pop(k, None)
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
