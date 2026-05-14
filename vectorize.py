"""
Floorplan raster -> vector skeleton converter.

Pipeline (per the spec in gemini-code-1778356910874.md):
  1. HSV color segmentation -> binary masks for {black wall, blue window, yellow door}.
  2. scikit-image morphology.skeletonize -> 1-pixel-wide centerline per mask.
  3. Decompose each skeleton into "branches" between junction/endpoint pixels,
     then fit each branch with cv2.approxPolyDP to recover straight segments.
  4. NetworkX node-snapping: cluster all segment endpoints (across all colors)
     within a tolerance radius, replace each cluster with its centroid so that
     T-junctions, L-corners and color-boundary handoffs share one exact point.
  5. Emit output.json and a debug preview PNG.

Run: python vectorize.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from canonical_line import compute_local_thickness
from audit import candidate_position as _audit_position

# Segment fields that are pipeline-internal — computed and used inside
# vectorize_bgr but stripped before the result leaves this module. Keeps
# the public JSON contract limited to ``{type, x1, y1, x2, y2}``.
_INTERNAL_SEG_FIELDS = ("local_thickness",)


SRC_DIR = os.path.join(os.path.dirname(__file__), "srcImg")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_SRC_NAME = "source.png"

# Reference image dimension (the size at which the default tolerance constants
# below were tuned — i.e. source.png, 1200x895). For larger images, geometric
# tolerances are scaled up by max(h, w) / REFERENCE_DIM so that absolute pixel
# gaps from skeletonization at thick walls (which scale with image size) get
# closed correctly. The factor is clamped to >= 1.0 so smaller images are not
# tightened (and source.png itself stays at 1.0 → no behaviour change).
REFERENCE_DIM_PX = 1200.0

# Snap tolerance for merging near-coincident endpoints across all color layers.
SNAP_TOLERANCE_PX = 6.0

# Drop branches whose pixel arc-length is below this — they are stubs / spurs
# left by skeletonization at thick corners.
MIN_BRANCH_LEN_PX = 6

# approxPolyDP epsilon (in pixels). Small value keeps fine joggles, large
# value smooths them away. 1.5 px works well for axis-aligned floorplans.
POLY_EPSILON_PX = 1.5

# Orthogonality: a segment whose angle is within this many degrees of pure
# horizontal/vertical is forced exactly axis-aligned. ±5° per the spec.
AXIS_SNAP_DEG = 5.0

# Tightened: only collapse x/y coordinates that are extremely close, so we
# don't flatten genuinely-distinct walls that happen to lie on adjacent grid
# lines. Set to 2-3 px per the user's spec.
COLINEAR_TOL_PX = 2.5

# Merge-collinear thresholds (same-type, same axis, same coordinate-line).
# Tightened: PERP_TOL governs "are these two on the same line?" — must be
# very small, otherwise visually-distinct walls 5-10 px apart get fused.
# GAP_TOL governs "are these endpoints close enough along the line to fuse?"
MERGE_PERP_TOL_PX = 2.5
MERGE_GAP_TOL_PX = 6.0

# Node-to-edge T-junction snapping: a free endpoint within this perpendicular
# distance of an orthogonal segment's body gets pulled onto that segment.
T_SNAP_TOL_PX = 12.0

# Extend-to-intersect: two perpendicular segments whose endpoints don't meet
# but whose lines cross within this extension distance from each endpoint
# get extended to the intersection.
L_EXTEND_TOL_PX = 14.0

# Final tail prune: a degree-1 endpoint sitting on an edge shorter than this
# is treated as a skeletonization spur and removed — but only when the spur's
# *non-free* end connects to a same-type segment (cross-type connections are
# preserved no matter how short, since they represent door/window jambs).
TAIL_PRUNE_LEN_PX = 8.0

# Type priority used when merging endpoint clusters. The lower the number, the
# higher the priority — its coordinate wins. Walls anchor doors/windows,
# never the other way around.
TYPE_PRIORITY = {"wall": 0, "window": 1, "door": 1}

# Manhattan routing: distance threshold for L-corner intersection snap and
# T-junction projection. An endpoint within this distance of the candidate
# corner / trunk locks onto the exact mathematical intersection.
MANHATTAN_SNAP_TOL_PX = 15.0

# Watertight closure thresholds.
# Two parallel same-type segments whose perpendicular distance is below this
# and whose body extents overlap are treated as one duplicated centerline
# (skeletonization artefact at thick walls) and merged into one centred line.
PARALLEL_MERGE_TOL_PX = 13.0

# Grid-snap tolerance: an endpoint within this perpendicular distance of an
# orthogonal reference line is extended (or shortened) onto that line exactly.
GRID_SNAP_TOL_PX = 15.0

# Ultimate gap-closing tolerance: the larger sweep used for degree-1 endpoints
# only (loose ends), aimed at sealing the last few floating-point gaps.
GAP_CLOSE_TOL_PX = 30.0

# Final polish: short stub deletion threshold for any tail that survived
# gap-closing because it had nothing to attach to.
GAP_FINAL_PRUNE_PX = 10.0

# Brute-force ray extension: search radius for the final closure pass and
# the "no nearby endpoint" tolerance for declaring an endpoint loose.
RAY_EXT_TOL_PX = 40.0
RAY_EXT_LOOSE_PX = 1.0
RAY_EXT_FUSE_PX = 2.0

# Missing-connector pass: when two loose endpoints share the same x (or y)
# within COLINEAR_LOOSE_TOL_PX and are no more than CONNECTOR_MAX_LEN_PX
# apart on the perpendicular axis, insert a new wall segment between them.
COLINEAR_LOOSE_TOL_PX = 3.0
CONNECTOR_MAX_LEN_PX = 80.0

# Trunk-extension-to-loose pass: a loose endpoint that's close to a trunk's
# axis line (perpendicular dist <= TRUNK_EXTEND_PERP_PX) but past the trunk's
# body by no more than TRUNK_EXTEND_GAP_PX triggers an extension of the
# trunk to swallow the joint.
TRUNK_EXTEND_PERP_PX = 8.0
TRUNK_EXTEND_GAP_PX = 60.0

# Mask-gated asymmetric L-extend: per-leg max extension for the closure pass
# that fixes thick-wall L-corners where the two skeleton centerlines miss
# each other in *both* axes simultaneously. The pass is gated by the wall
# mask so this can be relatively generous without inventing geometry.
L_EXT_ASYM_PX = 50.0


# Candidate-acceptance thresholds for the step-4 score-and-accept loop.
# Strict positive delta for "compound" candidates that immediately register
# their junction; relaxed near-zero for deferred-payoff snaps that rely on
# pseudo_junction or downstream merge to materialise the geometry.
CANDIDATE_MIN_ACCEPT_DELTA = 0.0
BRUTE_FORCE_MIN_ACCEPT_DELTA = -1e-6

# Cheap pre-score gate for synthetic gap-closing connectors: paths with
# mask support below this are not even built as candidates.
GAP_CONNECTOR_GATE_MIN = 0.30


# ---------------------------------------------------------------------------
# Step 1: color segmentation
# ---------------------------------------------------------------------------
#
# Wall detection now goes through a multi-detector evidence map instead of a
# single HSV threshold. The pieces:
#
#   D1  strong-black:        v < V_STRONG, hue/sat irrelevant     (weight 1.0)
#   D2  low-saturation dark: s < S_LOW AND v < adaptive Otsu       (weight 0.7)
#   D3  edge-supported:      v < V_EDGE_DARK AND grad > G_MIN      (weight 0.4)
#   D4  CC filter:           drop tiny / non-elongated components (suppression)
#
# The detectors are fused via element-wise max into a continuous 0.0-1.0
# evidence map, which is then thresholded back to binary via Otsu on the
# non-zero portion of the histogram. The binary mask is what the rest of
# the pipeline still consumes — wall_evidence is a side channel for future
# scoring work (step 4).
#
# Set the env var WALL_EVIDENCE_DEBUG_DIR=/path/to/dir to dump
# per-detector intermediates as PNGs when segment_colors() runs.

WALL_EVIDENCE_V_STRONG = 80
WALL_EVIDENCE_S_LOW = 40
WALL_EVIDENCE_OTSU_CAP = 160
WALL_EVIDENCE_DARK_FRAC_MIN = 0.02
WALL_EVIDENCE_EDGE_V_MAX = 140
WALL_EVIDENCE_EDGE_GRAD_MIN = 30.0
WALL_EVIDENCE_D1_WEIGHT = 1.0
WALL_EVIDENCE_D2_WEIGHT = 0.7
WALL_EVIDENCE_D3_WEIGHT = 0.4
WALL_EVIDENCE_PRE_BINARY_THR = 0.3   # threshold used only by the CC filter
WALL_EVIDENCE_CC_KEEP_AREA = 40      # area >= this → keep regardless of shape
WALL_EVIDENCE_CC_MIN_AREA = 4        # smaller than this → always drop
WALL_EVIDENCE_CC_MIN_ASPECT = 2.0    # below KEEP_AREA, require this aspect


def _wall_chromatic_mask(hsv: np.ndarray) -> np.ndarray:
    """Boolean (h, w) mask of pixels that are chromatic (door / window).

    Used to suppress wall evidence where the pixel is clearly a colored
    feature, since the anti-aliased dark fringe of a colored stroke would
    otherwise leak into the wall channel.
    """
    window = cv2.inRange(hsv, (90, 80, 80), (135, 255, 255))
    door = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
    return (window > 0) | (door > 0)


def compute_wall_evidence(bgr: np.ndarray,
                          debug: Dict[str, np.ndarray] | None = None
                          ) -> np.ndarray:
    """Return a continuous wall-evidence map in [0.0, 1.0] of shape (h, w).

    If ``debug`` is provided, intermediate per-detector layers are stashed
    in it for later inspection.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    chromatic = _wall_chromatic_mask(hsv)

    # D1: strong black.
    d1 = (v < WALL_EVIDENCE_V_STRONG).astype(np.float32) * WALL_EVIDENCE_D1_WEIGHT

    # D2: low-saturation dark with adaptive Otsu.
    d2 = np.zeros_like(d1)
    low_sat = (s < WALL_EVIDENCE_S_LOW)
    if low_sat.any():
        v_lowsat = v[low_sat]
        otsu_thr, _ = cv2.threshold(v_lowsat, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_frac = float((v_lowsat < otsu_thr).sum()) / float(v_lowsat.size)
        if (otsu_thr <= WALL_EVIDENCE_OTSU_CAP
                and dark_frac >= WALL_EVIDENCE_DARK_FRAC_MIN):
            d2 = (low_sat & (v < otsu_thr)).astype(np.float32) * WALL_EVIDENCE_D2_WEIGHT

    # D3: edge-supported dark stroke. Sobel magnitude on V channel.
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    d3 = (((v < WALL_EVIDENCE_EDGE_V_MAX)
           & (grad_mag > WALL_EVIDENCE_EDGE_GRAD_MIN))
          .astype(np.float32) * WALL_EVIDENCE_D3_WEIGHT)

    # Fuse via element-wise max.
    evidence = np.maximum.reduce([d1, d2, d3])
    evidence[chromatic] = 0.0

    # D4: connected-component shape filter. Drop blobs that are too small
    # to be wall (likely page noise) or small + non-elongated (likely text
    # characters / dots).
    binary_pre = (evidence > WALL_EVIDENCE_PRE_BINARY_THR).astype(np.uint8)
    if binary_pre.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_pre, connectivity=8)
        keep = np.ones(n, dtype=bool)
        keep[0] = True  # background label, value irrelevant
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if area >= WALL_EVIDENCE_CC_KEEP_AREA:
                keep[lab] = True
            elif (area >= WALL_EVIDENCE_CC_MIN_AREA
                  and aspect >= WALL_EVIDENCE_CC_MIN_ASPECT):
                keep[lab] = True
            else:
                keep[lab] = False
        if not keep.all():
            keep_mask = keep[labels]
            evidence = evidence * keep_mask.astype(np.float32)

    if debug is not None:
        debug["d1_strong_black"] = (d1 > 0).astype(np.uint8) * 255
        debug["d2_low_sat_dark"] = (d2 > 0).astype(np.uint8) * 255
        debug["d3_edge_supported"] = (d3 > 0).astype(np.uint8) * 255
        debug["chromatic"] = chromatic.astype(np.uint8) * 255
        debug["evidence"] = (np.clip(evidence, 0.0, 1.0) * 255).astype(np.uint8)
    return evidence


WALL_EVIDENCE_BINARY_THR = 0.5       # output binary at this evidence level


def evidence_to_binary(evidence: np.ndarray) -> np.ndarray:
    """Threshold a 0.0-1.0 evidence map to a 0/255 uint8 mask.

    The cutoff is fixed at 0.5 rather than computed via Otsu. With three
    discrete detector weights (D1=1.0, D2=0.7, D3=0.4):

      - 0.5 admits the same pixels as the old ``D1 | D2`` union: D1 and D2
        both sit above 0.5; D3 alone does not promote a pixel to wall.
      - D3 (edge-supported dark stroke) is still computed and contributes
        to the continuous evidence map, where step 4's candidate scoring
        can weigh it without binary commit.
      - Otsu was tried first and rejected: with three discrete weights and
        a near-empty background, Otsu's two-class split lands near 0.7 on
        clean images and silently drops D2 pixels — losing faint exterior
        walls the previous code caught. The earlier 0.3 cutoff was tried
        next but admitted D3, which fires on the dark anti-aliased fringe
        immediately outside chromatic strokes and tanked window IOU.
    """
    if evidence.size == 0 or evidence.max() <= 0.0:
        return np.zeros(evidence.shape, dtype=np.uint8)
    return ((evidence > WALL_EVIDENCE_BINARY_THR).astype(np.uint8)) * 255


def segment_colors(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Return {label: uint8 mask} for wall/window/door.

    Wall comes from the multi-detector evidence path; window / door are
    direct HSV inRange. Anti-aliased edge pixels are absorbed by a small
    morphological closing on each output mask.
    """
    debug_dir = os.environ.get("WALL_EVIDENCE_DEBUG_DIR")
    debug: Dict[str, np.ndarray] | None = {} if debug_dir else None

    evidence = compute_wall_evidence(bgr, debug=debug)
    wall = evidence_to_binary(evidence)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    window = cv2.inRange(hsv, (90, 80, 80), (135, 255, 255))
    door = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))

    # Belt-and-braces chromatic suppression after thresholding — closing
    # below could otherwise dilate wall into anti-aliased chromatic fringe.
    chromatic = cv2.bitwise_or(window, door)
    wall = cv2.bitwise_and(wall, cv2.bitwise_not(chromatic))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks = {}
    for name, m in (("wall", wall), ("window", window), ("door", door)):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
        masks[name] = m

    if debug is not None:
        os.makedirs(debug_dir, exist_ok=True)
        debug["wall_final"] = masks["wall"]
        debug["window"] = masks["window"]
        debug["door"] = masks["door"]
        for name, layer in debug.items():
            cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), layer)
    return masks


# ---------------------------------------------------------------------------
# Step 2: skeletonization
# ---------------------------------------------------------------------------

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """scikit-image skeletonize on a 0/255 mask, returns bool 2D array."""
    return skeletonize(mask > 0)


# ---------------------------------------------------------------------------
# Step 3: skeleton -> straight segments
# ---------------------------------------------------------------------------

# 8-connectivity neighbour offsets.
_NB = [(-1, -1), (-1, 0), (-1, 1),
       (0, -1),           (0, 1),
       (1, -1),  (1, 0),  (1, 1)]


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """For every skeleton pixel, count its 8-connected on-neighbours."""
    s = skel.astype(np.uint8)
    # Convolve with 3x3 ones, subtract self.
    kernel = np.ones((3, 3), dtype=np.uint8)
    cnt = cv2.filter2D(s, ddepth=cv2.CV_16S, kernel=kernel,
                       borderType=cv2.BORDER_CONSTANT)
    cnt = cnt - s  # remove the centre
    cnt[~skel] = 0
    return cnt.astype(np.int16)


def _trace_branch(skel: np.ndarray, visited_edges: set,
                  start: Tuple[int, int], first_step: Tuple[int, int],
                  is_node: np.ndarray) -> List[Tuple[int, int]]:
    """Walk along degree-2 pixels until we hit another node or endpoint.

    visited_edges holds frozenset({a, b}) of pixel pairs already consumed so
    we never traverse the same skeleton edge twice.
    """
    path = [start, first_step]
    visited_edges.add(frozenset({start, first_step}))
    prev, curr = start, first_step
    while not is_node[curr]:
        # curr is degree-2 (or 1 if endpoint, handled by while-cond fail above).
        next_px = None
        cy, cx = curr
        for dy, dx in _NB:
            ny, nx_ = cy + dy, cx + dx
            if (ny, nx_) == prev:
                continue
            if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1] and skel[ny, nx_]:
                edge_key = frozenset({curr, (ny, nx_)})
                if edge_key in visited_edges:
                    continue
                next_px = (ny, nx_)
                visited_edges.add(edge_key)
                break
        if next_px is None:
            break
        path.append(next_px)
        prev, curr = curr, next_px
    return path


def extract_branches(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Split the skeleton into ordered pixel paths between junctions/endpoints.

    A "node" pixel is one whose 8-neighbour count is != 2 (i.e. endpoints with
    1 neighbour, junctions with >=3). Paths between nodes are unambiguous.
    Pure cycles (no nodes) are also captured by seeding from any pixel.
    """
    deg = _neighbor_count(skel)
    is_node = np.zeros_like(skel, dtype=bool)
    is_node[skel & (deg != 2)] = True

    visited_edges: set = set()
    branches: List[List[Tuple[int, int]]] = []

    node_coords = np.argwhere(is_node)
    for ny, nx_ in node_coords:
        node = (int(ny), int(nx_))
        for dy, dx in _NB:
            yy, xx = node[0] + dy, node[1] + dx
            if 0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1] and skel[yy, xx]:
                edge_key = frozenset({node, (yy, xx)})
                if edge_key in visited_edges:
                    continue
                path = _trace_branch(skel, visited_edges, node, (yy, xx), is_node)
                if len(path) >= 2:
                    branches.append(path)

    # Catch isolated cycles (no node pixels).
    remaining = skel.copy()
    for path in branches:
        for y, x in path:
            remaining[y, x] = False
    while remaining.any():
        ys, xs = np.where(remaining)
        seed = (int(ys[0]), int(xs[0]))
        # Walk forward until we close back; treat the whole loop as one branch.
        path = [seed]
        remaining[seed] = False
        prev = None
        curr = seed
        while True:
            cy, cx = curr
            nxt = None
            for dy, dx in _NB:
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < remaining.shape[0] and 0 <= xx < remaining.shape[1] and remaining[yy, xx]:
                    nxt = (yy, xx)
                    break
            if nxt is None:
                break
            path.append(nxt)
            remaining[nxt] = False
            prev, curr = curr, nxt
        if len(path) >= 2:
            branches.append(path)
    return branches


def branches_to_segments(branches: List[List[Tuple[int, int]]]) -> List[Tuple[float, float, float, float]]:
    """For each branch, run approxPolyDP and emit straight (x1,y1,x2,y2) tuples."""
    segments: List[Tuple[float, float, float, float]] = []
    for path in branches:
        if len(path) < 2:
            continue
        # Cheap arc length filter to drop spurs.
        arc = 0.0
        for (y0, x0), (y1, x1) in zip(path, path[1:]):
            arc += float(np.hypot(y1 - y0, x1 - x0))
        if arc < MIN_BRANCH_LEN_PX:
            continue

        # approxPolyDP wants Nx1x2 int32 array of (x, y).
        contour = np.array([[x, y] for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(contour, epsilon=POLY_EPSILON_PX, closed=False)
        pts = approx.reshape(-1, 2)
        if len(pts) < 2:
            continue
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 == x1 and y0 == y1:
                continue
            segments.append((float(x0), float(y0), float(x1), float(y1)))
    return segments


# Step 3b alternate path (door/window via CC + minAreaRect instead of
# skeletonize) was prototyped but reverted in step 2 because the centerline
# drift it introduced cascaded into wrong wall topology. The dead code has
# been removed from this file; recover it from git history (commit fa63544)
# if a future step 5 candidate wants to retry the CC route.
# ---------------------------------------------------------------------------
# Step 4: NetworkX node snapping
# ---------------------------------------------------------------------------
#
# Legacy ``snap_endpoints`` was deleted in step 7 phase 6 (dead code after
# phase 5 migrated the call site to ``_accept_2d_cluster_candidates``).


# ---------------------------------------------------------------------------
# Step 4b: axis alignment (kill chamfered corners on orthogonal floorplans)
# ---------------------------------------------------------------------------

# Legacy ``axis_align_segments`` was deleted in step 9 phase 4 (dead code
# after phase 1 migrated its call site to ``_accept_axis_align_candidates``).


# Legacy ``snap_colinear_coords`` was deleted in step 7 phase 6 (dead code
# after phase 4 migrated its call site to ``_accept_fuse_candidates`` with
# ``masks=None`` and a fixed ``fallback_tol = colinear_tol``).


# ---------------------------------------------------------------------------
# Step 4c: merge collinear same-type segments
# ---------------------------------------------------------------------------

def _classify_axis(seg: Dict) -> str:
    """Return 'h' if exactly horizontal, 'v' if exactly vertical, else 'd'."""
    if seg["y1"] == seg["y2"] and seg["x1"] != seg["x2"]:
        return "h"
    if seg["x1"] == seg["x2"] and seg["y1"] != seg["y2"]:
        return "v"
    return "d"


# Legacy ``merge_collinear`` + ``_make_axis_seg`` + ``_length_weighted`` were
# deleted in step 8 phase 4 (dead code after phases 2/3 migrated both call
# sites to ``_accept_cluster_collinear_merge_candidates``). The cluster
# generator in ``generators.py`` reproduces the partition exactly, and the
# accompanying wrapper assembles output in legacy's iteration order so
# downstream order-sensitive passes (``t_junction_snap`` etc) see identical
# input.


# ---------------------------------------------------------------------------
# Step 4d: T-junction node-to-edge snap, and L-corner extend-to-intersect
# ---------------------------------------------------------------------------

# Legacy ``_point_on_axis_seg`` + ``t_junction_snap`` were deleted in step
# 10 phase 3 (dead code after phase 2 migrated the call site to
# ``_accept_t_junction_snap_candidates``; the helper is reproduced
# locally inside ``generators.t_junction_snap_candidates`` as
# ``_point_on_axis_seg_local``).








# Legacy ``truncate_overshoots`` was deleted in step 9 phase 4 (dead code
# after phase 2 migrated its call site to
# ``_accept_truncate_overshoot_candidates``).


# Legacy ``prune_tails`` was deleted alongside its step 4.8 call-site
# removal (see ``vectorize_bgr`` "prune_tails removed in step 4.8" note).


# ---------------------------------------------------------------------------
# Step 4f: STRICT MANHATTAN ROUTING (final pass before JSON)
#
# Contract of this stage: after it runs there are NO diagonals at all, every
# segment is exactly horizontal or exactly vertical, every L-corner is a
# mathematically perfect 90°, every T-junction sits exactly on its trunk's
# axis, and colinear segments of the same type are fused into the longest
# possible spans. No averaging of nearby coordinates — all snapping uses
# exact axis values from the participating segments.
# ---------------------------------------------------------------------------


# Legacy ``manhattan_force_axis`` was deleted in step 10 phase 3 (dead
# code after phase 1 migrated its call site to
# ``_accept_manhattan_force_axis_candidates``).


def _seg_axis_strict(seg: Dict) -> str:
    """After manhattan_force_axis, every segment is strictly h or v."""
    return "h" if seg["y1"] == seg["y2"] else "v"


# ---------------------------------------------------------------------------
# Step 4.9.4: thickness-aware snap tolerance
# ---------------------------------------------------------------------------
#
# A snap pulls one segment's endpoint onto another segment's line (or onto
# a shared corner). The geometrically meaningful tolerance for that pull
# is "how far into the host wall's body can the endpoint be drawn?" — i.e.
# proportional to the host wall's physical thickness. A 4 px thin partition
# should not attract an endpoint that is 12 px away (clearly belongs to a
# different wall); a 12 px thick load-bearing wall can legitimately pull a
# 4 px endpoint that happens to land near its edge.
#
# Per-segment tol = clamp(0.25 * local_thickness, 2, 6). Floor 2 px so
# sub-pixel drift always snaps; ceiling 6 px so genuinely separate walls
# can't be fused. ``compute_local_thickness`` samples the distance
# transform along a segment's body, taking the median of interior samples
# (>0.5 dt) so junction holes don't pull thick walls down. Door / window
# centerlines sit outside the wall mask so they sample as -1 and the
# caller falls back to ``fallback_tol`` (3.0).

def _thickness_aware_tol(thickness: float,
                         *,
                         abs_min: float = 2.0,
                         abs_max: float = 6.0,
                         thickness_frac: float = 0.25,
                         fallback: float = 3.0) -> float:
    """Step 4.9.4 per-segment snap tolerance from local wall thickness."""
    if thickness > 0:
        return max(abs_min, min(abs_max, thickness_frac * thickness))
    return fallback


def _compute_seg_tols(segments: List[Dict],
                      masks: Optional[Dict[str, np.ndarray]] = None,
                      *,
                      fallback: float = 3.0) -> List[float]:
    """Build a per-segment list of step-4.9.4 thickness-aware tols.

    Each segment is sampled against its OWN type's mask DT (a wall samples
    wall mask, a door samples door mask) — using the wrong-type mask would
    score zero interior and force every chromatic segment to the fallback.
    The per-type DT is computed once per call and cached locally; callers
    that hit multiple snap functions in sequence pay the cost three times,
    which is acceptable on the regression images (DT on 2048x2048 in ~5 ms).
    """
    if not segments:
        return []
    if masks is None:
        return [fallback] * len(segments)
    dts: Dict[str, np.ndarray] = {}
    for t, m in masks.items():
        if m is not None and m.size > 0:
            dts[t] = cv2.distanceTransform((m > 0).astype(np.uint8),
                                           cv2.DIST_L2, 3)
    out: List[float] = []
    for s in segments:
        dt = dts.get(s["type"])
        if dt is None:
            out.append(fallback)
            continue
        th = compute_local_thickness(s, dt)
        out.append(_thickness_aware_tol(th, fallback=fallback))
    return out


# Legacy ``manhattan_intersection_snap`` was deleted alongside its step
# 4.9.7 call-site removal (see ``vectorize_bgr`` step 4.9.7 note); pure
# NO-OP after canonical_line + thickness-aware manhattan_t_project picked
# up its work.


# Legacy ``manhattan_t_project`` was deleted in step 7 phase 6 (dead code
# after phase 2 migrated its call site to ``_accept_t_project_candidates``
# with thickness-aware masks).


# Legacy ``manhattan_ultimate_merge`` was deleted in step 8 phase 4 (dead
# code after step 6 phase 1 migrated its single call site to
# ``_accept_merge_candidates(perp_tol=0, gap_tol=0, junction_aware=True)``).


# ---------------------------------------------------------------------------
# Step 4g: WATERTIGHT CLOSURE
#
# After Manhattan routing, two failure modes still produce a non-watertight
# mesh: parallel duplicate centerlines (the skeleton produced two centerlines
# for one thick wall and they survived merging because their y/x didn't fall
# within the tight collinear tolerance) and tiny gaps where an endpoint
# stopped 1-2 px short of the perpendicular reference line.
#
# This stage runs strictly H/V geometry through three sequential passes:
#   1. Cluster and merge parallel duplicates per (type, axis).
#   2. Grid-snap every endpoint onto the nearest orthogonal reference line.
#   3. Force-close any L-corners whose H and V endpoints are mutually close
#      but whose lines don't cross (handled by re-running intersection snap).
# ---------------------------------------------------------------------------


# Legacy ``cluster_parallel_duplicates`` was deleted in step 8 phase 4 (dead
# code after step 6 phase 3 migrated its single call site to
# ``_accept_parallel_merge_candidates`` with ``skip_score=True``).


# Legacy ``grid_snap_endpoints`` was deleted in step 7 phase 6 (dead code
# after phase 3 migrated its call site to ``_accept_t_project_candidates``
# with ``masks=None`` and a fixed ``fallback_tol = grid_snap``).


# ---------------------------------------------------------------------------
# Step 4h: ULTIMATE GAP CLOSING (carpet-bombing pass for degree-1 endpoints)
#
# After watertight closure most joints already share an exact (x, y), but a
# small number of degree-1 endpoints can remain because they were just
# outside grid_snap's body-span check or just outside its tolerance. This
# pass uses an expanded 30 px sweep that ONLY considers loose ends, so we
# can be aggressive without corrupting already-closed geometry. The pass
# is also empowered to extend a trunk to swallow a projection that lies
# past its current body — the previous passes were not allowed to.
# ---------------------------------------------------------------------------


def _build_degree_map(segments: List[Dict],
                      quantize: float = 0.01) -> Tuple[Dict[Tuple[int, int], int],
                                                       Dict[Tuple[int, int], set]]:
    """Return (degree-by-quantized-point, types-by-quantized-point).

    ``quantize=0.01`` (the only value in use) keys at 0.01-px precision —
    delegated to ``geom_utils.endpoint_key(precision=2)``. Other quantize
    values are supported for back-compat but no current caller uses them.
    """
    from geom_utils import endpoint_key
    if quantize == 0.01:
        _key = lambda x, y: endpoint_key(x, y, precision=2)  # noqa: E731
    else:
        def _key(x: float, y: float) -> Tuple[int, int]:
            return (int(round(x / quantize)), int(round(y / quantize)))
    deg: Dict[Tuple[int, int], int] = defaultdict(int)
    types: Dict[Tuple[int, int], set] = defaultdict(set)
    for s in segments:
        for ex_key, ey_key in (("x1", "y1"), ("x2", "y2")):
            k = _key(s[ex_key], s[ey_key])
            deg[k] += 1
            types[k].add(s["type"])
    return deg, types


def _qkey(x: float, y: float, quantize: float = 0.01) -> Tuple[int, int]:
    """0.01-px endpoint key (delegates to ``geom_utils.endpoint_key`` for
    the default quantize=0.01 case; other values supported for back-compat)."""
    from geom_utils import endpoint_key
    if quantize == 0.01:
        return endpoint_key(x, y, precision=2)
    return (int(round(x / quantize)), int(round(y / quantize)))






def t_snap_with_extension(segments: List[Dict], tol: float,
                          masks: Dict[str, np.ndarray] = None,
                          min_support: float = 0.6,
                          *,
                          wall_evidence: np.ndarray = None,
                          door_mask: np.ndarray = None,
                          window_mask: np.ndarray = None,
                          audit_recorder=None) -> List[Dict]:
    """Candidate-based step-4 implementation.

    Same outer structure as the legacy (up to 6 sweep passes), but every
    accepted snap goes through Candidate generation + score evaluation:

      1. For each degree-1 endpoint of segment i / end e:
         - perpendicular distance to orthogonal trunk axis ≤ tol
         - projection on the trunk body, or past it by ≤ tol
         - wall-priority: walls never project onto chromatic, walls never
           get extended to accommodate chromatic
      2. If extending the trunk is needed, the trunk-extension stretch
         must have mask support ≥ min_support on the trunk's own type
         mask (cheap evidence gate).
      3. Each Candidate atomically: optional trunk-end mutate, loose-end
         mutate, opposite-end-of-loose-seg mutate (axis-keep).
      4. Score-and-accept: delta ≥ BRUTE_FORCE_MIN_ACCEPT_DELTA (matches
         the other deferred-payoff passes — the T-junction registers in
         a downstream pass).

    Returns a fresh list (legacy convention).
    """
    import candidates as C
    import scoring as S

    if not segments:
        return []

    if door_mask is None and masks is not None:
        door_mask = masks.get("door")
    if window_mask is None and masks is not None:
        window_mask = masks.get("window")

    segs = [dict(s) for s in segments]
    MAX_PASSES = 6

    for _pass in range(MAX_PASSES):
        deg, _ = _build_degree_map(segs)
        n = len(segs)
        axis = [_seg_axis_strict(s) for s in segs]

        cands: List[C.Candidate] = []
        for i in range(n):
            seg = segs[i]
            my_axis = axis[i]
            if my_axis not in ("h", "v"):
                continue
            my_prio = TYPE_PRIORITY.get(seg["type"], 99)
            for end in ("1", "2"):
                ex = float(seg[f"x{end}"])
                ey = float(seg[f"y{end}"])
                if deg[_qkey(ex, ey)] != 1:
                    continue

                best = None
                best_d = tol
                for j in range(n):
                    if i == j or axis[j] == my_axis or axis[j] not in ("h", "v"):
                        continue
                    trunk = segs[j]
                    trunk_prio = TYPE_PRIORITY.get(trunk["type"], 99)
                    if my_prio < trunk_prio:
                        continue
                    if axis[j] == "v":
                        line_x = float(trunk["x1"])
                        t_lo, t_hi = sorted((float(trunk["y1"]), float(trunk["y2"])))
                        dx = abs(ex - line_x)
                        if dx > tol:
                            continue
                        proj_y = ey
                        if t_lo <= proj_y <= t_hi:
                            need_extend = None
                        elif proj_y < t_lo and (t_lo - proj_y) <= tol:
                            need_extend = "lo"
                        elif proj_y > t_hi and (proj_y - t_hi) <= tol:
                            need_extend = "hi"
                        else:
                            continue
                        if need_extend is not None and trunk_prio < my_prio:
                            continue
                        d = dx
                        if d < best_d:
                            best_d = d
                            best = (j, line_x, proj_y, need_extend, "v")
                    else:
                        line_y = float(trunk["y1"])
                        t_lo, t_hi = sorted((float(trunk["x1"]), float(trunk["x2"])))
                        dy = abs(ey - line_y)
                        if dy > tol:
                            continue
                        proj_x = ex
                        if t_lo <= proj_x <= t_hi:
                            need_extend = None
                        elif proj_x < t_lo and (t_lo - proj_x) <= tol:
                            need_extend = "lo"
                        elif proj_x > t_hi and (proj_x - t_hi) <= tol:
                            need_extend = "hi"
                        else:
                            continue
                        if need_extend is not None and trunk_prio < my_prio:
                            continue
                        d = dy
                        if d < best_d:
                            best_d = d
                            best = (j, proj_x, line_y, need_extend, "h")

                if best is None:
                    continue

                trunk_idx, snap_x, snap_y, need_extend, trunk_axis = best
                trunk = segs[trunk_idx]

                # Evidence gate on trunk-extension stretch.
                if need_extend is not None and masks is not None:
                    trunk_mask = masks.get(trunk["type"])
                    if trunk_mask is not None:
                        if trunk_axis == "v":
                            old_lo, old_hi = sorted((float(trunk["y1"]),
                                                     float(trunk["y2"])))
                            if need_extend == "lo":
                                ext_lo, ext_hi = min(snap_y, old_lo), old_lo
                            else:
                                ext_lo, ext_hi = old_hi, max(snap_y, old_hi)
                            if ext_hi - ext_lo > 1.0:
                                support = _path_mask_support(
                                    trunk_mask, snap_x, ext_lo, snap_x, ext_hi)
                                if support < min_support:
                                    continue
                        else:
                            old_lo, old_hi = sorted((float(trunk["x1"]),
                                                     float(trunk["x2"])))
                            if need_extend == "lo":
                                ext_lo, ext_hi = min(snap_x, old_lo), old_lo
                            else:
                                ext_lo, ext_hi = old_hi, max(snap_x, old_hi)
                            if ext_hi - ext_lo > 1.0:
                                support = _path_mask_support(
                                    trunk_mask, ext_lo, snap_y, ext_hi, snap_y)
                                if support < min_support:
                                    continue

                # Build mutation list.
                mut: List[Tuple[int, str, float, float]] = []
                if need_extend is not None:
                    if trunk_axis == "v":
                        is_y1_lo = float(trunk["y1"]) <= float(trunk["y2"])
                        lo_end, hi_end = ("1", "2") if is_y1_lo else ("2", "1")
                        target_end = lo_end if need_extend == "lo" else hi_end
                        mut.append((trunk_idx, target_end,
                                    float(trunk[f"x{target_end}"]), snap_y))
                    else:
                        is_x1_lo = float(trunk["x1"]) <= float(trunk["x2"])
                        lo_end, hi_end = ("1", "2") if is_x1_lo else ("2", "1")
                        target_end = lo_end if need_extend == "lo" else hi_end
                        mut.append((trunk_idx, target_end,
                                    snap_x, float(trunk[f"y{target_end}"])))
                mut.append((i, end, snap_x, snap_y))
                other = "2" if end == "1" else "1"
                if my_axis == "h":
                    mut.append((i, other, float(seg[f"x{other}"]), snap_y))
                else:
                    mut.append((i, other, snap_x, float(seg[f"y{other}"])))

                cands.append(C.Candidate(
                    op="t_snap",
                    add=[],
                    mutate=mut,
                    meta={"d": best_d, "need_extend": need_extend,
                          "trunk_idx": trunk_idx},
                ))

        if not cands:
            return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]

        # No sort: candidates were generated in (i, end) order; legacy
        # processed loose endpoints in the same order, and matching that
        # iteration order keeps the legacy's tie-breaking on which
        # endpoint claims which trunk first.

        current = segs
        wall_mask_local = masks.get("wall") if masks is not None else None
        base_score = S.compute_score(current,
                                     wall_evidence=wall_evidence,
                                     door_mask=door_mask,
                                     window_mask=window_mask,
                                     wall_mask=wall_mask_local)
        used_mutations: Set[Tuple[int, str]] = set()
        any_accepted = False
        # Match legacy "trunk locked for rest of pass" semantics: once a
        # trunk has had any mutation accepted in this pass, neither of its
        # ends is open to further mutation until the next pass.
        locked_trunks: Set[int] = set()

        for cand in cands:
            # Skip if the candidate would touch a trunk already locked this
            # pass (legacy `done` set semantics).
            mut_idxs = {idx for (idx, _, _, _) in cand.mutate}
            if cand.meta["trunk_idx"] in locked_trunks:
                continue
            if any(li in locked_trunks for li in mut_idxs):
                continue
            if any((idx, end) in used_mutations
                   for (idx, end, _, _) in cand.mutate):
                continue
            trial = C.apply_candidate(current, cand)
            trial_score = S.compute_score(trial,
                                          wall_evidence=wall_evidence,
                                          door_mask=door_mask,
                                          window_mask=window_mask,
                                          wall_mask=wall_mask_local)
            delta = trial_score.total - base_score.total
            accept = delta > CANDIDATE_MIN_ACCEPT_DELTA
            if audit_recorder is not None:
                delta_terms = {k: trial_score.terms.get(k, 0.0) - base_score.terms.get(k, 0.0)
                               for k in set(trial_score.terms) | set(base_score.terms)}
                audit_recorder.record(
                    op=cand.op, accepted=accept, delta=delta,
                    delta_terms=delta_terms, meta=cand.meta,
                    reason="score_gate",
                    position=_audit_position(cand),
                )
            if accept:
                current = trial
                base_score = trial_score
                for (idx, end, _, _) in cand.mutate:
                    used_mutations.add((idx, end))
                # Legacy adds (trunk_idx, "1") and (trunk_idx, "2") to done
                # after EVERY accepted snap (regardless of extension), so
                # the trunk can't be re-snapped within the same pass. Mirror
                # that here.
                locked_trunks.add(cand.meta["trunk_idx"])
                any_accepted = True

        segs = current
        if not any_accepted:
            break

    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]




# ---------------------------------------------------------------------------
# Step 4i: BRUTE-FORCE RAY EXTENSION (final-final closure)
#
# Operates IN PLACE on the same list of dicts that will be serialized to
# JSON. No copies, no graph indirection — every coordinate change directly
# mutates the dict that ends up in the output.
#
# Algorithm per spec:
#   1. For every endpoint, decide if it's "loose" — no other endpoint
#      within RAY_EXT_LOOSE_PX (= 1 px).
#   2. For each loose endpoint:
#        - if its segment is horizontal, find the closest vertical segment
#          within RAY_EXT_TOL_PX (= 40 px) whose body the endpoint's y
#          falls within (with the same slack). Set the endpoint's x equal
#          to that vertical's x.
#        - symmetric for vertical endpoints.
#   3. After all extensions, fuse endpoints within RAY_EXT_FUSE_PX (= 2 px)
#      to a single canonical value (per axis) so micro-deltas vanish.
# ---------------------------------------------------------------------------




def _accept_bridge_candidates(lines: List[Dict],
                               *,
                               max_radius: float,
                               wall_mask: np.ndarray = None,
                               wall_evidence: np.ndarray = None,
                               door_mask: np.ndarray = None,
                               window_mask: np.ndarray = None,
                               audit_recorder=None,
                               ) -> List[Dict]:
    """Run ``generators.proximal_bridge_candidates`` and accept those whose
    pipeline-score delta is strictly positive.

    Returns the new segments list (the input is *not* mutated in place).

    Acceptance order: shorter bridges first, then higher-mask-support
    first. Greedy: once a pair of endpoints has been bridged, both source
    endpoints are locked from being re-used by another bridge in the
    same call. Subsequent candidates that reference a locked endpoint
    are skipped — the next outer call (if any) regenerates.
    """
    import candidates as C
    import generators as G
    import scoring as S

    if not lines:
        return list(lines)
    if wall_mask is None:
        return list(lines)

    # Use the binary wall mask as a float evidence map fallback when the
    # caller didn't supply a continuous one. Keeps wall_evidence_integral
    # interpretable as "fraction of pixels on mask".
    if wall_evidence is None:
        wall_evidence = (wall_mask > 0).astype(np.float32)

    cands = G.proximal_bridge_candidates(
        lines, wall_mask=wall_mask, max_radius=max_radius)
    if not cands:
        return list(lines)

    # Shorter bridges (less wall invented) first, then higher mask support.
    cands.sort(key=lambda c: (c.meta["total_len"], -c.meta["min_support_observed"]))

    current = list(lines)
    base_score = S.compute_score(current,
                                 wall_evidence=wall_evidence,
                                 door_mask=door_mask,
                                 window_mask=window_mask,
                                 wall_mask=wall_mask)
    used_endpoints: set = set()

    for cand in cands:
        ea, eb = cand.meta["pair_endpoints"]
        ka = (int(round(ea[0])), int(round(ea[1])))
        kb = (int(round(eb[0])), int(round(eb[1])))
        if ka in used_endpoints or kb in used_endpoints:
            if audit_recorder is not None:
                audit_recorder.record(
                    op=cand.op, accepted=False, delta=0.0,
                    meta=cand.meta, reason="used_endpoint",
                    position=_audit_position(cand),
                )
            continue
        trial = C.apply_candidate(current, cand)
        trial_score = S.compute_score(trial,
                                      wall_evidence=wall_evidence,
                                      door_mask=door_mask,
                                      window_mask=window_mask,
                                      wall_mask=wall_mask)
        delta = trial_score.total - base_score.total
        accept = delta > CANDIDATE_MIN_ACCEPT_DELTA
        if audit_recorder is not None:
            delta_terms = {k: trial_score.terms.get(k, 0.0) - base_score.terms.get(k, 0.0)
                           for k in set(trial_score.terms) | set(base_score.terms)}
            audit_recorder.record(
                op=cand.op, accepted=accept, delta=delta,
                delta_terms=delta_terms, meta=cand.meta,
                reason="score_gate",
                position=_audit_position(cand),
            )
        if accept:
            current = trial
            base_score = trial_score
            used_endpoints.add(ka)
            used_endpoints.add(kb)
    return current


# ---------------------------------------------------------------------------
# Score-gate policy: three classes
# ---------------------------------------------------------------------------
#
# Every candidate-based pass falls into one of three classes that determine
# its score-gate policy. Adopted post-step-15 to make the architecture
# seam (score is local-decision, pipeline is global-recovery; see
# step 15 diagnostic in todo.md) an explicit framework instead of an
# ad-hoc per-call decision.
#
#   Class A — pure simplification
#     The candidate folds redundancy without changing what the JSON
#     renders to (e.g. two collinear touching segments become one).
#     ``delta`` is typically 0 because the score's positive merge
#     signals (``duplicate``, ``free_endpoint``) target body-overlap
#     or loose-end merges and don't reward touching-only clean-up.
#     Policy: ``delta >= 0`` accept (tie = accept on score-neutral
#     simplification). Score still acts as safety net for any
#     candidate that *would* regress.
#     Examples: ``_accept_merge_candidates`` (touching-collinear),
#               ``_accept_cluster_collinear_merge_candidates``
#               (cluster-based merge_collinear).
#
#   Class B — topology-recovery primitive
#     The candidate's standalone effect can look like a regression
#     to local score (destroys a T-junction, or moves an endpoint
#     sub-pixel) because the pipeline relies on a downstream pass
#     to recover the lost structure. Step 15 quantified this for
#     parallel_merge (mean junction delta -1.15 to -1.28 across
#     ~100 rejected candidates) and confirmed via monkey-patch that
#     letting score gate these *does* drop wall IOU by 3-27%.
#     Policy: ``skip_score=True`` permanent. Geometric gate is the
#     entire safety net. The wrapper still computes & records score
#     for audit signal so future ranking work has the data.
#     Examples: ``_accept_parallel_merge_candidates``,
#               ``_accept_fuse_candidates``.
#
#   Class C — destructive mutation
#     A pass that can move a *real* T-junction node, or change
#     cross-type topology (e.g. attach a wall to a door body in
#     a way that would change how the door anchors). No such pass
#     currently exists; this slot is reserved so future generators
#     are evaluated against it explicitly.
#     Policy: ``skip_score=True`` NEVER. Must pass score gate.
#     If a new generator naturally lands in this class, score
#     must be made strict enough to gate it correctly.
#
# Migration rule: when adding a new candidate-based pass, classify it
# *before* picking a score policy. Don't reach for ``skip_score=True``
# because the regression looks easier with it -- check first whether
# the pass actually fits Class B (downstream recovery, not local
# correctness).


def _run_merge_loop(lines: List[Dict],
                    *,
                    regenerate,           # callable(segments) -> List[Candidate]
                    sort_key=None,        # callable(Candidate) -> sortable
                    skip_score: bool = False,  # Class B (topology-recovery): True. Class A (pure simplification): False with delta >= 0 in the merge wrappers below.
                    wall_evidence: np.ndarray = None,
                    door_mask: np.ndarray = None,
                    window_mask: np.ndarray = None,
                    wall_mask: np.ndarray = None,
                    audit_recorder=None,
                    ) -> List[Dict]:
    """Shared fixed-point loop for any merge-style candidate stream.

    Phase 1 (collinear, exact-line) and phase 3 (parallel, perp-tol) both
    use this. Single-accept-then-regenerate (per todo.md L54), ``delta >=
    0`` accept rule (merges are pure simplifications — see
    ``_accept_merge_candidates``'s docstring for the rationale).

    ``wall_mask`` enables the thick-wall-aware ``junction_count`` clustering
    in ``compute_score``: when passed, T-junction nodes within local wall
    half-thickness are counted as one physical junction. Without it, every
    degree-3+ node counts separately (legacy behaviour). Pass the wall
    mask when the upstream pipeline state can have skeleton-ridge ridges
    that this loop is allowed to collapse (parallel_merge case) so the
    score doesn't penalise the merge for "losing" a duplicate skeleton
    artefact junction.
    """
    import candidates as C
    import scoring as S

    if not lines:
        return list(lines)

    current = list(lines)
    base_score = S.compute_score(current,
                                 wall_evidence=wall_evidence,
                                 door_mask=door_mask,
                                 window_mask=window_mask,
                                 wall_mask=wall_mask)

    max_iter = 4 * len(lines) + 8
    for _iter in range(max_iter):
        cands = regenerate(current)
        if not cands:
            break
        if sort_key is not None:
            cands.sort(key=sort_key)

        accepted_this_iter = False
        for cand in cands:
            trial = C.apply_candidate(current, cand)
            trial_score = S.compute_score(trial,
                                          wall_evidence=wall_evidence,
                                          door_mask=door_mask,
                                          window_mask=window_mask,
                                          wall_mask=wall_mask)
            delta = trial_score.total - base_score.total
            accept = skip_score or delta >= CANDIDATE_MIN_ACCEPT_DELTA
            if audit_recorder is not None:
                delta_terms = {k: trial_score.terms.get(k, 0.0) - base_score.terms.get(k, 0.0)
                               for k in set(trial_score.terms) | set(base_score.terms)}
                audit_recorder.record(
                    op=cand.op,
                    accepted=accept,
                    delta=delta,
                    delta_terms=delta_terms,
                    meta=cand.meta,
                    reason="skip_score" if skip_score else "score_gate",
                    position=_audit_position(cand),
                )
            if accept:
                current = trial
                base_score = trial_score
                accepted_this_iter = True
                break
        if not accepted_this_iter:
            break
    return current


def _accept_merge_candidates(lines: List[Dict],
                             *,
                             perp_tol: float = 0.0,
                             gap_tol: float = 0.0,
                             junction_aware: bool = True,
                             wall_evidence: np.ndarray = None,
                             door_mask: np.ndarray = None,
                             window_mask: np.ndarray = None,
                             wall_mask: np.ndarray = None,
                             audit_recorder=None,
                             ) -> List[Dict]:
    """Run ``generators.collinear_merge_candidates`` to a fixed point.

    **Score-gate class: A (pure simplification).** See the framework
    block above ``_run_merge_loop``. Accepts on ``delta >= 0``;
    tie-accept is correct because:

      - Two collinear touching segments folding into one renders
        identically (the JSON change is pure simplification).
      - The score's positive merge signals — ``duplicate`` (only
        counts body-overlapping pairs, missing the touching case)
        and ``free_endpoint`` (only fires when loose ends merge,
        missing the interior-touch case) — produce ``delta = 0``
        on clean touching-merges. The strict ``> 0`` rule would
        reject these benign simplifications.
      - The generator's geometric gates already guarantee structural
        safety (same axis + junction-aware filter). Score acts as
        safety net only: a merge that would destroy a real
        T-junction shows ``delta < 0`` via the ``junction`` term
        and the gate catches it.

    Iteration: each pass regenerates candidates against current
    state, sorts by (smallest perp_dist, longest merge), and accepts
    the first that passes the gate. Terminates in 1-3 passes for
    ~150-segment inputs (linear in chain length).
    """
    import generators as G
    return _run_merge_loop(
        lines,
        regenerate=lambda segs: G.collinear_merge_candidates(
            segs,
            perp_tol=perp_tol,
            gap_tol=gap_tol,
            junction_aware=junction_aware,
        ),
        sort_key=lambda c: (c.meta["perp_dist"], -c.meta["merged_len"]),
        wall_evidence=wall_evidence,
        door_mask=door_mask,
        window_mask=window_mask,
        wall_mask=wall_mask,
        audit_recorder=audit_recorder,
    )


def _accept_parallel_merge_candidates(lines: List[Dict],
                                       *,
                                       perp_tol: float,
                                       touch_perp_tol: float = 12.0,
                                       min_overlap_ratio: float = 0.5,
                                       skip_score: bool = False,
                                       wall_evidence: np.ndarray = None,
                                       door_mask: np.ndarray = None,
                                       window_mask: np.ndarray = None,
                                       wall_mask: np.ndarray = None,
                                       audit_recorder=None,
                                       ) -> List[Dict]:
    """Step 6 phase 3: run ``generators.parallel_merge_candidates`` to a
    fixed point. Replaces ``cluster_parallel_duplicates``.

    **Score-gate class: B (topology-recovery primitive)** when the
    pipeline calls this with ``skip_score=True`` (the default
    production wiring; see vectorize_bgr). Step 15 diagnostic
    confirmed: ~99/115 source and ~138/150 sg2 parallel-merge
    candidates would be score-rejected with mean ``junction`` term
    -1.15 to -1.28; letting score gate them drops wall IOU 3-27%
    because the lost T-junctions are restored by downstream
    t_project / fuse. The geometric gates (same-type, same-axis,
    perp_tol, min_overlap_ratio) are the entire safety net.

    ``skip_score`` is a parameter (not hard-coded True) so the
    same wrapper can serve a future Class-A use that finds a
    score-gateable parallel-merge subset, without forking code.

    Iteration: single-accept-then-regenerate. Sort prefers Case-1
    (touching near-collinear) over Case-2 (thick-wall parallel
    duplicate), then tighter perp_dist, then longest merge — the
    cleanest / most-genuine merges land first.
    """
    import generators as G
    return _run_merge_loop(
        lines,
        regenerate=lambda segs: G.parallel_merge_candidates(
            segs,
            perp_tol=perp_tol,
            touch_perp_tol=touch_perp_tol,
            min_overlap_ratio=min_overlap_ratio,
        ),
        sort_key=lambda c: (c.meta["case"], c.meta["perp_dist"],
                            -c.meta["merged_len"]),
        skip_score=skip_score,
        wall_evidence=wall_evidence,
        door_mask=door_mask,
        window_mask=window_mask,
        wall_mask=wall_mask,
        audit_recorder=audit_recorder,
    )


def _accept_t_junction_snap_candidates(lines: List[Dict],
                                         *,
                                         tol: float,
                                         ) -> List[Dict]:
    """Step 10 phase 2: candidate-based ``t_junction_snap``.

    Same generator-internal cascade simulation as
    ``_accept_truncate_overshoot_candidates``: legacy mutates segs[i]
    in place and subsequent (i', end') iterations read mutated state
    when scanning trunks. The generator simulates that in a local copy
    and emits one candidate per mutated endpoint carrying the final
    coord; batch-applying those candidates to the original input
    reproduces legacy bit-identically.

    Closing zero-length filter mirrors legacy.
    """
    import candidates as C
    import generators as G
    if not lines:
        return list(lines)
    cands = G.t_junction_snap_candidates(lines, tol=tol)
    if not cands:
        return [s for s in lines
                if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    current = list(lines)
    for cand in cands:
        current = C.apply_candidate(current, cand)
    return [s for s in current
            if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


def _accept_manhattan_force_axis_candidates(lines: List[Dict]) -> List[Dict]:
    """Step 10 phase 1: candidate-based ``manhattan_force_axis``.

    Per-segment independent transformation. Same batch-apply pattern as
    ``_accept_axis_align_candidates``; preserves input order so the
    immediate downstream pass (``canonicalize_offsets``, which iterates
    segments to compute distance-transform thicknesses) sees the same
    sequence legacy produced.
    """
    import candidates as C
    import generators as G
    if not lines:
        return list(lines)
    cands = G.manhattan_force_axis_candidates(lines)
    if not cands:
        return [s for s in lines
                if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    remove_set: set = set()
    mutates_by_seg: Dict[int, List[Tuple[str, float, float]]] = {}
    for cand in cands:
        for ri in cand.remove:
            remove_set.add(ri)
        for (idx, end, x, y) in cand.mutate:
            mutates_by_seg.setdefault(idx, []).append((end, x, y))
    out: List[Dict] = []
    for i, seg in enumerate(lines):
        if i in remove_set:
            continue
        if i in mutates_by_seg:
            ns = dict(seg)
            for end, x, y in mutates_by_seg[i]:
                ns["x" + end] = float(x)
                ns["y" + end] = float(y)
            out.append(ns)
        else:
            out.append(seg)
    return out


def _accept_canonicalize_offset_candidates(
        lines: List[Dict],
        *,
        wall_mask: Optional[np.ndarray] = None,
        abs_min: float = 2.0,
        abs_max: float = 6.0,
        thickness_frac: float = 0.25,
        fallback_tol: float = 3.0,
        attach_thickness: bool = False,
        ) -> List[Dict]:
    """Step 9 phase 3: candidate-based ``canonicalize_offsets``.

    The wrapper handles two side-channels that the candidate API
    doesn't natively support:

      1. ``attach_thickness=True``: legacy attaches a ``local_thickness``
         field to *every* output segment, not only those in a cluster.
         apply_candidate's mutate writes existing fields; new fields
         must be added by the wrapper after candidate apply.

      2. Distance transform / per-segment thickness computation: legacy
         computes the DT once and samples it once per segment. The
         candidate generator needs the thicknesses to pick its adaptive
         tol, so the wrapper computes them up front and threads them
         through.

    No score gate, no fixed-point loop: clusters are disjoint by the
    legacy partition, batch apply is correct.
    """
    import candidates as C
    import generators as G
    from canonical_line import compute_local_thickness as _local_thickness

    if not lines:
        return list(lines)

    # Single distance-transform compute (matches legacy: one DT per call).
    if wall_mask is not None and wall_mask.size > 0:
        dt = cv2.distanceTransform((wall_mask > 0).astype(np.uint8),
                                   cv2.DIST_L2, 3)
    else:
        dt = None
    thicknesses = [_local_thickness(s, dt) for s in lines]

    cands = G.canonicalize_offset_candidates(
        lines,
        thicknesses=thicknesses,
        abs_min=abs_min, abs_max=abs_max,
        thickness_frac=thickness_frac, fallback_tol=fallback_tol,
    )

    current = list(lines)
    for cand in cands:
        current = C.apply_candidate(current, cand)

    if not attach_thickness:
        return current

    # Attach local_thickness field. Note: ``current`` segments may be
    # different dict instances than ``lines`` due to apply_candidate's
    # mutation-via-deepcopy, but their index alignment is preserved
    # (cluster candidates only mutate, never remove or add) so each
    # ``current[i]`` corresponds to ``lines[i]`` and therefore to
    # ``thicknesses[i]``.
    out: List[Dict] = []
    for i, seg in enumerate(current):
        ns = dict(seg) if "local_thickness" not in seg else seg
        ns["local_thickness"] = thicknesses[i]
        out.append(ns)
    return out


def _accept_chromatic_anchor_candidates(lines: List[Dict],
                                          *,
                                          wall_mask: np.ndarray = None,
                                          max_radius: float,
                                          min_support: float = 0.85,
                                          wall_evidence: np.ndarray = None,
                                          door_mask: np.ndarray = None,
                                          window_mask: np.ndarray = None,
                                          audit_recorder=None,
                                          ) -> List[Dict]:
    """Step 20: anchor floating chromatic (door/window) endpoints onto walls.

    Runs ``generators.chromatic_anchor_bridge_candidates`` and accepts
    each whose pipeline-score delta is strictly positive. The geometric
    gate (floating endpoint + wall_mask support along every new segment)
    is the first line of defence; the score gate is the second. Together
    they prevent the generator from inventing wall where the source
    image doesn't support one.

    Greedy accept order: shorter bridges first, then higher mask
    support, then the orientation preferred by the chromatic axis
    (vertical-first stub for an H chromatic, etc.). Once an endpoint
    has been anchored by one bridge, further candidates targeting the
    same endpoint are dropped — the endpoint is no longer floating.
    """
    import candidates as C
    import generators as G
    import scoring as S

    if not lines:
        return list(lines)
    if wall_mask is None:
        return list(lines)

    cands = G.chromatic_anchor_bridge_candidates(
        lines, wall_mask=wall_mask,
        max_radius=max_radius, min_support=min_support)
    if not cands:
        return list(lines)

    # Sort: shorter bridge first (less invented wall), then higher
    # support, then orientation preference (lower "prefer" wins; ties
    # broken by total_len).
    cands.sort(key=lambda c: (
        c.meta["prefer"],
        c.meta["total_len"],
        -c.meta["min_support_observed"],
    ))

    if wall_evidence is None:
        wall_evidence = (wall_mask > 0).astype(np.float32)

    current = list(lines)
    base_score = S.compute_score(current,
                                 wall_evidence=wall_evidence,
                                 door_mask=door_mask,
                                 window_mask=window_mask,
                                 wall_mask=wall_mask)
    consumed_chromatic_endpoints: Set[Tuple[int, str]] = set()

    for cand in cands:
        key = (cand.meta["chrom_seg_idx"], cand.meta["chrom_end"])
        if key in consumed_chromatic_endpoints:
            if audit_recorder is not None:
                audit_recorder.record(
                    op=cand.op, accepted=False, delta=0.0,
                    meta=cand.meta, reason="consumed_endpoint",
                    position=_audit_position(cand),
                )
            continue
        trial = C.apply_candidate(current, cand)
        trial_score = S.compute_score(trial,
                                      wall_evidence=wall_evidence,
                                      door_mask=door_mask,
                                      window_mask=window_mask,
                                      wall_mask=wall_mask)
        delta = trial_score.total - base_score.total
        accept = delta > CANDIDATE_MIN_ACCEPT_DELTA
        if audit_recorder is not None:
            delta_terms = {k: trial_score.terms.get(k, 0.0) - base_score.terms.get(k, 0.0)
                           for k in set(trial_score.terms) | set(base_score.terms)}
            audit_recorder.record(
                op=cand.op, accepted=accept, delta=delta,
                delta_terms=delta_terms, meta=cand.meta,
                reason="score_gate",
                position=_audit_position(cand),
            )
        if accept:
            current = trial
            base_score = trial_score
            consumed_chromatic_endpoints.add(key)
    return current


def _accept_trunk_split_candidates(lines: List[Dict]) -> List[Dict]:
    """Topology completion pass: split a host trunk at every interior
    point where a free endpoint sits exactly on its body.

    Fixed-point loop because multiple endpoints can target the same
    trunk -- after the first split the trunk's index changes, and any
    other endpoint still on the new sub-trunk's body will be picked up
    in the next regeneration. No score gate: the operation is pure
    topology (sub-trunks render identically to the original) and the
    geometric gate (degree=1 endpoint + perp_dist=0 + strictly interior)
    is the entire safety net.
    """
    import candidates as C
    import generators as G
    if not lines:
        return list(lines)
    current = list(lines)
    max_iter = 4 * len(current) + 8
    for _ in range(max_iter):
        cands = G.trunk_split_candidates(current)
        if not cands:
            break
        # Greedy: apply the first candidate, regenerate. Different
        # candidates can target the same trunk, so we cannot batch.
        current = C.apply_candidate(current, cands[0])
    return current


def _accept_truncate_overshoot_candidates(lines: List[Dict],
                                            *,
                                            tol: float,
                                            ) -> List[Dict]:
    """Step 9 phase 2: candidate-based ``truncate_overshoots``.

    One Candidate per endpoint that crosses an orthogonal trunk by < tol;
    candidate carries a single-coord mutate (the perpendicular coord onto
    the trunk's line). ``used_endpoints`` set prevents double-mutating
    the same endpoint -- matches legacy's iterative single-mutate-per-
    endpoint pattern (legacy *can* in theory mutate twice if a second
    trunk also satisfies the gate at the post-first-mutate coord, but
    that requires the new endpoint to still be in crossing relation
    with a different trunk's line, which is geometrically rare; the
    closest-trunk-wins gate captures the dominant case).

    Closing zero-length filter mirrors legacy's
    ``[s for s in segs if (s['x1'], s['y1']) != (s['x2'], s['y2'])]``.
    """
    import candidates as C
    import generators as G
    if not lines:
        return list(lines)
    cands = G.truncate_overshoot_candidates(lines, tol=tol)
    if not cands:
        return [s for s in lines
                if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # Generator already pre-computed the final post-multi-truncate
    # endpoint coord inside its (i, end) inner loop, so each candidate
    # is the complete final effect for that endpoint. Apply all in
    # batch (each candidate's mutate targets a different (i, end), so
    # no conflicts).
    current = list(lines)
    for cand in cands:
        current = C.apply_candidate(current, cand)
    return [s for s in current
            if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


def _accept_axis_align_candidates(lines: List[Dict],
                                   *,
                                   tol_deg: float,
                                   ) -> List[Dict]:
    """Step 9 phase 1: candidate-based ``axis_align_segments``.

    All candidates are independent per-segment mutations + zero-length
    prunes; no inter-candidate dependency, so we apply them in one batch
    while preserving legacy's input-order output (the segment list shape
    must match legacy's to keep downstream order-sensitive passes
    bit-identical -- ``snap_colinear`` / ``merge_collinear_1`` happen
    immediately after this).

    No score gate or loop: legacy is unconditional per-segment, and the
    geometric gate (angle within tol_deg of an axis) is reproduced
    exactly inside ``axis_align_candidates``.
    """
    import generators as G
    if not lines:
        return list(lines)
    cands = G.axis_align_candidates(lines, tol_deg=tol_deg)
    if not cands:
        # Still must filter zero-length even when no rotation is needed
        # (legacy's ``if dx == 0 and dy == 0: continue``). With no cands
        # the only filter needed is the same zero-length check.
        return [s for s in lines
                if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    remove_set: set = set()
    mutates_by_seg: Dict[int, List[Tuple[str, float, float]]] = {}
    for cand in cands:
        for ri in cand.remove:
            remove_set.add(ri)
        for (idx, end, x, y) in cand.mutate:
            mutates_by_seg.setdefault(idx, []).append((end, x, y))
    out: List[Dict] = []
    for i, seg in enumerate(lines):
        if i in remove_set:
            continue
        if i in mutates_by_seg:
            ns = dict(seg)
            for end, x, y in mutates_by_seg[i]:
                ns["x" + end] = float(x)
                ns["y" + end] = float(y)
            out.append(ns)
        else:
            out.append(seg)
    return out


def _accept_cluster_collinear_merge_candidates(
        lines: List[Dict],
        *,
        perp_tol: float,
        gap_tol: float,
        ) -> List[Dict]:
    """Step 8: candidate-based ``merge_collinear``.

    Single-pass apply (NOT the regenerate fixed-point loop used by other
    merge wrappers): the generator emits one candidate per along-group
    in the same bucket -> perp-cluster -> along-sweep iteration order
    legacy uses, including singletons. The output is assembled as

        diagonals (input order) + [c.add[0] for c in cands]

    which is exactly the order legacy's ``merge_collinear`` produces
    (``out = list(diagonals)`` then ``out.extend(_merge_group(...))``
    over each H type then each V type). Preserving that order is
    load-bearing -- downstream ``t_junction_snap`` and
    ``truncate_overshoots`` iterate segments and mutate ``segs[i]``
    in place; later iterations read the mutated state from earlier i,
    so segment-list ordering changes their output.

    The cluster generator is the right structure here (not the pair-
    based phase 1 generator): phase 1's transitive fixed-point uses
    union-length weighting after the first merge, which diverges sub-
    pixel from legacy's own-length weighting on 3+ chain merges. The
    cluster generator computes the canon_line from every member's own
    original length in one shot, matching legacy bit-identically.

    No score gate / accept loop: legacy's partition is the entire
    safety net (geometric gates: same-type + same-axis-bucket + perp
    cluster within perp_tol + along sweep within gap_tol).
    """
    import generators as G
    if not lines:
        return list(lines)
    cands = G.cluster_collinear_merge_candidates(
        lines, perp_tol=perp_tol, gap_tol=gap_tol)
    out: List[Dict] = [s for s in lines if _classify_axis(s) == "d"]
    for cand in cands:
        out.extend(dict(seg) for seg in cand.add)
    return out


def _accept_2d_cluster_candidates(lines: List[Dict],
                                   *,
                                   tol: float,
                                   ) -> List[Dict]:
    """Step 7 phase 5: candidate-based ``snap_endpoints``.

    Runs ``generators.endpoint_cluster_2d_candidates`` and applies each
    component-mutate in sequence. ``skip_score=True`` (legacy is
    unconditional; the safety net is the geometric gate -- 2D circular
    tol + wall-priority anchor -- reproduced exactly inside the
    generator).

    Final 4-decimal rounding on EVERY endpoint mirrors legacy, which
    rounds the canonical coord (even for singleton components) before
    writing the output segment. Without this post-step the cluster
    candidates would only round non-singleton members; the singletons
    would retain whatever sub-4-decimal precision earlier passes left
    behind, breaking bit-identical regression.

    Components are independent (union-find partitions are disjoint by
    construction), so the order of accept within a single generator
    pass doesn't change the final state. We don't re-generate after
    each accept -- legacy is single-pass and re-clustering would only
    create new components for endpoints that ALREADY snapped to a
    canonical, which legacy doesn't do.
    """
    import candidates as C
    import generators as G
    if not lines:
        return list(lines)

    cands = G.endpoint_cluster_2d_candidates(lines, tol=tol)
    current = list(lines)
    for cand in cands:
        current = C.apply_candidate(current, cand)

    # Bit-identical post-step: legacy ``snap_endpoints`` rounds EVERY
    # output endpoint to 4 decimals. Apply uniformly across the segment
    # list, including segments untouched by any candidate.
    rounded: List[Dict] = []
    for s in current:
        ns = dict(s)
        ns["x1"] = round(float(ns["x1"]), 4)
        ns["y1"] = round(float(ns["y1"]), 4)
        ns["x2"] = round(float(ns["x2"]), 4)
        ns["y2"] = round(float(ns["y2"]), 4)
        if (ns["x1"], ns["y1"]) == (ns["x2"], ns["y2"]):
            continue
        rounded.append(ns)
    return rounded


def _accept_t_project_candidates(lines: List[Dict],
                                  *,
                                  fallback_tol: float,
                                  masks: Optional[Dict[str, np.ndarray]] = None,
                                  ) -> List[Dict]:
    """Step 7 phase 2: candidate-based ``manhattan_t_project``.

    One Candidate per endpoint with a valid orthogonal-trunk projection;
    ``used_endpoints`` set prevents double-mutating the same endpoint, so
    each endpoint is projected at most once (matches legacy single-pass).

    No score gate. Legacy ``manhattan_t_project`` has no equivalent of
    score either — its safety net is the per-pair geometric gate (wall-
    priority + thickness-aware perpendicular tol + body-containment). The
    legacy gate is reproduced exactly inside ``t_project_candidates``;
    the loop here is structural plumbing only.

    Sort by projection distance: closer trunks first. Combined with the
    used_endpoints filter, this guarantees each endpoint picks the same
    trunk legacy would have (the smallest-distance candidate that's still
    available when it gets its turn). For endpoints that have *only one*
    valid trunk (the common case after canonical_line normalises lines),
    order doesn't matter at all.
    """
    import candidates as C
    import generators as G
    if not lines:
        return list(lines)

    seg_tols = _compute_seg_tols(lines, masks, fallback=fallback_tol) \
        if masks is not None else [fallback_tol] * len(lines)

    cands = G.t_project_candidates(lines, seg_tols=seg_tols)
    if not cands:
        return [s for s in lines
                if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    cands.sort(key=lambda c: (c.meta["projection_dist"],
                              c.meta["endpoint_key"]))

    current = list(lines)
    used_endpoints: set = set()
    for cand in cands:
        ek = cand.meta["endpoint_key"]
        if ek in used_endpoints:
            continue
        current = C.apply_candidate(current, cand)
        used_endpoints.add(ek)

    return [s for s in current
            if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


def _accept_fuse_candidates(lines: List[Dict],
                             *,
                             fallback_tol: float,
                             masks: Optional[Dict[str, np.ndarray]] = None,
                             wall_evidence: np.ndarray = None,
                             door_mask: np.ndarray = None,
                             window_mask: np.ndarray = None,
                             wall_mask: np.ndarray = None,
                             audit_recorder=None,
                             ) -> List[Dict]:
    """Step 7 phase 1: candidate-based ``fuse_close_endpoints``.

    **Score-gate class: B (topology-recovery primitive).** See the
    framework block above ``_run_merge_loop``. ``skip_score=True``
    hard-coded below for the same architectural reason as
    parallel_merge: sub-2-px endpoint mutations make integer-count
    score terms (``free_endpoint`` / ``junction`` / ``pseudo_junction``)
    jitter in ways that reject benign clusters (step 11 evaluation:
    enabling score gate drops source wall IOU 3%). The geometric gate
    (1D wall-priority cluster within tol, identical to legacy) is the
    entire safety net.

    Runs ``generators.endpoint_fuse_candidates`` to a fixed point.
    Each candidate is one 1D cluster's worth of mutates — the
    cluster definition (sort-and-walk on each of x and y
    independently with the pairwise min thickness-aware tol as join
    threshold, wall-priority mean as canonical) matches legacy
    ``fuse_close_endpoints`` exactly.

    ``seg_tols`` is computed once up front against the *input*
    segments, mirroring legacy. Fuse only mutates endpoint coords
    (never adds / removes segments), so the per-segment thickness
    sampled from the distance transform stays valid across
    iterations.
    """
    import generators as G
    if not lines:
        return list(lines)

    seg_tols_initial = _compute_seg_tols(lines, masks, fallback=fallback_tol) \
        if masks is not None else [fallback_tol] * len(lines)

    # Capture the initial tols so the regenerate lambda uses them across
    # iterations. Each iteration's regenerate must pass a tols list whose
    # indices align with the *current* segments list. Since fuse never
    # adds or removes segments, current and initial indices line up
    # by-position throughout the loop.
    return _run_merge_loop(
        lines,
        regenerate=lambda segs: G.endpoint_fuse_candidates(
            segs, seg_tols=seg_tols_initial),
        # Apply x-axis clusters before y, then larger clusters first
        # (more endpoints unified per accept → faster convergence).
        sort_key=lambda c: (0 if c.meta["axis_dim"] == "x" else 1,
                            -c.meta["cluster_size"],
                            -c.meta["moved_count"]),
        # ``skip_score=True`` (kept after step 11 evaluation) -- even with
        # thick-wall-aware junction_count, the score's other integer-count
        # terms (free_endpoint / pseudo_junction) move erratically on
        # sub-2-px endpoint mutations and reject benign fuse clusters
        # (source IOU 0.97 drop on regression). The geometric gate
        # (1D wall-priority cluster within tol) is identical to legacy
        # and is the entire safety net here; ``wall_mask`` still flows
        # in for any future score audit / training-data extraction.
        skip_score=True,
        wall_evidence=wall_evidence,
        door_mask=door_mask,
        window_mask=window_mask,
        wall_mask=wall_mask,
        audit_recorder=audit_recorder,
    )


def brute_force_ray_extend(lines: List[Dict],
                           tol: float,
                           loose_tol: float,
                           *,
                           wall_evidence: np.ndarray = None,
                           door_mask: np.ndarray = None,
                           window_mask: np.ndarray = None,
                           audit_recorder=None) -> None:
    """Candidate-based step-4 implementation of the brute-force ray pass.

    Each loose endpoint generates at most one Candidate: snap it onto the
    nearest orthogonal trunk's axis line within ``tol``, in the segment's
    free direction (so the snap never shrinks or inverts the host). The
    gates are purely geometric (axis match + free-direction check + tol);
    scoring then admits any candidate that does not regress total score.

    Same call signature as legacy, plus optional masks/evidence kwargs for
    scoring. Cascading is bounded to 4 passes, matching legacy behaviour.
    """
    import candidates as C
    import scoring as S

    if not lines:
        return

    def _is_loose(endpoints: List[Tuple[float, float]],
                  self_idx: int, ex: float, ey: float) -> bool:
        loose2 = loose_tol * loose_tol
        for k, (px, py) in enumerate(endpoints):
            if k == self_idx:
                continue
            if (px - ex) * (px - ex) + (py - ey) * (py - ey) <= loose2:
                return False
        return True

    for _pass in range(4):
        n = len(lines)
        if n < 2:
            return
        endpoints: List[Tuple[float, float]] = []
        for s in lines:
            endpoints.append((s["x1"], s["y1"]))
            endpoints.append((s["x2"], s["y2"]))
        axis = [_seg_axis_strict(s) for s in lines]

        cands: List[C.Candidate] = []
        for i in range(n):
            seg = lines[i]
            seg_ax = axis[i]
            if seg_ax not in ("h", "v"):
                continue
            for end_idx, end in enumerate(("1", "2")):
                slot = 2 * i + end_idx
                ex = float(seg[f"x{end}"])
                ey = float(seg[f"y{end}"])
                if not _is_loose(endpoints, slot, ex, ey):
                    continue
                other_end = "2" if end == "1" else "1"
                ox = float(seg[f"x{other_end}"])
                oy = float(seg[f"y{other_end}"])

                if seg_ax == "h":
                    free_dir = 1.0 if ex > ox else -1.0
                    best_d = tol
                    best_x = None
                    for j in range(n):
                        if i == j or axis[j] != "v":
                            continue
                        v = lines[j]
                        line_x = float(v["x1"])
                        if abs(line_x - ox) < 1e-6:
                            continue
                        if (line_x - ox) * free_dir <= 0:
                            continue
                        lo, hi = sorted((float(v["y1"]), float(v["y2"])))
                        if ey < lo - tol or ey > hi + tol:
                            continue
                        if ey < lo:
                            dy_c = lo - ey
                        elif ey > hi:
                            dy_c = ey - hi
                        else:
                            dy_c = 0.0
                        d = float(np.hypot(ex - line_x, dy_c))
                        if d < best_d:
                            best_d = d
                            best_x = line_x
                    if best_x is not None and best_x != ex:
                        cands.append(C.Candidate(
                            op="l_extend",
                            add=[],
                            mutate=[(i, end, best_x, ey)],
                            meta={"distance": best_d, "axis": "h_to_v"},
                        ))

                else:  # seg_ax == "v"
                    free_dir = 1.0 if ey > oy else -1.0
                    best_d = tol
                    best_y = None
                    for j in range(n):
                        if i == j or axis[j] != "h":
                            continue
                        h = lines[j]
                        line_y = float(h["y1"])
                        if abs(line_y - oy) < 1e-6:
                            continue
                        if (line_y - oy) * free_dir <= 0:
                            continue
                        lo, hi = sorted((float(h["x1"]), float(h["x2"])))
                        if ex < lo - tol or ex > hi + tol:
                            continue
                        if ex < lo:
                            dx_c = lo - ex
                        elif ex > hi:
                            dx_c = ex - hi
                        else:
                            dx_c = 0.0
                        d = float(np.hypot(dx_c, ey - line_y))
                        if d < best_d:
                            best_d = d
                            best_y = line_y
                    if best_y is not None and best_y != ey:
                        cands.append(C.Candidate(
                            op="l_extend",
                            add=[],
                            mutate=[(i, end, ex, best_y)],
                            meta={"distance": best_d, "axis": "v_to_h"},
                        ))

        if not cands:
            return

        cands.sort(key=lambda c: c.meta["distance"])

        current = list(lines)
        base_score = S.compute_score(current,
                                     wall_evidence=wall_evidence,
                                     door_mask=door_mask,
                                     window_mask=window_mask)
        used_mutations: Set[Tuple[int, str]] = set()
        any_accepted = False
        for cand in cands:
            if any((idx, end) in used_mutations
                   for (idx, end, _, _) in cand.mutate):
                if audit_recorder is not None:
                    audit_recorder.record(
                        op=cand.op, accepted=False, delta=0.0,
                        meta=cand.meta, reason="used_endpoint",
                        position=_audit_position(cand),
                    )
                continue
            trial = C.apply_candidate(current, cand)
            trial_score = S.compute_score(trial,
                                          wall_evidence=wall_evidence,
                                          door_mask=door_mask,
                                          window_mask=window_mask)
            delta = trial_score.total - base_score.total
            accept = delta > CANDIDATE_MIN_ACCEPT_DELTA
            if audit_recorder is not None:
                delta_terms = {k: trial_score.terms.get(k, 0.0) - base_score.terms.get(k, 0.0)
                               for k in set(trial_score.terms) | set(base_score.terms)}
                audit_recorder.record(
                    op=cand.op, accepted=accept, delta=delta,
                    delta_terms=delta_terms, meta=cand.meta,
                    reason="score_gate",
                    position=_audit_position(cand),
                )
            if accept:
                current = trial
                base_score = trial_score
                for (idx, end, _, _) in cand.mutate:
                    used_mutations.add((idx, end))
                any_accepted = True
        lines.clear()
        lines.extend(current)
        if not any_accepted:
            return






def _path_mask_support(mask: np.ndarray, x1: float, y1: float,
                       x2: float, y2: float, perp: int = 3) -> float:
    """Return the fraction of pixels along the (x1,y1)->(x2,y2) line whose
    `perp`-pixel-thick neighbourhood in `mask` is non-zero.

    Used to gate any pass that wants to *create or extend* geometry: if the
    proposed path has no support in the source mask, the change is rejected.
    A return value near 1.0 means the source has a continuous line there;
    near 0.0 means the path crosses white space.
    """
    if mask is None:
        return 1.0  # no mask available -> don't block (legacy callers)
    h, w = mask.shape[:2]
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1.0:
        return 1.0
    n = max(8, int(round(length)))
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    hit = 0
    for x, y in zip(xs, ys):
        ix = int(round(x))
        iy = int(round(y))
        x0 = max(0, ix - perp)
        x1c = min(w, ix + perp + 1)
        y0 = max(0, iy - perp)
        y1c = min(h, iy + perp + 1)
        if x1c <= x0 or y1c <= y0:
            continue
        if mask[y0:y1c, x0:x1c].any():
            hit += 1
    return hit / n




def insert_missing_connectors(lines: List[Dict],
                              colinear_tol: float,
                              max_len: float,
                              wall_mask: np.ndarray = None,
                              min_support: float = 0.6,
                              *,
                              wall_evidence: np.ndarray = None,
                              door_mask: np.ndarray = None,
                              window_mask: np.ndarray = None,
                              audit_recorder=None) -> None:
    """Candidate-based step-4 implementation of the gap-close pass.

    Generates one Candidate per (loose-endpoint, loose-endpoint) pair that
    passes the cheap gates (semantic = wall↔wall, evidence = mask support
    ≥ GAP_CONNECTOR_GATE_MIN). Sorts candidates by length ascending /
    support descending, then greedy-accepts those whose pipeline-score
    delta is positive. An endpoint that has already been used by an
    accepted candidate is not reused.

    Same call signature as the legacy implementation so the pipeline
    wiring in vectorize_bgr stays unchanged. ``min_support`` is kept as a
    parameter but no longer hard-rejects — it is now an audit floor that
    callers can tighten if they want a more conservative pass.
    """
    import candidates as C
    import scoring as S

    if not lines:
        return

    # Inventory all endpoints and their owners.
    from collections import Counter
    end_count: Counter = Counter()
    end_to_seg: Dict[Tuple[float, float], List[Tuple[int, str]]] = {}
    for i, s in enumerate(lines):
        for end in ("1", "2"):
            pt = (s["x" + end], s["y" + end])
            end_count[pt] += 1
            end_to_seg.setdefault(pt, []).append((i, end))

    # Loose wall endpoints only.
    loose_wall: List[Tuple[float, float]] = []
    for pt, cnt in end_count.items():
        if cnt != 1:
            continue
        sidx, _ = end_to_seg[pt][0]
        if lines[sidx].get("type") != "wall":
            continue
        loose_wall.append(pt)

    if len(loose_wall) < 2:
        return

    # Use the binary wall mask as a float evidence map when no continuous
    # one is supplied. Promotes ((mask>0)?1.0:0.0) so the score's
    # evidence_integral computes "fraction of pixels on the wall mask",
    # matching legacy semantics.
    if wall_evidence is None and wall_mask is not None:
        wall_evidence = (wall_mask > 0).astype(np.float32)

    gate_min = max(GAP_CONNECTOR_GATE_MIN, 0.0)

    # ---- Candidate generation -----------------------------------------
    cands: List[C.Candidate] = []
    for axis, fixed_pred in (
        ("v", lambda a, b: abs(a[0] - b[0])),  # shared x; perpendicular = y-gap
        ("h", lambda a, b: abs(a[1] - b[1])),  # shared y; perpendicular = x-gap
    ):
        for i in range(len(loose_wall)):
            xi, yi = loose_wall[i]
            for j in range(i + 1, len(loose_wall)):
                xj, yj = loose_wall[j]
                if axis == "v":
                    if abs(xi - xj) > colinear_tol:
                        continue
                    d = abs(yi - yj)
                    if not (1e-3 < d <= max_len):
                        continue
                    cx = 0.5 * (xi + xj)
                    ylo, yhi = sorted((yi, yj))
                    if wall_mask is not None:
                        support = C.mask_support_along(wall_mask, cx, ylo, cx, yhi)
                    else:
                        support = 1.0
                    if support < gate_min:
                        continue
                    new_seg = {"type": "wall", "x1": cx, "y1": ylo,
                               "x2": cx, "y2": yhi}
                    mut = []
                    for (idx, end) in end_to_seg[(xi, yi)]:
                        mut.append((idx, end, cx, yi))
                    for (idx, end) in end_to_seg[(xj, yj)]:
                        mut.append((idx, end, cx, yj))
                else:
                    if abs(yi - yj) > colinear_tol:
                        continue
                    d = abs(xi - xj)
                    if not (1e-3 < d <= max_len):
                        continue
                    cy = 0.5 * (yi + yj)
                    xlo, xhi = sorted((xi, xj))
                    if wall_mask is not None:
                        support = C.mask_support_along(wall_mask, xlo, cy, xhi, cy)
                    else:
                        support = 1.0
                    if support < gate_min:
                        continue
                    new_seg = {"type": "wall", "x1": xlo, "y1": cy,
                               "x2": xhi, "y2": cy}
                    mut = []
                    for (idx, end) in end_to_seg[(xi, yi)]:
                        mut.append((idx, end, xi, cy))
                    for (idx, end) in end_to_seg[(xj, yj)]:
                        mut.append((idx, end, xj, cy))

                cands.append(C.Candidate(
                    op="gap_close",
                    add=[new_seg],
                    mutate=mut,
                    meta={"length": d, "support": support, "axis": axis,
                          "endpoints": ((xi, yi), (xj, yj))},
                ))

    if not cands:
        return

    # Short, high-support repairs first — greedy commit order matters
    # because endpoints already consumed in one accepted candidate cannot
    # be re-used by another.
    cands.sort(key=lambda c: (c.meta["length"], -c.meta["support"]))

    # ---- Score-and-accept loop -----------------------------------------
    current = list(lines)
    base_score = S.compute_score(current,
                                 wall_evidence=wall_evidence,
                                 door_mask=door_mask,
                                 window_mask=window_mask,
                                 wall_mask=wall_mask)
    used_mutations: Set[Tuple[int, str]] = set()
    used_endpoints: Set[Tuple[float, float]] = set()

    for cand in cands:
        # Endpoint-level exclusion: each pair of loose endpoints can be
        # consumed by at most one accepted candidate.
        eps = cand.meta["endpoints"]
        if eps[0] in used_endpoints or eps[1] in used_endpoints:
            if audit_recorder is not None:
                audit_recorder.record(
                    op=cand.op, accepted=False, delta=0.0,
                    meta=cand.meta, reason="used_endpoint",
                    position=_audit_position(cand),
                )
            continue
        if any((idx, end) in used_mutations for (idx, end, _, _) in cand.mutate):
            if audit_recorder is not None:
                audit_recorder.record(
                    op=cand.op, accepted=False, delta=0.0,
                    meta=cand.meta, reason="used_mutation",
                    position=_audit_position(cand),
                )
            continue

        trial = C.apply_candidate(current, cand)
        trial_score = S.compute_score(trial,
                                      wall_evidence=wall_evidence,
                                      door_mask=door_mask,
                                      window_mask=window_mask,
                                      wall_mask=wall_mask)
        delta = trial_score.total - base_score.total
        accept = delta > CANDIDATE_MIN_ACCEPT_DELTA
        if audit_recorder is not None:
            delta_terms = {k: trial_score.terms.get(k, 0.0) - base_score.terms.get(k, 0.0)
                           for k in set(trial_score.terms) | set(base_score.terms)}
            audit_recorder.record(
                op=cand.op, accepted=accept, delta=delta,
                delta_terms=delta_terms, meta=cand.meta,
                reason="score_gate",
                position=_audit_position(cand),
            )
        if accept:
            current = trial
            base_score = trial_score
            for (idx, end, _, _) in cand.mutate:
                used_mutations.add((idx, end))
            used_endpoints.add(eps[0])
            used_endpoints.add(eps[1])

    # In-place replace so the caller's reference stays valid.
    lines.clear()
    lines.extend(current)


# Legacy ``fuse_close_endpoints`` was deleted in step 7 phase 6 (dead code
# after phase 1 migrated its call site to ``_accept_fuse_candidates`` with
# thickness-aware masks; the same generator covers ``snap_colinear_coords``
# via phase 4 with ``masks=None``).


# Legacy ``hard_axis_snap`` / ``graph_node_align`` /
# ``final_perpendicular_project`` were deleted as dead code: none had a
# call site in ``vectorize_bgr`` after the step 4.x candidate migration.


# ---------------------------------------------------------------------------
# Step 5: debug preview
# ---------------------------------------------------------------------------

def draw_debug_image(segments: List[Dict], shape: Tuple[int, int], path: str) -> None:
    """Render the vector output for visual diff against the source."""
    h, w = shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    color_map = {
        "wall":   (0, 0, 0),       # black
        "window": (255, 128, 0),   # blue (BGR)
        "door":   (0, 200, 255),   # yellow (BGR)
    }
    for seg in segments:
        c = color_map.get(seg["type"], (0, 0, 0))
        p1 = (int(round(seg["x1"])), int(round(seg["y1"])))
        p2 = (int(round(seg["x2"])), int(round(seg["y2"])))
        cv2.line(canvas, p1, p2, c, thickness=1, lineType=cv2.LINE_AA)
    # Mark all snapped node points (small dots) so closure can be eyeballed.
    nodes = set()
    for seg in segments:
        nodes.add((round(seg["x1"], 4), round(seg["y1"], 4)))
        nodes.add((round(seg["x2"], 4), round(seg["y2"], 4)))
    for x, y in nodes:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    cv2.imwrite(path, canvas)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_src_path(name: str) -> str:
    if os.path.isabs(name) and os.path.exists(name):
        return name
    candidate = os.path.join(SRC_DIR, name)
    if os.path.exists(candidate):
        return candidate
    if not os.path.splitext(name)[1]:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            alt = os.path.join(SRC_DIR, name + ext)
            if os.path.exists(alt):
                return alt
    return candidate


def vectorize_bgr(bgr: np.ndarray, *,
                  verbose: bool = False,
                  audit_path: Optional[str] = None) -> Dict:
    """Pure pipeline: BGR ndarray in, dict ``{"lines": [...], "stats": {...}}`` out.

    No filesystem I/O, no print spam (unless ``verbose=True``). This is the
    function the API calls — ``run_one`` is a thin I/O wrapper around it.

    ``audit_path``: if provided, create an :class:`audit.AuditRecorder`,
    thread it into every score-using wrapper, and dump the events as
    JSON to ``audit_path`` at the end. Default-off (None) so the pipeline
    is unchanged. Useful for training-data extraction for step 5 ranking
    model and for debugging which score terms drove which decisions.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image, got shape {bgr.shape}")

    audit_recorder = None
    if audit_path is not None:
        from audit import AuditRecorder
        audit_recorder = AuditRecorder()

    h, w = bgr.shape[:2]
    _log(f"Image size: {w} x {h} px")

    # Image-size-aware tolerance scale. Pegged to source.png (1200x895) so its
    # output is bit-identical to the tuned baseline; larger images get
    # proportionally larger geometric tolerances, since the gaps left by
    # skeletonization at thick walls scale with image size.
    scale = max(1.0, max(h, w) / REFERENCE_DIM_PX)
    _log(f"Tolerance scale: {scale:.3f}x")
    snap_tol = SNAP_TOLERANCE_PX * scale
    colinear_tol = COLINEAR_TOL_PX * scale
    merge_perp = MERGE_PERP_TOL_PX * scale
    merge_gap = MERGE_GAP_TOL_PX * scale
    t_snap = T_SNAP_TOL_PX * scale
    l_extend = L_EXTEND_TOL_PX * scale
    tail_prune = TAIL_PRUNE_LEN_PX * scale
    manhattan_tol = MANHATTAN_SNAP_TOL_PX * scale
    parallel_merge = PARALLEL_MERGE_TOL_PX * scale
    grid_snap = GRID_SNAP_TOL_PX * scale
    gap_close = GAP_CLOSE_TOL_PX * scale
    gap_final = GAP_FINAL_PRUNE_PX * scale
    ray_ext = RAY_EXT_TOL_PX * scale
    ray_fuse = RAY_EXT_FUSE_PX * scale
    colinear_loose = COLINEAR_LOOSE_TOL_PX * scale
    connector_max = CONNECTOR_MAX_LEN_PX * scale
    trunk_perp = TRUNK_EXTEND_PERP_PX * scale
    trunk_gap = TRUNK_EXTEND_GAP_PX * scale
    l_ext_asym = L_EXT_ASYM_PX * scale

    _log("Color segmentation...")
    masks = segment_colors(bgr)
    # Step 13: also compute the continuous wall-evidence map and thread it
    # to score-using wrappers. segment_colors internally calls
    # compute_wall_evidence then thresholds at 0.5 for the binary mask;
    # we recompute it here once to get the raw continuous map (millisecond
    # cost on the regression images). With continuous evidence flowing,
    # ``wall_evidence_integral`` and ``phantom_penalty`` see D3-only
    # edge-supported pixels at their true 0.4 weight instead of the
    # 0/1 binary collapse.
    wall_evidence_map = compute_wall_evidence(bgr)

    typed_segments: List[Dict] = []
    branch_stats = {}
    for label, mask in masks.items():
        # Walls, doors, windows all go through skeleton → branches →
        # approxPolyDP. The earlier step-2 detour that swapped door/window
        # to a CC + minAreaRect centerline saved no measurable time on
        # the three regression images and produced ~1 px centerline
        # drift relative to the medial axis. Downstream snap passes
        # cascaded that drift into the wall topology — observed as wrong
        # T-junctions and reshaped corners on sg2. ``door_window_to_segments``
        # is retained in this file as dead code for the candidate
        # architecture to reconsider later, but the pipeline no longer
        # calls it.
        _log(f"  Skeletonize ({label})...")
        skel = skeletonize_mask(mask)
        branches = extract_branches(skel)
        segs = branches_to_segments(branches)
        branch_stats[label] = (int(skel.sum()), len(branches), len(segs))
        for x1, y1, x2, y2 in segs:
            typed_segments.append({"type": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    _log("Geometric optimization (may take a while on large images)...")
    # --- Geometric optimization pipeline --------------------------------
    # (1) Force orthogonality (segments within ±5° of an axis become exactly H/V).
    # Step 9 phase 1: candidate-based axis-snap. Same per-segment angle
    # gate as legacy; per-seg mutates are independent so we apply in batch
    # while preserving input order for the downstream order-sensitive
    # passes (snap_colinear / merge_collin_1 / t_junction_snap).
    s1 = _accept_axis_align_candidates(typed_segments, tol_deg=AXIS_SNAP_DEG)
    # Tight wall-anchored coordinate cluster so colinear walls share an x or y,
    # without dragging genuinely-distinct walls together.
    # Step 7 phase 4: candidate-based wrapper around the same 1D cluster
    # logic. ``snap_colinear_coords`` and ``fuse_close_endpoints`` share
    # the exact 1D-cluster-per-axis algorithm; the only difference is tol
    # source (fixed colinear_tol vs thickness-aware). Passing
    # ``masks=None`` plus a uniform ``fallback_tol = colinear_tol``
    # reproduces snap_colinear_coords exactly via the phase-1 generator.
    # Despite running pre-canonical (where step 6 phase 2 found the
    # candidate / score pattern fragile for *merge*), the fuse generator
    # uses ``skip_score=True`` -- only the geometric gate matters, which
    # is identical between the legacy 1D cluster and the candidate-based
    # one.
    s1 = _accept_fuse_candidates(s1, fallback_tol=colinear_tol, masks=None,
                                  audit_recorder=audit_recorder)

    # (2) Merge collinear same-type segments into a single longest span.
    # Step 8 phase 2: candidate-based cluster_collinear_merge replaces
    # the legacy ``merge_collinear``. The pair-based phase-1 generator
    # diverges sub-pixel from legacy on 3+ chain merges (union-length vs
    # own-length weighting); the cluster-based generator partitions
    # identically to legacy (bucket -> perp-cluster -> along-sweep) and
    # emits one Candidate per along-group, so the union of accepted
    # candidates' output equals legacy's output bit-identically.
    s2 = _accept_cluster_collinear_merge_candidates(
        s1, perp_tol=merge_perp, gap_tol=merge_gap)

    # (3a) T-junction node-to-edge snap: chromatic endpoints onto wall bodies,
    #      same-type endpoints onto own bodies. Walls never snap to chromatic.
    # Step 10 phase 2: candidate-based wrapper. Generator simulates
    # legacy's order-dependent in-place mutation cascade in a local
    # copy and emits one Candidate per snapped endpoint with the
    # cascade's *final* coord; batch apply reproduces legacy bit-
    # identically. Full-pipeline load-bearing (min IOU 0.81 ablation).
    s3 = _accept_t_junction_snap_candidates(s2, tol=t_snap)
    # (3b) extend_to_intersect (legacy L-corner extend pass) removed in
    #      step 4.8: post-junction-aware-merge ablation showed IOU=0.9967
    #      and dFree=1 — the same corner-closure work is now done by the
    #      proximal_bridge generator's L-bridge case, which the scorer
    #      accepts when mask support is present.
    # (3c) Truncate any remaining overshoots: an endpoint sitting on the far
    #      side of a perpendicular wall by less than tol gets clipped back.
    # Step 9 phase 2: candidate-based wrapper. Same crossing-gate +
    # single-axis mutate semantics; ``used_endpoints`` set enforces one
    # mutation per endpoint (matches legacy's dominant single-truncate
    # behaviour). Case-specific NO-OP on source, load-bearing on sg2.
    s3 = _accept_truncate_overshoot_candidates(s3, tol=l_extend)
    # Step 16: second collinear-merge call removed (case-specific patch).
    # Ablation: source NO-OP, sg2 IOU=0.97 dN=0, Gemini IOU=0.95 dN+1.
    # The first merge_collin already collapses obvious chains; the final
    # ``_accept_merge_candidates`` at end of pipeline catches any new
    # post-bridge merges. This intermediate "re-merge after snap" is the
    # classic "fix-A-breaks-B" patch the refactor is meant to eliminate.

    # Wall-priority NetworkX node merge.
    # Step 7 phase 5: candidate-based wrapper around the 2D circular-tol
    # cluster. Union-find replaces the inline NetworkX dependency; the
    # cluster semantic (top-priority anchor mean, 4-decimal round on
    # every output endpoint) matches legacy ``snap_endpoints`` exactly.
    s4 = _accept_2d_cluster_candidates(s3, tol=snap_tol)

    # prune_tails removed in step 4.8: try removal and revert if FAIL.
    s5 = s4

    # --- STRICT MANHATTAN ROUTING (zero diagonals) ----------------------
    # (a) Force every segment to be exactly horizontal or exactly vertical.
    # Step 10 phase 1: candidate-based wrapper. Per-segment forced
    # axis classification (|dx| >= |dy| -> H else V) + perpendicular
    # collapse to midpoint. No tol, no gate; full-pipeline load-bearing
    # on every reference image. The wrapper batches independent
    # per-seg mutates and preserves input order.
    s6 = _accept_manhattan_force_axis_candidates(s5)
    # (a.5) Step 4.9: canonicalise parallel offsets. Per (type, axis)
    #       bucket, cluster segment offsets within an adaptive
    #       thickness-aware tolerance (2-6 px) and pull each member's
    #       perp coordinate to the cluster's length-weighted mean. Unlike
    #       ``cluster_parallel_duplicates`` downstream, this does not
    #       require body overlap — its purpose is to collapse the 1-3 px
    #       y / x drift that skeletonisation introduces between collinear
    #       pieces of one wall. Running it before T / L snap means those
    #       passes see a single canonical line per logical wall, instead
    #       of N nominally-distinct lines all within 3 px of each other.
    # Step 9 phase 3: candidate-based wrapper around the same offset-
    # cluster + length-weighted-median algorithm. Per (type, axis)
    # bucket -> adaptive thickness-aware tol -> 1D cluster -> mutate
    # all member perp coords to canonical. ``attach_thickness`` is
    # handled by the wrapper as a post-step (apply_candidate doesn't
    # add new fields).
    s6 = _accept_canonicalize_offset_candidates(
        s6, wall_mask=masks.get("wall"), attach_thickness=True)
    # (b) Intersection-based L-corner snap removed in step 4.9.7 (post-
    #     ablation cleanup): with canonicalize_offsets above and thickness-
    #     aware tols in manhattan_t_project below, ablation on source/sg2
    #     shows ``manhattan_intersection_snap`` as a pure NO-OP (dN=0,
    #     IOU=1.000 bit-identical on both reference images; Gemini also
    #     bit-identical). Its work is fully covered by the canonical-line
    #     + T-project + intersection-on-fuse path now.
    # (c) T-junction projection onto orthogonal trunks. Step 4.9.4: tol
    #     scales with the trunk's local thickness, fallback ``manhattan_tol``.
    # Step 7 phase 2: candidate-based wrapper around the same projection
    # logic. One Candidate per endpoint with a valid trunk match; the
    # ``used_endpoints`` filter mirrors legacy single-pass semantics
    # (each endpoint projected at most once). No score gate — legacy
    # didn't have one either; the per-pair geometric gate (wall-priority
    # + thickness-aware tol + body-containment) is reproduced exactly
    # inside ``t_project_candidates``.
    s6 = _accept_t_project_candidates(s6, fallback_tol=manhattan_tol,
                                      masks=masks)
    # (d) Ultimate collinear merge: post-step-4.7 ablation confirmed this
    #     first manhattan_ultimate_merge call is NO-OP after the upstream
    #     passes had their inputs unchanged by step 4 / 4.6 / 4.7 — the
    #     real merge work happens at lines below (post-watertight and
    #     post-gap-closing). Call removed.

    # --- WATERTIGHT CLOSURE (kill duplicates + close gaps) --------------
    # Step 6 phase 3: candidate-based parallel merge replaces
    # ``cluster_parallel_duplicates``. Same tols (perp_tol = 13 px *
    # scale; inner touch_perp_tol = 12 px hardcoded; min_overlap_ratio
    # = 0.5). Pair-based with fixed-point loop handles the legacy's
    # graph-component transitivity.
    #
    # ``skip_score=True`` (kept after step 11 evaluation) — the
    # ``junction`` score term was originally the documented blocker;
    # step 11 fixed ``junction_count`` to take a ``wall_mask`` and
    # cluster T-nodes within local half-thickness so the two ridges'
    # apparent T-junctions count as one physical junction (merge
    # becomes score-neutral on that term). However step 11's
    # regression test showed that *other* score terms (likely
    # ``duplicate``/``free_endpoint`` interactions with how the
    # parallel-merge candidates are sequenced) still push delta < 0
    # for some merges legacy accepted. The geometric gates
    # (perp_tol + min_overlap_ratio + same-type + same-axis) are
    # already strict; skip_score=True keeps that trust. We still
    # pass ``wall_mask`` so the (otherwise-unused) score computation
    # inside the loop is thick-wall-aware -- helpful for audit logs
    # and future ranking-model training data.
    s7 = _accept_parallel_merge_candidates(
        s6,
        perp_tol=parallel_merge,
        skip_score=True,
        wall_evidence=wall_evidence_map,
        door_mask=masks.get("door"),
        window_mask=masks.get("window"),
        wall_mask=masks.get("wall"),
        audit_recorder=audit_recorder,
    )
    # Step 7 phase 3: candidate-based wrapper. ``grid_snap_endpoints`` is
    # semantically identical to ``manhattan_t_project`` -- same wall-
    # priority gate, same body-containment, same perpendicular-distance
    # rule, same single-axis mutate. The only difference is tol source:
    # ``grid_snap_endpoints`` uses a fixed tol, ``manhattan_t_project``
    # uses thickness-aware. Passing ``masks=None`` makes the shared
    # ``_accept_t_project_candidates`` use the fixed ``fallback_tol``
    # everywhere, reproducing grid_snap behaviour exactly. Two legacy
    # passes now share one candidate generator.
    s7 = _accept_t_project_candidates(s7, fallback_tol=grid_snap, masks=None)
    # manhattan_ultimate_merge (post-watertight) removed step 4.8 cleanup:
    # latest ablation shows it as NO-OP across all 3 images now that the
    # final merge at the end of the pipeline is junction-aware.

    # --- ULTIMATE GAP CLOSING (degree-1 carpet bombing) -----------------
    # (1) force_close_free_l_corners (force-close pairs of free L-corners
    #     within gap_close px to their exact intersection). Removed during
    #     step 4.8 NO-OP cleanup: post-junction-aware-merge ablation
    #     showed max|dN|=1, max|dFree|=1, IOU=1.0000 on all 3 images, and
    #     the regression remained PASS bit-identical after removal. Earlier
    #     Gemini-regression observation that gated this against deletion
    #     no longer applies — the proximal_bridge_generator + junction-
    #     aware merge now closes the same L-corners directly.
    # (2) T-snap with trunk auto-extension: a loose endpoint that projects
    #     past a trunk's body within tol triggers an extension of the trunk.
    #     The masks gate prevents extending through white space.
    s8 = t_snap_with_extension(s7, gap_close, masks=masks,
                                audit_recorder=audit_recorder)
    # manhattan_ultimate_merge (post-gap-closing) removed step 4.8: NO-OP
    # in latest ablation.

    snapped = s8

    # --- BRUTE-FORCE RAY EXTENSION (final-final closure, IN PLACE) ------
    # Operates directly on the same dict objects that get serialized to JSON.
    # No copies, no graph indirection — every mutation lands in the output.
    brute_force_ray_extend(snapped, ray_ext, RAY_EXT_LOOSE_PX,
                            audit_recorder=audit_recorder)
    # Defensive filter: ray extension may have collapsed a short segment to
    # a single point (both endpoints coincide). These zero-length segments
    # corrupt the loose-endpoint detection downstream because they
    # contribute two endpoints at the same coordinate, falsely raising the
    # degree of that point.
    snapped = [s for s in snapped if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # extend_trunk_to_loose (trunk auto-extension toward a loose endpoint)
    # used to run here. Post-step-4.7 ablation showed it became NO-OP on
    # all 3 regression images: the same trunk-extension work is now
    # handled atomically by proximal_bridge_generator (its L-bridge case
    # emits a 2-seg bridge whose downstream junction-aware merge folds
    # into the host trunk, producing the same extended-trunk result that
    # this pass used to perform via mutate). Call removed.
    # mask_gated_l_extend (asymmetric L-corner closure via endpoint
    # mutation) used to run here. Subsumed by ``proximal_bridge_generator``
    # below: the bridge generator proposes an *add-L-bridge* candidate for
    # the same pair of endpoints, and the now junction-aware
    # ``manhattan_ultimate_merge`` folds the new bridge into the existing
    # H / V trunks, producing the same rendered pixels as the legacy
    # mutate. Function body retained as TODO dead code for future
    # candidate-architecture cross-checks.
    # Insert missing connectors: pairs of loose endpoints that share an x
    # (or y) but were never linked by a real segment get a synthetic wall
    # bridging them. Gated on the wall mask. Kept *despite* the bridge
    # generator above also proposing axis-bridges; the legacy pass also
    # collapses near-parallel V (or H) walls into a single bridge via a
    # silent "snap x → diagonal → drop in merge" side-effect, which the
    # generator deliberately does not replicate (the generator only
    # mutates endpoints along axis-preserving directions). The merge-
    # parallel behaviour belongs in a future generator analogous to
    # cluster_parallel_duplicates; until then both passes coexist.
    insert_missing_connectors(snapped, colinear_loose,
                              connector_max,
                              wall_mask=masks.get("wall"),
                              audit_recorder=audit_recorder)
    snapped = [s for s in snapped if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # Proximal bridge generator: for any pair of wall endpoints within
    # ``l_ext_asym`` × scale of each other that aren't already coincident,
    # propose either a 1-seg axis-bridge (when endpoints share a near-axis)
    # or a 2-seg L-bridge (when they're offset on both axes). Subsumes
    # ``insert_missing_connectors``' loose-endpoint axis-bridge case and
    # extends it to the degree-2-corner L-bridge case that no earlier
    # pass addressed. Score-and-accept gates ensure no candidate is taken
    # that worsens total geometry quality.
    snapped = _accept_bridge_candidates(
        snapped,
        max_radius=l_ext_asym,
        wall_mask=masks.get("wall"),
        wall_evidence=wall_evidence_map,  # step 13: continuous evidence
        door_mask=masks.get("door"),
        window_mask=masks.get("window"),
        audit_recorder=audit_recorder,
    )
    snapped = [s for s in snapped if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # Final 2-px endpoint fusion: snap any near-coincident endpoints to one
    # canonical (wall-priority) coordinate so micro-deltas vanish.
    # Step 4.9.4: per-endpoint tol is thickness-aware; ``ray_fuse`` is the
    # fallback only (and is below the 2 px floor anyway, so in practice
    # every endpoint resolves via clamp(0.25*thickness, 2, 6)).
    # Step 7 phase 1: candidate-based fixed-point fuse loop replaces the
    # in-place ``fuse_close_endpoints``. Same 1D cluster semantic
    # (independent x and y, wall-priority mean canonical, pairwise-min
    # thickness-aware join), but each cluster is now a Candidate so the
    # shared accept-loop / generator infrastructure picks it up. ``skip_
    # score=True`` because legacy fuse has no score gate at all.
    snapped = _accept_fuse_candidates(
        snapped,
        fallback_tol=ray_fuse,
        masks=masks,
        wall_evidence=wall_evidence_map,  # step 13: continuous evidence
        door_mask=masks.get("door"),
        window_mask=masks.get("window"),
        audit_recorder=audit_recorder,
    )
    snapped = [s for s in snapped if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # Step 6 phase 1: one last collinear merge in case ray extension /
    # bridge generator produced overlapping spans that can now coalesce.
    # Implemented as a candidate-based fixed-point loop with
    # ``perp_tol=0`` + ``gap_tol=0`` + junction-aware filter, matching
    # the legacy ``manhattan_ultimate_merge`` semantics exactly. The
    # ``delta >= 0`` accept rule (vs the strict ``> 0`` used by every
    # other candidate pass) is needed because clean touching-merges
    # produce delta=0 — the score's only positive merge signals are
    # body-overlap-aware (``duplicate`` term, only counts overlapping
    # bodies) and free-endpoint-aware (``free_endpoint``, only fires on
    # loose-end merges); neither rewards "two touching pieces become
    # one" cleanly. The generator's gates already enforce structural
    # safety, so accepting ties is correct here.
    snapped = _accept_merge_candidates(
        snapped,
        perp_tol=0.0,
        gap_tol=0.0,
        junction_aware=True,
        # NOTE: keeps wall_evidence=None (binary fallback) on purpose --
        # this wrapper uses the strict ``delta >= 0`` rule for touching-
        # collinear merges, and continuous evidence introduces sub-ULP
        # numerical drift in ``wall_evidence_integral`` that flips some
        # zero-delta merges to slightly-negative, rejecting clean JSON
        # simplifications (verified: with continuous evidence source goes
        # 102 -> 113 segs and sg2 124 -> 127, all IOUs 1.0000 -- rendering
        # bit-identical but baseline hash diverges). Other score-using
        # wrappers above keep continuous evidence (they use the strict
        # ``delta > 0`` rule, no zero-delta ambiguity).
        wall_evidence=None,
        door_mask=masks.get("door"),
        window_mask=masks.get("window"),
        wall_mask=masks.get("wall"),  # step 11: thick-wall-aware junction
        audit_recorder=audit_recorder,
    )

    # Step 17: topology completion. After all geometric passes converge,
    # a free endpoint can still sit *exactly* on another segment's body
    # interior -- the geometry is a T-junction but the segment list
    # doesn't represent it as one (host trunk is one continuous span).
    # Chain analysis (audit_view.py chain) confirmed source 8/12 and sg2
    # 22/24 free endpoints in this state, all at d=0.00. Splitting the
    # host trunk at the free endpoint's coord introduces no new geometry
    # (sub-trunks render identically to the original) but completes the
    # graph topology so node degree counts reflect reality.
    snapped = _accept_trunk_split_candidates(snapped)

    # Step 20: anchor any floating chromatic endpoints. Most images
    # finish with floating_openings == 0 because step 17 trunk_split
    # already handled the "endpoint sits on wall body interior" case.
    # This pass picks up the remainder: a door/window endpoint near
    # but not on a wall corner (Gemini #2, #3) gets an L-shaped wall
    # stub inserted that runs from the chromatic endpoint to the wall
    # corner. Both wall_mask support and score-gate must accept; if
    # neither does, the floating endpoint stays floating (the
    # invariant layer will report it).
    snapped = _accept_chromatic_anchor_candidates(
        snapped,
        wall_mask=masks.get("wall"),
        max_radius=ray_ext,          # ~40 px at REFERENCE_DIM
        wall_evidence=wall_evidence_map,
        door_mask=masks.get("door"),
        window_mask=masks.get("window"),
        audit_recorder=audit_recorder,
    )

    # Strip pipeline-internal fields (e.g. ``local_thickness`` from
    # canonical_line) so the public JSON payload stays canonical
    # ``{type, x1, y1, x2, y2}``. Most are already dropped by passes that
    # rebuild dicts from scratch (cluster_parallel_duplicates,
    # manhattan_ultimate_merge), but anything that survived in-place
    # mutation gets cleaned up here.
    for s in snapped:
        for k in _INTERNAL_SEG_FIELDS:
            s.pop(k, None)

    from collections import Counter
    nodes: Counter = Counter()
    for s in snapped:
        nodes[(s["x1"], s["y1"])] += 1
        nodes[(s["x2"], s["y2"])] += 1
    deg_hist = Counter(nodes.values())
    n_diag = sum(1 for s in snapped if _classify_axis(s) == "d")

    if verbose:
        _log("=== summary ===")
        for label, (npix, nbranch, nseg) in branch_stats.items():
            _log(f"  {label:7s}  skeleton_px={npix:6d}  branches={nbranch:4d}  raw_segs={nseg:4d}")
        _log(f"  raw segments        : {len(typed_segments)}")
        _log(f"  after orthogonalize : {len(s1)}")
        _log(f"  after merge_collin  : {len(s2)}")
        _log(f"  after T/L snap+merge: {len(s3)}")
        _log(f"  after endpoint snap : {len(s4)}")
        _log(f"  after tail prune    : {len(s5)}")
        _log(f"  after manhattan     : {len(s6)}")
        _log(f"  after watertight    : {len(s7)}")
        _log(f"  after gap-closing   : {len(s8)}")
        _log(f"  after ray-extension : {len(snapped)}")
        _log(f"  diagonal seg count  : {n_diag}  (must be 0)")
        _log(f"  endpoint degree hist: {dict(sorted(deg_hist.items()))}")
        _log(f"  free endpoints (d=1): {deg_hist.get(1, 0)}  (lower = more watertight)")

    if audit_recorder is not None and audit_path is not None:
        audit_recorder.dump_json(audit_path)
        if verbose:
            summary = audit_recorder.summary()
            _log(f"  audit events        : {summary['total']} -> {audit_path}")
            for op, counts in sorted(summary["by_op"].items()):
                _log(f"    {op:24s} accepted={counts['accepted']:4d}  rejected={counts['rejected']:4d}")

    return {
        "lines": snapped,
        "image_size": {"width": int(w), "height": int(h)},
        "stats": {
            "scale": scale,
            "segment_count": len(snapped),
            "diagonal_count": n_diag,
            "free_endpoints": deg_hist.get(1, 0),
            "endpoint_degree_histogram": dict(sorted(deg_hist.items())),
        },
    }


def run_one(src_path: str) -> None:
    """CLI wrapper: read image from disk, vectorize, write JSON + preview PNG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_json = os.path.join(OUT_DIR, f"{stem}.json")
    out_debug = os.path.join(OUT_DIR, f"{stem}_preview.png")

    print(f"Loading: {src_path}", flush=True)
    bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read source image: {src_path}")

    result = vectorize_bgr(bgr, verbose=True)
    snapped = result["lines"]

    print("Writing output...", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"lines": snapped}, f, ensure_ascii=False, indent=2)
    draw_debug_image(snapped, bgr.shape, out_debug)
    print(f"  -> {out_json}", flush=True)
    print(f"  -> {out_debug}", flush=True)
    print("Done.", flush=True)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = [DEFAULT_SRC_NAME]
    for name in args:
        src_path = _resolve_src_path(name)
        print(f"\n========== {name} ==========", flush=True)
        run_one(src_path)


if __name__ == "__main__":
    main()
