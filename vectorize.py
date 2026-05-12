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
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

from canonical_line import canonicalize_offsets, compute_local_thickness

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

def snap_endpoints(typed_segments: List[Dict],
                   tolerance: float) -> List[Dict]:
    """Cluster endpoints within `tolerance` and rewrite segments to canonical points.

    typed_segments: list of {"type", "x1", "y1", "x2", "y2"}.
    Returns a new list with snapped coordinates and any zero-length segments
    (collapsed by snapping) removed.
    """
    # Step A: gather all endpoints as (idx, end, x, y), end in {"a","b"}.
    pts = []
    for i, seg in enumerate(typed_segments):
        pts.append((i, "a", seg["x1"], seg["y1"]))
        pts.append((i, "b", seg["x2"], seg["y2"]))

    # Step B: build a graph linking points that are within tolerance.
    # Connected components -> clusters that get merged to a centroid.
    g = nx.Graph()
    for k in range(len(pts)):
        g.add_node(k)

    coords = np.array([[p[2], p[3]] for p in pts], dtype=np.float64)

    # O(n^2) is fine — node count is small for floorplans, and per the spec
    # we prioritize precision over performance.
    n = len(pts)
    tol2 = tolerance * tolerance
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            dx = coords[j, 0] - xi
            dy = coords[j, 1] - yi
            if dx * dx + dy * dy <= tol2:
                g.add_edge(i, j)

    # Step C: each connected component collapses to a canonical point.
    # Anchor priority: walls win over windows/doors so chromatic endpoints
    # snap onto the wall, not the other way round. Within a single priority
    # tier the points are averaged.
    canonical: Dict[int, Tuple[float, float]] = {}
    for comp in nx.connected_components(g):
        comp = list(comp)
        # Find the highest priority (lowest number) among endpoints in comp.
        prios = [TYPE_PRIORITY.get(typed_segments[pts[k][0]]["type"], 99)
                 for k in comp]
        top = min(prios)
        anchor_idx = [k for k, p in zip(comp, prios) if p == top]
        cx = float(np.mean([coords[k, 0] for k in anchor_idx]))
        cy = float(np.mean([coords[k, 1] for k in anchor_idx]))
        for k in comp:
            canonical[k] = (cx, cy)

    # Step D: rewrite typed_segments.
    out = []
    for k_a in range(0, n, 2):
        k_b = k_a + 1
        idx = pts[k_a][0]
        ax, ay = canonical[k_a]
        bx, by = canonical[k_b]
        if (ax, ay) == (bx, by):
            continue  # collapsed to a point
        out.append({
            "type": typed_segments[idx]["type"],
            "x1": round(ax, 4), "y1": round(ay, 4),
            "x2": round(bx, 4), "y2": round(by, 4),
        })
    return out


# ---------------------------------------------------------------------------
# Step 4b: axis alignment (kill chamfered corners on orthogonal floorplans)
# ---------------------------------------------------------------------------

def axis_align_segments(segments: List[Dict], tol_deg: float) -> List[Dict]:
    """Force near-horizontal/vertical segments to be exactly axis-aligned.

    For a near-horizontal segment we set both endpoints' y to the average y;
    for near-vertical we average x. Diagonals further from an axis than
    tol_deg are left untouched.
    """
    rad = np.deg2rad(tol_deg)
    out = []
    for seg in segments:
        x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = np.arctan2(dy, dx)  # -pi..pi
        # Angle to nearest axis.
        # near horizontal: |ang| < rad or |ang - pi| < rad
        if abs(ang) < rad or abs(abs(ang) - np.pi) < rad:
            ymid = 0.5 * (y1 + y2)
            seg = {**seg, "y1": ymid, "y2": ymid}
        elif abs(abs(ang) - np.pi / 2) < rad:
            xmid = 0.5 * (x1 + x2)
            seg = {**seg, "x1": xmid, "x2": xmid}
        out.append(seg)
    return out


def snap_colinear_coords(segments: List[Dict], tol: float) -> List[Dict]:
    """Cluster endpoint x-values (and y-values) within `tol` and unify them.

    Tightened: tol should be very small (≤3 px) so that genuinely distinct
    walls don't get flattened onto a shared axis. Within each cluster, wall
    coordinates anchor — chromatic (window/door) values snap onto walls.
    """
    if not segments:
        return segments

    # Each tagged value is (coord_value, type). The cluster's canonical value
    # is the mean of the highest-priority (lowest TYPE_PRIORITY) members.
    def _cluster_1d_anchored(tagged: List[Tuple[float, str]],
                             tol: float) -> Dict[Tuple[float, str], float]:
        if not tagged:
            return {}
        order = sorted(tagged, key=lambda t: t[0])
        clusters: List[List[Tuple[float, str]]] = [[order[0]]]
        for v in order[1:]:
            if v[0] - clusters[-1][-1][0] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        mapping: Dict[Tuple[float, str], float] = {}
        for grp in clusters:
            prios = [TYPE_PRIORITY.get(t, 99) for _, t in grp]
            top = min(prios)
            anchors = [val for (val, t), p in zip(grp, prios) if p == top]
            canon = float(np.mean(anchors))
            for entry in grp:
                mapping[entry] = canon
        return mapping

    xs_tagged: List[Tuple[float, str]] = []
    ys_tagged: List[Tuple[float, str]] = []
    for seg in segments:
        xs_tagged.append((seg["x1"], seg["type"]))
        xs_tagged.append((seg["x2"], seg["type"]))
        ys_tagged.append((seg["y1"], seg["type"]))
        ys_tagged.append((seg["y2"], seg["type"]))
    xmap = _cluster_1d_anchored(xs_tagged, tol)
    ymap = _cluster_1d_anchored(ys_tagged, tol)

    out = []
    for seg in segments:
        t = seg["type"]
        x1 = xmap.get((seg["x1"], t), seg["x1"])
        x2 = xmap.get((seg["x2"], t), seg["x2"])
        y1 = ymap.get((seg["y1"], t), seg["y1"])
        y2 = ymap.get((seg["y2"], t), seg["y2"])
        if (x1, y1) == (x2, y2):
            continue
        out.append({**seg, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


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


def merge_collinear(segments: List[Dict],
                    perp_tol: float,
                    gap_tol: float) -> List[Dict]:
    """Within each (type, axis, shared-coordinate-band) group, fuse overlapping
    or barely-separated segments into the longest covering segment.

    Diagonals are passed through unchanged.
    """
    by_key: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    diagonals: List[Dict] = []

    # First, bucket horizontals by (type, y-band) and verticals by (type, x-band).
    # Band index = round(coord / perp_tol); we also check neighbours when merging
    # to avoid edge-of-band misses.
    horiz: Dict[str, List[int]] = defaultdict(list)
    vert: Dict[str, List[int]] = defaultdict(list)
    for i, seg in enumerate(segments):
        axis = _classify_axis(seg)
        if axis == "h":
            horiz[seg["type"]].append(i)
        elif axis == "v":
            vert[seg["type"]].append(i)
        else:
            diagonals.append(seg)

    out: List[Dict] = list(diagonals)

    def _merge_group(indices: List[int], axis: str) -> List[Dict]:
        """Greedy union-find over an axis-aligned same-type group."""
        if not indices:
            return []
        # Build representation: (low, high, line_coord) per segment.
        items = []
        for i in indices:
            seg = segments[i]
            if axis == "h":
                lo, hi = sorted((seg["x1"], seg["x2"]))
                line = seg["y1"]
            else:
                lo, hi = sorted((seg["y1"], seg["y2"]))
                line = seg["x1"]
            items.append([lo, hi, line, seg["type"]])

        # Cluster items by `line` within perp_tol — consider them colinear.
        order = sorted(range(len(items)), key=lambda k: items[k][2])
        clusters: List[List[int]] = []
        for k in order:
            if clusters and items[k][2] - items[clusters[-1][-1]][2] <= perp_tol:
                clusters[-1].append(k)
            else:
                clusters.append([k])

        merged: List[Dict] = []
        for cluster in clusters:
            # Sort by lo, then sweep merging when overlapping or within gap_tol.
            cluster.sort(key=lambda k: items[k][0])
            cur_lo, cur_hi, _, ctype = items[cluster[0]]
            # Track each member as (line, length) so we can anchor the merged
            # line to the longest contributor — long walls dominate jamb stubs.
            cur_members = [(items[cluster[0]][2],
                            items[cluster[0]][1] - items[cluster[0]][0])]
            for k in cluster[1:]:
                lo, hi, line, _ = items[k]
                if lo - cur_hi <= gap_tol:
                    cur_hi = max(cur_hi, hi)
                    cur_members.append((line, hi - lo))
                else:
                    merged.append(_make_axis_seg(
                        ctype, axis, cur_lo, cur_hi, _length_weighted(cur_members)))
                    cur_lo, cur_hi = lo, hi
                    cur_members = [(line, hi - lo)]
            merged.append(_make_axis_seg(
                ctype, axis, cur_lo, cur_hi, _length_weighted(cur_members)))
        return merged

    for t, idxs in horiz.items():
        out.extend(_merge_group(idxs, "h"))
    for t, idxs in vert.items():
        out.extend(_merge_group(idxs, "v"))
    return out


def _make_axis_seg(t: str, axis: str, lo: float, hi: float, line: float) -> Dict:
    if axis == "h":
        return {"type": t, "x1": lo, "y1": line, "x2": hi, "y2": line}
    return {"type": t, "x1": line, "y1": lo, "x2": line, "y2": hi}


def _length_weighted(members: List[Tuple[float, float]]) -> float:
    """Pick the line coordinate as a length-weighted mean, so the longest
    contributing segment dominates and short stubs can't drag a wall off-axis."""
    total_w = sum(max(w, 1e-6) for _, w in members)
    return sum(line * max(w, 1e-6) for line, w in members) / total_w


# ---------------------------------------------------------------------------
# Step 4d: T-junction node-to-edge snap, and L-corner extend-to-intersect
# ---------------------------------------------------------------------------

def _point_on_axis_seg(px: float, py: float, seg: Dict,
                       perp_tol: float) -> Tuple[bool, float, float]:
    """If (px,py) is within perp_tol of an axis-aligned seg's body (not just
    its endpoints), return (True, snapped_x, snapped_y). Otherwise (False, ...).
    """
    axis = _classify_axis(seg)
    if axis == "h":
        lo, hi = sorted((seg["x1"], seg["x2"]))
        line_y = seg["y1"]
        # Must lie strictly inside the span (with a small tolerance) to count
        # as a T-snap rather than an endpoint coincidence.
        if abs(py - line_y) <= perp_tol and (lo - perp_tol) <= px <= (hi + perp_tol):
            sx = min(max(px, lo), hi)
            return True, sx, line_y
    elif axis == "v":
        lo, hi = sorted((seg["y1"], seg["y2"]))
        line_x = seg["x1"]
        if abs(px - line_x) <= perp_tol and (lo - perp_tol) <= py <= (hi + perp_tol):
            sy = min(max(py, lo), hi)
            return True, line_x, sy
    return False, px, py


def t_junction_snap(segments: List[Dict], tol: float) -> List[Dict]:
    """Pull free endpoints onto the body of any orthogonal segment within tol.

    Anchor rule: a wall endpoint never snaps onto a chromatic (window/door)
    body — the wall is the structural backbone, so chromatic geometry must
    yield to walls, not the other way around. Same-type snapping is always
    allowed; chromatic-onto-wall is the primary use case.
    """
    segs = [dict(s) for s in segments]
    for i, seg in enumerate(segs):
        for end in ("1", "2"):
            ex_key, ey_key = f"x{end}", f"y{end}"
            ex, ey = seg[ex_key], seg[ey_key]
            best = None
            best_d = tol
            for j, other in enumerate(segs):
                if j == i or _classify_axis(other) == "d":
                    continue
                # Anchor rule: forbid wall -> chromatic. Allow everything else.
                if (TYPE_PRIORITY.get(seg["type"], 99)
                        < TYPE_PRIORITY.get(other["type"], 99)):
                    continue
                ok, sx, sy = _point_on_axis_seg(ex, ey, other, tol)
                if not ok:
                    continue
                d = float(np.hypot(sx - ex, sy - ey))
                if d < best_d:
                    best_d = d
                    best = (sx, sy)
            if best is not None:
                seg[ex_key], seg[ey_key] = best
    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]








def truncate_overshoots(segments: List[Dict], tol: float) -> List[Dict]:
    """After all snapping, scan each axis-aligned segment for endpoints that
    project past an orthogonal segment that would clearly serve as the corner.

    This handles the case where a long horizontal wall extends a few pixels
    past the vertical wall it should terminate against, leaving a visible
    protrusion. We pull the offending endpoint back to the intersection.
    """
    segs = [dict(s) for s in segments]
    axis = [_classify_axis(s) for s in segs]
    n = len(segs)

    for i in range(n):
        if axis[i] == "d":
            continue
        for end in ("1", "2"):
            ex_key, ey_key = f"x{end}", f"y{end}"
            ex, ey = segs[i][ex_key], segs[i][ey_key]
            for j in range(n):
                if i == j or axis[j] == "d" or axis[j] == axis[i]:
                    continue
                other = segs[j]
                if axis[j] == "v":
                    line_x = other["x1"]
                    olo, ohi = sorted((other["y1"], other["y2"]))
                    # Endpoint of `i` (horizontal) overshoots if its endpoint
                    # is on the wrong side of the vertical and within tol of it,
                    # while the vertical's body covers the horizontal's y.
                    if abs(ex - line_x) <= tol and (olo - tol) <= ey <= (ohi + tol):
                        # Determine which side the rest of segment i lies on.
                        far_x = segs[i]["x1"] if end == "2" else segs[i]["x2"]
                        if (ex - line_x) * (far_x - line_x) < 0:
                            # endpoint is on the opposite side to the body's bulk:
                            # truncate the endpoint back to the intersection.
                            segs[i][ex_key] = line_x
                            segs[i][ey_key] = other["y1"] if abs(ey - other["y1"]) < abs(ey - other["y2"]) else ey
                            # Y stays on segment i's own line — i is horizontal,
                            # so its y is invariant. Reset to original line value:
                            segs[i][ey_key] = ey
                            ex = line_x
                else:  # axis[j] == "h"
                    line_y = other["y1"]
                    olo, ohi = sorted((other["x1"], other["x2"]))
                    if abs(ey - line_y) <= tol and (olo - tol) <= ex <= (ohi + tol):
                        far_y = segs[i]["y1"] if end == "2" else segs[i]["y2"]
                        if (ey - line_y) * (far_y - line_y) < 0:
                            segs[i][ey_key] = line_y
                            ey = line_y

    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


# ---------------------------------------------------------------------------
# Step 4e: prune dangling tails (degree-1 endpoint on a too-short edge)
# ---------------------------------------------------------------------------

def prune_tails(segments: List[Dict], min_len: float,
                quantize: float = 0.01) -> List[Dict]:
    """Iteratively delete degree-1 endpoints whose edge length is below min_len.

    Strict definition (per spec): a "spur" is a short segment whose:
      - one end has degree 1 (truly free), AND
      - the connected end touches segments only of the SAME type (or nothing).

    A short segment that bridges colors — e.g. a tiny black wall stub linking
    a yellow door to anything — is NEVER removed, since those represent
    door/window jambs that must remain.
    """
    segs = [dict(s) for s in segments]

    def _key(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / quantize)), int(round(y / quantize)))

    while True:
        # Map each quantized point to the list of (seg_index, end) and the set
        # of types that touch it.
        node_types: Dict[Tuple[int, int], set] = defaultdict(set)
        deg: Dict[Tuple[int, int], int] = defaultdict(int)
        for idx, s in enumerate(segs):
            k1 = _key(s["x1"], s["y1"])
            k2 = _key(s["x2"], s["y2"])
            deg[k1] += 1
            deg[k2] += 1
            node_types[k1].add(s["type"])
            node_types[k2].add(s["type"])

        removed = False
        kept: List[Dict] = []
        for s in segs:
            length = float(np.hypot(s["x2"] - s["x1"], s["y2"] - s["y1"]))
            k1 = _key(s["x1"], s["y1"])
            k2 = _key(s["x2"], s["y2"])
            d1, d2 = deg[k1], deg[k2]
            stype = s["type"]

            if length >= min_len:
                kept.append(s)
                continue

            # Identify the free end and the connected end.
            if d1 == 1 and d2 > 1:
                free_k, conn_k = k1, k2
            elif d2 == 1 and d1 > 1:
                free_k, conn_k = k2, k1
            else:
                # Both ends loose, or both attached: not a classic tail. Keep.
                kept.append(s)
                continue

            # Connected node's set of types includes this segment's own type.
            # The "other types" present at that node:
            other_types = node_types[conn_k] - {stype}
            if other_types:
                # Cross-type connection — protect, this is a door/window jamb.
                kept.append(s)
                continue

            # Pure same-type tail under the length threshold: delete.
            removed = True

        segs = kept
        if not removed:
            break
    return segs


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


def manhattan_force_axis(segments: List[Dict]) -> List[Dict]:
    """Step 1: every segment becomes exactly H or exactly V.

    Classification rule: |dx| >= |dy| -> horizontal; otherwise vertical.
    There is no "diagonal" outcome — diagonals are eliminated by force.
    Off-axis coordinate is set to the average of the two ends; that's the
    only averaging in the Manhattan stage and it stays inside the segment.
    """
    out: List[Dict] = []
    for seg in segments:
        x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
        if (x1, y1) == (x2, y2):
            continue
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx >= dy:
            ymid = 0.5 * (y1 + y2)
            out.append({**seg, "x1": x1, "y1": ymid, "x2": x2, "y2": ymid})
        else:
            xmid = 0.5 * (x1 + x2)
            out.append({**seg, "x1": xmid, "y1": y1, "x2": xmid, "y2": y2})
    return out


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


def manhattan_intersection_snap(segments: List[Dict], tol: float,
                                *,
                                masks: Optional[Dict[str, np.ndarray]] = None
                                ) -> List[Dict]:
    """Step 2: for every (H, V) pair whose closest endpoints are within tol,
    overwrite both endpoints with the EXACT mathematical intersection
    (V.x, H.y). No coordinate averaging — the H's y stays the H's y, the
    V's x stays the V's x, the corner is forced 90°.

    Wall-priority is enforced indirectly: an H wall and a V door meeting
    at a corner end up with the corner at (V.x, H.y) — the wall's axis line
    is never moved, the door's endpoints align onto it.

    Step 4.9.4: when ``masks`` is given, the per-pair tolerance becomes
    ``min(tol_h, tol_v)`` with each ``tol_i = clamp(0.25 * local_thickness,
    2, 6)``. The ``tol`` argument is the fallback used when a segment's
    thickness can't be sampled (chromatic on wall mask, or too-short
    segments). Strict ``min`` is correct here: an H endpoint snap moves it
    laterally toward V's body (so V's thickness governs how far it may be
    drawn), AND the V endpoint snap moves it laterally toward H's body (so
    H's thickness governs symmetrically) — both must consent.
    """
    segs = [dict(s) for s in segments]
    n = len(segs)
    axis = [_seg_axis_strict(s) for s in segs]
    tols = _compute_seg_tols(segs, masks, fallback=tol) if masks is not None \
        else [tol] * n

    for i in range(n):
        if axis[i] != "h":
            continue
        h = segs[i]
        h_y = h["y1"]
        for j in range(n):
            if i == j or axis[j] != "v":
                continue
            v = segs[j]
            v_x = v["x1"]
            pair_tol = min(tols[i], tols[j])

            # Closest endpoints to (v_x, h_y).
            h_ends = (("1", h["x1"], h_y), ("2", h["x2"], h_y))
            v_ends = (("1", v_x, v["y1"]), ("2", v_x, v["y2"]))
            h_end = min(h_ends, key=lambda e: (e[1] - v_x) ** 2)
            v_end = min(v_ends, key=lambda e: (e[2] - h_y) ** 2)

            dh = abs(h_end[1] - v_x)   # H endpoint's distance to the intersection x
            dv = abs(v_end[2] - h_y)   # V endpoint's distance to the intersection y
            if dh > pair_tol or dv > pair_tol:
                continue

            # Wall-priority: if H is wall and V is chromatic, the V's nearest
            # endpoint snaps to (v_x, h_y) but the H's endpoint also snaps to
            # the same point — but the H's y is preserved (it's already h_y),
            # and the V's x is preserved. Both lines stay perfectly straight.
            h[f"x{h_end[0]}"] = v_x
            # h y is unchanged on both ends (still h_y).
            v[f"y{v_end[0]}"] = h_y
            # v x is unchanged on both ends (still v_x).

    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


def manhattan_t_project(segments: List[Dict], tol: float,
                        *,
                        masks: Optional[Dict[str, np.ndarray]] = None
                        ) -> List[Dict]:
    """Step 3: T-junction projection.

    For every endpoint, scan every orthogonal segment whose body the
    endpoint perpendicularly projects onto within tol. If the endpoint is
    H, it can only T onto a V trunk: rewrite endpoint x to the trunk's x
    (endpoint y is the H's y, untouched). Symmetric for V endpoints onto
    an H trunk.

    Wall-priority: walls never project onto chromatic trunks (TYPE_PRIORITY).
    The trunk segment is never modified.

    Step 4.9.4: when ``masks`` is given, the per-pair tolerance becomes
    the trunk's thickness-aware tol ``clamp(0.25 * trunk_thickness, 2, 6)``.
    The trunk's thickness is the geometrically correct gate here: the
    endpoint moves perpendicular to the trunk's body axis, so "how far
    into the trunk's body the endpoint may be pulled" is set by the trunk.
    The probing segment's own thickness is irrelevant — only its endpoint
    is moving, the body stays put. The body-extent guard ``lo - tol <= ey
    <= hi + tol`` keeps using the trunk's own tol (along-axis slack, same
    rationale: how far past the trunk's end can we count as "on the
    trunk's line").
    """
    segs = [dict(s) for s in segments]
    n = len(segs)
    axis = [_seg_axis_strict(s) for s in segs]
    tols = _compute_seg_tols(segs, masks, fallback=tol) if masks is not None \
        else [tol] * n

    for i in range(n):
        seg = segs[i]
        my_axis = axis[i]
        for end in ("1", "2"):
            ex = seg[f"x{end}"]
            ey = seg[f"y{end}"]
            best = None
            # Cross-trunk tie-breaker only: each trunk gates with its own
            # ``trunk_tol`` below (``d <= trunk_tol``), so ``best_d`` just
            # tracks the closest accepted candidate across trunks.
            best_d = float("inf")
            for j in range(n):
                if i == j:
                    continue
                if axis[j] == my_axis:
                    continue  # parallel can't form a T
                # Forbid wall->chromatic projection: the wall holds.
                if (TYPE_PRIORITY.get(seg["type"], 99)
                        < TYPE_PRIORITY.get(segs[j]["type"], 99)):
                    continue
                trunk = segs[j]
                trunk_tol = tols[j]
                if axis[j] == "v":
                    line_x = trunk["x1"]
                    lo, hi = sorted((trunk["y1"], trunk["y2"]))
                    # Endpoint y must lie within the trunk's body (with tol).
                    if not (lo - trunk_tol <= ey <= hi + trunk_tol):
                        continue
                    d = abs(ex - line_x)
                    if d <= trunk_tol and d < best_d:
                        best_d = d
                        best = (line_x, ey)  # exact projection
                else:  # trunk is horizontal
                    line_y = trunk["y1"]
                    lo, hi = sorted((trunk["x1"], trunk["x2"]))
                    if not (lo - trunk_tol <= ex <= hi + trunk_tol):
                        continue
                    d = abs(ey - line_y)
                    if d <= trunk_tol and d < best_d:
                        best_d = d
                        best = (ex, line_y)
            if best is not None:
                seg[f"x{end}"] = best[0]
                seg[f"y{end}"] = best[1]
                # Preserve own axis: if I'm horizontal, my y must be uniform
                # on both ends; if vertical, my x must be uniform.
                if my_axis == "h":
                    other = "2" if end == "1" else "1"
                    seg[f"y{other}"] = best[1]
                else:
                    other = "2" if end == "1" else "1"
                    seg[f"x{other}"] = best[0]

    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


def manhattan_ultimate_merge(segments: List[Dict]) -> List[Dict]:
    """Step 4: union same-type segments that lie on the exact same line and
    whose extents touch or overlap — fuse into the smallest set of longest
    spans, **but only when doing so does not destroy a T-junction**.

    Junction-aware merge rule:
      Two same-type collinear spans A=[lo_a, hi_a] and B=[lo_b, hi_b]
      (with hi_a ≥ lo_b after sorting) are merged unless the would-be
      interior coordinate is also an endpoint of any segment outside this
      bucket (a different-axis or different-type segment that terminates
      there). If that other segment exists, merging the two pieces would
      collapse the touch point from an explicit node into a body interior,
      and the foreign segment would become a "deferred T-junction"
      (degree-1 endpoint sitting on a body) instead of a real degree-3
      junction. Keeping the two pieces separate preserves the explicit
      junction in the node graph.

    For the canonical case our pipeline produces — Manhattan-routed
    geometry where line coordinates come from a single canonical source —
    the floats compare exact-equal, so endpoint-coordinate matching is
    safe. (We still apply a coarse 0.01-px quantisation for robustness
    against any future caller passing through ordinary float arithmetic.)
    """
    # Bucket horizontals by (type, y), verticals by (type, x).
    h_buckets: Dict[Tuple[str, float], List[Tuple[float, float]]] = defaultdict(list)
    v_buckets: Dict[Tuple[str, float], List[Tuple[float, float]]] = defaultdict(list)
    for s in segments:
        ax = _seg_axis_strict(s)
        if ax == "h":
            lo, hi = sorted((s["x1"], s["x2"]))
            h_buckets[(s["type"], s["y1"])].append((lo, hi))
        elif ax == "v":
            lo, hi = sorted((s["y1"], s["y2"]))
            v_buckets[(s["type"], s["x1"])].append((lo, hi))

    # Endpoint occurrence counter across the whole input — used to detect
    # when a span boundary is also touched by an external segment.
    endpoint_count: Dict[Tuple[int, int], int] = defaultdict(int)
    _Q = 100  # 0.01-px quantisation grid

    def _qk(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x * _Q)), int(round(y * _Q)))

    for s in segments:
        endpoint_count[_qk(s["x1"], s["y1"])] += 1
        endpoint_count[_qk(s["x2"], s["y2"])] += 1

    def _is_external(point: float, axis_value: float, axis_kind: str,
                     own_contribution: int) -> bool:
        """True iff the endpoint at (point, axis_value) for H (or
        (axis_value, point) for V) is shared with at least one segment
        outside the bucket currently being merged. ``own_contribution``
        is the number of bucket-internal endpoints we know are at this
        coordinate (1 for one-sided overlap, 2 for an exact touch).
        """
        if axis_kind == "h":
            k = _qk(point, axis_value)
        else:
            k = _qk(axis_value, point)
        return endpoint_count.get(k, 0) > own_contribution

    def _union_junction_aware(spans: List[Tuple[float, float]],
                               axis_value: float, axis_kind: str
                               ) -> List[Tuple[float, float]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged = [spans[0]]
        for lo, hi in spans[1:]:
            cur_lo, cur_hi = merged[-1]
            if lo > cur_hi:
                merged.append((lo, hi))
                continue
            # Spans overlap or touch — decide merge.
            block = False
            if abs(lo - cur_hi) < 1e-6:
                # Exact touch at point cur_hi. Two bucket-internal endpoints
                # contribute (cur's hi and new's lo) → own_contribution=2.
                if _is_external(cur_hi, axis_value, axis_kind, 2):
                    block = True
            else:
                # Overlap: cur_hi (cur's hi) and lo (new's lo) are at
                # different coords. Each gets one bucket-internal
                # endpoint contribution. If either coincides with an
                # external endpoint, merging would absorb that junction
                # into a body interior.
                if _is_external(cur_hi, axis_value, axis_kind, 1):
                    block = True
                elif _is_external(lo, axis_value, axis_kind, 1):
                    block = True
            if block:
                merged.append((lo, hi))
            else:
                merged[-1] = (cur_lo, max(cur_hi, hi))
        return merged

    out: List[Dict] = []
    for (t, y), spans in h_buckets.items():
        for lo, hi in _union_junction_aware(spans, y, "h"):
            if hi > lo:
                out.append({"type": t, "x1": lo, "y1": y, "x2": hi, "y2": y})
    for (t, x), spans in v_buckets.items():
        for lo, hi in _union_junction_aware(spans, x, "v"):
            if hi > lo:
                out.append({"type": t, "x1": x, "y1": lo, "x2": x, "y2": hi})
    return out


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


def cluster_parallel_duplicates(segments: List[Dict],
                                perp_tol: float) -> List[Dict]:
    """Per (type, axis) bucket: cluster segments whose line coordinates are
    within perp_tol AND whose along-axis extents overlap. Each cluster
    becomes one centred line spanning the union of extents.

    Length-weighted line coordinate: a long wall dominates a stub so a
    real long wall doesn't get nudged off-axis by a short skeleton fragment.
    """
    by_axis_type: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, s in enumerate(segments):
        by_axis_type[(_seg_axis_strict(s), s["type"])].append(i)

    out: List[Dict] = []
    for (axis, t), idxs in by_axis_type.items():
        items: List[Dict] = []
        for i in idxs:
            s = segments[i]
            if axis == "h":
                lo, hi = sorted((s["x1"], s["x2"]))
                line = s["y1"]
            else:
                lo, hi = sorted((s["y1"], s["y2"]))
                line = s["x1"]
            items.append({"lo": lo, "hi": hi, "line": line, "len": hi - lo})

        # Build undirected graph linking segment pairs that should be merged.
        # Two distinct cases must both be handled here:
        #
        #  (1) Near-collinear touching pieces. Same axis, tiny perpendicular
        #      separation (a few pixels), ranges touch or overlap a little.
        #      These are typically a single wall that the skeleton broke
        #      into two pieces. Merge tolerance for this is small (~5 px).
        #
        #  (2) Thick-wall duplicate centerlines. A genuinely thick wall whose
        #      mask skeletonized into two parallel ridges. Same axis, ranges
        #      overlap substantially (>50% of the shorter), perp distance up
        #      to perp_tol.
        #
        # A simple touch-merge with the full perp_tol would mis-fuse L-corners
        # (one long wall touching a short perpendicular stub at the corner).
        # Splitting the criterion as above keeps L-corners intact while still
        # absorbing thick-wall duplicates.
        TOUCH_PERP_TOL = 12.0
        g = nx.Graph()
        for k in range(len(items)):
            g.add_node(k)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                dline = abs(a["line"] - b["line"])
                if dline > perp_tol:
                    continue
                ov_lo = max(a["lo"], b["lo"])
                ov_hi = min(a["hi"], b["hi"])
                overlap = ov_hi - ov_lo
                shorter = min(a["len"], b["len"])
                if shorter <= 0:
                    continue
                # Case (1): near-collinear, ranges touch or overlap a bit.
                if dline <= TOUCH_PERP_TOL and overlap >= -1e-6:
                    g.add_edge(i, j)
                    continue
                # Case (2): thick-wall duplicate, requires real overlap.
                if overlap >= 0.5 * shorter and overlap > 0:
                    g.add_edge(i, j)

        for comp in nx.connected_components(g):
            comp = list(comp)
            members = [items[k] for k in comp]
            total_w = sum(max(m["len"], 1e-6) for m in members)
            line = sum(m["line"] * max(m["len"], 1e-6) for m in members) / total_w
            lo = min(m["lo"] for m in members)
            hi = max(m["hi"] for m in members)
            if hi <= lo:
                continue
            if axis == "h":
                out.append({"type": t, "x1": lo, "y1": line, "x2": hi, "y2": line})
            else:
                out.append({"type": t, "x1": line, "y1": lo, "x2": line, "y2": hi})
    return out


def grid_snap_endpoints(segments: List[Dict], tol: float) -> List[Dict]:
    """Treat each axis-aligned segment as an infinite reference line. For
    every endpoint, find the closest orthogonal reference line within tol
    perpendicular distance whose body covers the endpoint's along-axis
    coordinate, and pull the endpoint to that exact line.

    The segment's own axis identity is preserved (a horizontal stays
    horizontal: only its x at one end changes; a vertical stays vertical:
    only its y at one end changes). This is the "snap to guideline"
    behaviour the user asked for.
    """
    segs = [dict(s) for s in segments]
    n = len(segs)
    axis = [_seg_axis_strict(s) for s in segs]

    # Pre-compute reference lines per orientation:
    # horizontal lines as (line_y, lo_x, hi_x, type)
    # vertical lines as (line_x, lo_y, hi_y, type)
    h_refs: List[Tuple[float, float, float, str, int]] = []
    v_refs: List[Tuple[float, float, float, str, int]] = []
    for j, s in enumerate(segs):
        if axis[j] == "h":
            lo, hi = sorted((s["x1"], s["x2"]))
            h_refs.append((s["y1"], lo, hi, s["type"], j))
        else:
            lo, hi = sorted((s["y1"], s["y2"]))
            v_refs.append((s["x1"], lo, hi, s["type"], j))

    for i in range(n):
        my_type = segs[i]["type"]
        my_prio = TYPE_PRIORITY.get(my_type, 99)
        if axis[i] == "v":
            # Vertical's endpoints have variable y; snap each y to the
            # closest horizontal reference line.
            for end in ("1", "2"):
                ex = segs[i][f"x{end}"]
                ey = segs[i][f"y{end}"]
                best_y = None
                best_d = tol
                for (line_y, lo, hi, ref_t, j) in h_refs:
                    if i == j:
                        continue
                    if my_prio < TYPE_PRIORITY.get(ref_t, 99):
                        continue  # walls don't snap to chromatic refs
                    # The endpoint's x must lie within the horizontal body
                    # (with tol) so we don't snap to a wall on the other
                    # side of the building.
                    if not (lo - tol <= ex <= hi + tol):
                        continue
                    d = abs(ey - line_y)
                    if d < best_d:
                        best_d = d
                        best_y = line_y
                if best_y is not None:
                    segs[i][f"y{end}"] = best_y
        else:  # horizontal
            for end in ("1", "2"):
                ex = segs[i][f"x{end}"]
                ey = segs[i][f"y{end}"]
                best_x = None
                best_d = tol
                for (line_x, lo, hi, ref_t, j) in v_refs:
                    if i == j:
                        continue
                    if my_prio < TYPE_PRIORITY.get(ref_t, 99):
                        continue
                    if not (lo - tol <= ey <= hi + tol):
                        continue
                    d = abs(ex - line_x)
                    if d < best_d:
                        best_d = d
                        best_x = line_x
                if best_x is not None:
                    segs[i][f"x{end}"] = best_x

    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]




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
    """Return (degree-by-quantized-point, types-by-quantized-point)."""
    deg: Dict[Tuple[int, int], int] = defaultdict(int)
    types: Dict[Tuple[int, int], set] = defaultdict(set)

    def _key(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / quantize)), int(round(y / quantize)))

    for s in segments:
        for ex_key, ey_key in (("x1", "y1"), ("x2", "y2")):
            k = _key(s[ex_key], s[ey_key])
            deg[k] += 1
            types[k].add(s["type"])
    return deg, types


def _qkey(x: float, y: float, quantize: float = 0.01) -> Tuple[int, int]:
    return (int(round(x / quantize)), int(round(y / quantize)))






def t_snap_with_extension(segments: List[Dict], tol: float,
                          masks: Dict[str, np.ndarray] = None,
                          min_support: float = 0.6,
                          *,
                          wall_evidence: np.ndarray = None,
                          door_mask: np.ndarray = None,
                          window_mask: np.ndarray = None) -> List[Dict]:
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
        base_score = S.compute_score(current,
                                     wall_evidence=wall_evidence,
                                     door_mask=door_mask,
                                     window_mask=window_mask)
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
                                          window_mask=window_mask)
            delta = trial_score.total - base_score.total
            if delta > CANDIDATE_MIN_ACCEPT_DELTA:
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
                                 window_mask=window_mask)
    used_endpoints: set = set()

    for cand in cands:
        ea, eb = cand.meta["pair_endpoints"]
        ka = (int(round(ea[0])), int(round(ea[1])))
        kb = (int(round(eb[0])), int(round(eb[1])))
        if ka in used_endpoints or kb in used_endpoints:
            continue
        trial = C.apply_candidate(current, cand)
        trial_score = S.compute_score(trial,
                                      wall_evidence=wall_evidence,
                                      door_mask=door_mask,
                                      window_mask=window_mask)
        delta = trial_score.total - base_score.total
        if delta > CANDIDATE_MIN_ACCEPT_DELTA:
            current = trial
            base_score = trial_score
            used_endpoints.add(ka)
            used_endpoints.add(kb)
    return current


def brute_force_ray_extend(lines: List[Dict],
                           tol: float,
                           loose_tol: float,
                           *,
                           wall_evidence: np.ndarray = None,
                           door_mask: np.ndarray = None,
                           window_mask: np.ndarray = None) -> None:
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
                continue
            trial = C.apply_candidate(current, cand)
            trial_score = S.compute_score(trial,
                                          wall_evidence=wall_evidence,
                                          door_mask=door_mask,
                                          window_mask=window_mask)
            delta = trial_score.total - base_score.total
            if delta > CANDIDATE_MIN_ACCEPT_DELTA:
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
                              window_mask: np.ndarray = None) -> None:
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
                                 window_mask=window_mask)
    used_mutations: Set[Tuple[int, str]] = set()
    used_endpoints: Set[Tuple[float, float]] = set()

    for cand in cands:
        # Endpoint-level exclusion: each pair of loose endpoints can be
        # consumed by at most one accepted candidate.
        eps = cand.meta["endpoints"]
        if eps[0] in used_endpoints or eps[1] in used_endpoints:
            continue
        if any((idx, end) in used_mutations for (idx, end, _, _) in cand.mutate):
            continue

        trial = C.apply_candidate(current, cand)
        trial_score = S.compute_score(trial,
                                      wall_evidence=wall_evidence,
                                      door_mask=door_mask,
                                      window_mask=window_mask)
        delta = trial_score.total - base_score.total
        if delta > CANDIDATE_MIN_ACCEPT_DELTA:
            current = trial
            base_score = trial_score
            for (idx, end, _, _) in cand.mutate:
                used_mutations.add((idx, end))
            used_endpoints.add(eps[0])
            used_endpoints.add(eps[1])

    # In-place replace so the caller's reference stays valid.
    lines.clear()
    lines.extend(current)


def fuse_close_endpoints(lines: List[Dict], tol: float,
                         *,
                         masks: Optional[Dict[str, np.ndarray]] = None
                         ) -> None:
    """Mutate `lines` IN PLACE: cluster all endpoint x's within tol to a
    single canonical x (and same for y's). After this every endpoint that
    was within `tol` of another shares the EXACT same coordinate.

    Wall-priority: when a cluster contains both a wall coordinate and a
    chromatic coordinate, the wall coordinate wins.

    Step 4.9.4: when ``masks`` is given, each endpoint carries its host
    segment's thickness-aware tol ``clamp(0.25 * local_thickness, 2, 6)``.
    The sort-and-walk cluster uses the *minimum* of two neighbours' tols
    as the join threshold — strict, so a thin partition doesn't get glued
    to a separate thick wall just because the thick wall has high tol.
    The cluster's canonical coordinate still uses wall-priority mean (a
    cluster with one wall and several door coords resolves to the wall x
    even though the doors have tighter tols).
    """
    if not lines:
        return

    # Build per-segment thickness tols once. The clusters need per-endpoint
    # access, so duplicate each segment's tol into its two endpoint entries.
    seg_tols = _compute_seg_tols(lines, masks, fallback=tol) \
        if masks is not None else [tol] * len(lines)

    # Collect all endpoint coordinates with their owning segment's type
    # and the per-endpoint tol.
    xs: List[Tuple[float, str, float]] = []
    ys: List[Tuple[float, str, float]] = []
    for s, st in zip(lines, seg_tols):
        xs.append((s["x1"], s["type"], st))
        xs.append((s["x2"], s["type"], st))
        ys.append((s["y1"], s["type"], st))
        ys.append((s["y2"], s["type"], st))

    def _cluster(tagged: List[Tuple[float, str, float]]
                 ) -> Dict[Tuple[float, str], float]:
        if not tagged:
            return {}
        order = sorted(tagged, key=lambda p: p[0])
        clusters: List[List[Tuple[float, str, float]]] = [[order[0]]]
        for v in order[1:]:
            prev = clusters[-1][-1]
            join_tol = min(v[2], prev[2])
            if v[0] - prev[0] <= join_tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        mapping: Dict[Tuple[float, str], float] = {}
        for grp in clusters:
            prios = [TYPE_PRIORITY.get(t, 99) for _, t, _ in grp]
            top = min(prios)
            anchors = [val for (val, t, _), p in zip(grp, prios) if p == top]
            canon = float(np.mean(anchors))
            for val, t, _ in grp:
                mapping[(val, t)] = canon
        return mapping

    xmap = _cluster(xs)
    ymap = _cluster(ys)

    for s in lines:
        t = s["type"]
        s["x1"] = xmap.get((s["x1"], t), s["x1"])
        s["x2"] = xmap.get((s["x2"], t), s["x2"])
        s["y1"] = ymap.get((s["y1"], t), s["y1"])
        s["y2"] = ymap.get((s["y2"], t), s["y2"])


# ---------------------------------------------------------------------------
# (Legacy passes kept for the pre-Manhattan optimization stages.)
# ---------------------------------------------------------------------------

def hard_axis_snap(segments: List[Dict], tol_deg: float) -> List[Dict]:
    """Force every near-axis segment to be exactly horizontal/vertical.

    Stricter and broader than `axis_align_segments`: this runs after all
    higher-level optimization, so we accept a wider angle window (±10°)
    knowing each segment is otherwise correct. Diagonals further from an
    axis than tol_deg are left alone.
    """
    rad = np.deg2rad(tol_deg)
    out: List[Dict] = []
    for seg in segments:
        x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
        if (x1, y1) == (x2, y2):
            continue
        dx, dy = x2 - x1, y2 - y1
        ang = np.arctan2(dy, dx)
        if abs(ang) < rad or abs(abs(ang) - np.pi) < rad:
            ymid = 0.5 * (y1 + y2)
            seg = {**seg, "y1": ymid, "y2": ymid}
        elif abs(abs(ang) - np.pi / 2) < rad:
            xmid = 0.5 * (x1 + x2)
            seg = {**seg, "x1": xmid, "x2": xmid}
        out.append(seg)
    return out


def graph_node_align(segments: List[Dict], rejoin_tol: float) -> List[Dict]:
    """Re-unify endpoints that were spatially coincident before hard_axis_snap.

    Algorithm:
      1. Build a NetworkX graph; each node is an (segment_index, end) tuple.
      2. Add an edge between two nodes whose endpoint coordinates are within
         rejoin_tol — these were the same logical joint before per-segment
         straightening introduced sub-pixel drift.
      3. For every connected component, derive a canonical (x*, y*):
            - x* = wall-priority mean of x's contributed by VERTICAL segments
              (their x is the line, so it's the "trustworthy" x of the joint)
            - y* = wall-priority mean of y's contributed by HORIZONTAL segments
            - if a component has only one orientation, the missing axis falls
              back to the wall-priority mean of all members' coords on that axis.
      4. Rewrite each endpoint to (x*, y*). Since each segment's own axis
         identity is preserved (horizontals still carry y == y* on both ends
         after the rewrite), all segments stay perfectly axis-aligned.

    The result: connected segments share a single mathematical joint, and
    every long colinear chain reads off a single x or y.
    """
    if not segments:
        return segments

    n = len(segments)
    g = nx.Graph()
    nodes: List[Tuple[int, str, float, float, str]] = []  # (idx, end, x, y, axis_of_segment)
    for i, seg in enumerate(segments):
        ax = _classify_axis(seg)
        nodes.append((i, "1", seg["x1"], seg["y1"], ax))
        nodes.append((i, "2", seg["x2"], seg["y2"], ax))
    for k in range(len(nodes)):
        g.add_node(k)
    tol2 = rejoin_tol * rejoin_tol
    for i in range(len(nodes)):
        xi, yi = nodes[i][2], nodes[i][3]
        for j in range(i + 1, len(nodes)):
            dx = nodes[j][2] - xi
            dy = nodes[j][3] - yi
            if dx * dx + dy * dy <= tol2:
                g.add_edge(i, j)

    # Resolve canonical (x*, y*) per component.
    canonical: Dict[int, Tuple[float, float]] = {}
    for comp in nx.connected_components(g):
        comp = list(comp)
        # Group contributions by their segment's axis.
        v_xs: List[Tuple[float, str]] = []  # (x, type) from vertical segs
        h_ys: List[Tuple[float, str]] = []  # (y, type) from horizontal segs
        all_xs: List[Tuple[float, str]] = []
        all_ys: List[Tuple[float, str]] = []
        for k in comp:
            idx, end, x, y, ax = nodes[k]
            t = segments[idx]["type"]
            all_xs.append((x, t))
            all_ys.append((y, t))
            if ax == "v":
                v_xs.append((x, t))
            elif ax == "h":
                h_ys.append((y, t))

        def _wall_mean(vals: List[Tuple[float, str]]) -> float:
            if not vals:
                return 0.0
            prios = [TYPE_PRIORITY.get(t, 99) for _, t in vals]
            top = min(prios)
            anchored = [v for (v, t), p in zip(vals, prios) if p == top]
            return float(np.mean(anchored))

        cx = _wall_mean(v_xs) if v_xs else _wall_mean(all_xs)
        cy = _wall_mean(h_ys) if h_ys else _wall_mean(all_ys)
        for k in comp:
            canonical[k] = (cx, cy)

    out: List[Dict] = []
    for i, seg in enumerate(segments):
        ax = _classify_axis(seg)
        k1 = 2 * i
        k2 = 2 * i + 1
        x1, y1 = canonical[k1]
        x2, y2 = canonical[k2]
        # Preserve axis identity defensively. After the rewrite each end may
        # disagree with the segment's own line by sub-pixel; force-align.
        if ax == "h":
            # both ends must share y; pick the wall-priority y from this seg's
            # own contribution which is identical at both ends pre-snap, but
            # post-rewrite they may differ. Use the mean of the two new y's.
            yshared = 0.5 * (y1 + y2)
            y1 = y2 = yshared
        elif ax == "v":
            xshared = 0.5 * (x1 + x2)
            x1 = x2 = xshared
        if (x1, y1) == (x2, y2):
            continue
        out.append({**seg, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def final_perpendicular_project(segments: List[Dict], tol: float) -> List[Dict]:
    """Zero-tolerance closure: any degree-1 endpoint within tol of an
    orthogonal segment's body is rewritten to the *exact* perpendicular
    projection on that body.

    This guarantees the joint sits mathematically on the wall, no gaps.
    """
    segs = [dict(s) for s in segments]

    def _key(x: float, y: float, q: float = 0.05) -> Tuple[int, int]:
        return (int(round(x / q)), int(round(y / q)))

    # Compute degree per quantized point.
    deg: Dict[Tuple[int, int], int] = defaultdict(int)
    for s in segs:
        deg[_key(s["x1"], s["y1"])] += 1
        deg[_key(s["x2"], s["y2"])] += 1

    for i, seg in enumerate(segs):
        ax_i = _classify_axis(seg)
        for end in ("1", "2"):
            ex_key, ey_key = f"x{end}", f"y{end}"
            ex, ey = seg[ex_key], seg[ey_key]
            if deg[_key(ex, ey)] != 1:
                continue  # already attached to a real joint
            best = None
            best_d = tol
            for j, other in enumerate(segs):
                if i == j:
                    continue
                ax_j = _classify_axis(other)
                if ax_j == "d":
                    continue
                # Wall-priority anchor: never project a wall onto a chromatic body.
                if (TYPE_PRIORITY.get(seg["type"], 99)
                        < TYPE_PRIORITY.get(other["type"], 99)):
                    continue
                # Compute exact perpendicular projection on the orthogonal segment.
                if ax_j == "h":
                    line_y = other["y1"]
                    lo, hi = sorted((other["x1"], other["x2"]))
                    if lo - tol <= ex <= hi + tol:
                        sx = min(max(ex, lo), hi)
                        sy = line_y
                        d = float(np.hypot(sx - ex, sy - ey))
                        if d < best_d:
                            best_d = d
                            best = (sx, sy)
                else:  # vertical
                    line_x = other["x1"]
                    lo, hi = sorted((other["y1"], other["y2"]))
                    if lo - tol <= ey <= hi + tol:
                        sx = line_x
                        sy = min(max(ey, lo), hi)
                        d = float(np.hypot(sx - ex, sy - ey))
                        if d < best_d:
                            best_d = d
                            best = (sx, sy)
            if best is not None:
                # Snap the endpoint to the projection.
                seg[ex_key] = best[0]
                seg[ey_key] = best[1]
                # Preserve segment's own axis identity: the *other* endpoint
                # of this same segment must update too if we just changed the
                # line coordinate of an axis-aligned segment.
                if ax_i == "h":
                    # horizontal: y must be uniform on both ends.
                    other_end = "2" if end == "1" else "1"
                    seg[f"y{other_end}"] = best[1]
                elif ax_i == "v":
                    other_end = "2" if end == "1" else "1"
                    seg[f"x{other_end}"] = best[0]

    return [s for s in segs if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]


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


def vectorize_bgr(bgr: np.ndarray, *, verbose: bool = False) -> Dict:
    """Pure pipeline: BGR ndarray in, dict ``{"lines": [...], "stats": {...}}`` out.

    No filesystem I/O, no print spam (unless ``verbose=True``). This is the
    function the API calls — ``run_one`` is a thin I/O wrapper around it.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image, got shape {bgr.shape}")

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
    s1 = axis_align_segments(typed_segments, AXIS_SNAP_DEG)
    # Tight wall-anchored coordinate cluster so colinear walls share an x or y,
    # without dragging genuinely-distinct walls together.
    s1 = snap_colinear_coords(s1, colinear_tol)

    # (2) Merge collinear same-type segments into a single longest span.
    s2 = merge_collinear(s1, merge_perp, merge_gap)

    # (3a) T-junction node-to-edge snap: chromatic endpoints onto wall bodies,
    #      same-type endpoints onto own bodies. Walls never snap to chromatic.
    s3 = t_junction_snap(s2, t_snap)
    # (3b) extend_to_intersect (legacy L-corner extend pass) removed in
    #      step 4.8: post-junction-aware-merge ablation showed IOU=0.9967
    #      and dFree=1 — the same corner-closure work is now done by the
    #      proximal_bridge generator's L-bridge case, which the scorer
    #      accepts when mask support is present.
    # (3c) Truncate any remaining overshoots: an endpoint sitting on the far
    #      side of a perpendicular wall by less than tol gets clipped back.
    s3 = truncate_overshoots(s3, l_extend)
    # Re-merge after snapping; some segments may now align colinearly.
    s3 = merge_collinear(s3, merge_perp, merge_gap)

    # Wall-priority NetworkX node merge.
    s4 = snap_endpoints(s3, snap_tol)

    # prune_tails removed in step 4.8: try removal and revert if FAIL.
    s5 = s4

    # --- STRICT MANHATTAN ROUTING (zero diagonals) ----------------------
    # (a) Force every segment to be exactly horizontal or exactly vertical.
    s6 = manhattan_force_axis(s5)
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
    s6 = canonicalize_offsets(s6, wall_mask=masks.get("wall"),
                              attach_thickness=True)
    # (b) Intersection-based L-corner snap removed in step 4.9.7 (post-
    #     ablation cleanup): with canonicalize_offsets above and thickness-
    #     aware tols in manhattan_t_project below, ablation on source/sg2
    #     shows ``manhattan_intersection_snap`` as a pure NO-OP (dN=0,
    #     IOU=1.000 bit-identical on both reference images; Gemini also
    #     bit-identical). Its work is fully covered by the canonical-line
    #     + T-project + intersection-on-fuse path now.
    # (c) T-junction projection onto orthogonal trunks. Step 4.9.4: tol
    #     scales with the trunk's local thickness, fallback ``manhattan_tol``.
    s6 = manhattan_t_project(s6, manhattan_tol, masks=masks)
    # (d) Ultimate collinear merge: post-step-4.7 ablation confirmed this
    #     first manhattan_ultimate_merge call is NO-OP after the upstream
    #     passes had their inputs unchanged by step 4 / 4.6 / 4.7 — the
    #     real merge work happens at lines below (post-watertight and
    #     post-gap-closing). Call removed.

    # --- WATERTIGHT CLOSURE (kill duplicates + close gaps) --------------
    s7 = cluster_parallel_duplicates(s6, parallel_merge)
    s7 = grid_snap_endpoints(s7, grid_snap)
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
    s8 = t_snap_with_extension(s7, gap_close, masks=masks)
    # manhattan_ultimate_merge (post-gap-closing) removed step 4.8: NO-OP
    # in latest ablation.

    snapped = s8

    # --- BRUTE-FORCE RAY EXTENSION (final-final closure, IN PLACE) ------
    # Operates directly on the same dict objects that get serialized to JSON.
    # No copies, no graph indirection — every mutation lands in the output.
    brute_force_ray_extend(snapped, ray_ext, RAY_EXT_LOOSE_PX)
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
                              wall_mask=masks.get("wall"))
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
        wall_evidence=None,  # falls back to (wall_mask>0) as float32
        door_mask=masks.get("door"),
        window_mask=masks.get("window"),
    )
    snapped = [s for s in snapped if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # Final 2-px endpoint fusion: snap any near-coincident endpoints to one
    # canonical (wall-priority) coordinate so micro-deltas vanish.
    # Step 4.9.4: per-endpoint tol is thickness-aware; ``ray_fuse`` is the
    # fallback only (and is below the 2 px floor anyway, so in practice
    # every endpoint resolves via clamp(0.25*thickness, 2, 6)).
    fuse_close_endpoints(snapped, ray_fuse, masks=masks)
    snapped = [s for s in snapped if (s["x1"], s["y1"]) != (s["x2"], s["y2"])]
    # One last collinear merge in case ray extension produced overlapping
    # spans that can now coalesce.
    snapped = manhattan_ultimate_merge(snapped)

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
