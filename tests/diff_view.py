"""Visual three-panel diff viewer for vectorize.py output.

Renders, side-by-side: the source image, the baseline output, the current
output, and a red/green diff overlay on top of the source. Designed for
human inspection when the metric-based regression layer (tests/metrics.py)
fails or hits the warn / edge of tolerance — staring at numbers isn't
enough to decide "this change is an optimization" vs "this change is a
regression"; the side-by-side panel makes the call.

Usage:
    py -3 -m tests.diff_view --case source
    py -3 -m tests.diff_view --case source --out diff.png
    py -3 -m tests.diff_view --case all

Without ``--out``, the panel is written to
``tests/current/<case>/triptych.png`` so it lands next to the existing
``diff_overlay.png`` from regression.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import regression as R  # noqa: E402


PANEL_BG = (255, 255, 255)         # white
LABEL_BG = (240, 240, 240)         # light grey
LABEL_FG = (40, 40, 40)
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICK = 1
LABEL_PAD = 8

# Drawing colours match write_overlay so the comparison is visually
# consistent across the two tools.
COLOR_BASELINE_ONLY = (0, 0, 255)   # red (BGR)
COLOR_CURRENT_ONLY = (0, 255, 0)    # green
COLOR_OVERLAP = (0, 0, 0)           # black on white panel


def _draw_lines(canvas: np.ndarray, lines: List[Dict],
                colour_by_type: Dict[str, Tuple[int, int, int]],
                thickness: int) -> None:
    """Draw segments onto ``canvas`` (BGR, in-place)."""
    for s in lines:
        t = s.get("type", "")
        if t not in colour_by_type:
            continue
        x1, y1 = int(round(float(s["x1"]))), int(round(float(s["y1"])))
        x2, y2 = int(round(float(s["x2"]))), int(round(float(s["y2"])))
        cv2.line(canvas, (x1, y1), (x2, y2), colour_by_type[t], thickness)


def _render_segments(shape: Tuple[int, int], lines: List[Dict],
                     thickness: int) -> np.ndarray:
    """Render ``lines`` onto a white canvas using the standard
    wall/door/window colour scheme used by the project's preview PNGs.
    """
    h, w = shape[:2]
    canvas = np.full((h, w, 3), PANEL_BG, dtype=np.uint8)
    _draw_lines(canvas, lines, {
        "wall":   (0, 0, 0),         # black
        "door":   (0, 255, 255),     # yellow
        "window": (255, 200, 0),     # blue-cyan
    }, thickness)
    return canvas


def _render_diff_overlay(shape: Tuple[int, int],
                         source_bgr: np.ndarray,
                         baseline_lines: List[Dict],
                         current_lines: List[Dict],
                         thickness: int) -> np.ndarray:
    """Render a red/green diff on top of a faded source image.

    Source pixels are dimmed to 30% so the diff stripes pop visually.
    Pixels reached by both baseline & current draw black; only-baseline
    red; only-current green. Untouched pixels show the source itself.
    """
    h, w = shape[:2]
    base_mask = np.zeros((h, w), dtype=np.uint8)
    cur_mask = np.zeros((h, w), dtype=np.uint8)
    for t in ("wall", "door", "window"):
        base_mask = cv2.bitwise_or(
            base_mask, R.rasterize_lines(baseline_lines, shape, t, thickness))
        cur_mask = cv2.bitwise_or(
            cur_mask, R.rasterize_lines(current_lines, shape, t, thickness))
    b = base_mask > 0
    c = cur_mask > 0

    # Source faded toward white at 30% opacity so the diff stripes stand out.
    faded = cv2.addWeighted(source_bgr, 0.3,
                            np.full_like(source_bgr, 255), 0.7, 0)
    out = faded.copy()
    out[b & c] = COLOR_OVERLAP
    out[b & ~c] = COLOR_BASELINE_ONLY
    out[~b & c] = COLOR_CURRENT_ONLY
    return out


def _add_label_bar(panel: np.ndarray, text: str) -> np.ndarray:
    """Stack a label strip above ``panel`` (BGR uint8) and return the new image."""
    (tw, th), bl = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICK)
    strip_h = th + 2 * LABEL_PAD + bl
    h, w = panel.shape[:2]
    strip = np.full((strip_h, w, 3), LABEL_BG, dtype=np.uint8)
    cv2.putText(strip, text,
                (LABEL_PAD, LABEL_PAD + th),
                LABEL_FONT, LABEL_SCALE, LABEL_FG, LABEL_THICK,
                cv2.LINE_AA)
    return np.vstack([strip, panel])


def make_triptych(source_bgr: np.ndarray,
                  baseline_lines: List[Dict],
                  current_lines: List[Dict],
                  *,
                  thickness: Optional[int] = None,
                  case_label: str = "",
                  baseline_label: str = "",
                  current_label: str = "",
                  ) -> np.ndarray:
    """Return a horizontal 4-panel image: source / baseline / current / diff.

    Each panel carries a label bar describing what it shows.
    """
    shape = source_bgr.shape[:2]
    if thickness is None:
        # Use the same per-image normal thickness regression.py uses, so
        # the diff overlay rasterizer agrees with the regression
        # overlay.
        masks = {"wall": np.zeros_like(source_bgr[..., 0])}
        thickness = max(2, int(round(
            R.estimate_stroke_width(masks["wall"]) or 2.0)))
    p_src = _add_label_bar(source_bgr.copy(),
                           f"source ({case_label})" if case_label else "source")
    p_base = _add_label_bar(_render_segments(shape, baseline_lines, thickness),
                            baseline_label or "baseline")
    p_cur = _add_label_bar(_render_segments(shape, current_lines, thickness),
                           current_label or "current")
    p_diff = _add_label_bar(
        _render_diff_overlay(shape, source_bgr,
                             baseline_lines, current_lines, thickness),
        "diff: red=baseline-only  green=current-only  black=overlap")
    return np.hstack([p_src, p_base, p_cur, p_diff])


def _load_case(case: str
               ) -> Tuple[np.ndarray, List[Dict], List[Dict], float]:
    """Returns (bgr, baseline_lines, current_lines, stroke_width).

    Falls back to the freshly-run current output if
    tests/current/<case>/lines.json doesn't exist yet.
    """
    with open(os.path.join(REPO, "tests", "cases", case, "manifest.json")) as f:
        mani = json.load(f)
    img_path = os.path.normpath(os.path.join(
        REPO, "tests", "cases", case, mani["input_image"]))
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image {img_path}")

    with open(os.path.join(REPO, "tests", "baseline", case, "lines.json")) as f:
        baseline_lines = json.load(f)["lines"]
    with open(os.path.join(REPO, "tests", "baseline", case, "metrics.json")) as f:
        stroke_width = float(json.load(f)["stroke_width"])

    cur_path = os.path.join(REPO, "tests", "current", case, "lines.json")
    if os.path.isfile(cur_path):
        with open(cur_path) as f:
            current_lines = json.load(f)["lines"]
    else:
        # No current run yet — render baseline twice with a notice.
        print(f"[note] no tests/current/{case}/lines.json found; "
              f"diff will be baseline vs baseline (run regression.py first)")
        current_lines = baseline_lines
    return bgr, baseline_lines, current_lines, stroke_width


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-panel visual diff viewer")
    ap.add_argument("--case", required=True,
                    help="case name (e.g. 'source', 'sg2', 'all')")
    ap.add_argument("--out", default=None,
                    help="output PNG path; defaults to tests/current/<case>/triptych.png")
    ap.add_argument("--thickness", type=int, default=None,
                    help="stroke thickness in px (defaults to normal "
                         "rasterization thickness for the case)")
    args = ap.parse_args()

    cases: List[str] = []
    if args.case == "all":
        cases_dir = os.path.join(REPO, "tests", "cases")
        cases = sorted(d for d in os.listdir(cases_dir)
                       if os.path.isdir(os.path.join(cases_dir, d)))
    else:
        cases = [args.case]

    for case in cases:
        try:
            bgr, base_lines, cur_lines, stroke = _load_case(case)
        except SystemExit as e:
            print(f"[skip] {case}: {e}")
            continue
        thk = args.thickness or max(2, int(round(stroke)))
        panel = make_triptych(
            bgr, base_lines, cur_lines,
            thickness=thk,
            case_label=case,
            baseline_label=f"baseline ({len(base_lines)} segs)",
            current_label=f"current ({len(cur_lines)} segs)",
        )
        if args.out and args.case != "all":
            out_path = args.out
        else:
            out_path = os.path.join(
                REPO, "tests", "current", case, "triptych.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, panel)
        print(f"[wrote] {out_path}  ({panel.shape[1]}x{panel.shape[0]} px)")


if __name__ == "__main__":
    main()
