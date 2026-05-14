"""Generate ``output/<stem>_preview.png`` from a vectorised JSON.

Re-renders the same debug overlay that ``vectorize.run_one`` writes
(``draw_debug_image``), but starting from an existing JSON + source
image instead of running the full pipeline. Useful when:

  * you have already vectorised an image and only want a fresh preview
    (e.g. after editing the JSON by hand or after pulling a baseline
    from ``tests/baseline/<case>/lines.json``);
  * you want to bulk-rebuild every preview in ``output/`` against the
    matching ``srcImg/`` originals.

Usage:
  py preview.py                            # default: rebuild every output/*.json
  py preview.py source                     # output/source.json   -> output/source_preview.png
  py preview.py output/source.json         # explicit JSON path
  py preview.py source --src srcImg/source.png --out tmp/out.png
  py preview.py --all                      # explicit form of the default

The source image is needed only to recover image dimensions (the JSON
output does not carry ``image_size``). If you already know the size
and have no source on disk, pass ``--shape HxW`` (e.g. ``--shape
895x1200``) instead of ``--src``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Reuse the canonical renderer so this script never drifts from what
# ``python vectorize.py`` writes.
from vectorize import draw_debug_image, SRC_DIR, OUT_DIR


SRC_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resolve_json(stem_or_path: str) -> str:
    """``output/<stem>.json`` if argument is a bare stem; else as-given."""
    if os.path.isabs(stem_or_path) or os.sep in stem_or_path or stem_or_path.endswith(".json"):
        return stem_or_path
    return os.path.join(OUT_DIR, f"{stem_or_path}.json")


def _resolve_src(stem: str, override: Optional[str]) -> Optional[str]:
    """Find the matching source image. ``override`` wins; otherwise look
    under ``srcImg/`` for any of ``<stem>.{png,jpg,...}``. Returns None
    if nothing was found (caller must then supply ``--shape``).
    """
    if override:
        return override
    for ext in SRC_EXTS:
        cand = os.path.join(SRC_DIR, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def _parse_shape(s: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """``"HxW"`` -> (H, W, 3). Returns None if ``s`` is None."""
    if not s:
        return None
    try:
        h, w = (int(p) for p in s.lower().split("x"))
    except Exception:
        raise SystemExit(f"--shape expects HxW (e.g. 895x1200), got {s!r}")
    return (h, w, 3)


def _load_lines(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "lines" in data:
        return data["lines"]
    if isinstance(data, list):  # tolerate bare lists from older tooling
        return data
    raise SystemExit(f"{json_path}: cannot find 'lines' key")


def render_one(stem_or_path: str, *,
               src_override: Optional[str] = None,
               out_override: Optional[str] = None,
               shape_override: Optional[Tuple[int, int, int]] = None,
               quiet: bool = False) -> str:
    """Render one preview. Returns the output path."""
    json_path = _resolve_json(stem_or_path)
    if not os.path.exists(json_path):
        raise SystemExit(f"JSON not found: {json_path}")

    stem = os.path.splitext(os.path.basename(json_path))[0]

    if shape_override is not None:
        shape = shape_override
    else:
        src_path = _resolve_src(stem, src_override)
        if src_path is None:
            raise SystemExit(
                f"no source image for stem {stem!r} under {SRC_DIR}/ — "
                f"pass --src PATH or --shape HxW")
        if not os.path.exists(src_path):
            raise SystemExit(f"source image not found: {src_path}")
        bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"failed to decode source image: {src_path}")
        shape = bgr.shape

    out_path = out_override or os.path.join(OUT_DIR, f"{stem}_preview.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines = _load_lines(json_path)
    draw_debug_image(lines, shape, out_path)
    if not quiet:
        print(f"  {json_path}  ->  {out_path}  ({len(lines)} segments)", flush=True)
    return out_path


def _iter_all_stems() -> List[str]:
    """Every JSON under ``output/`` that has a matching srcImg entry."""
    out: List[str] = []
    if not os.path.isdir(OUT_DIR):
        return out
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.endswith(".json"):
            continue
        stem = os.path.splitext(name)[0]
        if _resolve_src(stem, None) is None:
            continue
        out.append(stem)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?",
                    help="stem (e.g. 'source') or path to a JSON; omit with --all")
    ap.add_argument("--src", default=None,
                    help="source image path (default: srcImg/<stem>.<ext>)")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: output/<stem>_preview.png)")
    ap.add_argument("--shape", default=None,
                    help="HxW (e.g. 895x1200) if no source image is available")
    ap.add_argument("--all", action="store_true",
                    help="rebuild every output/*.json with a matching srcImg/ "
                         "(also the default when no target is given)")
    args = ap.parse_args(argv)

    shape_override = _parse_shape(args.shape)

    # No positional target -> implicit --all. This is the path you get
    # from a bare ``py preview.py`` invocation.
    do_all = args.all or args.target is None
    if do_all:
        if args.target is not None:
            ap.error("--all is mutually exclusive with a positional target")
        stems = _iter_all_stems()
        if not stems:
            print(f"no JSON files under {OUT_DIR}/ with matching srcImg/", flush=True)
            return 1
        print(f"Rebuilding {len(stems)} preview(s)...", flush=True)
        for stem in stems:
            render_one(stem, shape_override=shape_override)
        print("Done.", flush=True)
        return 0

    render_one(args.target,
               src_override=args.src,
               out_override=args.out,
               shape_override=shape_override)
    return 0


if __name__ == "__main__":
    sys.exit(main())
