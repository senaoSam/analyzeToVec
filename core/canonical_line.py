"""Step 4.9: local thickness sampling (canonical-line clustering moved out).

This module used to host ``canonicalize_offsets``, the per-(type, axis)
bucketed offset clusterer; step 9 phase 3 moved the cluster + mutate
logic into ``generators.canonicalize_offset_candidates`` (the candidate-
generator side) and ``vectorize._accept_canonicalize_offset_candidates``
(the apply wrapper, which also handles the ``attach_thickness`` side
channel). What remains here is the single helper
``compute_local_thickness``, used by both the new wrapper and by other
thickness-aware tol consumers across the pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_local_thickness(seg: Dict,
                            dist_transform: Optional[np.ndarray]) -> float:
    """Median ``2 * dt`` sampled along ``seg``'s body. Distance-transform
    values at or below 0.5 (background / junction interior) are dropped
    before the median so they don't pull thick walls down at junction
    holes. Returns ``-1.0`` when sampling is impossible or too sparse.
    """
    if dist_transform is None or dist_transform.size == 0:
        return -1.0
    h, w = dist_transform.shape
    x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
    seg_len = max(abs(x2 - x1), abs(y2 - y1))
    if seg_len <= 0:
        return -1.0
    n = max(3, int(round(seg_len)))
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    samples = dist_transform[yi, xi]
    interior = samples[samples > 0.5]
    if interior.size < max(3, n // 4):
        return -1.0
    return float(2.0 * np.median(interior))


# Legacy ``canonicalize_offsets`` + its private helpers ``_axis`` /
# ``_length`` / ``_adaptive_tol`` / ``_length_weighted_median`` were
# deleted in step 9 phase 4 (dead code after phase 3 migrated its
# call site to ``_accept_canonicalize_offset_candidates``, which lives
# in vectorize.py and calls ``generators.canonicalize_offset_candidates``
# for the cluster/mutation logic; this module now exposes only
# ``compute_local_thickness``, which the wrapper still uses for the
# per-segment thickness sampling).
