"""Score-driven accept loop drawing candidates from multiple generators.

The architectural target for step 23 (todo.md §6): replace the linear
17-pass cascade in vectorize_bgr with a single loop where the order
of acceptance is decided by score rather than by hand-tuned pass
sequence. Each iteration regenerates candidates from every registered
generator against current state, scores them, and accepts the best
one whose delta clears the gate.

This module ships the loop as a *pure utility*: it has no special
knowledge of the floorplan pipeline beyond the (segments, candidate)
data contract. The vectorize_bgr cascade does not call it yet —
step 23 phase 1 is infrastructure-only. Future phases will gradually
migrate stage groups (post-Manhattan topology, topology completion)
once step 22 produces continuous score signals strong enough to gate
the passes that currently use ``skip_score=True``.

Use it directly:

    from core.master_loop import master_accept_loop

    state = master_accept_loop(
        initial_segments,
        generators=[
            lambda segs: t_project_candidates(segs, seg_tols),
            lambda segs: bridge_candidates(segs, wall_mask=wm, max_radius=40),
            lambda segs: trunk_split_candidates(segs),
        ],
        score_fn=lambda segs: compute_score(segs,
                                            wall_evidence=ev,
                                            door_mask=dm,
                                            window_mask=wnm,
                                            wall_mask=wm),
    )

The loop terminates when no generator produces a candidate that
improves the score (or when ``max_iterations`` is reached).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from . import candidates as C


# Candidate stream contract: a generator is any callable that takes the
# current segment list and returns a list of Candidate objects. The
# caller bakes generator-specific args via closure / functools.partial
# so the master loop only ever sees this shape.
GeneratorFn = Callable[[List[Dict]], List[C.Candidate]]

# Score function contract: takes the segment list, returns any object
# with a ``total: float`` field. compute_score's PipelineScore matches;
# tests may pass simpler stubs.
ScoreFn = Callable[[List[Dict]], "ScoreLike"]


class ScoreLike:
    total: float  # noqa: A003


def master_accept_loop(
        segments: Sequence[Dict],
        *,
        generators: Sequence[GeneratorFn],
        score_fn: ScoreFn,
        accept_delta_min: float = 0.0,
        skip_score: bool = False,
        max_iterations: int = 100,
        audit_recorder=None,
) -> List[Dict]:
    """Iteratively accept the best candidate across all generators.

    Algorithm (per iteration):
      1. Gather candidates from every generator against current state.
      2. If none, terminate.
      3. Compute base score once.
      4. For each candidate, compute trial-state score and delta.
      5. Pick the candidate with the largest delta.
      6. If ``skip_score`` or best_delta meets ``accept_delta_min``,
         apply it and continue; otherwise terminate.

    Each iteration regenerates ALL candidates against the post-accept
    state. This is the safest contract: a generator that produces an
    invalid candidate in iteration N (because its inputs were stale)
    is safe because it re-emits in iteration N+1 with fresh state.

    Args:
        segments: starting segment list (not mutated; loop works on copies).
        generators: callables that return candidate lists. Order matters
            ONLY for tie-breaking — score determines the actual accept
            order. Empty list → no-op (returns ``list(segments)``).
        score_fn: scoring callable; must return an object with ``.total``.
        accept_delta_min: minimum delta to accept. ``0.0`` (default) =
            tie-accept (matches step 16's ``_accept_merge_candidates``
            policy). Use a small positive value for strict policy.
        skip_score: when True, every iteration applies the
            first-generator-listed candidate regardless of delta (the
            score is still computed and audited but doesn't gate). This
            mirrors today's ``skip_score=True`` behaviour for
            topology-recovery passes; the goal of step 22 is to let
            ``skip_score=False`` become correct everywhere.
        max_iterations: hard cap (defensive — score-driven termination
            should fire much earlier).
        audit_recorder: optional ``audit.AuditRecorder`` for delta logs.

    Returns:
        New segments list. Input is not mutated.
    """
    if not generators:
        return list(segments)

    current = list(segments)
    for it in range(max_iterations):
        cands: List[C.Candidate] = []
        for gen in generators:
            cands.extend(gen(current))
        if not cands:
            break

        base_score = score_fn(current)

        best_cand: Optional[C.Candidate] = None
        best_delta = float("-inf")
        best_trial: Optional[List[Dict]] = None
        best_trial_score = None

        for cand in cands:
            trial = C.apply_candidate(current, cand)
            trial_score = score_fn(trial)
            delta = trial_score.total - base_score.total
            if delta > best_delta:
                best_delta = delta
                best_cand = cand
                best_trial = trial
                best_trial_score = trial_score

        if best_cand is None:
            break

        accept = skip_score or best_delta >= accept_delta_min

        if audit_recorder is not None:
            try:
                from .audit import candidate_position
                pos = candidate_position(best_cand)
            except Exception:
                pos = None
            delta_terms = {}
            if best_trial_score is not None and hasattr(best_trial_score, "terms"):
                base_terms = getattr(base_score, "terms", {})
                trial_terms = getattr(best_trial_score, "terms", {})
                delta_terms = {k: trial_terms.get(k, 0.0) - base_terms.get(k, 0.0)
                               for k in set(trial_terms) | set(base_terms)}
            audit_recorder.record(
                op=best_cand.op,
                accepted=accept,
                delta=best_delta,
                delta_terms=delta_terms,
                meta=best_cand.meta,
                reason="master_skip_score" if skip_score else "master_score_gate",
                position=pos,
            )

        if not accept:
            break

        current = best_trial

    return current
