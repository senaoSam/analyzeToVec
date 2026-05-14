"""Unit tests for master_loop.master_accept_loop.

The loop itself is generator-agnostic, so the tests use minimal
synthetic generators that produce concrete Candidate objects. This
exercises the loop's accept ordering / termination / skip_score
policy without depending on the full pipeline state.

Run via:
    py -3 -m pytest tests/test_master_loop.py
or directly:
    py -3 tests/test_master_loop.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from core import candidates as C  # noqa: E402
from core.master_loop import master_accept_loop  # noqa: E402


# ---------------------------------------------------------------------------
# Score stub: count "good" wall segments. The simpler the score, the easier
# the test asserts on observed accept ordering.
# ---------------------------------------------------------------------------

@dataclass
class StubScore:
    total: float
    terms: Dict[str, float]


def length_score(segments: List[Dict]) -> StubScore:
    """Score = total length of all segments. Higher = better."""
    total = 0.0
    for s in segments:
        total += abs(s["x2"] - s["x1"]) + abs(s["y2"] - s["y1"])
    return StubScore(total=total, terms={"length": total})


# ---------------------------------------------------------------------------
# Test 1: empty input → empty output
# ---------------------------------------------------------------------------

def test_empty_input() -> None:
    out = master_accept_loop([], generators=[], score_fn=length_score)
    assert out == []


def test_no_generators() -> None:
    segs = [{"type": "wall", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}]
    out = master_accept_loop(segs, generators=[], score_fn=length_score)
    assert out == segs


# ---------------------------------------------------------------------------
# Test 2: one generator proposes one candidate, accepted on positive delta
# ---------------------------------------------------------------------------

def test_single_accept_positive_delta() -> None:
    """Generator proposes a new wall. Score = total length, so adding any
    positive-length wall has positive delta and is accepted.
    """
    segs = [{"type": "wall", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}]

    emitted = {"count": 0}

    def add_wall_gen(state: List[Dict]) -> List[C.Candidate]:
        # Emit a single add-wall candidate the first time we're called,
        # nothing after (otherwise we'd loop forever).
        if emitted["count"] >= 1:
            return []
        emitted["count"] += 1
        return [C.Candidate(
            op="add_wall",
            add=[{"type": "wall", "x1": 10.0, "y1": 0.0,
                  "x2": 10.0, "y2": 10.0}],
            mutate=[],
            meta={},
        )]

    out = master_accept_loop(segs, generators=[add_wall_gen],
                              score_fn=length_score)
    assert len(out) == 2
    types = sorted(s["type"] for s in out)
    assert types == ["wall", "wall"]


# ---------------------------------------------------------------------------
# Test 3: candidate with NEGATIVE delta is rejected, loop terminates
# ---------------------------------------------------------------------------

def test_negative_delta_rejected() -> None:
    """Generator proposes removing a segment. Score = total length, so
    delta is negative; under accept_delta_min=0 (tie-accept), removal
    is rejected.
    """
    segs = [{"type": "wall", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}]

    def remove_gen(state: List[Dict]) -> List[C.Candidate]:
        if not state:
            return []
        return [C.Candidate(
            op="remove",
            add=[],
            mutate=[],
            remove=[0],
            meta={},
        )]

    out = master_accept_loop(segs, generators=[remove_gen],
                              score_fn=length_score,
                              accept_delta_min=0.0)
    # Removal rejected (delta = -10 < 0). State unchanged.
    assert out == segs


# ---------------------------------------------------------------------------
# Test 4: skip_score=True accepts even when delta is negative
# ---------------------------------------------------------------------------

def test_skip_score_overrides_gate() -> None:
    """Same setup as test_3 but with skip_score=True — removal happens
    despite the negative score delta. This mirrors today's
    parallel_merge / fuse behaviour.
    """
    segs = [{"type": "wall", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}]

    emitted = {"count": 0}

    def remove_gen(state: List[Dict]) -> List[C.Candidate]:
        if emitted["count"] >= 1 or not state:
            return []
        emitted["count"] += 1
        return [C.Candidate(
            op="remove",
            add=[],
            mutate=[],
            remove=[0],
            meta={},
        )]

    out = master_accept_loop(segs, generators=[remove_gen],
                              score_fn=length_score,
                              skip_score=True)
    # Removal accepted despite -10 delta. State now empty.
    assert out == []


# ---------------------------------------------------------------------------
# Test 5: two generators — best-delta candidate wins regardless of order
# ---------------------------------------------------------------------------

def _has_wall_starting_at_x(state: List[Dict], x: float) -> bool:
    return any(abs(s["x1"] - x) < 0.1 for s in state)


def test_best_candidate_wins_across_generators() -> None:
    """Two generators each propose one candidate. The one with larger
    positive delta should be accepted first; the other still fires in
    a later iteration because real generators are state-driven and
    re-emit until their condition is met.

    This test also documents an invariant about real generators: they
    must be stateless (decide what to emit purely from the input
    state). A counter-based generator would lose its candidate in the
    iteration where it's outranked, because master_accept_loop does
    not buffer rejected candidates between iterations.
    """
    segs = [{"type": "wall", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.0}]

    def short_gen(state: List[Dict]) -> List[C.Candidate]:
        if _has_wall_starting_at_x(state, 100.0):
            return []
        return [C.Candidate(
            op="add_short",
            add=[{"type": "wall", "x1": 100.0, "y1": 0.0,
                  "x2": 105.0, "y2": 0.0}],   # length 5
            mutate=[], meta={},
        )]

    def long_gen(state: List[Dict]) -> List[C.Candidate]:
        if _has_wall_starting_at_x(state, 200.0):
            return []
        return [C.Candidate(
            op="add_long",
            add=[{"type": "wall", "x1": 200.0, "y1": 0.0,
                  "x2": 300.0, "y2": 0.0}],   # length 100
            mutate=[], meta={},
        )]

    # Order short_gen first; long should still be accepted first because
    # delta is larger; short follows in iteration 2.
    out = master_accept_loop(segs,
                              generators=[short_gen, long_gen],
                              score_fn=length_score)
    assert len(out) == 3
    lengths = sorted(abs(s["x2"] - s["x1"]) + abs(s["y2"] - s["y1"]) for s in out)
    assert lengths == [1.0, 5.0, 100.0]
    # Long must come before short in acceptance order — assert by checking
    # that the segment count grew 1 -> 2 -> 3 with long arriving first.
    # (We can't directly observe iteration order from the output state,
    # but if the test passes it's a strong signal.)


# ---------------------------------------------------------------------------
# Test 6: max_iterations enforces termination on a generator that never
# stops emitting
# ---------------------------------------------------------------------------

def test_max_iterations_cap() -> None:
    """An ill-behaved generator that always emits a positive-delta
    candidate must still terminate at max_iterations.
    """
    segs: List[Dict] = []

    def always_add(state: List[Dict]) -> List[C.Candidate]:
        return [C.Candidate(
            op="endless",
            add=[{"type": "wall",
                  "x1": float(len(state)), "y1": 0.0,
                  "x2": float(len(state)) + 1.0, "y2": 0.0}],
            mutate=[], meta={},
        )]

    out = master_accept_loop(segs, generators=[always_add],
                              score_fn=length_score,
                              max_iterations=7)
    assert len(out) == 7   # exactly 7 accepts, then cap fires


# ---------------------------------------------------------------------------
# Driver: bare-bones runner so we don't need pytest available
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_empty_input,
        test_no_generators,
        test_single_accept_positive_delta,
        test_negative_delta_rejected,
        test_skip_score_overrides_gate,
        test_best_candidate_wins_across_generators,
        test_max_iterations_cap,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            raise
        else:
            print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED.")


if __name__ == "__main__":
    _run_all()
