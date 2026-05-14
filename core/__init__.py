"""Pipeline-internal modules used by vectorize.py and the API.

Modules here form the vectorisation core: geometry primitives, candidate
generators, scoring, canonical-line thickness, the audit recorder, and the
master accept-loop utility. Top-level ``vectorize`` and ``api`` import from
this package; research/harness scripts live in ``tools/`` and import these
same modules via ``from core import X``.
"""
