# Regression Harness

Safety net for vectorize.py refactoring. Compares current pipeline output
against a frozen baseline across 3 reference images and reports a status of
`PASS` / `WARNING` / `FAIL`.

This harness does **not** decide whether the result is *good*. It only
detects whether a code change made the output diverge from what was previously
accepted as the baseline.

## Quick start

```bash
# First time only — interactively create the baselines for every case.
python regression.py --update-baseline

# Day-to-day: just run the harness. Exits non-zero if anything FAILs.
python regression.py

# Limit to one case.
python regression.py --case source
```

## Directory layout

```
tests/
  cases/
    <case>/manifest.json          # {"input_image": "<relative path>"}
  baseline/
    <case>/
      lines.json                  # frozen pipeline output (raw)
      hash.txt                    # sha1 of normalized lines
      metrics.json                # stroke_width, graph stats, etc.
      masks/{wall,window,door}_{thin,normal,loose}.png
  current/
    <case>/                       # rewritten every regression run
      lines.json
      hash.txt
      metrics.json
      report.json
      diff_overlay.png            # 3-color overlay vs baseline
      masks/...
```

The three cases shipped with the repo:

| Case               | Image                                                | Size       |
|--------------------|------------------------------------------------------|------------|
| `source`           | `srcImg/source.png`                                  | 1200×895   |
| `sg2`              | `srcImg/sg2.png`                                     | 2048×2048  |
| `Gemini_Generated` | `srcImg/Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png` | 2586×1664  |

## Status definitions

| Status    | Meaning                                                                              |
|-----------|--------------------------------------------------------------------------------------|
| `PASS`    | Output is identical or close enough that every threshold is met.                     |
| `WARNING` | A soft check tripped — output drifted but not enough to call broken. Investigate.    |
| `FAIL`    | A hard threshold was exceeded. Revert first, understand why, then decide what to do. |

If hash matches the baseline, status is `PASS` immediately (no IOU/distance
math needed).

## Thresholds

IOU is computed against the baseline mask at three rasterization thicknesses,
all derived from the baseline's stroke_width (the median wall thickness
measured at baseline-creation time, frozen ever since):

| Level    | Thickness (px)                       | Purpose                                    |
|----------|--------------------------------------|--------------------------------------------|
| `thin`   | `max(1, round(0.2 × stroke_width))`  | Most sensitive — sub-pixel drift = WARNING |
| `normal` | `max(2, round(0.4 × stroke_width))`  | Main FAIL threshold                        |
| `loose`  | `max(3, round(0.8 × stroke_width))`  | Detects only gross IOU regressions         |

IOU is reported as a **drop** from baseline (`baseline_iou(=1.0) − current_iou`):

| Type   | normal FAIL | loose FAIL | thin WARN |
|--------|-------------|------------|-----------|
| wall   | 0.010       | 0.004      | 0.030     |
| window | 0.020       | 0.008      | 0.050     |
| door   | 0.020       | 0.008      | 0.050     |

Other hard FAIL checks:

- Symmetric distance (current mask ↔ baseline mask):
  - mean > `max(1.0, 0.12 × stroke_width)` → FAIL
  - p95  > `max(2.0, 0.30 × stroke_width)` → FAIL
- `free_endpoints` increased by more than `5` → FAIL
- `num_segments` changed by more than `15%` → FAIL
- Total length (summed across types) changed by more than `10%` → FAIL

Soft WARNING checks:

- `num_short_segments` (length < 10 px) grew
- Hash changed but every IOU is still within thresholds

## `--update-baseline` — by design, slow

Updating the baseline is **intentionally noisy**. Running regression is fast;
changing the answer key is slow.

- First-time baseline creation: must type `CREATE <case>`.
- Update after `PASS` or `WARNING`: `y/N` prompt, default **No**.
- Update after `FAIL`: must type `UPDATE <case>` verbatim.
- `--all`: must also type `UPDATE ALL` up front.
- `--yes`: bypasses every prompt. Logs `baseline_updated_by: manual_yes_flag`
  + timestamp into `metrics.json`. Use sparingly.

## Output

The `diff_overlay.png` written under `tests/current/<case>/`:

- **Red** — pixels present in baseline but missing in current (lost geometry)
- **Green** — pixels present in current but missing in baseline (new geometry)
- **Black** — overlap
- White background

It is built from a combined wall+window+door rasterization at `normal`
thickness. The path is printed to stdout but the harness does not try to open
it (so it works on headless / CI / remote machines).

## When to update the baseline

The threshold is **a quality floor**, not the answer key. If regression
catches a drift you actually wanted:

1. First, `git revert` and re-run regression on master to confirm baseline
   itself is stable.
2. Eyeball `diff_overlay.png` for every changed case — count by count is not
   enough.
3. Only then re-apply your change and `--update-baseline`.

If you're tempted to relax a threshold to make regression pass, stop and ask
whether you're hiding a real regression. The thresholds were chosen on real
images — they aren't arbitrary.
