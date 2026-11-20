# GTA VI trophy rarity forecast

`main.py` (Python 3.13, standard library only) reads each case in the input directory —
every read the house made of Exophase's GTA VI PS5 trophy page before the checkpoint —
and writes, per trophy, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its earn rate at
the case's target. It is deterministic: two runs of a case are byte-identical.

## The model

A rate is earners over tracked players, and the two move differently: earners only grow,
while the tracked set grows in jumps (a page refresh) and dilutes every rate it touches.
The forecast is built per trophy on the logit scale — rates clamped to 0.01..99.99 first,
exactly as the scorer clamps them — against x = ln(1 + hours since 2026-11-19T00:00Z).

**Live page** (the tracked-player count or the earn total moved during the history). Two
medians are averaged:

- *Trend.* A recency-weighted (12 h half-life, 72 h window) least-squares line of
  logit(rate) on x. The slope is shrunk by its effective sample size (half believed at
  n_eff = 10) and blended 60/40 with the mean shrunk slope of the trophy's tier, because
  one trophy's slope is mostly noise and its family's is a better prior. The line is
  anchored 70% onto the last read, and the extrapolated move is damped by 0.85 and capped
  at 1 logit.
- *Count space.* The tracked-player count gets one shared line of log(players) on x —
  the best-estimated series in the case — and each trophy's earners (rate × players) get
  a shrunk, tier-blended log line of their own, floored so earners never fall below 80%
  of their last count and capped by the player forecast. The median rate is earners over
  players at the target, with the player growth over the horizon capped at 3× (floor
  0.9×).

The spread takes the larger of the two models' spreads — each combines its fit residual
with a random-walk scale estimated from consecutive-read logit steps across all trophies
(median squared step per root-day, floored at 0.1) grown over the horizon — and adds half
the two medians' disagreement in quadrature. Total sigma is floored at 0.22 and capped
at 2.

**Stale page** (neither the tracked-player count nor the earn total moved in the whole
history, as in both practice cases). The answering read comes from a refreshed page. The
median is computed in count space: the tracked set is assumed to multiply by 6 (the house
documents 638 → 3,868 on launch night), and the new players' earn fraction for a trophy
is its incumbent rate times a catch-up of 0.06 + 0.55 × rate — small, because the house's
own account of the practice (the stay-put baseline's typical miss was about 0.68 points)
is what earners standing still while the denominator grows sixfold produces. The spread
is a 0.5-logit regime term plus the counting noise of the expected earners at the target
(1/E′, capped). Two pins: a trophy already at 100% keeps a median at the clamp with a
0.35 spread (the house says the 100% rate did not fall, and at the clamp a logit point is
worth far more than any dilution it could suffer), and a trophy still at 0 gets the
mirror treatment with a 0.6 spread.

Edge cases: a trophy no read ever listed falls back to the case's median last rate; an
empty history yields a flat 1% forecast; quantiles are rounded to 4 decimals, half up.

## Validation (local; the server alone issues real scores)

- Both practice cases: schema exact (55 rows, ids in order, quantiles non-decreasing,
  ≤ 65,536 bytes), two runs byte-identical, ~0.1 s per run against the 8 s limit.
- Proxy practice evaluation, clearly labelled: proxy answers were built from the
  documented refresh (638 → 3,868 players, the 100% rate held) over a family of catch-up
  assumptions, and scored with the official pinball rule (2 × max(τd, (τ−1)d) on the
  logit scale). Across the family the entry scores 0.15–0.53 against the house
  baseline's 0.88–1.69; it loses only in the two worlds the documented facts exclude (a
  catch-up of 0.5 costs 0.53 vs 0.45, and 0.9 — which would have left the baseline's
  typical miss near 0.1 points, not the documented 0.68 — costs 0.96 vs 0.04). In the
  worlds matching the documented miss (catch-up 0–0.1) it scores 0.15–0.18 against the
  baseline's documented ~1.51.
- A synthetic live case (110 reads, checkpoint 2026-11-21, target 2026-11-26, truth from
  known generating formulas — directional only): the entry scores 0.080 against the
  stay-put baseline's 0.221 and the curve baseline's 0.218.
- A wider stale spread (0.6) was tried and reverted: it cost ~0.017 in the likely worlds
  to buy 0.006 in the unlikely one.

## Assumptions worth knowing

- The final case's history is expected to be live (it runs to the 21 Nov deadline), so
  the live branch carries the prize; the stale branch is calibrated to the practice
  regime and stands as insurance for flat stretches.
- Earners are treated as near-monotone (floor 0.8×) and player growth as decelerating in
  log-time; both are extrapolations of the launch-week recording, not knowledge of the
  answering read.
