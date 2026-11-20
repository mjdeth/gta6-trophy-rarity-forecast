"""GTA VI trophy rarity forecast: where each trophy's Exophase earn rate lands at the target read.

A rate is earners over tracked players, so the model forecasts the two forces separately
where it can, and blends that with a direct trend of the rate itself. All work is on the
logit scale (rates are percents, clamped to 0.01..99.99 before the logit, as the scorer
clamps them), against x = ln(1 + hours since 2026-11-19T00:00Z).

Live page (the tracked-player count or the earn total moved during the history), two
medians, averaged:
  * Trend: a recency-weighted least-squares line of logit(rate) on x, its slope shrunk by
    its effective sample size and toward the mean slope of the trophy's tier, anchored
    mostly on the last read, extrapolated with the move damped and capped.
  * Count space: the tracked-player count gets its own line on x (one series, shared by
    every trophy, so it is the best-estimated thing in the case) and each trophy's earners
    (rate x players) get a shrunk, tier-blended line of their own; the median rate is
    earners over players at the target. This separates earning, which only grows, from
    dilution, which is the tracked set growing around the earners.
  The spread takes the larger of the two models' spreads and adds half their disagreement,
  so a case the two models read differently is forecast honestly wide.

Stale page (neither the tracked-player count nor the earn total moved in the whole
history, as in the practice cases): the answering read comes from a refreshed page, and a
refresh multiplies the tracked players (the house documents 638 -> 3,868 on launch night,
a typical miss of about 0.68 points for a forecast that stays put, which is what earners
standing still while the denominator grows sixfold produces). The median is computed in
count space with a 6x growth prior and a small catch-up for the new players, who trail the
early trackers; the spread carries that regime uncertainty plus the counting noise of the
expected earners at the target. A trophy already at 100% is pinned at the top instead: the
house's account says the 100% rate did not fall, and at the clamp a point of logit is
worth far more than the dilution it might suffer.

Deterministic: every number comes from the case file, iteration is in file order, and the
only clock read is the case's own timestamps. Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The trend's clock: x = ln(1 + hours since this instant), the house curve baseline's axis.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

PCT_LO = 0.01
PCT_HI = 99.99

# Shared fit tuning.
HALF_LIFE_H = 12.0        # recency weighting of reads in the per-trophy fits
WINDOW_H = 72.0           # reads older than this before the last read do not enter the fits
SLOPE_SHRINK_N = 10.0     # effective sample size at which a slope is half believed
TIER_BLEND = 0.4          # weight of the tier-mean slope in a trophy's final slope
ANCHOR_LAST = 0.7         # how far a fit is pulled onto the last read's own value
MOVE_DAMP = 0.85          # damping on an extrapolated move
MOVE_CAP = 1.0            # hard cap on the trend's extrapolated logit move
E_MOVE_CAP = 1.1          # hard cap on the earners' extrapolated log move
SIGMA_FLOOR = 0.22
SIGMA_CAP = 2.0
STEP_FLOOR = 0.1          # random-walk scale floor, logit per root-day
STEP_CAP = 2.0
PLAYERS_GROWTH_CAP = 3.0  # the tracked set at most triples over the horizon
PLAYERS_SHRINK_FLOOR = 0.9

# Stale-branch tuning (the refresh prior).
STALE_GROWTH = 6.0        # tracked players multiply by about this at a refresh (638 -> 3,868)
STALE_SIGMA = 0.5         # regime uncertainty of the dilution, in logit units
TOP_PIN_SIGMA = 0.35      # spread for a trophy already at 100%
BOTTOM_PIN_SIGMA = 0.6    # spread for a trophy still at 0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def x_of(moment):
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def clamp_pct(p):
    return min(PCT_HI, max(PCT_LO, p))


def logit_pct(p):
    p = clamp_pct(p)
    return math.log(p / (100.0 - p))


def pct_of_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the scorer's convention), integral values as ints."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(scaled) if float(scaled).is_integer() else scaled


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, weighted residual RMS, n_eff)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx
    resid = math.sqrt(sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    n_eff = sw * sw / sum(w * w for w in ws)
    return intercept, slope, resid, n_eff


def recency_weights(moments, t_last):
    return [0.5 ** ((t_last - t).total_seconds() / 3600.0 / HALF_LIFE_H) for t in moments]


def shrunk_slopes(fits, tiers):
    """Per-index final slopes: own slope shrunk by n_eff, blended with the tier mean."""
    def shrunk(fit):
        return fit[1] * (fit[3] / (fit[3] + SLOPE_SHRINK_N)) if fit is not None else None

    tier_mean = {}
    for tier in set(tiers):
        members = [shrunk(fits[i]) for i, t in enumerate(tiers) if t == tier and fits[i] is not None]
        tier_mean[tier] = sum(members) / len(members) if members else 0.0
    present = [v for v in tier_mean.values() if v != 0.0]
    global_mean = sum(present) / len(present) if present else 0.0
    out = []
    for i, tier in enumerate(tiers):
        own = shrunk(fits[i])
        tier = tier_mean.get(tier, global_mean)
        out.append((1.0 - TIER_BLEND) * own + TIER_BLEND * tier if own is not None else tier)
    return out


def step_sigma(series):
    """Random-walk scale (logit per root-day) from consecutive-read logit steps, all trophies."""
    steps = []
    for points in series:
        for (t0, r0), (t1, r1) in zip(points, points[1:]):
            days = (t1 - t0).total_seconds() / 86400.0
            if days > 0:
                steps.append((logit_pct(r1) - logit_pct(r0)) ** 2 / days)
    if not steps:
        return STEP_FLOOR
    steps.sort()
    return min(STEP_CAP, max(STEP_FLOOR, math.sqrt(steps[len(steps) // 2])))


def is_stale(case):
    players = {read["players_tracked"] for read in case["history"]}
    earned = {read["trophies_earned"] for read in case["history"]}
    return len(players) == 1 and len(earned) == 1


def stale_forecast(rate_pct, players):
    """Median logit and sigma for one trophy when the page never moved in the history."""
    if rate_pct >= PCT_HI:
        return logit_pct(PCT_HI), TOP_PIN_SIGMA
    if rate_pct <= PCT_LO:
        return logit_pct(PCT_LO), BOTTOM_PIN_SIGMA
    p = rate_pct / 100.0
    # The new players' earn fraction for this trophy. On the documented launch-night
    # refresh the earners all but stood still while the tracked set grew sixfold, so the
    # catch-up is small; it rises with the incumbent rate, since the trophies everyone
    # has are the ones a new player picks up first.
    catch_up = p * (0.06 + 0.55 * p)
    p_median = (p + (STALE_GROWTH - 1.0) * catch_up) / STALE_GROWTH
    earners = max(p_median * players * STALE_GROWTH, 2.0)
    count_var = min(0.30, 1.0 / earners)
    return logit_pct(p_median * 100.0), math.sqrt(STALE_SIGMA ** 2 + count_var)


def forecast_case(case):
    levels = case["levels"]
    zs = [NormalDist().inv_cdf(tau) for tau in levels]
    trophies = case["trophies"]
    history = case["history"]

    if not history:
        flat = [round4(pct_of_logit(logit_pct(1.0) + z)) for z in zs]
        return {"trophies": [{"id": t["id"], "q": flat} for t in trophies]}

    moments = [parse_time(read["at"]) for read in history]
    t_last = moments[-1]
    target_moment = parse_time(case["target"])
    target_x = x_of(target_moment)
    horizon_days = max((target_moment - t_last).total_seconds() / 86400.0, 0.02)
    players_last = float(history[-1]["players_tracked"])
    stale = is_stale(case)

    # Per-trophy (moment, rate) series with nulls dropped, and last rates.
    series = [[] for _ in trophies]
    for read, moment in zip(history, moments):
        for i, rate in enumerate(read["rates"]):
            if rate is not None:
                series[i].append((moment, float(rate)))
    lasts = [pts[-1][1] if pts else None for pts in series]
    known = sorted(v for v in lasts if v is not None)
    fallback_rate = known[len(known) // 2] if known else 1.0
    lasts = [v if v is not None else fallback_rate for v in lasts]

    if stale:
        rows = []
        for i, trophy in enumerate(trophies):
            median, sigma = stale_forecast(lasts[i], players_last)
            rows.append({"id": trophy["id"],
                         "q": [round4(pct_of_logit(median + z * sigma)) for z in zs]})
        return {"trophies": rows}

    rw = step_sigma(series)
    tiers = [t["tier"] for t in trophies]
    x_last = x_of(t_last)

    def fit_series(points, values):
        """Recency-weighted line of values on x over the recent window, or None."""
        if len(points) < 3:
            return None
        window = [(t, v) for (t, _), v in zip(points, values)
                  if (t_last - t).total_seconds() <= WINDOW_H * 3600.0]
        if len(window) < 3:
            window = list(zip([t for t, _ in points], values))[-3:]
        xs = [x_of(t) for t, _ in window]
        ys = [v for _, v in window]
        ws = recency_weights([t for t, _ in window], t_last)
        return weighted_line(xs, ys, ws)

    # Trend model fits: logit(rate) on x.
    trend_fits = [fit_series(pts, [logit_pct(r) for _, r in pts]) for pts in series]
    trend_slopes = shrunk_slopes(trend_fits, tiers)

    # Count-space fits: log earners on x, and the shared log players line.
    players_at = {moment: float(read["players_tracked"]) for read, moment in zip(history, moments)}
    earner_fits = []
    for pts in series:
        values = [math.log(max(r / 100.0 * players_at[t], 0.25)) for t, r in pts]
        earner_fits.append(fit_series(pts, values))
    earner_slopes = shrunk_slopes(earner_fits, tiers)

    n_fit = weighted_line([x_of(t) for t in moments],
                          [math.log(max(players_at[t], 1.0)) for t in moments],
                          recency_weights(moments, t_last))
    if n_fit is not None:
        log_players_target = n_fit[0] + n_fit[1] * target_x
        players_target = math.exp(log_players_target)
        ratio = players_target / max(players_last, 1.0)
        ratio = min(PLAYERS_GROWTH_CAP, max(PLAYERS_SHRINK_FLOOR, ratio))
        players_target = players_last * ratio
        n_sigma = min(n_fit[2], 0.3)
    else:
        players_target = players_last
        n_sigma = 0.1

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = lasts[i]
        pts = series[i] if series[i] else [(t_last, last_rate)]
        p_last = clamp_pct(last_rate) / 100.0

        # Trend median and spread.
        fit = trend_fits[i]
        if fit is None:
            trend_median = logit_pct(last_rate)
            trend_sigma = math.sqrt(rw * rw * horizon_days + 0.15 ** 2)
        else:
            fitted = fit[0] + fit[1] * x_of(pts[-1][0])
            base = fitted + ANCHOR_LAST * (logit_pct(last_rate) - fitted)
            move = min(MOVE_CAP, max(-MOVE_CAP, trend_slopes[i] * (target_x - x_of(pts[-1][0])) * MOVE_DAMP))
            trend_median = base + move
            trend_sigma = math.sqrt(0.5 * min(fit[2], 1.2) ** 2 + rw * rw * horizon_days + 0.15 ** 2)

        # Count-space median and spread.
        efit = earner_fits[i]
        earners_last = max(last_rate / 100.0 * players_last, 0.25)
        if efit is None:
            count_median = trend_median
            count_sigma = trend_sigma
        else:
            fitted_e = efit[0] + efit[1] * x_of(pts[-1][0])
            base_e = fitted_e + ANCHOR_LAST * (math.log(earners_last) - fitted_e)
            move_e = min(E_MOVE_CAP, max(-E_MOVE_CAP,
                                         earner_slopes[i] * (target_x - x_of(pts[-1][0])) * MOVE_DAMP))
            earners_target = math.exp(base_e + move_e)
            earners_target = min(max(earners_target, earners_last * 0.8), players_target)
            rate_cs = clamp_pct(100.0 * earners_target / max(players_target, 1.0))
            count_median = logit_pct(rate_cs)
            resid_logit = min(efit[2], 1.2) / max(1.0 - p_last, 0.05)
            count_sigma = math.sqrt(0.5 * resid_logit ** 2 + rw * rw * horizon_days
                                    + n_sigma ** 2 + 0.15 ** 2)

        median = 0.5 * (trend_median + count_median)
        sigma = math.sqrt(max(trend_sigma, count_sigma) ** 2
                          + (0.5 * (trend_median - count_median)) ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        rows.append({"id": trophy["id"],
                     "q": [round4(pct_of_logit(median + z * sigma)) for z in zs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
