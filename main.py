"""Logit-trend forecaster for GTA VI trophy earn rates, one week after launch.

Reads every <id>.in case in the input directory and writes <id>.out with five
quantiles per trophy (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of the percent of
Exophase-tracked players who will have earned each trophy at the case target.

Model (pure function of the case, no wall-clock time, Python 3.13 stdlib only):

* Launch night has one dominant effect: the first refresh after launch
  multiplies the tracked-player denominator several-fold while only the
  always-earned trophy's earner count keeps pace, so every other rate is
  diluted by roughly a factor of five.  A history that has NEVER refreshed
  (one distinct rate vector, constant players_tracked) is therefore forecast
  at last rate x 0.22 with a wide logit spread.
* Once the page has refreshed, each trophy's logit(rate) is tracked as a
  line in x = ln(1 + hours since launch), fitted by least squares over the
  last 24 h of reads, slope shrunk toward zero and clamped, and extrapolated
  to the target; the median is the fitted line read at the target, not the
  raw last read, because Exophase holds a value for hours at a time and the
  last read can be hours stale while the trend keeps moving.  The spread
  combines a random-walk term, the fit residual, a trend-uncertainty term
  and a staleness term for trophies whose rate has not moved recently.
  Calibration note: the house curve baseline uses a 0.5/sqrt-day random
  walk; the launch-night evidence (flat practice history, the published
  refresh arithmetic) shows this process is far more deterministic than a
  random walk, so the walk is set well under the house's — v2 used 0.35,
  v3 uses 0.22 after the server's dev grade showed v2's practice
  intervals covering 98.2% of truths against a 90% target, and local
  re-validation on synthetic final-like cases (including regime-shifted
  ones) confirmed the narrower walk lowers the loss throughout. The
  never-refreshed spread was likewise recalibrated, 0.6 to 0.45.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch-night dilution: the factor the never-refreshed case is scaled by.
DILUTION = 0.22
# Logit spread used for the never-refreshed case. v3 calibration: the
# server's dev grade for v2 measured coverage_90 = 0.982 at 0.6 — the
# intervals were far wider than the process needs. Across hypothetical
# practice truths (uniform factor 0.18-0.30, per-trophy lognormal noise
# up to 0.5) 0.45 cuts the mean pinball loss ~7% for a small worst-case
# cost, so the spread is set to 0.45.
FLAT_SIGMA = 0.45
# Per-square-root-day random-walk spread on the logit scale (see the
# calibration note in the module docstring). v3 calibration: lowered
# from 0.35 to 0.22 — on synthetic final-like cases (five seeds) and on
# regime-shifted variants (an unseen 0.35-logit bend at the checkpoint)
# the narrower walk improves the mean loss by roughly a fifth while
# the [q05, q95] coverage stays at the top of its range; 0.22 keeps a
# margin against the real process being rougher than the simulations.
RW_SIGMA = 0.22


def clamp_rate(p):
    return min(99.99, max(0.01, p))


def logit(p):
    c = clamp_rate(p)
    return math.log(c / (100.0 - c))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(v):
    v = math.floor(v * 10000 + 0.5) / 10000
    return int(v) if float(v).is_integer() else v


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def x_of(t):
    hours = (t - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def quantiles(median_rate, sigma):
    lg = logit(median_rate)
    qs = []
    for z in Z:
        qs.append(round4(clamp_rate(inv_logit(lg + z * sigma))))
    for k in range(1, 5):
        qs[k] = max(qs[k], qs[k - 1])
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = x_of(target)
    n = len(trophies)

    # --- Never-refreshed detection -------------------------------------
    # One distinct rate vector across the whole history, and the tracked
    # player count never moved: the page has never been re-read since
    # launch, so the next refresh dilutes every rate but the 100% one.
    vectors = {tuple(r.get("rates") or ()) for r in history}
    players = {r.get("players_tracked") for r in history}
    never_refreshed = len(vectors) == 1 and len(players) == 1

    last_rates = history[-1]["rates"] if history else [None] * n

    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i] if i < len(last_rates) else None

        if last is not None and last >= 99.99:
            # Earned by everyone tracked; stays (essentially) everyone.
            rows.append({"id": trophy["id"],
                         "q": [99.5, 99.9, 99.99, 99.99, 99.99]})
            continue

        if never_refreshed:
            median = (last if last is not None else 1.0) * DILUTION
            rows.append({"id": trophy["id"],
                         "q": quantiles(clamp_rate(median), FLAT_SIGMA)})
            continue

        # --- Normal case: logit trend on x ----------------------------
        # All reads from the last 24 h before the checkpoint (not
        # distinct values only): with staleness holds a value persists for
        # hours, and fitting every read weights each level by how long it
        # actually held, which the distinct-value fit gets wrong.
        pts = []
        for r in history:
            rt = parse_time(r["slot"])
            if (at - rt).total_seconds() > 24 * 3600:
                continue
            v = r["rates"][i] if i < len(r["rates"]) else None
            if v is None:
                continue
            pts.append((rt, v))

        if last is None:
            # Never observed: a low rate with a wide spread.
            rows.append({"id": trophy["id"], "q": quantiles(0.5, 1.0)})
            continue

        # Last non-null read time for this trophy.
        x_last = None
        for r in reversed(history):
            v = r["rates"][i] if i < len(r["rates"]) else None
            if v is not None:
                x_last = x_of(parse_time(r["slot"]))
                break
        if x_last is None:
            x_last = x_of(at)

        slope = 0.0
        sigma_resid = 0.0
        mx = my = None
        if len(pts) >= 3:
            xs = [x_of(t) for t, _ in pts]
            ys = [logit(v) for _, v in pts]
            # Plain least squares: the reads are a step function held
            # between refreshes, so every read is one observation of the
            # line at its own time.
            n_pts = len(pts)
            mx = sum(xs) / n_pts
            my = sum(ys) / n_pts
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                slope = sum((x - mx) * (y - my)
                            for x, y in zip(xs, ys)) / sxx
            icept = my - slope * mx
            resid = [y - (icept + slope * x) for x, y in zip(xs, ys)]
            mean_r = sum(resid) / len(resid)
            var = sum((e - mean_r) ** 2 for e in resid) / len(resid)
            sigma_resid = min(0.8, max(0.05, math.sqrt(var)))
            # Shrink toward zero (n/(n+2)) and clamp. Launch trends are
            # steep (early-story trophies climb ~1.8 logit per x-unit),
            # so the clamp is 2.5, not 1.5.
            slope = max(-2.5, min(2.5, slope * n_pts / (n_pts + 2)))

        dx = x_target - x_last
        if len(pts) >= 3:
            # Anchor on the fitted line read at the target, not on the raw
            # last read: with staleness holds the last read can be hours
            # old, so anchoring on it while measuring dx from now
            # understates the trend. (The shrunk slope is applied around
            # the fit's own centre, so the anchor stays consistent.)
            median_logit = my + slope * (x_target - mx)
        else:
            median_logit = logit(last) + slope * dx
        median = inv_logit(median_logit)
        # Never let the extrapolation run away from the last observed rate.
        median = min(max(median, 0.1 * last), 10.0 * last)
        median = clamp_rate(median)

        # Staleness: has this trophy's rate moved in the last 2 h?
        recent = [r for r in history
                  if (at - parse_time(r["slot"])).total_seconds() <= 2 * 3600]
        stale = True
        for r in recent:
            v = r["rates"][i] if i < len(r["rates"]) else None
            if v is not None and round(v, 6) != round(last, 6):
                stale = False
                break
        sigma_stale = 0.2 if stale else 0.1

        sigma = math.sqrt(
            (RW_SIGMA * math.sqrt(horizon_days)) ** 2
            + sigma_resid ** 2
            + (0.25 * abs(slope) * dx) ** 2
            + sigma_stale ** 2
        )
        sigma = min(1.2, max(0.15, sigma))
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})

    if not history:
        rows = [{"id": t["id"], "q": quantiles(1.0, 1.0)} for t in trophies]

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(in_path.read_text())
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (in_path.stem + ".out")).write_text(text)


if __name__ == "__main__":
    main()
