# -*- coding: utf-8 -*-
"""
experiments.py — Replications, Confidence Intervals, Required Replications
and Scenario Comparison
=========================================================================

Design
------
The toll plaza is a **terminating simulation**: every replication is one
12-hour day that starts empty and idle. Output analysis therefore uses the
method of *independent replications* (no warm-up deletion): replication
``r`` runs with its own seed and yields one observation of every metric
(see ``metrics.py``).

1. **Pilot run** — ``n0`` replications (default 30) with the deterministic
   seed list ``base_seed, base_seed + 1, ...``.
2. **Confidence intervals** — Student-t intervals
   ``x̄ ± t_{1-α/2, n-1} · s / √n`` for every metric.
3. **Required replications for an absolute error margin β** —
   ``n* = ⌈(t_{1-α/2, n0-1} · s / β)²⌉`` for each *target* metric (average
   delay in each queue and in each lane group). If ``max n* > n0`` the
   missing replications are run with the next seeds of the same list and all
   intervals are recomputed on the enlarged sample ("adjusted" CIs).
   By default the baseline determines the replication count, and every
   scenario is then run with that same number of replications (same seeds),
   so the paired comparisons use all replications. The n* table is still
   reported for every scenario and every β. Scenario delays differ by orders
   of magnitude (≈0.01 s to ≈100 s), so a single absolute β cannot be met by
   every scenario with a practical number of replications; ``achieved
   half_width`` and ``beta_met`` show where it is met.
4. **Scenario comparison** — every scenario uses the same seed list, so
   replication ``r`` of two scenarios sees identical arrival and vehicle-type
   streams (common random numbers). Differences are tested with the
   **paired-t** test and reported with a paired-difference confidence
   interval.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from config import SimulationConfig
from metrics import compute_run_metrics, metrics_to_wide, split_key
from simulation import TollPlazaSimulation

# Default experiment settings
DEFAULT_N0 = 30
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BETA = 0.1              # primary absolute error margin (s)
DEFAULT_REPORT_BETAS = (1.0, 0.5, 0.25, 0.1)   # n* is tabulated for each
DEFAULT_N_MAX = 1000            # safety cap on replications per scenario
DEFAULT_BASE_SEED = 2025

# Target metrics for the β analysis: average delay in each queue / lane group
# over the whole day.
TARGET_METRIC = "avg_delay"
TARGET_LEVELS = ("lane", "lane_kind")
TARGET_PERIOD = "all"

METRIC_UNITS = {
    "avg_delay": "s",
    "max_delay": "s",
    "avg_time_in_system": "s",
    "pct_wait_over_120s": "%",
    "pct_wait_over_120s_incl_queued": "%",
    "n_vehicles": "count",
    "n_completed": "count",
    "n_censored_in_service": "count",
    "n_censored_in_queue": "count",
    "time_avg_queue": "vehicles",
    "max_queue": "vehicles",
    "utilization": "fraction",
    "time_avg_number_in_system": "vehicles",
}


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One experimental configuration."""

    name: str          # short identifier used in file names
    factor: str        # factor this scenario belongs to
    level: str         # human-readable factor level
    config: SimulationConfig


def build_scenarios(base: SimulationConfig | None = None) -> list[Scenario]:
    """Baseline plus the three controllable factors of the project brief.

    * Factor 1 — number of manual booths with 5 lanes in total: 2 / 3 / 4.
    * Factor 2 — ETC adoption rate: 40 % / 60 % / 80 % (3M + 2E).
    * Factor 3 — one manual lane converted into a hybrid lane (2M + 2E + 1H).
    The baseline (3M + 2E, p_etc = 0.60) is the middle level of factors 1-2
    and the "no hybrid" level of factor 3.
    """
    base = base if base is not None else SimulationConfig()
    return [
        Scenario("baseline", "baseline", "3M + 2E, p_etc = 0.60", base),
        Scenario("F1_2M3E", "F1: manual booths (5 lanes)", "2M + 3E",
                 replace(base, n_manual_lanes=2, n_etc_lanes=3, n_hybrid_lanes=0)),
        Scenario("F1_4M1E", "F1: manual booths (5 lanes)", "4M + 1E",
                 replace(base, n_manual_lanes=4, n_etc_lanes=1, n_hybrid_lanes=0)),
        Scenario("F2_p40", "F2: ETC adoption", "p_etc = 0.40",
                 replace(base, p_etc=0.40)),
        Scenario("F2_p80", "F2: ETC adoption", "p_etc = 0.80",
                 replace(base, p_etc=0.80)),
        Scenario("F3_hybrid", "F3: hybrid lane", "2M + 2E + 1H",
                 replace(base, n_manual_lanes=2, n_etc_lanes=2, n_hybrid_lanes=1)),
    ]


def replication_seeds(n: int, base_seed: int = DEFAULT_BASE_SEED, start: int = 0) -> list[int]:
    """Deterministic, distinct seeds for replications ``start .. start+n-1``."""
    return [base_seed + i for i in range(start, start + n)]


# ----------------------------------------------------------------------
# Running replications
# ----------------------------------------------------------------------
def run_single_replication(
    config: SimulationConfig, seed: int, arrival_method: str = "piecewise"
) -> dict[str, float]:
    """Run one replication and return its metrics as a flat dict."""
    result = TollPlazaSimulation(replace(config, seed=seed), arrival_method=arrival_method).run()
    row = metrics_to_wide(compute_run_metrics(result))
    row["seed"] = seed
    return row


def _run_single_star(args):
    return run_single_replication(*args)


def run_replications(
    config: SimulationConfig,
    seeds: list[int],
    jobs: int | None = None,
    arrival_method: str = "piecewise",
) -> pd.DataFrame:
    """Run one replication per seed.

    Returns a *wide* DataFrame: one row per replication (indexed by
    ``replication`` = 0, 1, ...; column ``seed``) and one column per metric
    key ``"metric|level|group|period"``.
    """
    jobs = jobs if jobs is not None else (os.cpu_count() or 1)
    tasks = [(config, s, arrival_method) for s in seeds]
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_run_single_star, tasks, chunksize=max(1, len(tasks) // (4 * jobs))))
    else:
        rows = [_run_single_star(t) for t in tasks]
    wide = pd.DataFrame(rows)
    cols = ["seed"] + [c for c in wide.columns if c != "seed"]
    wide = wide[cols]
    wide.index.name = "replication"
    return wide


# ----------------------------------------------------------------------
# Confidence intervals
# ----------------------------------------------------------------------
def t_critical(n: int, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided Student-t critical value t_{1-α/2, n-1}."""
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1))


def summarize_replications(
    wide: pd.DataFrame, confidence: float = DEFAULT_CONFIDENCE
) -> pd.DataFrame:
    """Mean, standard deviation and Student-t CI of every metric.

    Returns a long table with the columns ``metric, level, group, period,
    unit, n, mean, std, se, t_crit, half_width, ci_low, ci_high,
    rel_half_width_pct``.
    """
    rows = []
    for col in wide.columns:
        if col == "seed":
            continue
        x = wide[col].dropna().to_numpy(dtype=float)
        n = x.size
        metric, level, group, period = split_key(col)
        mean = float(x.mean()) if n else float("nan")
        std = float(x.std(ddof=1)) if n > 1 else float("nan")
        se = std / math.sqrt(n) if n > 1 else float("nan")
        tc = t_critical(n, confidence)
        hw = tc * se if n > 1 else float("nan")
        rows.append(
            {
                "metric": metric,
                "level": level,
                "group": group,
                "period": period,
                "unit": METRIC_UNITS.get(metric, ""),
                "n": n,
                "mean": mean,
                "std": std,
                "se": se,
                "t_crit": tc,
                "half_width": hw,
                "ci_low": mean - hw,
                "ci_high": mean + hw,
                "rel_half_width_pct": 100.0 * hw / abs(mean) if n > 1 and mean != 0 else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.insert(0, "confidence", confidence)
    return out


# ----------------------------------------------------------------------
# Required number of replications for an error margin β
# ----------------------------------------------------------------------
def is_target(summary: pd.DataFrame) -> pd.Series:
    """Boolean mask of the β-analysis target metrics in a summary table."""
    return (
        (summary["metric"] == TARGET_METRIC)
        & (summary["level"].isin(TARGET_LEVELS))
        & (summary["period"] == TARGET_PERIOD)
    )


def required_replications(
    summary: pd.DataFrame,
    betas=DEFAULT_REPORT_BETAS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """``n* = ⌈(t_{1-α/2, n0-1} · s / β)²⌉`` for every target metric and β.

    ``s`` and ``n0`` come from the pilot summary. ``n_final`` is
    ``max(n0, n*)`` — when n* ≤ n0 the pilot already meets the margin.
    """
    targets = summary[is_target(summary)]
    rows = []
    for r in targets.itertuples(index=False):
        tc = t_critical(r.n, confidence)
        for beta in betas:
            if r.n > 1 and not np.isnan(r.std):
                n_star = int(math.ceil((tc * r.std / beta) ** 2))
            else:
                n_star = r.n
            rows.append(
                {
                    "metric": r.metric,
                    "level": r.level,
                    "group": r.group,
                    "period": r.period,
                    "confidence": confidence,
                    "beta": beta,
                    "n0": r.n,
                    "t_crit_n0": tc,
                    "pilot_mean": r.mean,
                    "pilot_std": r.std,
                    "pilot_half_width": r.half_width,
                    "n_star": n_star,
                    "n_final": max(r.n, n_star),
                    "pilot_meets_beta": bool(r.half_width <= beta),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class ScenarioOutcome:
    """Everything produced for one scenario."""

    scenario: Scenario
    replications: pd.DataFrame        # wide, all replications (n_used rows)
    ci_initial: pd.DataFrame          # CIs from the first n0 replications
    ci_adjusted: pd.DataFrame         # CIs from all n_used replications
    required: pd.DataFrame            # n* table for every β
    n0: int
    n_star_pilot: int                 # n* from the pilot (primary β)
    n_required: int                   # n* at the primary β (pilot n*, raised by refinement)
    n_used: int                       # replications actually run
    capped: bool


def run_scenario(
    scenario: Scenario,
    n0: int = DEFAULT_N0,
    beta: float = DEFAULT_BETA,
    report_betas=DEFAULT_REPORT_BETAS,
    confidence: float = DEFAULT_CONFIDENCE,
    n_max: int = DEFAULT_N_MAX,
    base_seed: int = DEFAULT_BASE_SEED,
    jobs: int | None = None,
    n_override: int | None = None,
    max_refinements: int = 10,
    log=print,
) -> ScenarioOutcome:
    """Pilot run, β analysis, top-up replications and adjusted CIs.

    If ``n_override`` is given, the scenario is topped up to exactly that
    many replications (e.g. the baseline's n*) instead of its own n*.
    """
    betas = tuple(dict.fromkeys([beta, *report_betas]))

    log(f"[{scenario.name}] pilot: {n0} replications ...")
    pilot = run_replications(scenario.config, replication_seeds(n0, base_seed), jobs)
    ci_initial = summarize_replications(pilot, confidence)

    required = required_replications(ci_initial, betas, confidence)
    primary = required[required["beta"] == beta]
    n_required = int(primary["n_final"].max()) if not primary.empty else n0
    n_star_pilot = n_required
    if n_override is not None:
        n_used = max(n0, int(n_override))
        capped = False
    else:
        n_used = min(n_required, max(n_max, n0))
        capped = n_required > n_used

    if n_used > n0:
        reason = (f"common replication count n = {n_used}" if n_override is not None
                  else f"n* = {n_required} for beta = {beta} s"
                  + (f" (capped at {n_used})" if capped else ""))
        log(f"[{scenario.name}] {reason}; running {n_used - n0} more replications ...")
        extra = run_replications(
            scenario.config, replication_seeds(n_used - n0, base_seed, start=n0), jobs
        )
        extra.index = extra.index + n0
        extra.index.name = "replication"
        all_reps = pd.concat([pilot, extra])
    else:
        log(f"[{scenario.name}] n = n0 = {n0} replications are sufficient.")
        all_reps = pilot

    ci_adjusted = summarize_replications(all_reps, confidence)

    # Sequential refinement: the enlarged sample gives a new estimate of s;
    # if some target still misses β, recompute n* from the current sample
    # and add the missing replications (until met or the cap is reached).
    if n_override is None:
        for _ in range(max_refinements):
            targets = ci_adjusted[is_target(ci_adjusted)]
            if targets.empty or (targets["half_width"] <= beta).all():
                break
            n_now = len(all_reps)
            n_new = max(
                int(math.ceil((t_critical(r.n, confidence) * r.std / beta) ** 2))
                for r in targets.itertuples(index=False)
            )
            n_new = min(max(n_new, n_now + 1), max(n_max, n0))
            if n_new <= n_now:
                capped = True
                break
            log(f"[{scenario.name}] refinement: beta not yet met at n = {n_now}; "
                f"running {n_new - n_now} more replications ...")
            extra = run_replications(
                scenario.config, replication_seeds(n_new - n_now, base_seed, start=n_now), jobs
            )
            extra.index = extra.index + n_now
            extra.index.name = "replication"
            all_reps = pd.concat([all_reps, extra])
            ci_adjusted = summarize_replications(all_reps, confidence)
        n_used = len(all_reps)
        n_required = max(n_required, n_used)

    mask = is_target(ci_adjusted)
    ci_adjusted["beta_target"] = np.where(mask, beta, np.nan)
    ci_adjusted["beta_met"] = np.where(mask, ci_adjusted["half_width"] <= beta, pd.NA)

    return ScenarioOutcome(
        scenario=scenario,
        replications=all_reps,
        ci_initial=ci_initial,
        ci_adjusted=ci_adjusted,
        required=required,
        n0=n0,
        n_star_pilot=n_star_pilot,
        n_required=n_required,
        n_used=n_used,
        capped=capped,
    )


# ----------------------------------------------------------------------
# Paired-t comparison (common random numbers)
# ----------------------------------------------------------------------
COMPARISON_LEVELS = ("overall", "vehicle_type", "lane_kind", "plaza")
COMPARISON_METRICS = (
    "avg_delay",
    "max_delay",
    "avg_time_in_system",
    "pct_wait_over_120s",
    "pct_wait_over_120s_incl_queued",
    "time_avg_queue",
    "max_queue",
    "utilization",
    "time_avg_number_in_system",
)


def paired_comparison(
    reference: pd.DataFrame,
    alternative: pd.DataFrame,
    confidence: float = DEFAULT_CONFIDENCE,
    metrics=COMPARISON_METRICS,
    levels=COMPARISON_LEVELS,
) -> pd.DataFrame:
    """Paired-t test of ``alternative − reference`` on common seeds.

    Only metric keys present in both tables are compared (lane-level keys
    are skipped because lane layouts differ between scenarios).
    """
    ref = reference.set_index("seed")
    alt = alternative.set_index("seed")
    common = ref.index.intersection(alt.index)
    ref, alt = ref.loc[common], alt.loc[common]

    alpha = 1.0 - confidence
    rows = []
    for col in ref.columns.intersection(alt.columns):
        metric, level, group, period = split_key(col)
        if metric not in metrics or level not in levels:
            continue
        pair = pd.DataFrame({"ref": ref[col], "alt": alt[col]}).dropna()
        n = len(pair)
        if n < 2:
            continue
        d = (pair["alt"] - pair["ref"]).to_numpy(dtype=float)
        mean_d = float(d.mean())
        sd_d = float(d.std(ddof=1))
        tc = t_critical(n, confidence)
        hw = tc * sd_d / math.sqrt(n)
        if sd_d > 0:
            t_stat, p_value = stats.ttest_rel(pair["alt"], pair["ref"])
            t_stat, p_value = float(t_stat), float(p_value)
        else:  # identical in every replication
            t_stat, p_value = (0.0, 1.0) if mean_d == 0 else (math.copysign(math.inf, mean_d), 0.0)
        ref_mean = float(pair["ref"].mean())
        rows.append(
            {
                "metric": metric,
                "level": level,
                "group": group,
                "period": period,
                "unit": METRIC_UNITS.get(metric, ""),
                "n_pairs": n,
                "reference_mean": ref_mean,
                "alternative_mean": float(pair["alt"].mean()),
                "mean_difference": mean_d,
                "pct_change": 100.0 * mean_d / ref_mean if ref_mean else float("nan"),
                "sd_difference": sd_d,
                "t_crit": tc,
                "diff_ci_low": mean_d - hw,
                "diff_ci_high": mean_d + hw,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": bool(p_value < alpha),
            }
        )
    return pd.DataFrame(rows)


def compare_to_baseline(
    outcomes: list[ScenarioOutcome], confidence: float = DEFAULT_CONFIDENCE
) -> pd.DataFrame:
    """Paired-t comparison of every scenario against the baseline."""
    base = next(o for o in outcomes if o.scenario.name == "baseline")
    frames = []
    for o in outcomes:
        if o is base:
            continue
        cmp = paired_comparison(base.replications, o.replications, confidence)
        cmp.insert(0, "alternative", o.scenario.name)
        cmp.insert(0, "reference", base.scenario.name)
        cmp.insert(0, "level_label", o.scenario.level)
        cmp.insert(0, "factor", o.scenario.factor)
        frames.append(cmp)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Compact presentation table
# ----------------------------------------------------------------------
KEY_METRICS = [
    ("avg_delay", "lane_kind", "manual", "all"),
    ("avg_delay", "lane_kind", "etc", "all"),
    ("avg_delay", "lane_kind", "hybrid", "all"),
    ("avg_delay", "vehicle_type", "manual", "all"),
    ("avg_delay", "vehicle_type", "etc", "all"),
    ("avg_delay", "vehicle_type", "manual", "rush"),
    ("avg_delay", "vehicle_type", "etc", "rush"),
    ("avg_time_in_system", "overall", "all", "all"),
    ("pct_wait_over_120s", "overall", "all", "all"),
    ("pct_wait_over_120s_incl_queued", "overall", "all", "all"),
    ("time_avg_queue", "plaza", "all", "all"),
    ("max_queue", "plaza", "all", "all"),
    ("utilization", "lane_kind", "manual", "all"),
    ("utilization", "lane_kind", "etc", "all"),
    ("utilization", "lane_kind", "hybrid", "all"),
    ("time_avg_number_in_system", "plaza", "all", "all"),
    ("n_censored_in_queue", "overall", "all", "all"),
]


def key_metrics_table(outcomes: list[ScenarioOutcome]) -> pd.DataFrame:
    """One row per scenario × key metric (adjusted CIs) for quick plotting."""
    rows = []
    for o in outcomes:
        ci = o.ci_adjusted.set_index(["metric", "level", "group", "period"])
        for k in KEY_METRICS:
            if k not in ci.index:
                continue
            r = ci.loc[k]
            rows.append(
                {
                    "scenario": o.scenario.name,
                    "factor": o.scenario.factor,
                    "level_label": o.scenario.level,
                    "metric": k[0],
                    "level": k[1],
                    "group": k[2],
                    "period": k[3],
                    "unit": r["unit"],
                    "n": int(r["n"]),
                    "mean": r["mean"],
                    "half_width": r["half_width"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_N0",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_BETA",
    "DEFAULT_REPORT_BETAS",
    "DEFAULT_N_MAX",
    "DEFAULT_BASE_SEED",
    "Scenario",
    "ScenarioOutcome",
    "build_scenarios",
    "replication_seeds",
    "run_single_replication",
    "run_replications",
    "t_critical",
    "summarize_replications",
    "required_replications",
    "run_scenario",
    "paired_comparison",
    "compare_to_baseline",
    "key_metrics_table",
]
