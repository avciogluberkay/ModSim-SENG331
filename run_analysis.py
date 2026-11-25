# -*- coding: utf-8 -*-
"""
run_analysis.py — Full Statistical Analysis in One Command
==========================================================

Runs the complete output analysis and writes every table to ``results/``:

1. One sample replication per scenario: raw ``vehicle_records``,
   ``lane_state_events`` and ``timeline_events`` (for Q(t) / B(t) plots)
   plus its single-run metrics.
2. Baseline: ``n0`` pilot replications → Student-t CIs → required number of
   replications ``n*`` for the error margin β → top-up → adjusted CIs.
3. Every scenario of the three factors with the same replication count and
   the same seeds (common random numbers), its own n* table, initial and
   adjusted CIs.
4. Paired-t tests (scenario − baseline) with paired-difference CIs.

Usage
-----
    python run_analysis.py                       # defaults: n0=30, beta=0.1 s, 95 %
    python run_analysis.py --beta 0.25 --jobs 4
    python run_analysis.py --per-scenario-nstar  # each scenario uses its own n*
    python run_analysis.py --quick               # smoke test (n0=5, no top-up)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import (
    DEFAULT_BASE_SEED,
    DEFAULT_BETA,
    DEFAULT_CONFIDENCE,
    DEFAULT_N0,
    DEFAULT_N_MAX,
    DEFAULT_REPORT_BETAS,
    build_scenarios,
    compare_to_baseline,
    key_metrics_table,
    run_scenario,
)
from metrics import compute_run_metrics, lane_time_averages, littles_law_check
from simulation import TollPlazaSimulation


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.10g")


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def _tag(df: pd.DataFrame, scenario) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "level_label", scenario.level)
    df.insert(0, "factor", scenario.factor)
    df.insert(0, "scenario", scenario.name)
    return df


def export_sample_runs(scenarios, seed: int, outdir: Path) -> dict:
    """Raw tables + metrics of one replication per scenario."""
    checks = {}
    for sc in scenarios:
        result = TollPlazaSimulation(replace(sc.config, seed=seed)).run()
        folder = outdir / "sample_run" / sc.name
        _write_csv(result.vehicle_records, folder / "vehicle_records.csv")
        _write_csv(result.lane_state_events, folder / "lane_state_events.csv")
        _write_csv(result.timeline_events, folder / "timeline_events.csv")
        _write_csv(compute_run_metrics(result), folder / "run_metrics.csv")
        _write_csv(lane_time_averages(result.lane_state_events, result.config),
                   folder / "lane_summary.csv")
        checks[sc.name] = littles_law_check(result)
    return checks


def _print_table(title: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.4g}".format):
        print(df[columns].to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SENG-331 Project 14 — statistical analysis and experiments."
    )
    parser.add_argument("--n0", type=int, default=DEFAULT_N0,
                        help="pilot replications (default 30)")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help="primary absolute error margin for average delay, s (default 0.1)")
    parser.add_argument("--report-betas", type=float, nargs="+", default=list(DEFAULT_REPORT_BETAS),
                        help="additional beta values tabulated in required_replications.csv")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--n-max", type=int, default=DEFAULT_N_MAX,
                        help="safety cap on replications per scenario (default 1000)")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel worker processes (default: all CPU cores)")
    parser.add_argument("--per-scenario-nstar", action="store_true",
                        help="top up each scenario to its own n* (capped by --n-max) "
                             "instead of the baseline's replication count")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: n0 = 5 and no top-up replications")
    args = parser.parse_args(argv)

    if args.quick:
        args.n0 = 5
        args.n_max = 5

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    scenarios = build_scenarios()

    # 1. Sample replication per scenario (raw data for Q(t)/B(t) plots)
    print("Exporting sample runs ...")
    littles = export_sample_runs(scenarios, args.base_seed, outdir)

    # 2. Baseline: pilot -> n* -> top-up -> adjusted CIs
    baseline = scenarios[0]
    common = dict(n0=args.n0, beta=args.beta, report_betas=tuple(args.report_betas),
                  confidence=args.confidence, n_max=args.n_max,
                  base_seed=args.base_seed, jobs=args.jobs)
    outcomes = [run_scenario(baseline, **common)]
    n_common = outcomes[0].n_used

    # 3. Factor scenarios
    for sc in scenarios[1:]:
        override = None if args.per_scenario_nstar else n_common
        outcomes.append(run_scenario(sc, n_override=override, **common))

    # 4. Paired-t comparisons against the baseline
    comparisons = compare_to_baseline(outcomes, args.confidence)

    # ---------------- Write results ----------------------------------
    ci_initial = pd.concat([_tag(o.ci_initial, o.scenario) for o in outcomes], ignore_index=True)
    ci_adjusted = pd.concat([_tag(o.ci_adjusted, o.scenario) for o in outcomes], ignore_index=True)
    required = pd.concat([_tag(o.required, o.scenario) for o in outcomes], ignore_index=True)
    key_table = key_metrics_table(outcomes)

    _write_csv(ci_initial, outdir / "ci_initial.csv")
    _write_csv(ci_adjusted, outdir / "ci_adjusted.csv")
    _write_csv(required, outdir / "required_replications.csv")
    _write_csv(key_table, outdir / "scenario_key_metrics.csv")
    _write_csv(comparisons, outdir / "paired_t_tests.csv")
    for o in outcomes:
        reps = o.replications.reset_index()
        _write_csv(reps, outdir / "replications" / f"replications_{o.scenario.name}.csv")

    base_out = outcomes[0]
    base_req = base_out.required[base_out.required["beta"] == args.beta]
    beta_analysis = {
        "confidence": args.confidence,
        "beta_seconds": args.beta,
        "target_metrics": "avg_delay per lane and per lane kind, whole day",
        "formula": "n* = ceil((t_{1-alpha/2, n0-1} * s / beta)^2)",
        "n0": args.n0,
        "baseline_n_star_pilot": base_out.n_star_pilot,
        "baseline_n_star_after_refinement": base_out.n_required,
        "baseline_n_used": base_out.n_used,
        "driving_metric": (
            base_req.sort_values("n_star", ascending=False).iloc[0][["level", "group"]].to_dict()
            if not base_req.empty else None
        ),
        "replications_per_scenario": {o.scenario.name: o.n_used for o in outcomes},
        "own_pilot_n_star_per_scenario": {o.scenario.name: o.n_star_pilot for o in outcomes},
        "capped": {o.scenario.name: o.capped for o in outcomes},
        "mode": "per-scenario n*" if args.per_scenario_nstar else "baseline n* for all scenarios",
    }
    _write_json(beta_analysis, outdir / "beta_analysis.json")

    try:
        import simpy
        simpy_version = getattr(simpy, "__version__", "unknown")
    except Exception:  # pragma: no cover
        simpy_version = "unavailable"
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - t_start, 1),
        "arguments": vars(args),
        "seeds": f"base_seed + r for replication r = 0 .. n-1 (base_seed = {args.base_seed})",
        "sample_run_seed": args.base_seed,
        "scenarios": [
            {"name": s.name, "factor": s.factor, "level": s.level,
             "config": {k: v for k, v in asdict(s.config).items() if k != "seed"}}
            for s in scenarios
        ],
        "littles_law_check_sample_runs": littles,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "simpy": simpy_version,
        },
    }
    _write_json(manifest, outdir / "run_manifest.json")

    # ---------------- Console summary --------------------------------
    _print_table(
        f"Required replications, baseline (confidence {args.confidence:.0%})",
        base_out.required.sort_values(["level", "group", "beta"], ascending=[False, True, False]),
        ["group", "level", "beta", "pilot_mean", "pilot_std", "pilot_half_width", "n_star"],
    )
    print(f"\nBaseline: pilot n* = {base_out.n_star_pilot}, after refinement "
          f"{base_out.n_required} for beta = {args.beta} s -> "
          f"{base_out.n_used} replications used.")
    _print_table("Key metrics per scenario (adjusted CIs)",
                 key_table[key_table["period"] == "all"],
                 ["scenario", "metric", "level", "group", "n", "mean", "half_width"])
    sig = comparisons[(comparisons["period"] == "all")
                      & (comparisons["metric"].isin(["avg_delay", "time_avg_number_in_system"]))]
    _print_table("Paired-t vs baseline (whole day)", sig,
                 ["alternative", "metric", "level", "group", "mean_difference",
                  "diff_ci_low", "diff_ci_high", "p_value", "significant"])
    print(f"\nAll results written to: {outdir.resolve()}  "
          f"({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
