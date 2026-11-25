# -*- coding: utf-8 -*-
"""
metrics.py — Single-Run Performance Metrics
===========================================

Turns the raw tables of one ``SimulationResult`` (``vehicle_records``,
``lane_state_events``) into every performance measure required by the
project brief:

* average / maximum delay in queue — per lane (``M1``, ``E1``, ``H1`` ...),
  per lane kind (manual / etc / hybrid), per vehicle type and overall;
  each for the whole day, the rush window and the off-peak hours
  (period assigned by the vehicle's **arrival time**);
* average time in system — same breakdown;
* time-average and maximum queue length per lane (``Q̄_i``, ``max Q_i``);
* server utilization per lane and per lane kind (``B̄_i``);
* time-average number of vehicles in the plaza
  ``L̄ = Σ_i (Q̄_i + B̄_i)``;
* share of vehicles that wait longer than 120 s (in %);
* censoring counts (vehicles still in queue / in service at the horizon).

Time averages are computed **exactly** by integrating the piecewise-constant
step functions Q_i(t) and B_i(t) logged in ``lane_state_events``: each row
holds the state *after* the event, valid until the next row of the same lane;
the ``final_state`` row closes the series at ``t = horizon``.

Censoring rule
--------------
* ``avg_delay`` / ``max_delay`` / ``avg_time_in_system`` use completed
  vehicles only (``completed == True``).
* ``pct_wait_over_120s`` (primary definition) uses every vehicle whose wait
  is known, i.e. every vehicle that started service — including vehicles
  still in service at the horizon.
* ``pct_wait_over_120s_incl_queued`` (supplementary) additionally counts
  vehicles still waiting in queue at the horizon: those that had already
  waited more than 120 s are counted as long waiters; the denominator is all
  vehicles that joined a lane.
* Censored vehicle counts are reported separately (``n_censored_*``).

Output format
-------------
:func:`compute_run_metrics` returns a *long* DataFrame with the columns
``metric, level, group, period, value, unit``. The ``key`` helper builds the
string ``"metric|level|group|period"`` used as a column name when many
replications are stacked side by side (see ``experiments.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SimulationConfig, VEHICLE_ETC, VEHICLE_MANUAL

# Long-threshold for the "waited more than 2 minutes" measure (seconds).
LONG_WAIT_THRESHOLD = 120.0

# Period labels
PERIOD_ALL = "all"
PERIOD_RUSH = "rush"
PERIOD_OFFPEAK = "offpeak"
PERIODS = (PERIOD_ALL, PERIOD_RUSH, PERIOD_OFFPEAK)

# Aggregation levels
LEVEL_LANE = "lane"              # a single lane: M1, E2, H1, ...
LEVEL_LANE_KIND = "lane_kind"    # manual / etc / hybrid lane group
LEVEL_VEHICLE_TYPE = "vehicle_type"  # manual / etc vehicles
LEVEL_OVERALL = "overall"        # all vehicles
LEVEL_PLAZA = "plaza"            # whole toll plaza (sum over lanes)

METRIC_COLUMNS = ["metric", "level", "group", "period", "value", "unit"]


def key(metric: str, level: str, group: str, period: str) -> str:
    """Build the flat metric identifier ``"metric|level|group|period"``."""
    return f"{metric}|{level}|{group}|{period}"


def split_key(metric_key: str) -> tuple[str, str, str, str]:
    """Inverse of :func:`key`."""
    metric, level, group, period = metric_key.split("|")
    return metric, level, group, period


# ----------------------------------------------------------------------
# Step-function helpers
# ----------------------------------------------------------------------
def step_function_stats(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> tuple[float, float]:
    """Time average and maximum of a right-continuous step function on
    ``[start, end)``.

    ``values[k]`` holds on ``[times[k], times[k+1])``; the last value holds
    until ``end`` (or until ``times[-1]`` if that is later). Zero-length
    states inside the window still count for the maximum.

    Returns
    -------
    (time_average, maximum)
    """
    if end <= start:
        raise ValueError(f"Empty window: [{start}, {end})")
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size == 0:
        return 0.0, 0.0

    next_times = np.append(times[1:], max(times[-1], end))
    seg_start = np.clip(times, start, end)
    seg_end = np.clip(next_times, start, end)
    durations = seg_end - seg_start
    average = float(np.sum(values * durations) / (end - start))

    in_window = (times < end) & ((next_times > start) | (times >= start))
    maximum = float(values[in_window].max()) if in_window.any() else 0.0
    return average, maximum


def _period_windows(config: SimulationConfig) -> dict[str, list[tuple[float, float]]]:
    """Time windows (list of intervals) that make up each period."""
    windows = {PERIOD_ALL: [(0.0, config.horizon)]}
    if config.has_rush_period:
        windows[PERIOD_RUSH] = [(config.rush_start, config.rush_end)]
        off = []
        if config.rush_start > 0.0:
            off.append((0.0, config.rush_start))
        if config.rush_end < config.horizon:
            off.append((config.rush_end, config.horizon))
        windows[PERIOD_OFFPEAK] = off
    else:
        windows[PERIOD_OFFPEAK] = [(0.0, config.horizon)]
    return windows


def _windowed_stats(times, values, intervals) -> tuple[float, float]:
    """Time average / maximum of a step function over a union of intervals."""
    total_length = sum(b - a for a, b in intervals)
    if total_length <= 0:
        return float("nan"), float("nan")
    area = 0.0
    maximum = 0.0
    for a, b in intervals:
        avg, mx = step_function_stats(times, values, a, b)
        area += avg * (b - a)
        maximum = max(maximum, mx)
    return area / total_length, maximum


# ----------------------------------------------------------------------
# Lane-level time averages (Q̄_i, B̄_i, max Q_i)
# ----------------------------------------------------------------------
def lane_time_averages(
    lane_state_events: pd.DataFrame, config: SimulationConfig
) -> pd.DataFrame:
    """Exact time-weighted queue / server statistics for every lane and period.

    Returns one row per (lane, period) with the columns
    ``lane_id, lane_kind, period, time_avg_queue, max_queue, utilization,
    max_busy, time_avg_number_in_lane``.
    """
    windows = _period_windows(config)
    rows = []
    # Preserve the lane construction order (M..., E..., H...).
    lane_order = list(dict.fromkeys(lane_state_events["lane_id"]))
    for lane_id in lane_order:
        lane_events = lane_state_events[lane_state_events["lane_id"] == lane_id]
        lane_events = lane_events.sort_values("time", kind="stable")
        times = lane_events["time"].to_numpy(dtype=float)
        q = lane_events["queue_length"].to_numpy(dtype=float)
        b = lane_events["busy_servers"].to_numpy(dtype=float)
        lane_kind = str(lane_events["lane_kind"].iloc[0])
        for period, intervals in windows.items():
            q_avg, q_max = _windowed_stats(times, q, intervals)
            b_avg, b_max = _windowed_stats(times, b, intervals)
            rows.append(
                {
                    "lane_id": lane_id,
                    "lane_kind": lane_kind,
                    "period": period,
                    "time_avg_queue": q_avg,
                    "max_queue": q_max,
                    "utilization": b_avg,
                    "max_busy": b_max,
                    "time_avg_number_in_lane": q_avg + b_avg,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Vehicle-level preparation
# ----------------------------------------------------------------------
def prepare_vehicles(
    vehicle_records: pd.DataFrame,
    lane_state_events: pd.DataFrame,
    config: SimulationConfig,
) -> pd.DataFrame:
    """Add helper columns: ``lane_kind``, ``period`` (by arrival time),
    ``censor_status`` and ``elapsed_wait_at_horizon``."""
    v = vehicle_records.copy()
    lane_kind_map = (
        lane_state_events.drop_duplicates("lane_id").set_index("lane_id")["lane_kind"]
    )
    v["lane_kind"] = v["lane_id"].map(lane_kind_map)

    arrival = v["arrival_time"].astype(float)
    if config.has_rush_period:
        rush = (arrival >= config.rush_start) & (arrival < config.rush_end)
    else:
        rush = pd.Series(False, index=v.index)
    v["period"] = np.where(rush, PERIOD_RUSH, PERIOD_OFFPEAK)

    completed = v["completed"].astype(bool)
    started = v["service_start_time"].notna()
    v["completed"] = completed
    v["censor_status"] = np.select(
        [completed, started & ~completed],
        ["completed", "in_service_at_horizon"],
        default="in_queue_at_horizon",
    )
    # Waiting time already accumulated by vehicles still queued at the end.
    v["elapsed_wait_at_horizon"] = np.where(
        v["censor_status"] == "in_queue_at_horizon",
        config.horizon - v["queue_entry_time"].astype(float),
        np.nan,
    )
    return v


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def compute_run_metrics(result) -> pd.DataFrame:
    """Compute every performance metric of a single replication.

    Parameters
    ----------
    result:
        A ``SimulationResult`` returned by ``TollPlazaSimulation.run()``.

    Returns
    -------
    pandas.DataFrame
        Long table with the columns ``metric, level, group, period, value,
        unit`` (see the module docstring for the metric list).
    """
    config: SimulationConfig = result.config
    lanes = lane_time_averages(result.lane_state_events, config)
    vehicles = prepare_vehicles(
        result.vehicle_records, result.lane_state_events, config
    )

    records: list[dict] = []

    def add(metric, level, group, period, value, unit):
        records.append(
            {
                "metric": metric,
                "level": level,
                "group": str(group),
                "period": period,
                "value": float(value) if value is not None else np.nan,
                "unit": unit,
            }
        )

    lane_ids = list(lanes["lane_id"].unique())
    lane_kinds = list(dict.fromkeys(lanes["lane_kind"]))
    vehicle_types = [VEHICLE_MANUAL, VEHICLE_ETC]

    # ---------------- Vehicle-based measures ----------------------------
    groupings = [
        (LEVEL_LANE, "lane_id", lane_ids),
        (LEVEL_LANE_KIND, "lane_kind", lane_kinds),
        (LEVEL_VEHICLE_TYPE, "vehicle_type", vehicle_types),
        (LEVEL_OVERALL, None, ["all"]),
    ]

    for period in PERIODS:
        vp = vehicles if period == PERIOD_ALL else vehicles[vehicles["period"] == period]
        for level, column, groups in groupings:
            for group in groups:
                sub = vp if column is None else vp[vp[column] == group]
                done = sub[sub["completed"]]

                # Delay and time in system: completed vehicles only.
                if len(done):
                    add("avg_delay", level, group, period, done["wait_time"].mean(), "s")
                    add("max_delay", level, group, period, done["wait_time"].max(), "s")
                    add("avg_time_in_system", level, group, period,
                        done["time_in_system"].mean(), "s")
                else:
                    add("avg_delay", level, group, period, np.nan, "s")
                    add("max_delay", level, group, period, np.nan, "s")
                    add("avg_time_in_system", level, group, period, np.nan, "s")

                # > 120 s waiters (primary: every vehicle whose wait is known).
                known = sub["wait_time"].dropna()
                pct = 100.0 * (known > LONG_WAIT_THRESHOLD).mean() if len(known) else np.nan
                add("pct_wait_over_120s", level, group, period, pct, "%")

                # Supplementary: also count vehicles still queued at the horizon
                # that had already waited more than 120 s.
                if len(sub):
                    long_known = (sub["wait_time"] > LONG_WAIT_THRESHOLD).sum()
                    long_queued = (sub["elapsed_wait_at_horizon"] > LONG_WAIT_THRESHOLD).sum()
                    pct_incl = 100.0 * (long_known + long_queued) / len(sub)
                else:
                    pct_incl = np.nan
                add("pct_wait_over_120s_incl_queued", level, group, period, pct_incl, "%")

                # Counts (only for the coarse levels, to keep the table compact).
                if level in (LEVEL_VEHICLE_TYPE, LEVEL_OVERALL, LEVEL_LANE):
                    add("n_vehicles", level, group, period, len(sub), "count")
                    add("n_completed", level, group, period, int(sub["completed"].sum()), "count")
                    add("n_censored_in_service", level, group, period,
                        int((sub["censor_status"] == "in_service_at_horizon").sum()), "count")
                    add("n_censored_in_queue", level, group, period,
                        int((sub["censor_status"] == "in_queue_at_horizon").sum()), "count")

    # ---------------- Time-weighted measures -----------------------------
    for period in PERIODS:
        lp = lanes[lanes["period"] == period]
        if lp.empty:
            continue
        for row in lp.itertuples(index=False):
            add("time_avg_queue", LEVEL_LANE, row.lane_id, period, row.time_avg_queue, "vehicles")
            add("max_queue", LEVEL_LANE, row.lane_id, period, row.max_queue, "vehicles")
            add("utilization", LEVEL_LANE, row.lane_id, period, row.utilization, "fraction")
            add("time_avg_number_in_system", LEVEL_LANE, row.lane_id, period,
                row.time_avg_number_in_lane, "vehicles")

        for kind in lane_kinds:
            lk = lp[lp["lane_kind"] == kind]
            # Lane group: queues add up, utilization is the mean over servers.
            add("time_avg_queue", LEVEL_LANE_KIND, kind, period, lk["time_avg_queue"].sum(), "vehicles")
            add("max_queue", LEVEL_LANE_KIND, kind, period, lk["max_queue"].max(), "vehicles")
            add("utilization", LEVEL_LANE_KIND, kind, period, lk["utilization"].mean(), "fraction")
            add("time_avg_number_in_system", LEVEL_LANE_KIND, kind, period,
                lk["time_avg_number_in_lane"].sum(), "vehicles")

        # Whole plaza: L̄ = Σ (Q̄_i + B̄_i), L̄_q = Σ Q̄_i.
        add("time_avg_queue", LEVEL_PLAZA, "all", period, lp["time_avg_queue"].sum(), "vehicles")
        add("max_queue", LEVEL_PLAZA, "all", period, lp["max_queue"].max(), "vehicles")
        add("utilization", LEVEL_PLAZA, "all", period, lp["utilization"].mean(), "fraction")
        add("time_avg_number_in_system", LEVEL_PLAZA, "all", period,
            lp["time_avg_number_in_lane"].sum(), "vehicles")

    return pd.DataFrame(records, columns=METRIC_COLUMNS)


def metrics_to_wide(metrics: pd.DataFrame) -> dict[str, float]:
    """Flatten a long metrics table into ``{"metric|level|group|period": value}``."""
    return {
        key(r.metric, r.level, r.group, r.period): r.value
        for r in metrics.itertuples(index=False)
    }


def littles_law_check(result) -> dict:
    """Consistency check between the two independent data hooks.

    The area under the plaza occupancy curve N(t) = Σ_i (Q_i + B_i) must
    equal the total time all vehicles spent in the plaza up to the horizon,
    Σ_v (min(departure_v, T) − arrival_v). Returns both sides and the
    relative difference (should be ~1e-12).
    """
    config = result.config
    T = config.horizon
    lanes = lane_time_averages(result.lane_state_events, config)
    area = float(lanes.loc[lanes["period"] == PERIOD_ALL, "time_avg_number_in_lane"].sum()) * T

    v = result.vehicle_records
    exit_time = v["departure_time"].astype(float).fillna(T).clip(upper=T)
    vehicle_time = float((exit_time - v["arrival_time"].astype(float)).sum())
    rel = abs(area - vehicle_time) / vehicle_time if vehicle_time else 0.0
    return {
        "area_under_N_t": area,
        "sum_of_vehicle_times": vehicle_time,
        "relative_difference": rel,
    }


__all__ = [
    "LONG_WAIT_THRESHOLD",
    "PERIOD_ALL",
    "PERIOD_RUSH",
    "PERIOD_OFFPEAK",
    "PERIODS",
    "METRIC_COLUMNS",
    "key",
    "split_key",
    "step_function_stats",
    "lane_time_averages",
    "prepare_vehicles",
    "compute_run_metrics",
    "metrics_to_wide",
    "littles_law_check",
]
