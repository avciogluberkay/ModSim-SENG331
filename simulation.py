# -*- coding: utf-8 -*-
"""
simulation.py — Main Simulation Engine
======================================

Sets up the ``simpy.Environment``, wires the lanes and the arrival process,
runs the simulation, and returns structured raw data for Person 2
(statistics) and Person 3 (visualization).

``SimulationResult`` holds three tables
---------------------------------------
``vehicle_records``
    One row per vehicle: type, assigned lane, arrival / queue entry /
    service start / departure times, service time, waiting time, time in
    system and the ``completed`` flag.

``lane_state_events``
    Lane state transitions: ``initial_state``, ``enqueue``, ``start_service``,
    ``departure``, ``final_state``. Each row carries the queue length (Q) and
    the number of busy servers (B) AFTER the event -> B(t) and Q(t) step
    functions can be produced directly from this table.

``timeline_events``
    System-level events: ``simulation_start``, ``rush_start``, ``rush_end``,
    ``simulation_end``.

Command line usage
------------------
    python simulation.py --seed 2025 --etc-share 0.6
    python simulation.py --hybrid --outdir output
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from arrivals import ArrivalGenerator, classify_vehicle
from config import (
    SimulationConfig,
    VEHICLE_ETC,
    VEHICLE_MANUAL,
)
from lanes import (
    LANE_ETC,
    LANE_HYBRID,
    LANE_MANUAL,
    Lane,
    LaneRouter,
    Vehicle,
)

# DataFrame column schemas (for consistency even with empty results)
VEHICLE_COLUMNS = [
    "vehicle_id",
    "vehicle_type",
    "arrival_time",
    "lane_id",
    "queue_entry_time",
    "service_start_time",
    "departure_time",
    "service_time",
    "wait_time",
    "time_in_system",
    "completed",
]

LANE_EVENT_COLUMNS = [
    "time",
    "lane_id",
    "lane_kind",
    "event",
    "vehicle_id",
    "queue_length",
    "busy_servers",
]

TIMELINE_COLUMNS = ["time", "event"]


@dataclass
class SimulationResult:
    """Output of a single simulation replication."""

    config: SimulationConfig
    vehicle_records: pd.DataFrame
    lane_state_events: pd.DataFrame
    timeline_events: pd.DataFrame

    # ------------------------------------------------------------------
    # Quick summary (for validation and notebook exploration)
    # ------------------------------------------------------------------
    def basic_summary(self) -> dict:
        """Basic counts and mean metrics (not Person 2's formal analysis;
        this is a quick health check)."""
        vehicles = self.vehicle_records
        completed = vehicles[vehicles["completed"]]

        summary: dict = {
            "seed": int(self.config.seed),
            "horizon": float(self.config.horizon),
            "total_vehicles": int(len(vehicles)),
            "completed_vehicles": int(len(completed)),
            "incomplete_vehicles": int(len(vehicles) - len(completed)),
            "vehicles_by_type": {
                str(k): int(v) for k, v in vehicles["vehicle_type"].value_counts().items()
            },
            "mean_wait_by_type": {},
            "mean_time_in_system_by_type": {},
        }

        for vehicle_type, group in completed.groupby("vehicle_type"):
            summary["mean_wait_by_type"][str(vehicle_type)] = float(
                group["wait_time"].mean()
            )
            summary["mean_time_in_system_by_type"][str(vehicle_type)] = float(
                group["time_in_system"].mean()
            )

        final_states = self.lane_state_events[
            self.lane_state_events["event"] == "final_state"
        ]
        summary["final_lane_state"] = {
            row.lane_id: {
                "queue_length": int(row.queue_length),
                "busy_servers": int(row.busy_servers),
            }
            for row in final_states.itertuples(index=False)
        }
        return summary


class TollPlazaSimulation:
    """Discrete-event simulation engine for the toll plaza complex.

    Parameters
    ----------
    config:
        Simulation configuration; if omitted, the default scenario is used.
    arrival_method:
        ``"piecewise"`` (default) or ``"thinning"``.
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        arrival_method: str = "piecewise",
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.arrival_method = arrival_method

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_lanes(self, env: simpy.Environment, lane_rngs: list) -> list[Lane]:
        """Build the lanes in configuration order:
        first the manual lanes, then ETC, and finally the hybrid lanes.

        Lane identifiers: ``M1..M{n}``, ``E1..E{n}``, ``H1..H{n}``.
        """
        cfg = self.config
        lanes: list[Lane] = []
        rng_index = 0

        for i in range(cfg.n_manual_lanes):
            lanes.append(
                Lane(env, f"M{i + 1}", LANE_MANUAL, cfg, lane_rngs[rng_index])
            )
            rng_index += 1

        for i in range(cfg.n_etc_lanes):
            lanes.append(Lane(env, f"E{i + 1}", LANE_ETC, cfg, lane_rngs[rng_index]))
            rng_index += 1

        for i in range(cfg.n_hybrid_lanes):
            lanes.append(
                Lane(env, f"H{i + 1}", LANE_HYBRID, cfg, lane_rngs[rng_index])
            )
            rng_index += 1

        return lanes

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        """Run the simulation until the end of the horizon and return the
        raw records."""
        cfg = self.config
        env = simpy.Environment()

        # --- Reproducible RNG streams -------------------------------
        # Independent streams are derived from a single master seed:
        # arrivals, vehicle type classification and the service times of
        # each lane.
        seed_sequence = np.random.SeedSequence(cfg.seed)
        streams = seed_sequence.spawn(2 + cfg.total_lanes)
        arrival_rng = np.random.default_rng(streams[0])
        classification_rng = np.random.default_rng(streams[1])
        lane_rngs = [np.random.default_rng(s) for s in streams[2:]]

        # --- Resources and router ------------------------------------
        lanes = self._build_lanes(env, lane_rngs)
        router = LaneRouter(lanes)

        # --- Arrival process -----------------------------------------
        arrivals = ArrivalGenerator(cfg, arrival_rng, method=self.arrival_method)
        vehicles: list[Vehicle] = []
        timeline_events: list[dict] = [
            {"time": 0.0, "event": "simulation_start"}
        ]

        def arrival_process():
            """On each arrival: classify the type -> choose a lane -> start
            the service process."""
            for arrival_time in arrivals.arrival_times():
                # Wait until the vehicle's exact arrival time.
                yield env.timeout(arrival_time - env.now)

                vehicle_type = classify_vehicle(classification_rng, cfg.p_etc)
                lane = router.choose_lane(vehicle_type)

                vehicle = Vehicle(
                    vehicle_id=len(vehicles) + 1,
                    vehicle_type=vehicle_type,
                    arrival_time=float(env.now),
                    lane_id=lane.lane_id,
                )
                vehicles.append(vehicle)
                env.process(lane.service_process(vehicle))

        def marker_process(event_name: str, event_time: float):
            """Add the rush period start/end markers to the timeline."""
            yield env.timeout(event_time - env.now)
            timeline_events.append(
                {"time": float(env.now), "event": event_name}
            )

        # --- Start the processes --------------------------------------
        env.process(arrival_process())

        for event_name, event_time in (
            ("rush_start", cfg.rush_start),
            ("rush_end", cfg.rush_end),
        ):
            if cfg.has_rush_period and event_time < cfg.horizon:
                env.process(marker_process(event_name, event_time))

        # --- Run ------------------------------------------------------
        env.run(until=cfg.horizon)

        # End of horizon: close the timeline and record the final state of
        # each lane.
        timeline_events.append({"time": float(env.now), "event": "simulation_end"})
        for lane in lanes:
            lane.record_final_state()

        # --- Structured outputs ----------------------------------------
        vehicle_records = pd.DataFrame(
            [vehicle.to_dict() for vehicle in vehicles], columns=VEHICLE_COLUMNS
        )

        lane_state_events = pd.DataFrame(
            [event for lane in lanes for event in lane.state_events],
            columns=LANE_EVENT_COLUMNS,
        )
        if not lane_state_events.empty:
            # Record order is preserved for identical timestamps (stable sort).
            lane_state_events = lane_state_events.sort_values(
                "time", kind="stable"
            ).reset_index(drop=True)

        timeline = pd.DataFrame(timeline_events, columns=TIMELINE_COLUMNS)
        timeline = timeline.sort_values("time", kind="stable").reset_index(drop=True)

        return SimulationResult(
            config=cfg,
            vehicle_records=vehicle_records,
            lane_state_events=lane_state_events,
            timeline_events=timeline,
        )


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------
def _build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    n_manual = args.manual_lanes
    n_hybrid = args.hybrid_lanes

    # --hybrid shortcut: one manual lane is converted into a hybrid lane.
    if args.hybrid:
        if n_manual < 1:
            raise SystemExit("--hybrid requires at least one manual lane.")
        n_manual -= 1
        n_hybrid += 1

    return SimulationConfig(
        horizon=args.horizon,
        p_etc=args.etc_share,
        n_manual_lanes=n_manual,
        n_etc_lanes=args.etc_lanes,
        n_hybrid_lanes=n_hybrid,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SENG-331 Project 14 — Toll plaza simulation (Person 1 core model)."
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--etc-share", type=float, default=0.60, help="ETC vehicle share (default 0.60)"
    )
    parser.add_argument("--manual-lanes", type=int, default=3)
    parser.add_argument("--etc-lanes", type=int, default=2)
    parser.add_argument("--hybrid-lanes", type=int, default=0)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Convert one manual lane into a hybrid lane (scenario experiment).",
    )
    parser.add_argument("--horizon", type=float, default=43_200.0)
    parser.add_argument(
        "--method",
        choices=("piecewise", "thinning"),
        default="piecewise",
        help="Arrival generation method.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Folder to write the raw CSV outputs to, if provided.",
    )
    args = parser.parse_args(argv)

    config = _build_config_from_args(args)
    result = TollPlazaSimulation(config, arrival_method=args.method).run()

    print(json.dumps(result.basic_summary(), indent=2, ensure_ascii=False))

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        result.vehicle_records.to_csv(
            out / "vehicle_records.csv", index=False, encoding="utf-8"
        )
        result.lane_state_events.to_csv(
            out / "lane_state_events.csv", index=False, encoding="utf-8"
        )
        result.timeline_events.to_csv(
            out / "timeline_events.csv", index=False, encoding="utf-8"
        )
        print(f"\nCSV outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
