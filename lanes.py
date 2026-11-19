# -*- coding: utf-8 -*-
"""
lanes.py — Lanes, Vehicles and Routing
======================================

Each lane is modeled as a SimPy ``Resource`` with its own independent FCFS
queue. This module is responsible for:

* ``Vehicle``: holds the timestamps across a vehicle's lifecycle
  (arrival -> queue entry -> service start -> departure).
* ``Lane``: ``simpy.Resource(capacity=1)`` + the service process + queue and
  server state-change records (raw data for Q(t) and B(t)).
* ``LaneRouter``: applying the "join the shortest total length among the
  allowed lane group" rule; ties are broken by choosing the first eligible
  lane.

Design rules
------------
* ETC vehicles may only use ETC and hybrid lanes.
* Manual (cash/card) vehicles may only use manual and hybrid lanes.
* Lane total length = vehicles waiting in queue + vehicles in service.
* There is no jockeying (moving between queues); each vehicle is served in
  FCFS order in the lane it selected.

Data collection hooks (Person 2 / Person 3 integration)
-------------------------------------------------------
On every state change, each ``Lane`` object appends a record with the
following schema to its ``state_events`` list::

    {
        "time": float,           # event time (seconds)
        "lane_id": str,          # "M1", "E1", "H1" ...
        "lane_kind": str,        # "manual" | "etc" | "hybrid"
        "event": str,            # initial_state/enqueue/start_service/departure/final_state
        "vehicle_id": int | None,
        "queue_length": int,     # queue length AFTER the event
        "busy_servers": int,     # number of busy servers AFTER the event (0/1)
    }

From these records Person 3 can produce Q(t) and B(t) step-function plots
directly, while Person 2 can compute time-averaged values via trapezoid /
step-function integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import simpy

from config import (
    SimulationConfig,
    VEHICLE_ETC,
    VEHICLE_MANUAL,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from typing import Any, Iterator

# Lane kinds
LANE_MANUAL = "manual"
LANE_ETC = "etc"
LANE_HYBRID = "hybrid"


@dataclass
class Vehicle:
    """Timestamps of a single vehicle within the simulation.

    Vehicles with ``completed=False`` are those still in queue or in service
    when the simulation horizon ends; they must be treated separately in the
    statistics (especially for time in system).
    """

    vehicle_id: int
    vehicle_type: str                 # "etc" | "manual"
    arrival_time: float               # arrival at the toll plaza area (s)

    lane_id: str | None = None        # assigned lane
    queue_entry_time: float | None = None
    service_start_time: float | None = None
    departure_time: float | None = None
    service_time: float | None = None
    completed: bool = False

    # ------------------------------------------------------------------
    # Derived measurements
    # ------------------------------------------------------------------
    @property
    def wait_time(self) -> float | None:
        """Queue waiting time = service start - queue entry."""
        if self.service_start_time is None or self.queue_entry_time is None:
            return None
        return self.service_start_time - self.queue_entry_time

    @property
    def time_in_system(self) -> float | None:
        """Time in system = departure - arrival (completed vehicles only)."""
        if not self.completed or self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time

    def to_dict(self) -> dict:
        """Return a flat dictionary for DataFrame/CSV production."""
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "arrival_time": self.arrival_time,
            "lane_id": self.lane_id,
            "queue_entry_time": self.queue_entry_time,
            "service_start_time": self.service_start_time,
            "departure_time": self.departure_time,
            "service_time": self.service_time,
            "wait_time": self.wait_time,
            "time_in_system": self.time_in_system,
            "completed": self.completed,
        }


class Lane:
    """A single toll booth / pass-through point: an independent FCFS queue
    and its server.

    Uses a SimPy ``Resource(capacity=1)``: incoming requests are served in
    the order they were created (FCFS) and only one vehicle is served at a
    time.
    """

    def __init__(
        self,
        env: simpy.Environment,
        lane_id: str,
        lane_kind: str,
        config: SimulationConfig,
        rng,
    ) -> None:
        if lane_kind not in (LANE_MANUAL, LANE_ETC, LANE_HYBRID):
            raise ValueError(f"Unknown lane kind: {lane_kind!r}")

        self.env = env
        self.lane_id = lane_id
        self.lane_kind = lane_kind
        self.config = config
        self.rng = rng  # lane-specific, reproducible service time stream

        self.resource = simpy.Resource(env, capacity=1)

        # Raw data hooks
        self.state_events: list[dict] = []
        self._record_state("initial_state")

    # ------------------------------------------------------------------
    # Instantaneous state properties
    # ------------------------------------------------------------------
    @property
    def queue_length(self) -> int:
        """Number of vehicles waiting in queue (not yet in service)."""
        return len(self.resource.queue)

    @property
    def busy_servers(self) -> int:
        """Number of vehicles currently in service (0 or 1 for capacity=1)."""
        return self.resource.count

    @property
    def occupancy(self) -> int:
        """Total lane length = queue + vehicles in service."""
        return self.queue_length + self.busy_servers

    def serves(self, vehicle_type: str) -> bool:
        """Can this lane serve the given vehicle type?"""
        if self.lane_kind == LANE_MANUAL:
            return vehicle_type == VEHICLE_MANUAL
        if self.lane_kind == LANE_ETC:
            return vehicle_type == VEHICLE_ETC
        # A hybrid lane is open to both vehicle types.
        return vehicle_type in (VEHICLE_MANUAL, VEHICLE_ETC)

    # ------------------------------------------------------------------
    # State recording
    # ------------------------------------------------------------------
    def _record_state(self, event: str, vehicle: Vehicle | None = None) -> None:
        """Record the instantaneous queue/server state AFTER the event."""
        self.state_events.append(
            {
                "time": float(self.env.now),
                "lane_id": self.lane_id,
                "lane_kind": self.lane_kind,
                "event": event,
                "vehicle_id": None if vehicle is None else vehicle.vehicle_id,
                "queue_length": self.queue_length,
                "busy_servers": self.busy_servers,
            }
        )

    def record_final_state(self) -> None:
        """Record the remaining state at the end of the simulation horizon
        (closes the Q(t)/B(t) series)."""
        self._record_state("final_state")

    # ------------------------------------------------------------------
    # Service process
    # ------------------------------------------------------------------
    def service_process(self, vehicle: Vehicle):
        """SimPy process running the entire lifecycle of a vehicle in this lane.

        Flow
        ----
        1. The vehicle enters the queue (``enqueue`` record).
        2. Service starts once the SimPy resource is available
           (``start_service``).
        3. The service time is sampled from the distribution and waited out.
        4. On departure the resource is released (``departure`` record).
        """
        vehicle.lane_id = self.lane_id
        vehicle.queue_entry_time = float(self.env.now)

        with self.resource.request() as request:
            # Request created; the vehicle is queued (or served immediately).
            self._record_state("enqueue", vehicle)

            yield request

            # The server has been allocated to this vehicle.
            vehicle.service_start_time = float(self.env.now)
            self._record_state("start_service", vehicle)

            spec = self.config.service_spec_for(vehicle.vehicle_type)
            vehicle.service_time = spec.sample(self.rng)

            yield self.env.timeout(vehicle.service_time)

            vehicle.departure_time = float(self.env.now)
            vehicle.completed = True
        # Exiting the with block releases the resource.

        # State record AFTER the release: busy_servers shows 0.
        self._record_state("departure", vehicle)


class LaneRouter:
    """Routes an incoming vehicle to the shortest lane among its allowed
    lane group.

    Rule: among the selectable lanes, the one with the smallest ``occupancy``
    (queue + in service) is chosen. On a tie, the first eligible lane in the
    lane list wins (deterministic behavior of ``min``).
    """

    def __init__(self, lanes: list[Lane]) -> None:
        if not lanes:
            raise ValueError("At least one lane is required for the router.")
        self.lanes: list[Lane] = list(lanes)

    def allowed_lanes(self, vehicle_type: str) -> list[Lane]:
        """Lanes the given vehicle type is allowed to use."""
        allowed = [lane for lane in self.lanes if lane.serves(vehicle_type)]
        if not allowed:
            raise RuntimeError(
                f"No eligible lane is defined for vehicle type {vehicle_type!r}. "
                "Check the lane count configuration."
            )
        return allowed

    def choose_lane(self, vehicle_type: str) -> Lane:
        """Return the allowed lane with the shortest total length."""
        return min(self.allowed_lanes(vehicle_type), key=lambda lane: lane.occupancy)

    def state_snapshot(self) -> list[dict]:
        """Instantaneous state of all lanes (for animation / debugging)."""
        return [
            {
                "lane_id": lane.lane_id,
                "lane_kind": lane.lane_kind,
                "queue_length": lane.queue_length,
                "busy_servers": lane.busy_servers,
                "occupancy": lane.occupancy,
            }
            for lane in self.lanes
        ]


__all__ = [
    "LANE_MANUAL",
    "LANE_ETC",
    "LANE_HYBRID",
    "Vehicle",
    "Lane",
    "LaneRouter",
]
