# -*- coding: utf-8 -*-
"""
config.py — Central Simulation Configuration
============================================

Collects every parameter of the SENG-331 Project 14 (Highway Toll Plaza with
Mixed Payment Lanes) simulation in a single place. Person 2 (statistics /
experiment design) and Person 3 (visualization / animation) can import this
class and derive scenario variants via ``dataclasses.replace``; the simulation
code itself contains no magic numbers.

Example usage
-------------
>>> from dataclasses import replace
>>> base = SimulationConfig(seed=7)
>>> scenario = replace(base, n_manual_lanes=2, n_hybrid_lanes=1)
>>> scenario.total_lanes
5

Experiment design notes
-----------------------
* For the "total lane count held constant" rule, adjust ``n_etc_lanes``
  accordingly when changing ``n_manual_lanes`` (default total = 5).
* Hybrid scenario: a manual lane is converted so it serves both vehicle
  types -> e.g. ``replace(cfg, n_manual_lanes=2, n_hybrid_lanes=1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vehicle types (managed from a single source across the system)
# ---------------------------------------------------------------------------
VEHICLE_ETC = "etc"        # Vehicle with an electronic toll tag (OGS/HGS-like)
VEHICLE_MANUAL = "manual"  # Vehicle paying by cash / card
VEHICLE_TYPES = (VEHICLE_ETC, VEHICLE_MANUAL)

# ---------------------------------------------------------------------------
# Service time distribution kinds
# ---------------------------------------------------------------------------
SERVICE_TRIANGULAR = "triangular"
SERVICE_UNIFORM = "uniform"


@dataclass(frozen=True)
class ServiceTimeSpec:
    """Service time distribution definition for a service point.

    Parameters
    ----------
    distribution:
        ``"triangular"`` or ``"uniform"``.
    params:
        ``(min, mode, max)`` for triangular; ``(min, max)`` for uniform.

    Note
    ----
    The ``sample`` method draws from a ``numpy.random.Generator`` passed in
    from the outside, so reproducibility is preserved through centralized
    RNG streams.
    """

    distribution: str
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        # Store parameters as an immutable (hashable, comparable) tuple.
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))

        if self.distribution == SERVICE_TRIANGULAR:
            if len(self.params) != 3:
                raise ValueError(
                    "Triangular distribution requires 3 parameters: (min, mode, max)."
                )
            low, mode, high = self.params
            if not (0.0 < low <= mode <= high):
                raise ValueError(
                    "Triangular parameters must satisfy 0 < min <= mode <= max; "
                    f"got: {self.params}"
                )
        elif self.distribution == SERVICE_UNIFORM:
            if len(self.params) != 2:
                raise ValueError(
                    "Uniform distribution requires 2 parameters: (min, max)."
                )
            low, high = self.params
            if not (0.0 <= low < high):
                raise ValueError(
                    "Uniform parameters must satisfy 0 <= min < max; "
                    f"got: {self.params}"
                )
        else:
            raise ValueError(
                f"Unknown distribution kind: {self.distribution!r}. "
                f"Supported: {SERVICE_TRIANGULAR!r}, {SERVICE_UNIFORM!r}."
            )

    def sample(self, rng) -> float:
        """Draw a single service time sample from the distribution (seconds)."""
        if self.distribution == SERVICE_TRIANGULAR:
            low, mode, high = self.params
            return float(rng.triangular(low, mode, high))
        if self.distribution == SERVICE_UNIFORM:
            low, high = self.params
            return float(rng.uniform(low, high))
        # __post_init__ prevents this; kept as a defensive guard.
        raise ValueError(f"Unknown distribution kind: {self.distribution!r}")


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration carrying every simulation parameter.

    Attributes
    ----------
    horizon:
        Simulation horizon (seconds). Default is 12 hours = 43,200 s.
    rush_start / rush_end:
        Start and end times of the rush period (seconds).
        Default: between t = 14,400 (4th hour) and t = 21,600 (6th hour).
        If ``rush_start == rush_end`` there is no rush period.
    mean_iat_offpeak / mean_iat_rush:
        Mean inter-arrival time (seconds): 8 s off-peak, 3 s during the rush
        period (exponential distribution).
    p_etc:
        Share of ETC (OGS/HGS) vehicles determined by Bernoulli at arrival.
    n_manual_lanes / n_etc_lanes / n_hybrid_lanes:
        Lane counts. A hybrid lane serves both vehicle types.
    manual_service / etc_service:
        Service time distributions.
    seed:
        Master random seed; every RNG stream is derived from it.
    """

    horizon: float = 43_200.0                   # 12 hours
    rush_start: float = 14_400.0                # 4th hour
    rush_end: float = 21_600.0                  # 6th hour

    mean_iat_offpeak: float = 8.0               # seconds
    mean_iat_rush: float = 3.0                  # seconds

    p_etc: float = 0.60                         # 60% ETC / 40% manual

    n_manual_lanes: int = 3
    n_etc_lanes: int = 2
    n_hybrid_lanes: int = 0

    manual_service: ServiceTimeSpec = field(
        default_factory=lambda: ServiceTimeSpec(SERVICE_TRIANGULAR, (8.0, 14.0, 25.0))
    )
    etc_service: ServiceTimeSpec = field(
        default_factory=lambda: ServiceTimeSpec(SERVICE_UNIFORM, (2.0, 4.0))
    )

    seed: int = 2025

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive; got: {self.horizon}")

        if self.rush_start != self.rush_end:
            if not (0.0 <= self.rush_start < self.rush_end <= self.horizon):
                raise ValueError(
                    "Rush period must satisfy 0 <= rush_start < rush_end <= horizon; "
                    f"got: [{self.rush_start}, {self.rush_end}], "
                    f"horizon={self.horizon}"
                )

        if self.mean_iat_offpeak <= 0 or self.mean_iat_rush <= 0:
            raise ValueError("Mean inter-arrival times must be positive.")

        if not (0.0 <= self.p_etc <= 1.0):
            raise ValueError(f"p_etc must be in [0, 1]; got: {self.p_etc}")

        for name in ("n_manual_lanes", "n_etc_lanes", "n_hybrid_lanes"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer; got: {value!r}"
                )

        if self.total_lanes <= 0:
            raise ValueError("At least one lane must be defined.")

    # ------------------------------------------------------------------
    # Helper properties and methods
    # ------------------------------------------------------------------
    @property
    def total_lanes(self) -> int:
        """Total lane count (manual + ETC + hybrid)."""
        return self.n_manual_lanes + self.n_etc_lanes + self.n_hybrid_lanes

    @property
    def has_rush_period(self) -> bool:
        """Is a rush period defined in this simulation?"""
        return self.rush_start != self.rush_end

    def is_rush(self, t: float) -> bool:
        """Are we inside the rush period at time ``t``? (rush_start <= t < rush_end)"""
        if not self.has_rush_period:
            return False
        return self.rush_start <= t < self.rush_end

    def mean_iat_at(self, t: float) -> float:
        """Mean inter-arrival time at time ``t`` (seconds).

        Defines the piecewise-constant rate of the non-stationary Poisson
        process: rho(t) = 1 / mean_iat_at(t).
        """
        return self.mean_iat_rush if self.is_rush(t) else self.mean_iat_offpeak

    def service_spec_for(self, vehicle_type: str) -> ServiceTimeSpec:
        """Return the service time distribution that applies to a vehicle type."""
        if vehicle_type == VEHICLE_MANUAL:
            return self.manual_service
        if vehicle_type == VEHICLE_ETC:
            return self.etc_service
        raise ValueError(f"Unknown vehicle type: {vehicle_type!r}")


__all__ = [
    "VEHICLE_ETC",
    "VEHICLE_MANUAL",
    "VEHICLE_TYPES",
    "SERVICE_TRIANGULAR",
    "SERVICE_UNIFORM",
    "ServiceTimeSpec",
    "SimulationConfig",
]
