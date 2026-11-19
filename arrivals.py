# -*- coding: utf-8 -*-
"""
arrivals.py — Time-Dependent Arrival Process and Vehicle Type Assignment
========================================================================

This module provides the two core components within Person 1's "Model and
Core Simulation" responsibility:

1. **Non-stationary Poisson arrival process** — the mean inter-arrival time
   varies with time (off-peak: 8 s, rush: 3 s).
2. **Bernoulli vehicle type assignment** — at arrival, each vehicle is
   independently classified as ETC with probability ``p_etc``, otherwise as
   manual/card type.

Because the arrival rate is piecewise constant, two methods are supported:

``method="piecewise"`` (default, exact)
    Sampling is drawn from the exponential distribution within each time
    piece. If a sample crosses a piece boundary, no arrival is counted; the
    clock jumps to the boundary and a new sample is drawn at the new rate.
    Thanks to the memoryless property of the exponential distribution, this
    approach is **statistically exact**.

``method="thinning"``
    Candidate arrivals are generated at a constant upper-bound rate
    ``lambda_max`` and each candidate is accepted with probability
    ``lambda(t) / lambda_max``. Provided as a second method for validation
    purposes.

Both methods use the same ``numpy.random.Generator`` stream; the simulation
engine manages the seed centrally (see ``simulation.py``).
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterator

from config import SimulationConfig, VEHICLE_ETC, VEHICLE_MANUAL

# Supported arrival generation methods
METHOD_PIECEWISE = "piecewise"
METHOD_THINNING = "thinning"


def classify_vehicle(rng, p_etc: float) -> str:
    """Determine the vehicle type at arrival via a Bernoulli distribution.

    Parameters
    ----------
    rng:
        A ``numpy.random.Generator`` instance (reproducible stream).
    p_etc:
        Probability of being an ETC (OGS/HGS) vehicle, in [0, 1].

    Returns
    -------
    str
        ``"etc"`` or ``"manual"``.
    """
    if not (0.0 <= p_etc <= 1.0):
        raise ValueError(f"p_etc must be in [0, 1]; got: {p_etc}")
    return VEHICLE_ETC if float(rng.random()) < p_etc else VEHICLE_MANUAL


class ArrivalGenerator:
    """Class generating time-dependent Poisson arrival times.

    Parameters
    ----------
    config:
        Simulation configuration (horizon, rush window and mean
        inter-arrival times are read from here).
    rng:
        ``numpy.random.Generator`` dedicated to arrivals.
    method:
        ``"piecewise"`` (default) or ``"thinning"``.

    Examples
    --------
    >>> import numpy as np
    >>> gen = ArrivalGenerator(SimulationConfig(seed=1),
    ...                        np.random.default_rng(1))
    >>> first = next(gen.arrival_times())
    >>> 0 <= first < 43200
    True
    """

    def __init__(self, config: SimulationConfig, rng, method: str = METHOD_PIECEWISE) -> None:
        if method not in (METHOD_PIECEWISE, METHOD_THINNING):
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Supported: {METHOD_PIECEWISE!r}, {METHOD_THINNING!r}."
            )
        self.config = config
        self.rng = rng
        self.method = method

        # Interior boundaries where the rate changes (piecewise-constant rate
        # transitions).
        self._boundaries: list[float] = []
        if config.has_rush_period:
            if 0.0 < config.rush_start < config.horizon:
                self._boundaries.append(config.rush_start)
            if 0.0 < config.rush_end < config.horizon:
                self._boundaries.append(config.rush_end)
            self._boundaries.sort()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def arrival_times(self) -> Iterator[float]:
        """Yield arrival times in increasing order over the horizon (seconds)."""
        if self.method == METHOD_PIECEWISE:
            yield from self._arrival_times_piecewise()
        else:
            yield from self._arrival_times_thinning()

    # ------------------------------------------------------------------
    # Method 1: Piecewise-constant rate (exact sampling)
    # ------------------------------------------------------------------
    def _arrival_times_piecewise(self) -> Iterator[float]:
        """Exact sampling from a piecewise-constant rate Poisson process.

        Algorithm
        ---------
        1. Take the mean inter-arrival time ``mu(t)`` for the current ``t``.
        2. Compute the candidate arrival time ``t + Exponential(mu(t))``.
        3. If the candidate crosses the next rate boundary: no arrival is
           counted, the clock jumps to the boundary and step 1 repeats at the
           new rate.
        4. Otherwise the candidate is accepted and the process continues
           until the horizon.
        """
        horizon = self.config.horizon
        t = 0.0

        while t < horizon:
            mean_iat = self.config.mean_iat_at(t)
            candidate = t + float(self.rng.exponential(mean_iat))

            boundary = self._next_boundary(t)
            if boundary is not None and candidate >= boundary:
                # No arrivals up to the rate-change point -> jump to the
                # boundary. (Exact, thanks to the memoryless property of the
                # exponential distribution.)
                t = boundary
                continue

            if candidate >= horizon:
                break

            t = candidate
            yield t

    def _next_boundary(self, t: float) -> float | None:
        """Return the first rate boundary strictly greater than ``t``."""
        idx = bisect_right(self._boundaries, t)
        if idx < len(self._boundaries):
            return self._boundaries[idx]
        return None

    # ------------------------------------------------------------------
    # Method 2: Thinning (generate candidates + accept/reject)
    # ------------------------------------------------------------------
    def _arrival_times_thinning(self) -> Iterator[float]:
        """Time-dependent Poisson arrivals via the thinning method.

        Candidate arrivals are generated at the highest rate over all times,
    ``lambda_max``; each candidate is accepted with probability
    ``lambda(t)/lambda_max``.
        """
        cfg = self.config
        horizon = cfg.horizon

        # Rate (arrivals/s) = 1 / mean inter-arrival time.
        lambda_max = max(1.0 / cfg.mean_iat_offpeak, 1.0 / cfg.mean_iat_rush)
        t = 0.0

        while True:
            t += float(self.rng.exponential(1.0 / lambda_max))
            if t >= horizon:
                break

            lam = 1.0 / cfg.mean_iat_at(t)
            if float(self.rng.random()) < lam / lambda_max:
                yield t


__all__ = [
    "METHOD_PIECEWISE",
    "METHOD_THINNING",
    "classify_vehicle",
    "ArrivalGenerator",
]
