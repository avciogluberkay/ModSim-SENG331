# ModSim-SENG331 — Project 14: Highway Toll Plaza with Mixed Payment Lanes

A discrete-event simulation (SimPy) of a highway toll plaza with mixed payment
lanes, built for SENG-331 (Modeling & Simulation). This repository contains
the **core simulation engine and the event-graph design** (Phase 1). The
statistical analysis and visualization phases consume its output tables
through the documented Python API.

## 1. Project Overview

Vehicles arrive at a toll plaza according to a **non-stationary Poisson
process** and are routed to the shortest eligible lane. Each lane is a
single-server FCFS queue. The plaza serves two vehicle types — **ETC**
(electronic toll tag) and **manual** (cash/card) — with dedicated and
optional hybrid lanes.

### System parameters (default scenario)

| Parameter | Value | Notes |
|---|---|---|
| Horizon | 43,200 s (12 h) | Simulation ends at `t = 43,200`; unfinished vehicles are censored (`completed = False`) |
| Rush period | 14,400–21,600 s (2 h) | Hours 4–6; mean inter-arrival time drops from 8 s to 3 s |
| Mean inter-arrival time | 8 s off-peak / 3 s rush | Exponential; piecewise-constant rate `rho(t) = 1/mu(t)` |
| ETC share `p_etc` | 0.60 | Vehicle type drawn i.i.d. at each arrival via Bernoulli |
| Lanes | 3 Manual + 2 ETC (total 5) | Optional hybrid lanes serve both types |
| Manual service time | Triangular(8, 14, 25) s | |
| ETC service time | Uniform(2, 4) s | Hybrid lanes apply the distribution of the vehicle's type |
| Routing rule | Join the shortest lane | `U_i = Q_i + B_i` (queue + in-service) among **allowed** lanes; ties broken deterministically; no jockeying |

### Controllable experiment factors

* `seed` — master random seed (drives all RNG streams)
* `p_etc` / `--etc-share` — ETC vehicle share
* Lane configuration — `n_manual_lanes`, `n_etc_lanes`, `n_hybrid_lanes`
  (the "total lane count held constant" experiments swap manual ↔ hybrid
  lanes, e.g. `2 manual + 1 hybrid + 2 ETC`)
* `horizon`, rush window, arrival method (`piecewise` or `thinning`)

## 2. Repository Structure

| File | Responsibility |
|---|---|
| `config.py` | Immutable `SimulationConfig` / `ServiceTimeSpec` dataclasses holding every parameter; scenario variants are derived via `dataclasses.replace` |
| `arrivals.py` | Non-stationary Poisson arrival process (exact piecewise sampling + thinning validation method) and Bernoulli vehicle-type classification |
| `lanes.py` | `Vehicle` lifecycle records, `Lane` (SimPy `Resource(capacity=1)` FCFS server with state-event logging), `LaneRouter` (shortest-allowed-lane rule) |
| `simulation.py` | `TollPlazaSimulation` engine: RNG stream management, process wiring, `SimulationResult` output tables, CLI |
| `event_graph.md` | Event-graph design: state variables, events E0–E7, transitions, pseudocode, data-hook semantics |
| `event_graph.png` | Rendered event-graph diagram (300 DPI) |
| `_draw_event_graph.py` | Matplotlib script that renders `event_graph.png` |
| `metrics.py` | Single-run performance metrics: delays, time in system, exact time-weighted Q̄/B̄, max queue, L̄, share of waits > 120 s, censoring counts |
| `experiments.py` | Scenarios (baseline + 3 factors), independent replications, Student-t CIs, required replications for an error margin β, paired-t comparisons |
| `run_analysis.py` | Runs the complete analysis and writes every table to `results/` |
| `results/` | Analysis outputs (CSV / JSON), see section 6 |
| `requirements.txt` | Runtime dependencies (simpy, numpy, pandas, scipy) |

## 3. Setup & Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the default scenario (prints a JSON summary)
python simulation.py

# Examples
python simulation.py --seed 42 --etc-share 0.5
python simulation.py --manual-lanes 2 --hybrid          # 2 manual + 1 hybrid + 2 ETC
python simulation.py --method thinning                  # cross-validate arrivals
python simulation.py --outdir output                     # also write raw CSVs

# 4. Full statistical analysis (writes results/)
python run_analysis.py                                   # n0=30, beta=0.1 s, 95 % CIs
python run_analysis.py --quick                           # fast smoke test
```

Key CLI options: `--seed`, `--etc-share`, `--manual-lanes`, `--etc-lanes`,
`--hybrid-lanes`, `--hybrid`, `--horizon`, `--method`, `--outdir`.

## 4. Integration Guide for Next Phases (Statistical Analysis & Visualization)

### 4.1 Running experiments through the Python API

Do not rely on the CLI for experiments; instantiate the engine directly and
derive scenario configs with `dataclasses.replace`:

```python
from dataclasses import replace

from config import SimulationConfig
from simulation import TollPlazaSimulation

base = SimulationConfig(seed=2025)                     # default 3M + 2ETC
hybrid = replace(base, n_manual_lanes=2, n_hybrid_lanes=1)   # lane-swap experiment

result = TollPlazaSimulation(hybrid).run()             # -> SimulationResult
result.vehicle_records, result.lane_state_events, result.timeline_events
result.basic_summary()                                 # quick health check (JSON dict)
```

`SimulationConfig` is frozen and validated, so scenario definitions are
safe to reuse across replications.

### 4.2 Output table schemas

`SimulationResult` holds three pandas DataFrames (full semantics and the
event-to-code mapping: see `event_graph.md`, sections 6–7).

**`vehicle_records`** — one row per vehicle:

| Column | Meaning |
|---|---|
| `vehicle_id`, `vehicle_type` | Sequential ID (`"etc"` / `"manual"`) |
| `arrival_time`, `queue_entry_time`, `service_start_time`, `departure_time` | Lifecycle timestamps (s) |
| `lane_id` | `"M1"`, `"E1"`, `"H1"`, … |
| `service_time` | Sampled service duration |
| `wait_time` | `service_start − queue_entry` (`None` if never served) |
| `time_in_system` | `departure − arrival` (`None` unless `completed`) |
| `completed` | `True` iff the vehicle departed within the horizon |

**`lane_state_events`** — one row per lane state change, carrying the state
**after** the event: `time`, `lane_id`, `lane_kind`
(`manual`/`etc`/`hybrid`), `event`
(`initial_state`/`enqueue`/`start_service`/`departure`/`final_state`),
`vehicle_id`, `queue_length` (Q), `busy_servers` (B). The
`initial_state`/`final_state` rows close the step functions at `t = 0` and
`t = horizon`.

**`timeline_events`** — system-level markers: `simulation_start`,
`rush_start`, `rush_end`, `simulation_end`.

### 4.3 Critical technical notes for the analysis phase

* **Censoring rule.** Vehicles still in queue or in service at `t = horizon`
  have `completed = False`, no `departure_time`, and `None` for
  `wait_time`/`time_in_system`. Filter on
  `vehicle_records[vehicle_records["completed"]]` before computing
  per-vehicle response statistics (means, quantiles, confidence
  intervals); report the censored count separately.
* **Time-weighted Q(t) / B(t).** `lane_state_events` is a complete step-function
  log per lane. Time-averaged queue length and utilization follow directly
  from step-function integration,
  `Q̄ = (1/T)∫₀ᵀ Q(t)dt` and `B̄ = (1/T)∫₀ᵀ B(t)dt` (each row's state holds
  until the next row's `time` for the same lane; `final_state` closes the
  interval).
* **Common Random Numbers (CRN).** All RNG streams are spawned from a single
  `numpy.random.SeedSequence(seed)`: stream 0 → arrivals, stream 1 → vehicle
  classification, streams 2… → per-lane service times (in M, E, H order).
  The arrival and classification streams depend **only** on `seed`, so
  scenario comparisons that share a seed observe identical arrival patterns
  (CRN). Per-lane service streams are index-aligned: they stay aligned when
  the lane layout prefix is unchanged (e.g. the `2 manual + 1 hybrid` swap
  keeps M1's stream but shifts E-lane streams by one index).
* `basic_summary()` is a convenience health check only — it is **not** the
  formal statistical analysis.

## 5. Project Roadmap & Status

| Phase | Scope | Status |
|---|---|---|
| 1. Core simulation & event graph | `config.py`, `arrivals.py`, `lanes.py`, `simulation.py`, event-graph design and diagram | **Completed** |
| 2. Statistical analysis & experiment design | Replications, scenario comparison (lane configurations, `p_etc`), CIs on wait/time-in-system, time-weighted Q̄/B̄ | **Completed** |
| 3. Visualization / notebook | Q(t)/B(t) step plots, lane utilization, comparative dashboards from the raw tables | Next |

## 6. Statistical Analysis (`run_analysis.py`)

**Method.** Each replication is one 12-hour day that starts empty, so the
model is a terminating simulation analysed with independent replications
(no warm-up deletion). Replication `r` uses seed `2025 + r`; all scenarios
share these seeds (common random numbers).

1. Pilot: `n0 = 30` replications → 95 % Student-t CIs for every metric.
2. Required replications for the error margin β on the average delay of each
   queue / lane group: `n* = ⌈(t₀.₉₇₅,ₙ₀₋₁ · s / β)²⌉`. Missing replications
   are added (and n* re-estimated until β is met), then CIs are recomputed.
   Default β = 0.1 s (baseline delays are ≈ 2 s manual, ≈ 0.1 s ETC);
   n* is also tabulated for β = 1, 0.5 and 0.25 s.
3. Every scenario runs with the baseline's final replication count and is
   compared with the baseline by a paired-t test (difference CI + p-value).
   Use `--per-scenario-nstar` to size each scenario by its own n* instead.

| Scenario | Configuration |
|---|---|
| `baseline` | 3M + 2E, `p_etc = 0.60` |
| `F1_2M3E`, `F1_4M1E` | 2M + 3E, 4M + 1E (5 lanes fixed) |
| `F2_p40`, `F2_p80` | `p_etc = 0.40`, `0.80` (3M + 2E) |
| `F3_hybrid` | 2M + 2E + 1H |

**Output files (`results/`).** Long tables use the columns
`metric, level, group, period`; `level` ∈ `lane / lane_kind / vehicle_type /
overall / plaza`, `period` ∈ `all / rush / offpeak` (by arrival time for
vehicle-based metrics).

| File | Content |
|---|---|
| `ci_initial.csv`, `ci_adjusted.csv` | Mean, std, 95 % CI of every metric per scenario — pilot (`n0`) and final sample; `beta_met` marks the β targets |
| `required_replications.csv` | n* per scenario, target metric and β |
| `beta_analysis.json` | Summary of the β procedure (pilot n*, final n, driving metric) |
| `scenario_key_metrics.csv` | Compact scenario × key-metric table for plots |
| `paired_t_tests.csv` | Scenario − baseline: mean difference, difference CI, t, p-value, `significant` |
| `replications/replications_<scenario>.csv` | One row per replication, one column per `metric\|level\|group\|period` |
| `sample_run/<scenario>/` | Raw `vehicle_records`, `lane_state_events`, `timeline_events` of replication 0, its `run_metrics.csv` and `lane_summary.csv` — input for Q(t)/B(t) plots |
| `run_manifest.json` | Arguments, scenario definitions, library versions, Little's-law check |

**Notes.**
* Average delay and time in system use completed vehicles only.
  `pct_wait_over_120s` uses every vehicle that started service (including
  those still in service at the horizon); `pct_wait_over_120s_incl_queued`
  also counts vehicles still queued at the horizon that had already waited
  more than 120 s. Censored counts: `n_censored_in_service`,
  `n_censored_in_queue`.
* Lane-group utilization is the mean over the group's servers; group and
  plaza queue lengths are sums over lanes.
* In `F3_hybrid` the `lane_kind = manual` group has 2 lanes instead of 3;
  compare scenarios through `vehicle_type` metrics for like-for-like results.
* `2M + 3E` and `p_etc = 0.40` overload the manual lanes during the rush
  hour (load ≈ 1.04), so their delays are large and highly variable; their
  CIs are wider than β (see `beta_met`).
