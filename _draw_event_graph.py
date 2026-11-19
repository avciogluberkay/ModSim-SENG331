# -*- coding: utf-8 -*-
"""Render script for event_graph.png (300 DPI)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.set_xlim(0, 160)
ax.set_ylim(0, 100)
ax.axis("off")

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
MAIN_FILL, MAIN_EDGE = "#cfe6ff", "#1f4e79"   # flow events
MARK_FILL, MARK_EDGE = "#ffe3c2", "#b05a00"   # rush markers
END_FILL,  END_EDGE  = "#e8e8e8", "#555555"   # terminator
BUS_COLOR = "#888888"
FLOW_COLOR = "black"

def node(x, y, w, h, code, name, lines, fill, edge, double=False, fs_code=9,
         fs_name=8, fs_lines=7.2, gap=2.1):
    """Draw an event node (ellipse) with code, name and state updates."""
    if double:
        ax.add_patch(Ellipse((x, y), w + 2.4, h + 2.4, facecolor=fill,
                             edgecolor=edge, lw=1.4, zorder=3))
    ax.add_patch(Ellipse((x, y), w, h, facecolor=fill, edgecolor=edge,
                         lw=1.6, zorder=4))
    ax.text(x, y + h * 0.28, code, ha="center", va="center",
            fontsize=fs_code, fontweight="bold", color=edge, zorder=5)
    ax.text(x, y + h * 0.10, name, ha="center", va="center",
            fontsize=fs_name, fontweight="bold", zorder=5)
    for k, ln in enumerate(lines):
        ax.text(x, y - h * 0.14 - k * gap, ln, ha="center", va="center",
                fontsize=fs_lines, zorder=5)

def arrow(p1, p2, rad=0.0, color=FLOW_COLOR, ls="-", lw=1.5, ms=13):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))

def line(p1, p2, color=BUS_COLOR, ls=(0, (5, 4)), lw=1.2):
    ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                       linestyle=ls, lw=lw, zorder=1))

# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(80, 96.5, "Event Graph — Project 14: Highway Toll Plaza with Mixed Payment Lanes",
        ha="center", va="center", fontsize=13, fontweight="bold")

# ----------------------------------------------------------------------
# Main flow row (y = 42)
# ----------------------------------------------------------------------
Y = 42
node(8.5, Y, 16, 11, "E0", "SimStart", ["t = 0", "Qᵢ = Bᵢ = 0 ∀i"],
     MAIN_FILL, MAIN_EDGE)
node(37, Y, 20, 14.5, "E1", "Arrival",
     ["t_a = t", "type ~ Bernoulli(p_etc)", "N += 1"], MAIN_FILL, MAIN_EDGE,
     gap=1.95)
node(65, Y, 20, 13, "E2", "Route & Enqueue",
     ["lane = argmin Uᵢ", "Q_lane += 1;  t_q = t"], MAIN_FILL, MAIN_EDGE)
node(97, Y, 20, 13, "E3", "StartService",
     ["Qᵢ −= 1;  Bᵢ = 1", "t_s = t;  s ~ F(type)"], MAIN_FILL, MAIN_EDGE)
node(127, Y, 20, 13, "E4", "Departure",
     ["Bᵢ = 0;  t_d = t"], MAIN_FILL, MAIN_EDGE)

# Top marker row (y = 76) + terminator
node(29, 76, 17, 11, "E5", "RushStart", ["µ ← 3 s"], MARK_FILL, MARK_EDGE)
node(56, 76, 17, 11, "E6", "RushEnd", ["µ ← 8 s"], MARK_FILL, MARK_EDGE)
node(141, 76, 17, 11, "E7", "SimEnd", ["t = 43,200", "stop"],
     END_FILL, END_EDGE, double=True)

# ----------------------------------------------------------------------
# Scheduling bus (dashed, from E0 at t = 0)
# ----------------------------------------------------------------------
line((8.5, 48.2), (8.5, 90))
line((11, 90), (141, 90))
for x in (29, 56, 141):
    arrow((x, 90), (x, 82.2), color=BUS_COLOR, ls=(0, (5, 4)), lw=1.2, ms=11)
ax.text(76, 92.6, "marker events scheduled from SimStart (t = 0)",
        ha="center", va="center", fontsize=7.5, color="#555555", style="italic")
ax.text(33.5, 86.4, "@ t = 14,400", ha="left", va="center", fontsize=7.2,
        color="#555555")
ax.text(60.5, 86.4, "@ t = 21,600", ha="left", va="center", fontsize=7.2,
        color="#555555")
ax.text(145.5, 86.4, "@ t = 43,200", ha="left", va="center", fontsize=7.2,
        color="#555555")

# ----------------------------------------------------------------------
# Flow edges (main row)
# ----------------------------------------------------------------------
arrow((16.8, Y), (26.6, Y))
ax.text(21.7, 45.3, "schedule first:\n(t₁ ~ Exp(µ(0)))", ha="center",
        va="center", fontsize=7.2)

arrow((47.4, Y), (54.6, Y))
ax.text(51, 46.8, "[true] / 0", ha="center", va="center", fontsize=7.5)

arrow((75.4, Y), (86.6, Y))
ax.text(81, 46.8, "[Bᵢ = 0] / 0", ha="center", va="center", fontsize=7.5)

arrow((107.4, Y), (116.6, Y), rad=-0.15)
ax.text(112, 46.8, "(s)", ha="center", va="center", fontsize=7.5)

# E4 -> E3 : service of the next FCFS vehicle (bows down, label right below)
arrow((127, 35.2), (97, 35.2), rad=-0.32)
ax.text(112, 28.6, "[Qᵢ > 0] / 0  (head of the FCFS queue)",
        ha="center", va="center", fontsize=7.5)

# E2 busy branch note
ax.text(65, 31.8, "[Bᵢ = 1] → vehicle waits in Qᵢ (no jockeying)",
        ha="center", va="center", fontsize=7.2, color="#444444", style="italic")

# E1 self-loop (schedules the next arrival): leaves and re-enters the
# bottom edge of the E1 ellipse, hanging below it
arrow((33, 35.4), (41, 35.4), rad=1.3)
ax.text(37, 28, "next arrival:\n(t + Exp(µ(t)))", ha="center", va="center",
        fontsize=7.2)

# ----------------------------------------------------------------------
# Rate-change edges (dashed orange): E5/E6 -> E1
# ----------------------------------------------------------------------
arrow((25, 70.2), (32.8, 48.6), rad=-0.18, color=MARK_EDGE, ls=(0, (5, 3)), lw=1.4)
ax.text(19.5, 60, "µ ← 3 s", ha="center", va="center", fontsize=7.5,
        color=MARK_EDGE)
arrow((52, 70.2), (41.2, 48.6), rad=0.18, color=MARK_EDGE, ls=(0, (5, 3)), lw=1.4)
ax.text(56.5, 60, "µ ← 8 s", ha="center", va="center", fontsize=7.5,
        color=MARK_EDGE)
ax.text(85, 66, "rate change affects\nsubsequent arrivals only", ha="center",
        va="center", fontsize=7.2, color=MARK_EDGE, style="italic")

# E7 note
ax.text(141, 65.5, "simulation stops;\nremaining vehicles\ncompleted = False",
        ha="center", va="top", fontsize=7.2, color="#444444", style="italic")

# ----------------------------------------------------------------------
# Legend
# ----------------------------------------------------------------------
ax.add_patch(Rectangle((6, 2), 148, 18, facecolor="#f7f7f7",
                       edgecolor="#bbbbbb", lw=1.0))
ax.text(9, 17.6, "State variables", fontsize=8.5, fontweight="bold",
        va="center")
ax.text(9, 14.6,
        "t : clock    µ(t) : mean inter-arrival time (rate ρ(t) = 1/µ(t))    "
        "N : vehicle counter",
        fontsize=7.2, va="center")
ax.text(9, 11.6,
        "Qᵢ : queue length of lane i    Bᵢ : busy servers (0/1)    "
        "Uᵢ = Qᵢ + Bᵢ : total lane length",
        fontsize=7.2, va="center")
ax.text(9, 8.6,
        "type ∈ {etc, manual}    t_a, t_q, t_s, t_d : arrival / queue entry / "
        "service start / departure    s : service time",
        fontsize=7.2, va="center")
ax.text(9, 5.4,
        "F(type) : Triangular(8, 14, 25) s for manual, Uniform(2, 4) s for ETC "
        "(hybrid lane: by vehicle type)",
        fontsize=7.2, va="center")

ax.text(106, 17.6, "Notation", fontsize=8.5, fontweight="bold", va="center")
ax.text(106, 14.6, "edge label = [condition] / (time delay);  "
                   "unconditional edges use [true]", fontsize=7.2, va="center")
ax.text(106, 11.6, "dashed gray = fixed-time markers scheduled at t = 0;  "
                   "dashed orange = arrival-rate change", fontsize=7.2, va="center")
ax.text(106, 8.6, "[Qᵢ = 0] after a departure → server stays idle;  "
                  "each lane has its own FCFS queue", fontsize=7.2, va="center")
ax.text(106, 5.4, "double ellipse = terminator event;  ETC vehicles use ETC/hybrid "
                  "lanes, manual vehicles use manual/hybrid lanes",
        fontsize=7.2, va="center")

fig.savefig("event_graph.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("saved")
