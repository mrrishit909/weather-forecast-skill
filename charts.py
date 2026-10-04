"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_horizon.png, 02_methods.png, 03_seasons.png, 04_bias.png, 05_rain.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
SK = pd.read_csv(R / "skill.csv"); HZ = pd.read_csv(R / "horizon.csv"); BI = pd.read_csv(R / "bias.csv"); SE = pd.read_csv(R / "seasons.csv"); RA = pd.read_csv(R / "rain.csv"); SM = pd.read_csv(R / "summary.csv").set_index("key")["value"]; CL = pd.read_csv(R / "climatology.csv")
STATIONS = list(pd.read_csv(R / "stations.csv")["station"]); HS = (1, 2, 3, 5, 7, 14)


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def lines(ax, el="TMAX", fs=8):
    g = SK[SK["element"] == el]
    for s in STATIONS:
        d = g[g["station"] == s]; ax.plot(d["horizon"], d["skill_damped"], color="#5a5a58", linewidth=1.1, zorder=2)
    m = g.groupby("horizon")["skill_damped"].mean(); ax.plot(m.index, m.values, color=BLUE, linewidth=2.8, label="mean of the 12 stations", zorder=3)
    p = g.groupby("horizon")["skill_persistence"].mean(); ax.plot(p.index, p.values, color=ORANGE, linewidth=2.2, linestyle="--", label="tomorrow = today (mean)", zorder=3)
    ax.axhline(0, color=DIM, linewidth=0.9); ax.axhline(0.05, color=GREEN, linestyle=":", linewidth=1.2); ax.text(14, 0.06, "5%", color=GREEN, fontsize=8.5, ha="right", va="bottom")
    ax.set_ylim(-1.1, 0.7); ax.set_xlim(1, 14); ax.set_xticks([1, 2, 3, 5, 7, 10, 14]); ax.set_xlabel("days ahead"); ax.set_ylabel("skill against the calendar (1 = perfect, 0 = no better)"); ax.legend(frameon=False, labelcolor=INK, fontsize=fs + 0.5, loc="lower left")


def grouped(ax, el="TMAX"):
    x = np.arange(len(HS)); w = 0.38; g = SK[SK["element"] == el]; dm = [g[g["horizon"] == h]["skill_damped"].mean() for h in HS]; pm = [g[g["horizon"] == h]["skill_persistence"].mean() for h in HS]
    ax.bar(x - w / 2, pm, width=w * 0.92, color=ORANGE, label="tomorrow = today"); ax.bar(x + w / 2, dm, width=w * 0.92, color=BLUE, label="damped: calendar + a share of today's departure")
    for xi, v in list(zip(x - w / 2, pm)) + list(zip(x + w / 2, dm)):
        ax.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels([f"{h} day" + ("s" if h > 1 else "") for h in HS], fontsize=9); ax.set_ylim(-1.2, 0.65); ax.set_ylabel("mean skill against the calendar"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower left")


def horizon(axes):
    lines(axes[0]); axes[0].set_title("Daily high: skill by days ahead", fontsize=12)
    ax = axes[1]; d = HZ[HZ["element"] == "TMAX"].set_index("station"); order = list(d["first_horizon_damped_below_5pct"].sort_values().index); y = np.arange(len(order)); t = HZ[HZ["element"] == "TMIN"].set_index("station").loc[order]
    for yi, s in zip(y, order):
        ax.plot([d.loc[s, "first_horizon_damped_below_5pct"], t.loc[s, "first_horizon_damped_below_5pct"]], [yi, yi], color="#3a3a3a", linewidth=1.2, zorder=1)
    ax.scatter(d.loc[order, "first_horizon_damped_below_5pct"], y, s=46, color=BLUE, zorder=3, label="daily high"); ax.scatter(t["first_horizon_damped_below_5pct"], y, s=46, color=ORANGE, zorder=3, label="daily low")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9); ax.set_xlabel("first day on which the skill is under 5%"); ax.set_xlim(0, 14.5); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right"); ax.grid(axis="y", visible=False); ax.set_title("How many days the calendar can be beaten", fontsize=12)


def methods(axes):
    ax = axes[0]; g = SK[SK["element"] == "TMAX"]
    for k, lab, c, ls in (("skill_persistence", "tomorrow = today", ORANGE, "--"), ("skill_damped", "damped persistence", BLUE, "-"), ("skill_week", "today and last week", GREEN, "-")):
        m = g.groupby("horizon")[k].mean(); ax.plot(m.index, m.values, color=c, linewidth=2.2, linestyle=ls, label=lab)
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_ylim(-1.1, 0.55); ax.set_xlim(1, 14); ax.set_xticks([1, 2, 3, 5, 7, 10, 14]); ax.set_xlabel("days ahead"); ax.set_ylabel("mean skill, daily high"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="center right"); ax.set_title("Three forecasts from the station's own past", fontsize=12)
    ax = axes[1]; d = SK[(SK["horizon"] == 1)].pivot(index="station", columns="element", values="skill_damped").loc[STATIONS]; order = list(d["TMAX"].sort_values().index); y = np.arange(len(order)); w = 0.38
    ax.barh(y - w / 2, d.loc[order, "TMAX"], height=w * 0.92, color=BLUE, label="daily high"); ax.barh(y + w / 2, d.loc[order, "TMIN"], height=w * 0.92, color=ORANGE, label="daily low")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9); ax.set_xlabel("skill one day ahead (damped persistence)"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right"); ax.grid(axis="y", visible=False); ax.set_title("Tomorrow's weather, station by station", fontsize=12)


def seasons(axes):
    names = ["winter", "spring", "summer", "autumn"]
    for ax, h in zip(axes, (1, 3)):
        d = SE[(SE["element"] == "TMAX") & (SE["horizon"] == h)].pivot(index="station", columns="season", values="skill_damped").loc[STATIONS, names]; vmax = 0.65 if h == 1 else 0.22; im = ax.imshow(d.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=INK if v > 0.45 * vmax else BG)
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9); ax.set_yticks(range(len(STATIONS))); ax.set_yticklabels(STATIONS if h == 1 else [], fontsize=9); ax.grid(False); ax.set_title(f"Daily high, {h} day{'s' if h > 1 else ''} ahead (darker = more skill)", fontsize=11)
        for sp in ax.spines.values(): sp.set_visible(False)


def bias(axes):
    for ax, el in zip(axes, ("TMAX", "TMIN")):
        d = BI[BI["element"] == el].set_index("station").loc[STATIONS]; x = np.arange(len(STATIONS)); w = 0.27
        for dx, col, lab, c in ((-w, "bias_stale_seasonal_only", "seasonal cycle, fitted 1991-2010", GRAY), (0, "bias_stale_trend", "plus a trend, fitted 1991-2010", ORANGE), (w, "bias_rolling", "refitted on the last 20 years", BLUE)):
            ax.bar(x + dx, d[col], width=w * 0.92, color=c, label=lab)
        ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels([s.replace("Dallas-Fort Worth", "Dallas").replace("Los Angeles", "L.A.").replace("Minneapolis", "Minneap.") for s in STATIONS], rotation=45, ha="right", fontsize=8); ax.set_ylim(-0.8, 1.7); ax.grid(axis="x", visible=False)
        ax.set_ylabel("mean error, observed minus calendar (°C)"); ax.set_title("Daily high" if el == "TMAX" else "Daily low", fontsize=12)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper right")


def rain(axes):
    ax = axes[0]
    for s in STATIONS:
        d = RA[RA["station"] == s]; ax.plot(d["horizon"], 100 * d["skill_two_state"], color="#5a5a58", linewidth=1.1)
    m = RA.groupby("horizon")["skill_two_state"].mean(); ax.plot(m.index, 100 * m.values, color=BLUE, linewidth=2.8, label="mean of the 12 stations"); ax.axhline(0, color=DIM, linewidth=0.9)
    for s in ("Los Angeles", "Seattle"):
        d = RA[(RA["station"] == s) & (RA["horizon"] == 1)]; ax.text(1.06, 100 * d["skill_two_state"].iloc[0], s, fontsize=8.5, color=INK, va="center")
    ax.set_xticks(range(1, 6)); ax.set_xlabel("days ahead"); ax.set_ylabel("Brier skill against the seasonal probability (%)"); ax.set_ylim(-1, 12); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right"); ax.set_title("Will it rain (1 mm or more)? Two-state model", fontsize=12)
    ax = axes[1]; g = RA[RA["horizon"] == 1]; v = [g["brier_climatology"].mean(), g["brier_two_state"].mean(), g["brier_persistence"].mean()]; x = np.arange(3); ax.bar(x, v, color=[GRAY, BLUE, ORANGE], width=0.6)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.005, f"{vi:.3f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["seasonal\nprobability", "two-state\nmodel", "wet if\nwet today"], fontsize=9); ax.set_ylabel("Brier score tomorrow (lower is better)"); ax.set_ylim(0, 0.32); ax.grid(axis="x", visible=False); ax.set_title("Tomorrow's rain: mean Brier score", fontsize=12)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[1.15, 1]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    lines(a, fs=10); a.set_title("Daily high: skill against the calendar, 12 stations", fontsize=14); grouped(b); b.set_title("Mean skill, persistence and damped persistence", fontsize=14)
    d3 = SK[(SK["element"] == "TMAX") & (SK["horizon"] == 3)]["skill_damped"].mean()
    fig.suptitle(f"Today's weather beats the calendar for about three days: {100 * SM['TMAX_mean_skill_damped_h1']:.0f}% skill tomorrow, {100 * d3:.0f}% in three, then next to nothing", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); horizon(axes); save(fig, "01_horizon.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); methods(axes); save(fig, "02_methods.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3)); seasons(axes); save(fig, "03_seasons.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); bias(axes); save(fig, "04_bias.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8)); rain(axes); save(fig, "05_rain.png")


if __name__ == "__main__":
    main()
