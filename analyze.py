"""Steps 2-4: audit the station records, then ask how far ahead a station's own recent weather beats the calendar.

    ./venv/bin/python analyze.py
        -> results/stations.csv  skill.csv  horizon.csv  climatology.csv  bias.csv  seasons.csv  rain.csv  summary.csv

Data: GHCN-Daily TMAX, TMIN (degrees C) and PRCP (mm) at twelve airport stations, 1 January 1991 to 31 December 2025, days with a quality flag dropped (flagged values are counted within 1991 to 2025). For every test year Y from 2011 to 2025 and every station, the models are fitted on the 20 years before Y only (a rolling
walk-forward), and forecast from each day t of Y for horizons h = 1 to 14 days (rain: 1 to 5):
  climatology   the seasonal cycle (mean and three annual harmonics) plus a linear trend, evaluated at day t+h; the calendar alone;
  persistence   the temperature of day t again;
  damped        climatology at t+h plus phi_h times today's departure from it (phi_h fitted by least squares on the 20-year window);
  week          climatology at t+h plus two terms fitted on the window: today's departure and the mean departure of the last seven days.
All four are scored on the same days (t, t+h and the seven days to t observed) by root-mean-square error, and skill is 1 - MSE / MSE of climatology. The rain forecasts are probabilities that day t+h has 1 mm or more: the climatological probability (a seasonal
curve with two harmonics), persistence (wet if wet today), and a seasonal two-state model (a separate seasonal curve for days after a wet and after a dry day), scored by the Brier score. "Stale" climatology is fitted once on 1991 to 2010 and never refitted.
"""
import gzip
import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
START, END, ORIGIN = date(1991, 1, 1), date(2025, 12, 31), date(2000, 1, 1)
TEST, WIN, H, HR, K, KR = range(2011, 2026), 20, range(1, 15), range(1, 6), 3, 2
STATIONS = {"USW00094728": "New York", "USW00094846": "Chicago", "USW00024233": "Seattle", "USW00023183": "Phoenix", "USW00012839": "Miami", "USW00013874": "Atlanta", "USW00014739": "Boston", "USW00014922": "Minneapolis", "USW00023174": "Los Angeles", "USW00003927": "Dallas-Fort Worth", "USW00022521": "Honolulu", "USW00023050": "Albuquerque"}
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
N = (END - START).days + 1
DAYS = np.arange(N) + (START - ORIGIN).days                       # days since 1 January 2000
DATES = pd.date_range(START, END); YEAR = DATES.year.to_numpy(); MONTH = DATES.month.to_numpy()


def design(days, k, trend):
    w = 2 * np.pi * days / 365.2425; cols = [np.ones(len(days))] + ([days / 3652.5] if trend else [])
    for j in range(1, k + 1): cols += [np.cos(j * w), np.sin(j * w)]
    return np.column_stack(cols)


def fit(X, y):
    ok = np.isfinite(y); A = X[ok]; return np.linalg.solve(A.T @ A, A.T @ y[ok])


def load(sid):
    raw = pd.read_csv(D / f"{sid}.csv.gz", header=None, names=["id", "date", "element", "value", "m", "q", "s", "t"], dtype={"date": str, "q": str, "m": str, "s": str, "t": str}, keep_default_na=False)
    raw = raw[raw["element"].isin(["TMAX", "TMIN", "PRCP"])]; raw = raw.assign(idx=(pd.to_datetime(raw["date"], format="%Y%m%d") - pd.Timestamp(START)).dt.days); raw = raw[(raw["idx"] >= 0) & (raw["idx"] < N)]
    dropped = raw[raw["q"] != ""].groupby("element").size().to_dict(); raw = raw[raw["q"] == ""]; out = {}
    for e in ("TMAX", "TMIN", "PRCP"):
        m = (raw["element"] == e).to_numpy(); a = np.full(N, np.nan); a[raw["idx"].to_numpy()[m]] = raw["value"].to_numpy()[m].astype(float) / 10; out[e] = a
    return out, dropped


def roll7(a):
    c = np.full(N, np.nan); s = np.lib.stride_tricks.sliding_window_view(a, 7); c[6:] = s.mean(axis=1); return c


def main():
    R.mkdir(exist_ok=True)
    meta = {l[:11]: (l[12:20].strip(), l[21:30].strip(), l[38:40], l[41:71].strip()) for l in open(D / "ghcnd-stations.txt")}
    st_rows, sk_rows, cl_rows, bi_rows, se_rows, ra_rows, hz_rows = [], [], [], [], [], [], []
    for sid, name in STATIONS.items():
        obs, dropped = load(sid); X = design(DAYS, K, True); X0 = design(DAYS, K, False); XR = design(DAYS, KR, False)
        st_rows.append({"station": name, "id": sid, "ghcn_name": meta[sid][3], "state": meta[sid][2], "latitude": float(meta[sid][0]), "longitude": float(meta[sid][1]), "days": N, "tmax_observed": int(np.isfinite(obs["TMAX"]).sum()), "tmin_observed": int(np.isfinite(obs["TMIN"]).sum()), "prcp_observed": int(np.isfinite(obs["PRCP"]).sum()),
                        "tmax_flagged": int(dropped.get("TMAX", 0)), "tmin_flagged": int(dropped.get("TMIN", 0)), "prcp_flagged": int(dropped.get("PRCP", 0)), "wet_day_share": float(np.nanmean(obs["PRCP"] >= 1.0)), "mean_tmax": float(np.nanmean(obs["TMAX"])), "mean_tmin": float(np.nanmean(obs["TMIN"]))})
        for el in ("TMAX", "TMIN"):
            y = obs[el]; acc = {h: np.zeros(8) for h in H}; sea = {(h, s): np.zeros(3) for h in (1, 3, 7) for s in ("winter", "spring", "summer", "autumn")}; h1 = {"roll": [], "obs": [], "stale_t": [], "stale_s": []}
            bst = fit(X[(YEAR >= 1991) & (YEAR <= 2010)], y[(YEAR >= 1991) & (YEAR <= 2010)]); bss = fit(X0[(YEAR >= 1991) & (YEAR <= 2010)], y[(YEAR >= 1991) & (YEAR <= 2010)]); cst, css = X @ bst, X0 @ bss
            for Y in TEST:
                win = (YEAR >= Y - WIN) & (YEAR <= Y - 1); b = fit(X[win], y[win]); clim = X @ b; a = y - clim; a7 = roll7(a); widx = np.flatnonzero(win); origin = np.flatnonzero(YEAR == Y)
                for h in H:
                    s = widx[widx + h <= widx[-1]]; ok = np.isfinite(a[s]) & np.isfinite(a[s + h]); phi = float((a[s][ok] * a[s + h][ok]).sum() / (a[s][ok] ** 2).sum())
                    s2 = s[np.isfinite(a[s]) & np.isfinite(a7[s]) & np.isfinite(a[s + h])]; M = np.array([[(a[s2] ** 2).sum(), (a[s2] * a7[s2]).sum()], [(a[s2] * a7[s2]).sum(), (a7[s2] ** 2).sum()]]); v = np.array([(a[s2] * a[s2 + h]).sum(), (a7[s2] * a[s2 + h]).sum()]); b1, b2 = np.linalg.solve(M, v)
                    t = origin[origin + h < N]; tt = t[np.isfinite(y[t]) & np.isfinite(a7[t]) & np.isfinite(y[t + h])]; obs_h = y[tt + h]; c = clim[tt + h]
                    e1, e2, e3, e4 = obs_h - c, obs_h - y[tt], obs_h - (c + phi * a[tt]), obs_h - (c + b1 * a[tt] + b2 * a7[tt])
                    acc[h] += np.array([len(tt), (e1 ** 2).sum(), (e2 ** 2).sum(), (e3 ** 2).sum(), (e4 ** 2).sum(), e1.sum(), obs_h.sum(), (obs_h ** 2).sum()])
                    if h in (1, 3, 7):
                        for sname in ("winter", "spring", "summer", "autumn"):
                            mk = np.array([SEASONS[m] == sname for m in MONTH[tt + h]]); sea[(h, sname)] += np.array([mk.sum(), (e1[mk] ** 2).sum(), (e3[mk] ** 2).sum()])
                    if h == 1: h1["roll"].append(e1); h1["stale_t"].append(obs_h - cst[tt + h]); h1["stale_s"].append(obs_h - css[tt + h])
            for h in H:
                n, s1, s2, s3, s4, sb, so, soo = acc[h]; m = [s1 / n, s2 / n, s3 / n, s4 / n]
                sk_rows.append({"station": name, "element": el, "horizon": h, "n": int(n), "rmse_climatology": np.sqrt(m[0]), "rmse_persistence": np.sqrt(m[1]), "rmse_damped": np.sqrt(m[2]), "rmse_week": np.sqrt(m[3]), "skill_persistence": 1 - m[1] / m[0], "skill_damped": 1 - m[2] / m[0], "skill_week": 1 - m[3] / m[0]})
                if h == 1: cl_rows.append({"station": name, "element": el, "cases": int(n), "sd_observed": float(np.sqrt(soo / n - (so / n) ** 2)), "rmse_climatology": float(np.sqrt(m[0])), "r2_climatology": float(1 - (s1 / n) / (soo / n - (so / n) ** 2))})
            for (h, sname), v in sea.items(): se_rows.append({"station": name, "element": el, "horizon": h, "season": sname, "n": int(v[0]), "skill_damped": float(1 - v[2] / v[1])})
            r, a_, b_ = (np.concatenate(h1[k]) for k in ("roll", "stale_t", "stale_s")); bi_rows.append({"station": name, "element": el, "cases": len(r), "bias_rolling": float(r.mean()), "bias_stale_trend": float(a_.mean()), "bias_stale_seasonal_only": float(b_.mean()), "rmse_rolling": float(np.sqrt((r ** 2).mean())), "rmse_stale_trend": float(np.sqrt((a_ ** 2).mean())), "rmse_stale_seasonal_only": float(np.sqrt((b_ ** 2).mean()))})
        wet = np.where(np.isfinite(obs["PRCP"]), (obs["PRCP"] >= 1.0).astype(float), np.nan); racc = {h: np.zeros(5) for h in HR}
        for Y in TEST:
            win = (YEAR >= Y - WIN) & (YEAR <= Y - 1); widx = np.flatnonzero(win); origin = np.flatnonzero(YEAR == Y); pc = np.clip(XR @ fit(XR[win], wet[win]), 0.001, 0.999)
            for h in HR:
                s = widx[widx + h <= widx[-1]]; s = s[np.isfinite(wet[s]) & np.isfinite(wet[s + h])]; p = {}
                for state in (0.0, 1.0):
                    q = s[wet[s] == state]; p[state] = np.clip(XR @ np.linalg.solve(XR[q + h].T @ XR[q + h], XR[q + h].T @ wet[q + h]), 0.001, 0.999)
                t = origin[origin + h < N]; tt = t[np.isfinite(wet[t]) & np.isfinite(wet[t + h])]; o = wet[tt + h]; pm = np.where(wet[tt] == 1.0, p[1.0][tt + h], p[0.0][tt + h])
                racc[h] += np.array([len(tt), o.sum(), ((pc[tt + h] - o) ** 2).sum(), ((wet[tt] - o) ** 2).sum(), ((pm - o) ** 2).sum()])
        for h in HR:
            n, w, b0, bp, bm = racc[h]; ra_rows.append({"station": name, "horizon": h, "n": int(n), "wet_share": w / n, "brier_climatology": b0 / n, "brier_persistence": bp / n, "brier_two_state": bm / n, "skill_persistence": 1 - bp / b0, "skill_two_state": 1 - bm / b0})
    SK = pd.DataFrame(sk_rows)
    for (name, el), g in SK.groupby(["station", "element"]):
        g = g.sort_values("horizon"); below = g[g["skill_damped"] < 0.05]; pneg = g[g["skill_persistence"] < 0]
        hz_rows.append({"station": name, "element": el, "first_horizon_damped_below_5pct": int(below["horizon"].iloc[0]) if len(below) else 0, "first_horizon_persistence_below_climatology": int(pneg["horizon"].iloc[0]) if len(pneg) else 0, "skill_damped_h1": float(g["skill_damped"].iloc[0]), "skill_damped_h14": float(g["skill_damped"].iloc[-1])})
    pd.DataFrame(st_rows).to_csv(R / "stations.csv", index=False); SK.to_csv(R / "skill.csv", index=False); pd.DataFrame(hz_rows).to_csv(R / "horizon.csv", index=False); pd.DataFrame(cl_rows).to_csv(R / "climatology.csv", index=False); pd.DataFrame(bi_rows).to_csv(R / "bias.csv", index=False); pd.DataFrame(se_rows).to_csv(R / "seasons.csv", index=False); pd.DataFrame(ra_rows).to_csv(R / "rain.csv", index=False)
    summ = {"stations": len(STATIONS), "test_years": len(TEST), "days": N}
    for el in ("TMAX", "TMIN"):
        for h in (1, 2, 3, 5, 7, 14):
            g = SK[(SK["element"] == el) & (SK["horizon"] == h)]; summ[f"{el}_mean_skill_damped_h{h}"] = float(g["skill_damped"].mean()); summ[f"{el}_mean_skill_persistence_h{h}"] = float(g["skill_persistence"].mean()); summ[f"{el}_mean_skill_week_h{h}"] = float(g["skill_week"].mean())
    pd.DataFrame(list(summ.items()), columns=["key", "value"]).to_csv(R / "summary.csv", index=False)
    for f in ("stations", "horizon", "climatology", "bias", "rain", "summary"):
        print("==", f); print(pd.read_csv(R / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
