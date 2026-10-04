"""Step 5: the project's check. Recompute with different code, in plain Python straight from the downloaded files (gzip, csv, math, datetime and loops: no pandas, numpy or scipy), and compare with results/:

    ./venv/bin/python check.py          (CHECK_ONLY=audit, temperature or rain runs one section)

the station audit (days observed, quality-flagged values, wet-day shares, means); then for every station, element and test year the seasonal-plus-trend climatology fitted on the 20 years before it (normal equations solved by Gaussian elimination), the anomalies, the damped
and two-term regressions for each horizon, and the pooled errors of the four forecasts (skill, climatology R-squared, bias of the rolling and the stale climatologies, skill by season, the horizons at which skill falls below 5%); and the rain probabilities (a seasonal curve and a
seasonal two-state model, fitted on the window) and their Brier scores. The temperature statistics are compared at a relative tolerance of 1e-8 because the two programs sum in a different order.
"""
import csv
import gzip
import math
import os
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
START, END, ORIGIN = date(1991, 1, 1), date(2025, 12, 31), date(2000, 1, 1)
TEST, WIN, H, HR, K, KR = list(range(2011, 2026)), 20, list(range(1, 15)), list(range(1, 6)), 3, 2
STATIONS = {"USW00094728": "New York", "USW00094846": "Chicago", "USW00024233": "Seattle", "USW00023183": "Phoenix", "USW00012839": "Miami", "USW00013874": "Atlanta", "USW00014739": "Boston", "USW00014922": "Minneapolis", "USW00023174": "Los Angeles", "USW00003927": "Dallas-Fort Worth", "USW00022521": "Honolulu", "USW00023050": "Albuquerque"}
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
N = (END - START).days + 1
DATES = [START + timedelta(days=i) for i in range(N)]; YEAR = [d.year for d in DATES]; MONTH = [d.month for d in DATES]; DAYS = [(d - ORIGIN).days for d in DATES]
FIRST = {y: YEAR.index(y) for y in set(YEAR)}; LAST = {y: N - 1 - YEAR[::-1].index(y) for y in set(YEAR)}
ONLY = os.environ.get("CHECK_ONLY")
n_ok = 0


def same(label, got, want, rel=1e-8):
    global n_ok
    n_ok += 1; want = float(want)
    assert abs(got - want) <= rel * abs(want) + 1e-12, f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def design_rows(k, trend):
    rows = []
    for d in DAYS:
        w = 2 * math.pi * d / 365.2425; r = [1.0] + ([d / 3652.5] if trend else [])
        for j in range(1, k + 1):
            r += [math.cos(j * w), math.sin(j * w)]
        rows.append(r)
    return rows


def solve(a, b):
    n = len(b); m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c])); m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def lsq(X, y, idx):
    """least squares on the rows idx where y is observed"""
    k = len(X[0]); a = [[0.0] * k for _ in range(k)]; b = [0.0] * k
    for i in idx:
        yi = y[i]
        if yi is None:
            continue
        xi = X[i]
        for p in range(k):
            xp = xi[p]; b[p] += xp * yi; ap = a[p]
            for q in range(p, k):
                ap[q] += xp * xi[q]
    for p in range(k):
        for q in range(p):
            a[p][q] = a[q][p]
    return solve(a, b)


def load(sid):
    obs = {"TMAX": [None] * N, "TMIN": [None] * N, "PRCP": [None] * N}; flagged = {"TMAX": 0, "TMIN": 0, "PRCP": 0}
    for r in csv.reader(gzip.open(D / f"{sid}.csv.gz", "rt")):
        e = r[2]
        if e not in obs:
            continue
        dd = date(int(r[1][:4]), int(r[1][4:6]), int(r[1][6:8])); i = (dd - START).days
        if not 0 <= i < N:
            continue
        if r[5]:
            flagged[e] += 1; continue
        obs[e][i] = int(r[3]) / 10
    return obs, flagged


def mean(v):
    return sum(v) / len(v)


def main():
    meta = {l[:11]: l for l in open(D / "ghcnd-stations.txt")}
    data = {sid: load(sid) for sid in STATIONS}
    if ONLY in (None, "audit"):
        for r in table("stations"):
            sid = r["id"]; obs, fl = data[sid]; assert STATIONS[sid] == r["station"]; l = meta[sid]; assert r["ghcn_name"] == l[41:71].strip() and r["state"] == l[38:40]; same("lat", float(l[12:20]), r["latitude"]); same("lon", float(l[21:30]), r["longitude"]); same("days", N, r["days"])
            for e, c in (("TMAX", "tmax"), ("TMIN", "tmin"), ("PRCP", "prcp")):
                same(f"{sid} {e} observed", sum(1 for v in obs[e] if v is not None), r[f"{c}_observed"]); same(f"{sid} {e} flagged", fl[e], r[f"{c}_flagged"])
            same("wet share", mean([1.0 if v >= 1.0 else 0.0 for v in obs["PRCP"] if v is not None]), r["wet_day_share"]); same("mean tmax", mean([v for v in obs["TMAX"] if v is not None]), r["mean_tmax"]); same("mean tmin", mean([v for v in obs["TMIN"] if v is not None]), r["mean_tmin"])
        sm = {r["key"]: float(r["value"]) for r in table("summary")}; same("stations", len(STATIONS), sm["stations"]); same("test years", len(TEST), sm["test_years"]); same("days", N, sm["days"])
    X = design_rows(K, True) if ONLY in (None, "temperature") else None
    skill, clim_rows, bias_rows, sea_rows = {}, {}, {}, {}
    if ONLY in (None, "temperature"):
        X0 = design_rows(K, False)
        for sid, name in STATIONS.items():
            obs, _ = data[sid]
            for el in ("TMAX", "TMIN"):
                y = obs[el]; acc = {h: [0.0] * 8 for h in H}; sea = {(h, s): [0.0] * 3 for h in (1, 3, 7) for s in ("winter", "spring", "summer", "autumn")}; h1 = {"roll": [], "stale_t": [], "stale_s": []}
                tr = range(FIRST[1991], LAST[2010] + 1); bst = lsq(X, y, tr); bss = lsq(X0, y, tr)
                cst = [sum(b * x for b, x in zip(bst, X[i])) for i in range(N)]; css = [sum(b * x for b, x in zip(bss, X0[i])) for i in range(N)]
                for Y in TEST:
                    w0, w1 = FIRST[Y - WIN], LAST[Y - 1]; b = lsq(X, y, range(w0, w1 + 1)); clim = [sum(bb * x for bb, x in zip(b, X[i])) for i in range(N)]; a = [None if y[i] is None else y[i] - clim[i] for i in range(N)]
                    a7 = [None] * N
                    for i in range(6, N):
                        seg = a[i - 6:i + 1]
                        if None not in seg:
                            a7[i] = sum(seg) / 7
                    for h in H:
                        sxy = sxx = 0.0; m11 = m12 = m22 = v1 = v2 = 0.0
                        for s in range(w0, w1 - h + 1):
                            as_, at = a[s], a[s + h]
                            if as_ is None or at is None:
                                continue
                            sxy += as_ * at; sxx += as_ * as_
                            b7 = a7[s]
                            if b7 is not None:
                                m11 += as_ * as_; m12 += as_ * b7; m22 += b7 * b7; v1 += as_ * at; v2 += b7 * at
                        phi = sxy / sxx; det = m11 * m22 - m12 * m12; b1 = (v1 * m22 - m12 * v2) / det; b2 = (m11 * v2 - m12 * v1) / det
                        cnt = s1 = s2 = s3 = s4 = sb = so = soo = 0.0
                        for t in range(FIRST[Y], LAST[Y] + 1):
                            if t + h >= N:
                                break
                            yt, y7, yh = y[t], a7[t], y[t + h]
                            if yt is None or y7 is None or yh is None:
                                continue
                            c = clim[t + h]; at = a[t]; e1 = yh - c; e2 = yh - yt; e3 = yh - (c + phi * at); e4 = yh - (c + b1 * at + b2 * y7)
                            cnt += 1; s1 += e1 * e1; s2 += e2 * e2; s3 += e3 * e3; s4 += e4 * e4; sb += e1; so += yh; soo += yh * yh
                            if h in (1, 3, 7):
                                q = sea[(h, SEASONS[MONTH[t + h]])]; q[0] += 1; q[1] += e1 * e1; q[2] += e3 * e3
                            if h == 1:
                                h1["roll"].append(e1); h1["stale_t"].append(yh - cst[t + h]); h1["stale_s"].append(yh - css[t + h])
                        q = acc[h]
                        for j, v in enumerate((cnt, s1, s2, s3, s4, sb, so, soo)):
                            q[j] += v
                skill[(name, el)] = acc; sea_rows[(name, el)] = sea; bias_rows[(name, el)] = h1
        for r in table("skill"):
            n_, s1, s2, s3, s4, sb, so, soo = skill[(r["station"], r["element"])][int(r["horizon"])]; m = [s1 / n_, s2 / n_, s3 / n_, s4 / n_]; same("n", n_, r["n"])
            for j, k in enumerate(("climatology", "persistence", "damped", "week")):
                same(f"rmse {k}", math.sqrt(m[j]), r[f"rmse_{k}"])
            for j, k in ((1, "persistence"), (2, "damped"), (3, "week")):
                same(f"skill {k} {r['station']} {r['element']} {r['horizon']}", 1 - m[j] / m[0], r[f"skill_{k}"])
        for r in table("climatology"):
            n_, s1, _, _, _, _, so, soo = skill[(r["station"], r["element"])][1]; var = soo / n_ - (so / n_) ** 2; same("cases", n_, r["cases"]); same("sd", math.sqrt(var), r["sd_observed"]); same("rmse clim", math.sqrt(s1 / n_), r["rmse_climatology"]); same("r2 clim", 1 - (s1 / n_) / var, r["r2_climatology"])
        for r in table("bias"):
            h1 = bias_rows[(r["station"], r["element"])]; same("cases", len(h1["roll"]), r["cases"])
            for k, c in (("roll", "rolling"), ("stale_t", "stale_trend"), ("stale_s", "stale_seasonal_only")):
                same(f"bias {c}", mean(h1[k]), r[f"bias_{c}"], 1e-7); same(f"rmse {c}", math.sqrt(mean([e * e for e in h1[k]])), r[f"rmse_{c}"])
        for r in table("seasons"):
            v = sea_rows[(r["station"], r["element"])][(int(r["horizon"]), r["season"])]; same("season n", v[0], r["n"]); same(f"season skill {r['station']} {r['element']} {r['horizon']} {r['season']}", 1 - v[2] / v[1], r["skill_damped"])
        for r in table("horizon"):
            acc = skill[(r["station"], r["element"])]; sd = {h: 1 - acc[h][3] / acc[h][1] for h in H}; sp = {h: 1 - acc[h][2] / acc[h][1] for h in H}
            below = [h for h in H if sd[h] < 0.05]; pneg = [h for h in H if sp[h] < 0]
            same("first below 5%", below[0] if below else 0, r["first_horizon_damped_below_5pct"]); same("persistence below", pneg[0] if pneg else 0, r["first_horizon_persistence_below_climatology"]); same("h1", sd[1], r["skill_damped_h1"]); same("h14", sd[14], r["skill_damped_h14"])
        sm = {r["key"]: float(r["value"]) for r in table("summary")}
        for el in ("TMAX", "TMIN"):
            for h in (1, 2, 3, 5, 7, 14):
                for j, k in ((3, "damped"), (2, "persistence"), (4, "week")):
                    same(f"summary {el} {k} {h}", mean([1 - skill[(nm, el)][h][j] / skill[(nm, el)][h][1] for nm in STATIONS.values()]), sm[f"{el}_mean_skill_{k}_h{h}"])
    if ONLY in (None, "rain"):
        XR = design_rows(KR, False); rain = {}
        for sid, name in STATIONS.items():
            obs, _ = data[sid]; wet = [None if v is None else (1.0 if v >= 1.0 else 0.0) for v in obs["PRCP"]]; racc = {h: [0.0] * 5 for h in HR}
            for Y in TEST:
                w0, w1 = FIRST[Y - WIN], LAST[Y - 1]; b = lsq(XR, wet, range(w0, w1 + 1)); pc = [min(0.999, max(0.001, sum(bb * x for bb, x in zip(b, XR[i])))) for i in range(N)]
                for h in HR:
                    pairs = [s for s in range(w0, w1 - h + 1) if wet[s] is not None and wet[s + h] is not None]; p = {}
                    for state in (0.0, 1.0):
                        q = [s + h for s in pairs if wet[s] == state]; bb = lsq(XR, wet, q); p[state] = [min(0.999, max(0.001, sum(c * x for c, x in zip(bb, XR[i])))) for i in range(N)]
                    cnt = w = b0 = bp = bm = 0.0
                    for t in range(FIRST[Y], LAST[Y] + 1):
                        if t + h >= N or wet[t] is None or wet[t + h] is None:
                            continue
                        o = wet[t + h]; pm = p[wet[t]][t + h]; cnt += 1; w += o; b0 += (pc[t + h] - o) ** 2; bp += (wet[t] - o) ** 2; bm += (pm - o) ** 2
                    for j, v in enumerate((cnt, w, b0, bp, bm)):
                        racc[h][j] += v
            rain[name] = racc
        for r in table("rain"):
            n_, w, b0, bp, bm = rain[r["station"]][int(r["horizon"])]; same("rain n", n_, r["n"]); same("wet share", w / n_, r["wet_share"]); same("brier clim", b0 / n_, r["brier_climatology"]); same("brier persistence", bp / n_, r["brier_persistence"]); same("brier two-state", bm / n_, r["brier_two_state"])
            same(f"rain skill persistence {r['station']} {r['horizon']}", 1 - bp / b0, r["skill_persistence"]); same(f"rain skill two-state {r['station']} {r['horizon']}", 1 - bm / b0, r["skill_two_state"], 1e-7)
    print(f"check.py: {n_ok} recomputed numbers agree with results/ (sections: {ONLY or 'all'}); everything is written out in plain Python from the downloaded files")


if __name__ == "__main__":
    main()
