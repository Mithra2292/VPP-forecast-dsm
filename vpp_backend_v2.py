"""
VPP backend v2 — grid-connected, forecast-driven, DSM-aware.

Changes from the original (islanded) vpp_backend.py:
  1. CSV ingestion for user-supplied solar/wind/load/EV history, and a
     DSM flexible-load spec CSV (shiftable + curtailable tasks).
  2. Forecasting layer (forecasting.py) turns that history into a 24-hour
     forecasted resource/load/EV profile BEFORE dispatch ever runs.
  3. Grid import is a real, TOU-priced, capacity-capped source in the
     dispatch priority stack (capacity cap defaults to the real HT-IIA
     1750 kVA contract demand) - not a VOLL-penalized absence.
  4. Reliability objective is EENS (kWh/yr unserved) instead of LOLP (%).
     LCOE is unchanged.
  5. Decommissioning cost is generalized to every technology, not just SMR.
  6. The coalition game is reframed as PROFIT allocation (matching your
     architecture diagram's "Shapley Value (Profit allocation)"):
       profit(U) = cost(emptyset) - cost(U)
     cost(emptyset) = cost of buying 100% of demand from the grid at TOU
     tariff (the "build nothing" baseline, now a real, not hypothetical,
     option since the VPP is grid-connected).
     EV remains a DSM/V2G flexibility resource feeding the dispatch - it
     is NOT a 5th NSGA-II sizing variable or 5th Shapley player here.
"""

import itertools
from math import factorial

import numpy as np
import pandas as pd

import forecasting as fcst

HOURS_PER_YEAR = 8760
PLAYERS = ["PV", "WT", "SMR", "H2"]

# ----------------------------------------------------------------------
# 1. CSV INGESTION
# ----------------------------------------------------------------------
def load_solar_csv(file) -> pd.DataFrame:
    """Expects columns: timestamp, ghi (W/m^2)."""
    df = pd.read_csv(file, parse_dates=["timestamp"]).set_index("timestamp")
    return df[["ghi"]].sort_index()


def load_wind_csv(file) -> pd.DataFrame:
    """Expects columns: timestamp, wind_speed (m/s)."""
    df = pd.read_csv(file, parse_dates=["timestamp"]).set_index("timestamp")
    return df[["wind_speed"]].sort_index()


def load_demand_csv(file) -> pd.DataFrame:
    """Expects columns: timestamp, load_kw."""
    df = pd.read_csv(file, parse_dates=["timestamp"]).set_index("timestamp")
    return df[["load_kw"]].sort_index()


def load_ev_csv(file, fleet_size) -> pd.DataFrame:
    """Expects either:
      (a) an hourly log: timestamp, n_present, avg_soc_frac  -- used as-is, or
      (b) an event log: vehicle_id, plug_in, plug_out, soc_in, soc_out
          -- aggregated into the hourly (n_present, avg_soc_frac) form.
    Auto-detects which shape was given."""
    df_raw = pd.read_csv(file)
    if "n_present" in df_raw.columns:
        df = df_raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")[["n_present", "avg_soc_frac"]].sort_index()

    df_raw["plug_in"] = pd.to_datetime(df_raw["plug_in"])
    df_raw["plug_out"] = pd.to_datetime(df_raw["plug_out"])
    start = df_raw["plug_in"].min().floor("h")
    end = df_raw["plug_out"].max().ceil("h")
    hours = pd.date_range(start, end, freq="h")
    n_present = pd.Series(0, index=hours)
    soc_sum = pd.Series(0.0, index=hours)
    soc_cnt = pd.Series(0, index=hours)
    for _, row in df_raw.iterrows():
        mask = (hours >= row["plug_in"]) & (hours < row["plug_out"])
        n_present[mask] += 1
        frac = np.linspace(row["soc_in"], row["soc_out"], mask.sum()) if mask.sum() > 0 else []
        soc_sum[mask] += frac
        soc_cnt[mask] += 1
    avg_soc = (soc_sum / soc_cnt.replace(0, np.nan)).ffill().fillna(0.5)
    return pd.DataFrame({"n_present": n_present.clip(0, fleet_size), "avg_soc_frac": avg_soc})


def load_dsm_flex_csv(file) -> pd.DataFrame:
    """Expects columns: task_id, bus_or_zone, type (shiftable|curtailable),
    power_kw, energy_kwh, earliest_hour, latest_hour, priority (1=high..5=low).
    This is the user-declared DSM participation pool - what they're
    offering to shift/curtail, not something the model assumes."""
    df = pd.read_csv(file)
    required = {"task_id", "type", "power_kw", "energy_kwh", "earliest_hour", "latest_hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DSM flexible-load CSV missing columns: {missing}")
    if "priority" not in df.columns:
        df["priority"] = 3
    return df


# ----------------------------------------------------------------------
# 2. FORECASTING WRAPPER - history CSVs -> 24h forecasted hourly arrays
# ----------------------------------------------------------------------
def build_forecasts(solar_df=None, wind_df=None, load_df=None, ev_df=None,
                     fleet_size=0, horizon=24):
    """Returns a dict of 24h-ahead forecasted series (numpy arrays). Falls
    back to the original synthetic generators for any input the user
    didn't upload, so the app still runs end-to-end without real data."""
    out = {}

    if solar_df is not None and len(solar_df) >= 20:
        pv_frac, _, pv_model = fcst.forecast_pv(solar_df, horizon=horizon)
        out["pv_kwp_frac"] = pv_frac.values
        out["pv_source"] = "forecast"
    else:
        hrs = np.arange(horizon)
        daylight = (hrs >= 6) & (hrs <= 18)
        theta = np.where(daylight, np.pi * (hrs - 6) / 12, 0)
        out["pv_kwp_frac"] = np.where(daylight, np.sin(theta), 0.0)
        out["pv_source"] = "synthetic (no solar CSV uploaded)"

    if wind_df is not None and len(wind_df) >= 20:
        wt_frac, _, wt_model = fcst.forecast_wind(wind_df, horizon=horizon)
        out["wt_frac"] = wt_frac.values
        out["wt_source"] = "forecast"
    else:
        rng = np.random.default_rng(2)
        out["wt_frac"] = np.clip(rng.weibull(2.0, horizon) * 0.3, 0, 1)
        out["wt_source"] = "synthetic (no wind CSV uploaded)"

    if load_df is not None and len(load_df) >= 20:
        load_kw, _, load_model = fcst.forecast_load(load_df, horizon=horizon)
        out["load_kw"] = load_kw.values
        out["load_source"] = "forecast"
    else:
        hrs = np.arange(horizon)
        out["load_kw"] = 150 + 350 * np.clip(np.sin((hrs - 7) / 24 * 2 * np.pi), 0, None)
        out["load_source"] = "synthetic (no load CSV uploaded)"

    if ev_df is not None and len(ev_df) >= 20 and fleet_size > 0:
        n_present, _, _ = fcst.forecast_ev_presence(ev_df, fleet_size, horizon=horizon)
        soc, _, _ = fcst.forecast_ev_soc(ev_df, horizon=horizon)
        out["ev_n_present"] = n_present.values
        out["ev_avg_soc"] = soc.values
        out["ev_source"] = "forecast"
    else:
        hrs = np.arange(horizon)
        out["ev_n_present"] = np.where((hrs >= 18) | (hrs < 8), fleet_size * 0.8, fleet_size * 0.1)
        out["ev_avg_soc"] = np.where((hrs >= 18) | (hrs < 8), 0.3 + 0.5 * (hrs % 24) / 12, 0.8)
        out["ev_source"] = "synthetic (no EV CSV uploaded)"

    return out


# ----------------------------------------------------------------------
# 3. TECHNOLOGY PARAMETERS - decommissioning now generalized to every tech
# ----------------------------------------------------------------------
DEFAULT_TECH_PARAMS = {
    "PV":   dict(capex_per_kw=45000, om_pct=0.01, lifetime=25, decomm_frac=0.04),
    "WT":   dict(capex_per_kw=75000, om_pct=0.02, lifetime=20, decomm_frac=0.07),
    "SMR":  dict(capex_per_kw=650000, om_pct=0.025, lifetime=60,
                 fuel_cost_per_kwh=0.9, base_frac=0.55, max_ramp_pct=0.25,
                 decomm_frac=0.12),
    "EL":   dict(capex_per_kw=55000, om_pct=0.02, lifetime=12, eta=0.68, decomm_frac=0.05),
    "TANK": dict(capex_per_kwh=6000, om_pct=0.01, lifetime=22, decomm_frac=0.05),
    "FC":   dict(capex_per_kw=95000, om_pct=0.03, lifetime=15, eta=0.55,
                 max_ramp_pct=1.0, decomm_frac=0.08),
}

DISCOUNT_RATE = 0.08
VOLL = 120.0  # Rs/kWh - only applies to genuine curtailment beyond the grid cap
GRID_CAPACITY_KW = 1750.0 * 0.95  # HT-IIA 1750 kVA contract demand, ~0.95 PF -> kW


def crf(rate, n):
    if rate == 0:
        return 1.0 / n
    return rate * (1 + rate) ** n / ((1 + rate) ** n - 1)


def sinking_fund_factor(rate, n):
    if rate == 0:
        return 1.0 / n
    return rate / ((1 + rate) ** n - 1)


def tou_tariff(hour):
    if 18 <= hour < 22:
        return 10.0
    if 7 <= hour < 10:
        return 7.5
    return 5.0


# ----------------------------------------------------------------------
# 4. DISPATCH - grid-connected, DSM-aware, forecast-driven
# ----------------------------------------------------------------------
def dispatch_simulate(x, forecasts, dsm_tasks, tech=DEFAULT_TECH_PARAMS,
                       grid_cap_kw=GRID_CAPACITY_KW):
    """x = [C_PV, C_WT, P_SMR, P_EL, C_TANK, P_FC]
    forecasts: dict from build_forecasts() (24h arrays)
    dsm_tasks: DataFrame from load_dsm_flex_csv(), or None/empty."""
    C_PV, C_WT, P_SMR, P_EL, C_TANK, P_FC = x
    n = len(forecasts["load_kw"])

    p_pv = C_PV * forecasts["pv_kwp_frac"]
    p_wt = C_WT * forecasts["wt_frac"]
    load = forecasts["load_kw"]

    smr_base = tech["SMR"]["base_frac"] * P_SMR
    smr_ramp_step = tech["SMR"]["max_ramp_pct"] * P_SMR
    fc_ramp_step = tech["FC"]["max_ramp_pct"] * P_FC

    tank_soc = 0.5 * C_TANK
    smr_out_prev = smr_base
    fc_out_prev = 0.0

    # DSM task pool: remaining shiftable/curtailable energy per task, reset daily
    if dsm_tasks is not None and len(dsm_tasks) > 0:
        shift_tasks = dsm_tasks[dsm_tasks["type"] == "shiftable"].copy()
        curt_tasks = dsm_tasks[dsm_tasks["type"] == "curtailable"].copy()
    else:
        shift_tasks = pd.DataFrame(columns=["energy_kwh", "power_kw", "earliest_hour", "latest_hour"])
        curt_tasks = pd.DataFrame(columns=["energy_kwh", "power_kw", "earliest_hour", "latest_hour"])
    shift_remaining = shift_tasks["energy_kwh"].to_dict() if len(shift_tasks) else {}
    curt_remaining = curt_tasks["energy_kwh"].to_dict() if len(curt_tasks) else {}

    unmet = np.zeros(n)
    curtailed_re = np.zeros(n)
    smr_trace = np.zeros(n)
    fc_trace = np.zeros(n)
    grid_trace = np.zeros(n)
    shift_trace = np.zeros(n)
    dsm_curt_trace = np.zeros(n)
    grid_cost_trace = np.zeros(n)
    tank_trace = np.zeros(n)

    for h in range(n):
        re_avail = p_pv[h] + p_wt[h]
        smr_out = smr_base
        net = load[h] - re_avail - smr_out
        hour_of_day = h % 24

        if net > 0:
            # fuel cell first (fast response)
            fc_target = min(net, P_FC, fc_out_prev + fc_ramp_step)
            fc_actual = min(fc_target, tank_soc * tech["FC"]["eta"])
            tank_soc -= fc_actual / max(tech["FC"]["eta"], 1e-6)
            net -= fc_actual

            # SMR ramp (slow)
            if net > 0 and P_SMR > smr_base:
                smr_extra = min(net, P_SMR - smr_base, smr_out_prev + smr_ramp_step - smr_base)
                smr_extra = max(smr_extra, 0)
                smr_out += smr_extra
                net -= smr_extra

            # DSM load shifting (cheapest non-physical lever, tried before grid)
            shift_kw = 0.0
            if net > 0:
                for tid in list(shift_remaining.keys()):
                    row = shift_tasks.loc[tid]
                    if row["earliest_hour"] <= hour_of_day <= row["latest_hour"] and shift_remaining[tid] > 0:
                        take = min(net - shift_kw, row["power_kw"], shift_remaining[tid])
                        if take > 0:
                            shift_kw += take
                            shift_remaining[tid] -= take
                    if shift_kw >= net:
                        break
                net -= shift_kw

            # grid import (TOU-priced, capacity-capped)
            grid_kw = 0.0
            if net > 0:
                grid_kw = min(net, grid_cap_kw)
                net -= grid_kw

            # DSM curtailment (last resort, only if grid cap exceeded)
            dsm_curt_kw = 0.0
            if net > 0:
                for tid in list(curt_remaining.keys()):
                    row = curt_tasks.loc[tid]
                    if row["earliest_hour"] <= hour_of_day <= row["latest_hour"] and curt_remaining[tid] > 0:
                        take = min(net - dsm_curt_kw, row["power_kw"], curt_remaining[tid])
                        if take > 0:
                            dsm_curt_kw += take
                            curt_remaining[tid] -= take
                    if dsm_curt_kw >= net:
                        break
                net -= dsm_curt_kw

            unmet[h] = max(net, 0)  # genuinely unserved -> EENS
            el_out = 0.0
            fc_trace[h] = fc_actual
            grid_trace[h] = grid_kw
            shift_trace[h] = shift_kw
            dsm_curt_trace[h] = dsm_curt_kw
            grid_cost_trace[h] = grid_kw * tou_tariff(hour_of_day)
        else:
            surplus = -net
            el_in = min(surplus, P_EL)
            h2_produced = el_in * tech["EL"]["eta"]
            room = C_TANK - tank_soc
            h2_stored = min(h2_produced, room)
            el_out = h2_stored / max(tech["EL"]["eta"], 1e-6) if tech["EL"]["eta"] > 0 else 0
            tank_soc += h2_stored
            curtailed_re[h] = surplus - el_out
            fc_trace[h] = 0.0
            grid_trace[h] = 0.0

        smr_trace[h] = smr_out
        tank_trace[h] = tank_soc
        smr_out_prev = smr_out
        fc_out_prev = fc_trace[h]

    served = load - unmet
    return dict(p_pv=p_pv, p_wt=p_wt, smr=smr_trace, fc=fc_trace, grid=grid_trace,
                shift=shift_trace, dsm_curt=dsm_curt_trace, grid_cost=grid_cost_trace,
                unmet=unmet, curtailed_re=curtailed_re, tank_soc=tank_trace,
                load=load, served=served)


# ----------------------------------------------------------------------
# 5. COST, LCOE, EENS
# ----------------------------------------------------------------------
def annualized_fixed_cost(x, tech, r):
    C_PV, C_WT, P_SMR, P_EL, C_TANK, P_FC = x
    breakdown = {}
    breakdown["PV_capex_om"] = C_PV * tech["PV"]["capex_per_kw"] * (crf(r, tech["PV"]["lifetime"]) + tech["PV"]["om_pct"])
    breakdown["WT_capex_om"] = C_WT * tech["WT"]["capex_per_kw"] * (crf(r, tech["WT"]["lifetime"]) + tech["WT"]["om_pct"])
    breakdown["SMR_capex_om"] = P_SMR * tech["SMR"]["capex_per_kw"] * (crf(r, tech["SMR"]["lifetime"]) + tech["SMR"]["om_pct"])
    breakdown["EL_capex_om"] = P_EL * tech["EL"]["capex_per_kw"] * (crf(r, tech["EL"]["lifetime"]) + tech["EL"]["om_pct"])
    breakdown["Tank_capex_om"] = C_TANK * tech["TANK"]["capex_per_kwh"] * (crf(r, tech["TANK"]["lifetime"]) + tech["TANK"]["om_pct"])
    breakdown["FC_capex_om"] = P_FC * tech["FC"]["capex_per_kw"] * (crf(r, tech["FC"]["lifetime"]) + tech["FC"]["om_pct"])

    # decommissioning, now for every technology (sinking-fund annuity)
    caps = dict(PV=C_PV, WT=C_WT, SMR=P_SMR, EL=P_EL, TANK=C_TANK, FC=P_FC)
    capex_key = dict(PV="capex_per_kw", WT="capex_per_kw", SMR="capex_per_kw",
                      EL="capex_per_kw", TANK="capex_per_kwh", FC="capex_per_kw")
    for tname in ["PV", "WT", "SMR", "EL", "TANK", "FC"]:
        cap_cost = caps[tname] * tech[tname][capex_key[tname]]
        breakdown[f"{tname}_decomm"] = cap_cost * tech[tname]["decomm_frac"] * sinking_fund_factor(r, tech[tname]["lifetime"])

    return sum(breakdown.values()), breakdown


def compute_objectives(x, forecasts, dsm_tasks, tech=DEFAULT_TECH_PARAMS, r=DISCOUNT_RATE,
                        scale_factor=365.0, voll=VOLL, grid_cap_kw=GRID_CAPACITY_KW):
    """scale_factor: how many days/year this forecasted profile represents
    (default 365 - one forecasted day standing in for the whole year;
    pass a smaller number if you're stitching together several
    forecasted representative days instead)."""
    sim = dispatch_simulate(x, forecasts, dsm_tasks, tech, grid_cap_kw)

    E_load_yr = sim["load"].sum() * scale_factor
    E_served_yr = sim["served"].sum() * scale_factor
    E_unmet_yr = sim["unmet"].sum() * scale_factor  # <-- this IS the EENS
    annual_smr_energy = sim["smr"].sum() * scale_factor
    annual_grid_cost = sim["grid_cost"].sum() * scale_factor

    fixed_cost, breakdown = annualized_fixed_cost(x, tech, r)
    smr_fuel_cost = annual_smr_energy * tech["SMR"]["fuel_cost_per_kwh"]
    breakdown["SMR_fuel"] = smr_fuel_cost
    breakdown["Grid_import"] = annual_grid_cost

    cost_yr = fixed_cost + smr_fuel_cost + annual_grid_cost
    lcoe = cost_yr / max(E_served_yr, 1e-6)
    scalar_cost = cost_yr + voll * E_unmet_yr  # genuine curtailment only

    return dict(lcoe=lcoe, eens_kwh=E_unmet_yr, cost_yr=cost_yr, scalar_cost=scalar_cost,
                E_load_yr=E_load_yr, E_served_yr=E_served_yr, breakdown=breakdown, sim=sim)


# ----------------------------------------------------------------------
# 6. NSGA-II (objectives: LCOE, EENS) - mechanics unchanged from before
# ----------------------------------------------------------------------
def fast_nondominated_sort(objs):
    n = len(objs)
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def crowding_distance(objs, front):
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = np.inf
        return dist
    for m in range(objs.shape[1]):
        idx_sorted = sorted(front, key=lambda i: objs[i, m])
        dist[idx_sorted[0]] = np.inf
        dist[idx_sorted[-1]] = np.inf
        fmin, fmax = objs[idx_sorted[0], m], objs[idx_sorted[-1], m]
        if fmax - fmin < 1e-12:
            continue
        for k in range(1, len(idx_sorted) - 1):
            dist[idx_sorted[k]] += (objs[idx_sorted[k + 1], m] - objs[idx_sorted[k - 1], m]) / (fmax - fmin)
    return dist


def tournament_select(pop, rank, crowd, k=2):
    idx = np.random.randint(0, len(pop), size=k)
    best = idx[0]
    for i in idx[1:]:
        if rank[i] < rank[best] or (rank[i] == rank[best] and crowd.get(i, 0) > crowd.get(best, 0)):
            best = i
    return pop[best].copy()


def sbx_crossover(p1, p2, lb, ub, eta=15, pcross=0.9):
    if np.random.rand() > pcross:
        return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if ub[i] <= lb[i]:
            continue
        if np.random.rand() < 0.5 and abs(p1[i] - p2[i]) > 1e-9:
            x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            rand = np.random.rand()
            beta = 1 + (2 * (x1 - lb[i]) / max(x2 - x1, 1e-9))
            alpha = 2 - beta ** (-(eta + 1))
            if rand <= 1 / alpha:
                betaq = (rand * alpha) ** (1 / (eta + 1))
            else:
                betaq = (1 / (2 - rand * alpha)) ** (1 / (eta + 1))
            c1[i] = np.clip(0.5 * ((x1 + x2) - betaq * (x2 - x1)), lb[i], ub[i])
            c2[i] = np.clip(0.5 * ((x1 + x2) + betaq * (x2 - x1)), lb[i], ub[i])
    return c1, c2


def polynomial_mutation(x, lb, ub, eta=20, pmut=0.15):
    y = x.copy()
    for i in range(len(x)):
        if ub[i] <= lb[i]:
            continue
        if np.random.rand() < pmut:
            u = np.random.rand()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] = np.clip(x[i] + delta * (ub[i] - lb[i]), lb[i], ub[i])
    return y


def run_nsga2(forecasts, dsm_tasks, lb, ub, pop_size=30, generations=20,
              tech=DEFAULT_TECH_PARAMS, scale_factor=365.0, voll=VOLL, r=DISCOUNT_RATE,
              grid_cap_kw=GRID_CAPACITY_KW, progress_cb=None):
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    ndim = len(lb)
    pop = np.array([lb + np.random.rand(ndim) * (ub - lb) for _ in range(pop_size)])

    def evaluate(P):
        objs = np.zeros((len(P), 2))
        for i, ind in enumerate(P):
            res = compute_objectives(ind, forecasts, dsm_tasks, tech, r, scale_factor, voll, grid_cap_kw)
            objs[i] = [res["lcoe"], res["eens_kwh"]]
        return objs

    objs = evaluate(pop)
    history = []
    for gen in range(generations):
        fronts, rank = fast_nondominated_sort(objs)
        crowd = {}
        for f in fronts:
            crowd.update(crowding_distance(objs, f))

        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, rank, crowd)
            p2 = tournament_select(pop, rank, crowd)
            c1, c2 = sbx_crossover(p1, p2, lb, ub)
            c1 = polynomial_mutation(c1, lb, ub)
            c2 = polynomial_mutation(c2, lb, ub)
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])
        off_objs = evaluate(offspring)

        combined = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, off_objs])
        fronts, rank = fast_nondominated_sort(combined_objs)
        new_pop_idx = []
        for f in fronts:
            if len(new_pop_idx) + len(f) <= pop_size:
                new_pop_idx.extend(f)
            else:
                crowd_f = crowding_distance(combined_objs, f)
                remaining = pop_size - len(new_pop_idx)
                sorted_f = sorted(f, key=lambda i: -crowd_f[i])
                new_pop_idx.extend(sorted_f[:remaining])
                break
        pop = combined[new_pop_idx]
        objs = combined_objs[new_pop_idx]
        history.append((gen + 1, objs[:, 0].min(), objs[:, 1].min()))
        if progress_cb:
            progress_cb(gen + 1, generations)

    fronts, rank = fast_nondominated_sort(objs)
    return pop[fronts[0]], objs[fronts[0]], history


def select_representative_point(pareto_x, pareto_objs, eens_target_kwh):
    meets = pareto_objs[:, 1] <= eens_target_kwh
    if np.any(meets):
        idx_candidates = np.where(meets)[0]
        best = idx_candidates[np.argmin(pareto_objs[idx_candidates, 0])]
        return best, True
    return np.argmin(pareto_objs[:, 1]), False


# ----------------------------------------------------------------------
# 7. FEASIBILITY (EENS instead of LOLP)
# ----------------------------------------------------------------------
def feasibility_verdict(lcoe, eens_kwh, eens_target_kwh, baseline_tariff, E_load_yr):
    cost_ok = lcoe < baseline_tariff
    reliability_ok = eens_kwh <= eens_target_kwh
    feasible = cost_ok and reliability_ok
    savings_pct = 100.0 * (baseline_tariff - lcoe) / baseline_tariff
    eens_pct_of_load = 100.0 * eens_kwh / max(E_load_yr, 1e-6)
    return dict(feasible=feasible, cost_ok=cost_ok, reliability_ok=reliability_ok,
                savings_pct=savings_pct, eens_pct_of_load=eens_pct_of_load)


# ----------------------------------------------------------------------
# 8. COALITION MASKING + PROFIT-BASED SHAPLEY ALLOCATION
# ----------------------------------------------------------------------
_PLAYER_GENE_IDX = {"PV": [0], "WT": [1], "SMR": [2], "H2": [3, 4, 5]}


def bounds_for_coalition(coalition, full_lb, full_ub):
    lb = np.array(full_lb, dtype=float).copy()
    ub = np.array(full_ub, dtype=float).copy()
    for player, idxs in _PLAYER_GENE_IDX.items():
        if player not in coalition:
            for i in idxs:
                lb[i] = 0.0
                ub[i] = 0.0
    return lb, ub


def all_coalitions(players=PLAYERS):
    subsets = []
    for k in range(len(players) + 1):
        subsets.extend(itertools.combinations(players, k))
    return subsets


def cost_for_coalition(coalition, forecasts, dsm_tasks, full_lb, full_ub,
                        tech=DEFAULT_TECH_PARAMS, r=DISCOUNT_RATE, scale_factor=365.0,
                        voll=VOLL, grid_cap_kw=GRID_CAPACITY_KW, pop_size=20, generations=15,
                        progress_cb=None):
    """Empty coalition: cost of buying 100% of demand from the grid at TOU
    tariff, capacity-capped at grid_cap_kw (same real backstop every other
    coalition also has access to) - now a genuinely achievable baseline,
    not a hypothetical blackout."""
    E_load_yr = forecasts["load_kw"].sum() * scale_factor
    if len(coalition) == 0:
        hours = np.arange(len(forecasts["load_kw"]))
        hourly_grid_served = np.minimum(forecasts["load_kw"], grid_cap_kw)
        hourly_unmet = forecasts["load_kw"] - hourly_grid_served
        cost = (hourly_grid_served * np.array([tou_tariff(h % 24) for h in hours])).sum() * scale_factor
        eens = hourly_unmet.sum() * scale_factor
        cost += voll * eens
        return dict(coalition=coalition, cost=cost, x=None, lcoe=None, eens_kwh=eens)

    lb_c, ub_c = bounds_for_coalition(coalition, full_lb, full_ub)
    pareto_x, pareto_objs, _ = run_nsga2(forecasts, dsm_tasks, lb_c, ub_c, pop_size=pop_size,
                                          generations=generations, tech=tech, scale_factor=scale_factor,
                                          voll=voll, r=r, grid_cap_kw=grid_cap_kw, progress_cb=progress_cb)
    best_cost, best_x, best_res = None, None, None
    for xi in pareto_x:
        res = compute_objectives(xi, forecasts, dsm_tasks, tech, r, scale_factor, voll, grid_cap_kw)
        if best_cost is None or res["scalar_cost"] < best_cost:
            best_cost, best_x, best_res = res["scalar_cost"], xi, res
    return dict(coalition=coalition, cost=best_cost, x=best_x, lcoe=best_res["lcoe"], eens_kwh=best_res["eens_kwh"])


def run_profit_allocation_game(forecasts, dsm_tasks, full_lb, full_ub, tech=DEFAULT_TECH_PARAMS,
                                r=DISCOUNT_RATE, scale_factor=365.0, voll=VOLL,
                                grid_cap_kw=GRID_CAPACITY_KW, pop_size=20, generations=15,
                                progress_cb=None):
    """profit(U) = cost(emptyset) - cost(U) : how much the coalition SAVES
    versus buying everything from the grid. Shapley-allocated across
    {PV, WT, SMR, H2} - this is the 'Shapley Value (Profit allocation)'
    step in your architecture diagram."""
    coalitions = all_coalitions(PLAYERS)
    results = {}
    for n_done, U in enumerate(coalitions):
        res = cost_for_coalition(U, forecasts, dsm_tasks, full_lb, full_ub, tech, r, scale_factor,
                                  voll, grid_cap_kw, pop_size, generations)
        results[frozenset(U)] = res
        if progress_cb:
            progress_cb(n_done + 1, len(coalitions), U)

    cost_empty = results[frozenset()]["cost"]
    profit = {frozenset(U): (0.0 if len(U) == 0 else cost_empty - results[frozenset(U)]["cost"]) for U in coalitions}
    phi = shapley_values(PLAYERS, profit)
    return results, profit, phi


def shapley_values(players, values):
    n = len(players)
    phi = {p: 0.0 for p in players}
    for i in players:
        others = [p for p in players if p != i]
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                Sset, Si = frozenset(S), frozenset(set(S) | {i})
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi[i] += weight * (values[Si] - values[Sset])
    return phi
