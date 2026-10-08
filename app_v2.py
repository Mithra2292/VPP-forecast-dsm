"""
VPP Sizing, DSM & Profit-Allocation Tool (v2) — Streamlit front end
=====================================================================
Grid-connected, forecast-driven version. New vs. the original app:
  - Sidebar CSV uploads: solar/wind/load history, EV log, DSM flexible-
    load pool (shiftable + curtailable tasks the user declares)
  - Forecasting runs automatically on whatever history is uploaded
    (falls back to synthetic profiles for anything not provided)
  - Grid import (TOU-priced, HT-IIA capacity-capped) in the dispatch stack
  - EENS (kWh/yr) replaces LOLP as the reliability objective; LCOE unchanged
  - Coalition game reframed as profit allocation (Shapley-split savings
    vs. buying 100% from the grid)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting as fcst
import vpp_backend_v2 as vb

st.set_page_config(page_title="VPP Sizing, DSM & Profit Allocation", layout="wide", page_icon="⚡")

for key, default in [("results", None), ("game_results", None), ("forecasts", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("⚡ VPP Sizing, DSM & Profit-Allocation Tool")
st.caption(
    "Grid-connected · forecast-driven · DSM-aware. Upload your own solar/wind/load/EV "
    "history and a declared DSM flexible-load pool — the app forecasts the next day, "
    "schedules dispatch around it, sizes with NSGA-II (LCOE vs. EENS), and splits the "
    "savings fairly with a Shapley profit-allocation game."
)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ Historical data (for forecasting)")
    st.caption("Upload a few weeks of hourly history for each. Anything left blank falls back to a synthetic profile.")

    solar_file = st.file_uploader("Solar — timestamp, ghi (W/m²)", type="csv", key="solar")
    wind_file = st.file_uploader("Wind — timestamp, wind_speed (m/s)", type="csv", key="wind")
    load_file = st.file_uploader("Load — timestamp, load_kw", type="csv", key="load")
    ev_file = st.file_uploader("EV — hourly (timestamp, n_present, avg_soc_frac) or event log (vehicle_id, plug_in, plug_out, soc_in, soc_out)", type="csv", key="ev")
    fleet_size = st.number_input("EV fleet size", 0, 1000, 75, 5)

    st.header("2️⃣ DSM participation (flexible loads)")
    st.caption("Declare what you're offering to shift or curtail — columns: task_id, type (shiftable/curtailable), power_kw, energy_kwh, earliest_hour, latest_hour, priority (optional)")
    dsm_file = st.file_uploader("DSM flexible-load pool CSV", type="csv", key="dsm")
    use_dsm_template = st.checkbox("Use a small example DSM pool if none uploaded", value=True)

    st.header("3️⃣ Site constraints")
    max_pv = st.number_input("Max PV (kWp)", 0.0, 20000.0, 800.0, 50.0)
    max_wt = st.number_input("Max Wind (kW)", 0.0, 10000.0, 300.0, 50.0)
    max_smr = st.number_input("Max SMR (kW)", 0.0, 5000.0, 350.0, 25.0)
    max_el = st.number_input("Max Electrolyzer (kW)", 0.0, 2000.0, 150.0, 10.0)
    max_tank = st.number_input("Max Tank (kWh-eq)", 0.0, 10000.0, 1200.0, 50.0)
    max_fc = st.number_input("Max Fuel Cell (kW)", 0.0, 2000.0, 150.0, 10.0)
    grid_cap_kw = st.number_input("Grid import capacity (kW)", 0.0, 10000.0, vb.GRID_CAPACITY_KW, 50.0,
                                   help="Defaults to the HT-IIA 1750 kVA contract demand at ~0.95 PF.")

    st.header("4️⃣ Economics & reliability")
    eens_target = st.number_input("Target EENS (kWh/yr, max acceptable)", 0.0, 2_000_000.0, 50000.0, 1000.0)
    baseline_tariff = st.number_input("Baseline all-grid tariff (₹/kWh, for comparison)", 1.0, 30.0, 9.0, 0.5)
    discount_rate = st.slider("Discount rate", 0.02, 0.15, 0.08, 0.01)
    voll = st.number_input("VOLL (₹/kWh, for genuine curtailment only)", 5.0, 1000.0, 120.0, 5.0)
    scale_factor = st.number_input("Days/year this forecast represents", 1.0, 365.0, 365.0, 1.0,
                                    help="365 = treat the forecasted day as typical of the whole year.")

    st.header("5️⃣ Optimizer settings")
    pop_size = st.slider("NSGA-II population", 10, 80, 30, 2)
    generations = st.slider("NSGA-II generations", 5, 80, 25, 1)
    run_btn = st.button("🚀 Run forecast → schedule → size", type="primary", width='stretch')

    st.header("6️⃣ Profit allocation (Shapley)")
    game_pop = st.slider("Per-coalition population", 6, 40, 16, 2)
    game_gens = st.slider("Per-coalition generations", 4, 40, 12, 1)
    run_game_btn = st.button("🧮 Run profit-allocation game", width='stretch')

tech = vb.DEFAULT_TECH_PARAMS
lb = [0, 0, 0, 0, 0, 0]
ub = [max_pv, max_wt, max_smr, max_el, max_tank, max_fc]

# ----------------------------------------------------------------------
# Build forecasts (always, so the preview tab works pre-run too)
# ----------------------------------------------------------------------
solar_df = vb.load_solar_csv(solar_file) if solar_file else None
wind_df = vb.load_wind_csv(wind_file) if wind_file else None
load_df = vb.load_demand_csv(load_file) if load_file else None
ev_df = vb.load_ev_csv(ev_file, fleet_size) if ev_file else None

try:
    forecasts = vb.build_forecasts(solar_df, wind_df, load_df, ev_df, fleet_size=fleet_size, horizon=24)
except Exception as e:
    st.sidebar.error(f"Forecasting error: {e}")
    forecasts = vb.build_forecasts(None, None, None, None, fleet_size=fleet_size, horizon=24)
st.session_state.forecasts = forecasts

if dsm_file:
    dsm_tasks = vb.load_dsm_flex_csv(dsm_file)
elif use_dsm_template:
    dsm_tasks = pd.DataFrame([
        dict(task_id="pump1", bus_or_zone="community-pump", type="shiftable", power_kw=20, energy_kwh=60, earliest_hour=9, latest_hour=16, priority=2),
        dict(task_id="ev_charge_flex", bus_or_zone="ev-cluster", type="shiftable", power_kw=40, energy_kwh=120, earliest_hour=0, latest_hour=7, priority=1),
        dict(task_id="hvac_defer", bus_or_zone="clubhouse", type="curtailable", power_kw=15, energy_kwh=30, earliest_hour=18, latest_hour=21, priority=4),
    ])
else:
    dsm_tasks = pd.DataFrame(columns=["task_id", "type", "power_kw", "energy_kwh", "earliest_hour", "latest_hour", "priority"])

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Forecast Preview", "🧬 Optimization & Pareto Front",
    "🏗️ Recommended Sizing & Dispatch", "✅ Feasibility Verdict",
    "🤝 Profit Allocation (Shapley)",
])

# ----------------------------------------------------------------------
# TAB 1 — Forecast preview
# ----------------------------------------------------------------------
with tab1:
    st.subheader("24-hour forecast (next day)")
    sources = {
        "Solar": forecasts["pv_source"], "Wind": forecasts["wt_source"],
        "Load": forecasts["load_source"], "EV": forecasts["ev_source"],
    }
    cols = st.columns(4)
    for c, (k, v) in zip(cols, sources.items()):
        c.metric(k, "Forecasted" if v == "forecast" else "Synthetic", help=v)

    h = np.arange(24)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h, y=forecasts["load_kw"], name="Load (kW)", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=h, y=forecasts["pv_kwp_frac"] * max_pv, name="PV @ max capacity (kW)", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=h, y=forecasts["wt_frac"] * max_wt, name="Wind @ max capacity (kW)", line=dict(color="#1f77b4")))
    fig.update_layout(xaxis_title="Hour", yaxis_title="kW", height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h, y=forecasts["ev_n_present"], name="EVs present", line=dict(color="#2ca02c")))
    fig2.add_trace(go.Scatter(x=h, y=np.array(forecasts["ev_avg_soc"]) * 100, name="Avg SOC (%)", yaxis="y2", line=dict(color="#9467bd")))
    fig2.update_layout(xaxis_title="Hour", yaxis_title="Vehicles present",
                        yaxis2=dict(title="SOC (%)", overlaying="y", side="right"), height=320)
    st.plotly_chart(fig2, width='stretch')

    st.subheader("Declared DSM flexible-load pool")
    if len(dsm_tasks) > 0:
        st.dataframe(dsm_tasks, width='stretch', hide_index=True)
    else:
        st.info("No DSM tasks declared — upload a CSV in the sidebar, or enable the example pool.")

# ----------------------------------------------------------------------
# Run NSGA-II sizing
# ----------------------------------------------------------------------
if run_btn:
    progress = st.progress(0.0, text="Running NSGA-II...")

    def cb(gen, total):
        progress.progress(gen / total, text=f"Generation {gen}/{total}")

    pareto_x, pareto_objs, history = vb.run_nsga2(
        forecasts, dsm_tasks, lb, ub, pop_size=pop_size, generations=generations,
        tech=tech, scale_factor=scale_factor, voll=voll, r=discount_rate,
        grid_cap_kw=grid_cap_kw, progress_cb=cb)
    progress.empty()

    best_idx, met_target = vb.select_representative_point(pareto_x, pareto_objs, eens_target)
    x_star = pareto_x[best_idx]
    res_star = vb.compute_objectives(x_star, forecasts, dsm_tasks, tech, discount_rate, scale_factor, voll, grid_cap_kw)
    verdict = vb.feasibility_verdict(res_star["lcoe"], res_star["eens_kwh"], eens_target, baseline_tariff, res_star["E_load_yr"])

    st.session_state.results = dict(
        pareto_x=pareto_x, pareto_objs=pareto_objs, history=history, x_star=x_star,
        lcoe_star=res_star["lcoe"], eens_star=res_star["eens_kwh"], sim=res_star["sim"],
        breakdown=res_star["breakdown"], cost_yr=res_star["cost_yr"], met_target=met_target,
        eens_target=eens_target, baseline_tariff=baseline_tariff, verdict=verdict,
    )

# ----------------------------------------------------------------------
# TAB 2 — Pareto front
# ----------------------------------------------------------------------
with tab2:
    res = st.session_state.results
    if res is None:
        st.warning("Run the sizing from the sidebar to see the Pareto front.")
    else:
        st.subheader("Pareto front — LCOE vs. EENS")
        fig2 = go.Figure(go.Scatter(x=res["pareto_objs"][:, 1], y=res["pareto_objs"][:, 0],
                                     mode="markers", marker=dict(size=9, color="#2c3e50")))
        star_idx = np.where((res["pareto_x"] == res["x_star"]).all(axis=1))[0]
        if len(star_idx):
            fig2.add_trace(go.Scatter(x=[res["pareto_objs"][star_idx[0], 1]], y=[res["pareto_objs"][star_idx[0], 0]],
                                       mode="markers", marker=dict(size=16, color="#e74c3c", symbol="star"), name="Selected"))
        fig2.update_layout(xaxis_title="EENS (kWh/yr)", yaxis_title="LCOE (₹/kWh)", height=440)
        st.plotly_chart(fig2, width='stretch')

        hist = np.array(res["history"])
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=hist[:, 0], y=hist[:, 1], name="Best LCOE"))
        fig_c.add_trace(go.Scatter(x=hist[:, 0], y=hist[:, 2], name="Best EENS (kWh)", yaxis="y2"))
        fig_c.update_layout(xaxis_title="Generation", yaxis_title="Best LCOE (₹/kWh)",
                             yaxis2=dict(title="Best EENS (kWh)", overlaying="y", side="right"), height=360)
        st.plotly_chart(fig_c, width='stretch')

# ----------------------------------------------------------------------
# TAB 3 — Sizing & dispatch
# ----------------------------------------------------------------------
with tab3:
    res = st.session_state.results
    if res is None:
        st.warning("Run the sizing from the sidebar to see the recommended sizing.")
    else:
        names = ["Solar PV (kWp)", "Wind (kW)", "SMR (kW)", "Electrolyzer (kW)", "Tank (kWh-eq)", "Fuel Cell (kW)"]
        st.subheader("Recommended sizing")
        cols = st.columns(6)
        for c, n, v in zip(cols, names, res["x_star"]):
            c.metric(n, f"{v:,.0f}")

        bd = dict(res["breakdown"])
        fig3 = go.Figure(go.Pie(labels=list(bd.keys()), values=list(bd.values()), hole=0.4))
        fig3.update_layout(height=440, title="Annualized cost breakdown (includes grid import + decommissioning for every technology)")
        st.plotly_chart(fig3, width='stretch')

        sim = res["sim"]
        hrs = np.arange(len(sim["load"]))
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=hrs, y=sim["p_pv"], name="Solar PV", stackgroup="one", line=dict(color="#f39c12")))
        fig4.add_trace(go.Scatter(x=hrs, y=sim["p_wt"], name="Wind", stackgroup="one", line=dict(color="#3498db")))
        fig4.add_trace(go.Scatter(x=hrs, y=sim["smr"], name="SMR", stackgroup="one", line=dict(color="#8e44ad")))
        fig4.add_trace(go.Scatter(x=hrs, y=sim["fc"], name="Fuel Cell", stackgroup="one", line=dict(color="#c0392b")))
        fig4.add_trace(go.Scatter(x=hrs, y=sim["grid"], name="Grid import", stackgroup="one", line=dict(color="#1E6FD9")))
        fig4.add_trace(go.Scatter(x=hrs, y=sim["load"], name="Load", line=dict(color="black", dash="dash")))
        fig4.update_layout(xaxis_title="Hour", yaxis_title="kW", height=420, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig4, width='stretch')

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=hrs, y=sim["shift"], name="DSM shift (kW)", marker_color="#2E8B57"))
        fig5.add_trace(go.Bar(x=hrs, y=sim["dsm_curt"], name="DSM curtailment (kW)", marker_color="#D1495B"))
        fig5.add_trace(go.Bar(x=hrs, y=sim["unmet"], name="Genuine unmet (EENS, kW)", marker_color="#555555"))
        fig5.update_layout(barmode="stack", xaxis_title="Hour", yaxis_title="kW", height=320, title="DSM actions & residual unmet load")
        st.plotly_chart(fig5, width='stretch')

# ----------------------------------------------------------------------
# TAB 4 — Feasibility
# ----------------------------------------------------------------------
with tab4:
    res = st.session_state.results
    if res is None:
        st.warning("Run the sizing from the sidebar to get a feasibility verdict.")
    else:
        v = res["verdict"]
        if v["feasible"]:
            st.success("### ✅ VPP is FEASIBLE for this community")
        else:
            st.error("### ⚠️ VPP does NOT currently meet both cost and reliability targets")

        c1, c2, c3 = st.columns(3)
        c1.metric("Optimized LCOE", f"₹{res['lcoe_star']:.2f}/kWh", f"{v['savings_pct']:.1f}% vs. baseline")
        c2.metric("EENS (energy not supplied)", f"{res['eens_star']:,.0f} kWh/yr", f"target ≤ {res['eens_target']:,.0f}")
        c3.metric("Baseline all-grid tariff", f"₹{res['baseline_tariff']:.2f}/kWh")

        st.caption(f"EENS as a share of annual demand: {v['eens_pct_of_load']:.2f}%")
        if not res["met_target"]:
            st.warning("No sizing on the Pareto front met your EENS target — showing the best-reliability option found instead. "
                       "Consider raising capacity ceilings, the grid import cap, or your DSM participation pool.")

        report_df = pd.DataFrame({
            "Metric": ["LCOE (₹/kWh)", "EENS (kWh/yr)", "Baseline tariff (₹/kWh)", "Feasible"],
            "Value": [f"{res['lcoe_star']:.2f}", f"{res['eens_star']:.0f}", f"{res['baseline_tariff']:.2f}", "Yes" if v["feasible"] else "No"],
        })
        st.download_button("⬇️ Download feasibility report (CSV)", report_df.to_csv(index=False).encode(),
                            file_name="vpp_feasibility_report.csv", mime="text/csv")

# ----------------------------------------------------------------------
# Run profit-allocation game
# ----------------------------------------------------------------------
if run_game_btn:
    progress = st.progress(0.0, text="Running profit-allocation game...")

    def game_cb(done, total, U):
        label = "{" + ", ".join(U) + "}" if U else "∅"
        progress.progress(done / total, text=f"Coalition {done}/{total}: {label}")

    results_c, profit, phi = vb.run_profit_allocation_game(
        forecasts, dsm_tasks, lb, ub, tech=tech, r=discount_rate, scale_factor=scale_factor,
        voll=voll, grid_cap_kw=grid_cap_kw, pop_size=game_pop, generations=game_gens, progress_cb=game_cb)
    progress.empty()
    st.session_state.game_results = dict(results=results_c, profit=profit, phi=phi)

# ----------------------------------------------------------------------
# TAB 5 — Profit allocation
# ----------------------------------------------------------------------
with tab5:
    gr = st.session_state.game_results
    if gr is None:
        st.info("Click **'Run profit-allocation game'** in the sidebar to see each technology's fair share "
                "of the savings versus buying 100% of demand from the grid.")
    else:
        phi = gr["phi"]
        total = sum(phi.values())
        fig_phi = go.Figure(go.Bar(x=list(phi.keys()), y=list(phi.values()),
                                    marker_color=["#f39c12", "#3498db", "#8e44ad", "#16a085"],
                                    text=[f"₹{v:,.0f}/yr" for v in phi.values()], textposition="outside"))
        fig_phi.update_layout(yaxis_title="Shapley-allocated profit (₹/yr)", height=420,
                               title="Profit allocation — savings vs. 100% grid purchase")
        st.plotly_chart(fig_phi, width='stretch')

        cols = st.columns(4)
        for c, (p, val) in zip(cols, phi.items()):
            pct = 100 * val / total if total else 0
            c.metric(p, f"₹{val:,.0f}/yr", f"{pct:.1f}% of total")

        rows = []
        for U in vb.all_coalitions():
            fs = frozenset(U)
            r = gr["results"][fs]
            rows.append({
                "Coalition": "{" + ", ".join(U) + "}" if U else "∅ (100% grid purchase)",
                "cost(U) ₹/yr": f"{r['cost']:,.0f}",
                "profit(U) ₹/yr": f"{gr['profit'][fs]:,.0f}",
                "LCOE ₹/kWh": f"{r['lcoe']:.2f}" if r["lcoe"] is not None else "—",
                "EENS kWh/yr": f"{r['eens_kwh']:.0f}" if r["eens_kwh"] is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.markdown("""
#### How to read this
- **cost(U)** = annualized cost of serving the community using only coalition U's technologies
  (plus grid import at TOU tariff, capacity-capped at your contract demand).
- **profit(U)** = cost(∅) − cost(U) — how much cheaper this is than buying 100% of demand
  from the grid. This is a real, achievable baseline now, not a hypothetical blackout.
- **Shapley value (φᵢ)** splits the grand coalition's total profit fairly across PV, WT, SMR
  and H2, based on each one's average marginal contribution across every possible arrival order.
- **EV is not a Shapley player here** — it participates as a DSM/V2G flexibility resource that
  shapes the net load the sizing problem sees, not as a sizeable capacity decision.
""")

st.markdown("---")
st.caption("VPP Sizing, DSM & Profit-Allocation Tool · Forecast-driven · Grid-connected · NSGA-II + Shapley")
