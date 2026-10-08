"""
Forecasting module for PV, Wind, Load, EV-presence and EV-SOC.

Follows the same baseline/index structure as the Unified Forecasting
Framework (Chapter 5): every target is normalized against a day-type-matched
physical or statistical baseline into an index, the index is predicted by a
small feed-forward neural network, and the prediction is denormalized back
to real units.

  Target        Baseline                         Index
  ------        --------                          -----
  PV (GHI)      clear-sky GHI (solar geometry)     CSI(t) = GHI(t)/GHI_clear(t)
  Wind (speed)  7-day moving average, same hour    AI(t)  = V(t)/V_ref(t)
  Load (kW)     day-type-matched moving average    LI(t)  = D(t)/D_ref(t)
  EV presence   day-type-matched moving average    EI(t)  = N_present(t)/N_ref(t)
  EV SOC        day-type-matched moving average    SI(t)  = SOC(t)/SOC_ref(t)

NN CHOICE: the thesis trains its latent-code predictor with MOST (Monte
Carlo Stochastic Optimization, gradient-free). This module instead uses a
standard backprop-trained small MLP (sklearn MLPRegressor, Adam optimizer)
as the practical substitute discussed - faster to get right, well-tested,
appropriate given these targets are low-dimensional (no spatial grid to
compress, so no autoencoder stage either - see docstring note below).

NO AUTOENCODER: PV/wind in the thesis need compression because they start
from a 100-point SPATIAL grid across many ground stations. Load, EV
presence and EV SOC here are each a single scalar time series (one
community, one fleet) - there is no spatial grid, so the encoder/decoder
stage is skipped entirely for all five targets as implemented here. If you
later have multiple separately-metered zones/buildings, re-introduce
spatial compression for load specifically.

Every forecaster is benchmarked against first-order persistence
(Eq. 5.11-style) before being trusted - see `benchmark()`.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

N_LAGS = 4          # how many past index values feed the predictor
HORIZON = 24         # hours ahead to forecast (next-day, hour by hour)


# ----------------------------------------------------------------------
# Day-type-matched baseline
# ----------------------------------------------------------------------
def day_type(timestamps):
    """0 = weekday, 1 = weekend. Swap in a real holiday calendar if available."""
    dow = pd.DatetimeIndex(timestamps).dayofweek
    return (dow >= 5).astype(int)


def build_baseline(df, value_col, window_days=14):
    """df must have a DatetimeIndex. Returns a Series `ref(t)` = the
    day-type-matched, same-hour moving average over the last `window_days`
    days (separately for weekday/weekend) - the Eq. 5.4-style reference
    curve, generalized with a day-type split."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["daytype"] = day_type(df.index)
    ref = pd.Series(index=df.index, dtype=float)
    for (h, dt), grp in df.groupby(["hour", "daytype"]):
        grp = grp.sort_index()
        roll = grp[value_col].rolling(window=window_days, min_periods=1).mean().shift(1)
        ref.loc[grp.index] = roll.values
    ref = ref.fillna(df[value_col].mean())
    return ref


# ----------------------------------------------------------------------
# Clear-sky GHI (Ineichen-style simplified) - deterministic, no history needed
# ----------------------------------------------------------------------
def clear_sky_ghi(timestamps, latitude=11.32, ghi_peak=925.0):
    """Simplified clear-sky curve: a daylight sine bell scaled to a
    measured local peak (925 W/m^2, matching Table 5.1's ~925 W/m^2 at
    solar noon for NITC). Good enough as the CSI baseline; swap in a full
    Ineichen model (pvlib) for production-grade clear-sky values."""
    ts = pd.DatetimeIndex(timestamps)
    hour_frac = ts.hour + ts.minute / 60.0
    sunrise, sunset = 6.0, 18.5
    daylight = (hour_frac >= sunrise) & (hour_frac <= sunset)
    theta = np.pi * (hour_frac - sunrise) / (sunset - sunrise)
    ghi = np.where(daylight, np.sin(theta) * ghi_peak, 0.0)
    return pd.Series(ghi, index=ts)


# ----------------------------------------------------------------------
# Generic index forecaster (shared by all five targets)
# ----------------------------------------------------------------------
class IndexForecaster:
    def __init__(self, name, n_lags=N_LAGS, hidden=(16, 8), random_state=0):
        self.name = name
        self.n_lags = n_lags
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh",
                                   solver="adam", max_iter=2000, random_state=random_state,
                                   early_stopping=True, n_iter_no_change=20)
        self.fitted = False

    def _make_supervised(self, index_series):
        """lag features (+ hour-of-day, day-type) -> next-step index."""
        vals = index_series.values
        ts = index_series.index
        X, y = [], []
        for i in range(self.n_lags, len(vals) - 1):
            lags = vals[i - self.n_lags:i]
            hour_sin = np.sin(2 * np.pi * ts[i].hour / 24)
            hour_cos = np.cos(2 * np.pi * ts[i].hour / 24)
            dt = day_type([ts[i]])[0]
            X.append(np.concatenate([lags, [hour_sin, hour_cos, dt]]))
            y.append(vals[i + 1])
        return np.array(X), np.array(y)

    def fit(self, index_series):
        X, y = self._make_supervised(index_series)
        if len(X) < 20:
            raise ValueError(f"[{self.name}] not enough history to train "
                              f"({len(X)} samples) - need several weeks of data.")
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.fitted = True
        return self

    def predict_next(self, index_series):
        """One-step-ahead prediction from the most recent `n_lags` index values."""
        vals = index_series.values[-self.n_lags:]
        ts = index_series.index[-1]
        next_ts = ts + pd.Timedelta(hours=1)
        hour_sin = np.sin(2 * np.pi * next_ts.hour / 24)
        hour_cos = np.cos(2 * np.pi * next_ts.hour / 24)
        dt = day_type([next_ts])[0]
        x = np.concatenate([vals, [hour_sin, hour_cos, dt]]).reshape(1, -1)
        xs = self.scaler.transform(x)
        return float(self.model.predict(xs)[0])

    def forecast_horizon(self, index_series, horizon=HORIZON):
        """Iteratively forecasts `horizon` steps ahead, feeding each
        prediction back in as the next lag (standard recursive multi-step
        forecasting)."""
        series = index_series.copy()
        preds = []
        for _ in range(horizon):
            p = self.predict_next(series)
            next_ts = series.index[-1] + pd.Timedelta(hours=1)
            series = pd.concat([series, pd.Series([p], index=[next_ts])])
            preds.append(p)
        future_ts = series.index[-horizon:]
        return pd.Series(preds, index=future_ts)


# ----------------------------------------------------------------------
# Persistence benchmarks (Section 5.9's discipline, generalized)
# ----------------------------------------------------------------------
def persistence_forecast(index_series, horizon=HORIZON, order=1):
    """order=1: next value = last value (Eq. 5.11).
    Higher orders extrapolate the recent trend (Eq. 5.12-5.13 style) but
    are usually noisier - included for completeness, not recommended."""
    vals = list(index_series.values[-3:])
    preds = []
    for _ in range(horizon):
        if order == 1 or len(vals) < 2:
            p = vals[-1]
        elif order == 2 or len(vals) < 3:
            p = 2 * vals[-1] - vals[-2]
        else:
            p = 2.5 * vals[-1] - 2 * vals[-2] + 0.5 * vals[-3]
        preds.append(p)
        vals.append(p)
    future_ts = pd.date_range(index_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.Series(preds, index=future_ts)


def benchmark(model_preds, naive_preds, actuals):
    """RMSE comparison - the model only 'earns' its complexity if it beats
    naive persistence (Section 5.9's stated discipline)."""
    rmse_model = float(np.sqrt(np.mean((model_preds - actuals) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((naive_preds - actuals) ** 2)))
    return dict(rmse_model=rmse_model, rmse_naive=rmse_naive,
                model_beats_naive=rmse_model < rmse_naive,
                improvement_pct=100 * (rmse_naive - rmse_model) / max(rmse_naive, 1e-9))


# ----------------------------------------------------------------------
# Per-target wrappers: raw series in -> forecasted real-unit series out
# ----------------------------------------------------------------------
def forecast_pv(hist_df, horizon=HORIZON, pr=0.784):
    """hist_df: DataFrame with DatetimeIndex, column 'ghi' (W/m^2).
    Returns forecasted P(t) in kW/kWp (Eq. 5.14)."""
    ghi_clear = clear_sky_ghi(hist_df.index)
    csi = (hist_df["ghi"] / ghi_clear.replace(0, np.nan)).clip(0, 1.3).fillna(0)
    fc = IndexForecaster("PV").fit(csi)
    csi_fcst = fc.forecast_horizon(csi, horizon)
    ghi_clear_fcst = clear_sky_ghi(csi_fcst.index)
    ghi_fcst = csi_fcst * ghi_clear_fcst.values
    p_fcst = (ghi_fcst / 1000.0) * pr
    return p_fcst.clip(lower=0), csi, fc


def forecast_wind(hist_df, horizon=HORIZON, cut_in=2.5, rated=11.0, cut_out=25.0):
    """hist_df: DataFrame with DatetimeIndex, column 'wind_speed' (m/s).
    Returns forecasted turbine output fraction f(t) in [0,1]."""
    v_ref = build_baseline(hist_df, "wind_speed")
    ai = (hist_df["wind_speed"] / v_ref.replace(0, np.nan)).fillna(1.0).clip(0, 3)
    fc = IndexForecaster("Wind").fit(ai)
    ai_fcst = fc.forecast_horizon(ai, horizon)
    v_ref_fcst = pd.Series(v_ref.values[-horizon:] if len(v_ref) >= horizon else
                            np.full(horizon, v_ref.mean()), index=ai_fcst.index)
    v_fcst = ai_fcst * v_ref_fcst.values
    ramp = (v_fcst >= cut_in) & (v_fcst < rated)
    rated_band = (v_fcst >= rated) & (v_fcst < cut_out)
    f = pd.Series(0.0, index=v_fcst.index)
    f[ramp] = ((v_fcst[ramp] - cut_in) / (rated - cut_in)) ** 3
    f[rated_band] = 1.0
    return f, ai, fc


def forecast_load(hist_df, horizon=HORIZON):
    """hist_df: DataFrame with DatetimeIndex, column 'load_kw'."""
    d_ref = build_baseline(hist_df, "load_kw")
    li = (hist_df["load_kw"] / d_ref.replace(0, np.nan)).fillna(1.0).clip(0.3, 3)
    fc = IndexForecaster("Load").fit(li)
    li_fcst = fc.forecast_horizon(li, horizon)
    d_ref_fcst = pd.Series(d_ref.values[-horizon:] if len(d_ref) >= horizon else
                            np.full(horizon, d_ref.mean()), index=li_fcst.index)
    d_fcst = li_fcst * d_ref_fcst.values
    return d_fcst.clip(lower=0), li, fc


def forecast_ev_presence(hist_df, fleet_size, horizon=HORIZON):
    """hist_df: DataFrame with DatetimeIndex, column 'n_present' (vehicles
    plugged in that hour)."""
    n_ref = build_baseline(hist_df, "n_present")
    ei = (hist_df["n_present"] / n_ref.replace(0, np.nan)).fillna(1.0).clip(0, 2)
    fc = IndexForecaster("EV-presence").fit(ei)
    ei_fcst = fc.forecast_horizon(ei, horizon)
    n_ref_fcst = pd.Series(n_ref.values[-horizon:] if len(n_ref) >= horizon else
                            np.full(horizon, n_ref.mean()), index=ei_fcst.index)
    n_fcst = (ei_fcst * n_ref_fcst.values).clip(0, fleet_size)
    return n_fcst.round(), ei, fc


def forecast_ev_soc(hist_df, horizon=HORIZON):
    """hist_df: DataFrame with DatetimeIndex, column 'avg_soc_frac'
    (fleet-average SOC fraction among present vehicles, that hour)."""
    s_ref = build_baseline(hist_df, "avg_soc_frac")
    si = (hist_df["avg_soc_frac"] / s_ref.replace(0, np.nan)).fillna(1.0).clip(0, 2)
    fc = IndexForecaster("EV-SOC").fit(si)
    si_fcst = fc.forecast_horizon(si, horizon)
    s_ref_fcst = pd.Series(s_ref.values[-horizon:] if len(s_ref) >= horizon else
                            np.full(horizon, s_ref.mean()), index=si_fcst.index)
    soc_fcst = (si_fcst * s_ref_fcst.values).clip(0, 1)
    return soc_fcst, si, fc
