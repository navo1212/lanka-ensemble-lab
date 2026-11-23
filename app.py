"""
LankaEnsemble Lab — Forecasting Colombo's weather with ensemble learning (one real Sri Lankan dataset)
B.N.V. Gunathilaka (224059P) — Faculty of IT, University of Moratuwa

Run locally:  pip install -r requirements.txt  &&  streamlit run app.py
AI declaration: the dashboard UI and parts of this code were generated with the help of Claude (Anthropic)
and reviewed/tested by the student.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, StackingRegressor, VotingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA

SEED = 42
NJ = 2            # Streamlit Community Cloud has ~2 CPUs; n_jobs=-1 would start one process per *host* CPU
TEST_DAYS = 365
ORDER = (2, 0, 2)  # chosen by AIC in the notebook
HEAT_C = 28.5
DATA = Path(__file__).parent / "data"
CSV = DATA / "colombo_weather_daily_used.csv"
FAM_COL = {"Baseline": "#adb5bd", "Bagging": "#2a9d8f", "Boosting": "#e76f51", "Stacking": "#6a4c93", "Hybrid": "#264653"}
VARS = ("temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max,shortwave_radiation_sum")

st.set_page_config(page_title="LankaEnsemble Lab · Colombo weather", page_icon="🌦️", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#8d153a 0%,#eb7400 55%,#ffbe29 100%); color:white; padding:1.6rem 2rem;
       border-radius:16px; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2.2rem;} .hero p {margin:.3rem 0 0 0; font-size:1.05rem; opacity:.95}
.card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:1rem 1.1rem; height:100%;}
.tag {display:inline-block; padding:2px 10px; border-radius:20px; color:white; font-size:.8rem; margin-right:4px}
</style>""", unsafe_allow_html=True)


# =====================================================================================
# DATA
# =====================================================================================
def download(start, end):
    import requests
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=6.9271&longitude=79.8612"
           f"&start_date={start}&end_date={end}&daily={VARS}&timezone=Asia%2FColombo")
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})


def tidy(d):
    d = d.copy(); d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date").asfreq("D").interpolate(limit_direction="both")


@st.cache_data
def load_history():
    """The fixed dataset used in the notebook: 1 Jan 2015 – 31 Oct 2026."""
    d = pd.read_csv(CSV) if CSV.exists() else download("2015-01-01", "2026-10-31")
    return tidy(d).loc["2015-01-01":"2026-10-31"]


@st.cache_data
def notebook_results():
    """Headline numbers written by the executed notebook (results_summary.json), if present."""
    try:
        return json.loads((Path(__file__).parent / "results_summary.json").read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading the latest Colombo weather from Open-Meteo…")
def load_live():
    end = (pd.Timestamp.now(tz="Asia/Colombo").normalize() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    try:
        return tidy(download("2015-01-01", end)), "REAL · Open-Meteo archive (ERA5) — downloaded live"
    except Exception:
        return load_history(), "saved dataset (Open-Meteo was unreachable)"


def make_features(df):
    y = df["temperature_2m_mean"]
    f = pd.DataFrame(index=df.index)
    for L in [1, 2, 3, 4, 5, 6, 7, 14]:
        f[f"temp_lag{L}"] = y.shift(L)
    f["temp_roll7"] = y.shift(1).rolling(7).mean()
    f["temp_roll30"] = y.shift(1).rolling(30).mean()
    f["temp_last_year"] = y.shift(365)
    f["tmax_lag1"] = df["temperature_2m_max"].shift(1)
    f["tmin_lag1"] = df["temperature_2m_min"].shift(1)
    f["diurnal_range_lag1"] = f["tmax_lag1"] - f["tmin_lag1"]
    for L in [1, 2, 3]:
        f[f"rain_lag{L}"] = df["precipitation_sum"].shift(L)
        f[f"wind_lag{L}"] = df["wind_speed_10m_max"].shift(L)
        f[f"rad_lag{L}"] = df["shortwave_radiation_sum"].shift(L)
    f["rain_3d"] = df["precipitation_sum"].shift(1).rolling(3).sum()
    f["rain_7d"] = df["precipitation_sum"].shift(1).rolling(7).sum()
    doy = df.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["month"] = df.index.month
    f["sw_monsoon"] = df.index.month.isin([5, 6, 7, 8, 9]).astype(int)
    f["ne_monsoon"] = df.index.month.isin([12, 1, 2]).astype(int)
    return f


def resid_features(df, resid, y):
    f = pd.DataFrame(index=df.index)
    for L in range(1, 8):
        f[f"resid_lag{L}"] = resid.shift(L)
    for L in (1, 2, 3):
        f[f"rain_lag{L}"] = df["precipitation_sum"].shift(L)
        f[f"wind_lag{L}"] = df["wind_speed_10m_max"].shift(L)
        f[f"rad_lag{L}"] = df["shortwave_radiation_sum"].shift(L)
        f[f"temp_lag{L}"] = y.shift(L)
    f["rain_3d_sum"] = df["precipitation_sum"].shift(1).rolling(3).sum()
    f["rain_7d_sum"] = df["precipitation_sum"].shift(1).rolling(7).sum()
    f["diurnal_range_lag1"] = (df["temperature_2m_max"] - df["temperature_2m_min"]).shift(1)
    f["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    f["month"] = df.index.month
    f["sw_monsoon"] = df.index.month.isin([5, 6, 7, 8, 9]).astype(int)
    f["ne_monsoon"] = df.index.month.isin([12, 1, 2]).astype(int)
    return f


def fourier(idx, start):
    doy = idx.dayofyear.values
    return pd.DataFrame({"s1": np.sin(2 * np.pi * doy / 365.25), "c1": np.cos(2 * np.pi * doy / 365.25),
                         "s2": np.sin(4 * np.pi * doy / 365.25), "c2": np.cos(4 * np.pi * doy / 365.25),
                         "trend": (idx - start).days.values / 3652.5}, index=idx)


def metrics(y, p):
    return {"MAE °C": mean_absolute_error(y, p), "RMSE °C": float(np.sqrt(mean_squared_error(y, p))), "R²": r2_score(y, p)}


def ann(hidden=(64, 32)):
    return TransformedTargetRegressor(MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-2, learning_rate_init=1e-3,
                                                   max_iter=800, early_stopping=True, validation_fraction=0.15,
                                                   n_iter_no_change=30, random_state=SEED), transformer=StandardScaler())


def split():
    wx = load_history(); y = wx["temperature_2m_mean"]; X = make_features(wx)
    ok = X.dropna().index; te = wx.index[-TEST_DAYS:]; tr = ok[ok < te[0]]
    return wx, y, X, tr, te


# =====================================================================================
# MODELS (cached — trained once per server start)
# =====================================================================================
@st.cache_resource(show_spinner="Training 12 ensemble models on 10 years of Colombo weather… (first load ≈ 1 min)")
def train_arena():
    wx, y, X, tr, te = split()
    base = [("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)),
            ("xgb", xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=NJ)),
            ("lgbm", lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=NJ)),
            ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0)))]
    zoo = {
        "Ridge regression": ("Baseline", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "Decision Tree": ("Baseline", DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=SEED)),
        "Bagging (Trees)": ("Bagging", BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=100, random_state=SEED, n_jobs=NJ)),
        "Random Forest": ("Bagging", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)),
        "Extra Trees": ("Bagging", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7, random_state=SEED, n_jobs=NJ)),
        "AdaBoost": ("Boosting", AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, learning_rate=0.05, random_state=SEED)),
        "Gradient Boosting": ("Boosting", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)),
        "XGBoost": ("Boosting", xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=NJ)),
        "LightGBM": ("Boosting", lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=NJ)),
        "Stacking (RF+XGB+LGBM+Ridge)": ("Stacking", StackingRegressor(base, final_estimator=RidgeCV(), cv=KFold(5), n_jobs=NJ)),
        "Voting (average)": ("Stacking", VotingRegressor(base, n_jobs=NJ)),
    }
    naive = y.shift(1).loc[te]
    rows = [{"Model": "Naive (tomorrow = today)", "Family": "Baseline", **metrics(y.loc[te], naive), "Train s": 0.0}]
    preds, models = {"Naive (tomorrow = today)": naive.values}, {}
    for name, (fam, m) in zoo.items():
        t0 = time.time(); m.fit(X.loc[tr], y.loc[tr]); secs = time.time() - t0
        preds[name], models[name] = m.predict(X.loc[te]), m
        rows.append({"Model": name, "Family": fam, **metrics(y.loc[te], preds[name]), "Train s": secs})
    board = pd.DataFrame(rows).sort_values("RMSE °C").reset_index(drop=True)
    naive_rmse = board.set_index("Model").loc["Naive (tomorrow = today)", "RMSE °C"]
    board["Skill vs naive %"] = 100 * (1 - board["RMSE °C"] / naive_rmse)
    # RF vs XGBoost: error vs number of trees
    Ns = [10, 25, 50, 100, 200, 300, 400, 600]
    rf_w = RandomForestRegressor(min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ, warm_start=True)
    rf_curve = []
    for n in Ns:
        rf_w.set_params(n_estimators=n); rf_w.fit(X.loc[tr], y.loc[tr])
        rf_curve.append(np.sqrt(mean_squared_error(y.loc[te], rf_w.predict(X.loc[te]))))
    xgb_curve = [np.sqrt(mean_squared_error(y.loc[te], models["XGBoost"].predict(X.loc[te], iteration_range=(0, n)))) for n in Ns]
    curve = pd.DataFrame({"Random Forest": rf_curve, "XGBoost": xgb_curve}, index=Ns)
    # RF vs XGBoost: time-series cross-validation (each fold tests on the next unseen year)
    cv = {"Random Forest": [], "XGBoost": []}
    for f_tr, f_va in TimeSeriesSplit(n_splits=5, test_size=365).split(X.loc[tr]):
        Xa, ya = X.loc[tr].iloc[f_tr], y.loc[tr].iloc[f_tr]; Xb, yb = X.loc[tr].iloc[f_va], y.loc[tr].iloc[f_va]
        for nm, mdl in [("Random Forest", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)),
                        ("XGBoost", xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=NJ))]:
            cv[nm].append(np.sqrt(mean_squared_error(yb, mdl.fit(Xa, ya).predict(Xb))))
    imp = pd.DataFrame({"Random Forest": models["Random Forest"].feature_importances_,
                        "XGBoost": models["XGBoost"].feature_importances_}, index=X.columns)
    return dict(board=board, preds=preds, models=models, te=te, y_te=y.loc[te], curve=curve, cv=pd.DataFrame(cv), imp=imp / imp.sum())


@st.cache_resource(show_spinner="Fitting ARIMA and the hybrid ARIMA + Random Forest…")
def train_hybrid():
    wx, y, X, tr, te = split()
    F = fourier(wx.index, wx.index[0])
    ar = ARIMA(y.iloc[:-TEST_DAYS], exog=F.iloc[:-TEST_DAYS], order=ORDER, trend="c").fit()
    full = ar.append(y.iloc[-TEST_DAYS:], exog=F.iloc[-TEST_DAYS:], refit=False)
    fc = full.fittedvalues; resid = y - fc
    R = resid_features(wx, resid, y); valid = R.dropna().index; h_tr = valid[valid < te[0]][30:]
    # same settings as the notebook (500 trees, seed 42), so the dashboard shows exactly the notebook's hybrid result
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ).fit(R.loc[h_tr], resid.loc[h_tr])
    hybrid = fc.loc[te] + rf.predict(R.loc[te])
    A = train_arena()
    pred = pd.DataFrame({"Actual": y.loc[te], "Naive (tomorrow = today)": y.shift(1).loc[te], "ARIMA + Fourier": fc.loc[te],
                         "Random Forest": A["preds"]["Random Forest"], "XGBoost": A["preds"]["XGBoost"],
                         "Stacking": A["preds"]["Stacking (RF+XGB+LGBM+Ridge)"], "★ Hybrid ARIMA + RF": hybrid}, index=te)
    board = pd.DataFrame({c: metrics(pred.Actual, pred[c]) for c in pred.columns[1:]}).T.sort_values("RMSE °C")
    season = np.select([te.month.isin([12, 1, 2]), te.month.isin([3, 4]), te.month.isin([5, 6, 7, 8, 9])],
                       ["NE monsoon (Dec–Feb)", "1st inter-monsoon (Mar–Apr)", "SW monsoon (May–Sep)"], "2nd inter-monsoon (Oct–Nov)")
    seas = pd.DataFrame({c: [float(np.sqrt(np.mean((pred.Actual.values[season == s] - pred[c].values[season == s]) ** 2)))
                             for s in pd.unique(season)] for c in pred.columns[1:]}, index=pd.unique(season))
    err = (pred.Actual - hybrid).values
    imp = pd.Series(rf.feature_importances_, index=R.columns).sort_values(ascending=False)
    return dict(pred=pred, board=board, seas=seas, q=np.quantile(err, [0.10, 0.90]), imp=imp)


@st.cache_resource(show_spinner="Training Random Forest + neural network hybrids…")
def train_rf_ann():
    wx, y, X, tr, te = split()
    Xtr, ytr, Xte = X.loc[tr], y.loc[tr], X.loc[te]
    sc = StandardScaler().fit(Xtr); Str, Ste = sc.transform(Xtr), sc.transform(Xte)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)
    nn = ann()
    oof_rf = cross_val_predict(rf, Xtr, ytr, cv=KFold(5), n_jobs=NJ); oof_nn = cross_val_predict(nn, Str, ytr, cv=KFold(5))
    rf.fit(Xtr, ytr); nn.fit(Str, ytr)
    p_rf, p_nn = rf.predict(Xte), nn.predict(Ste)
    ws = np.linspace(0, 1, 101)
    w = float(ws[np.argmin([mean_squared_error(ytr, a * oof_rf + (1 - a) * oof_nn) for a in ws])])
    res = ann((32,)).fit(np.column_stack([Str, oof_rf]), ytr - oof_rf)
    stack = StackingRegressor([("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)),
                               ("ann", make_pipeline(StandardScaler(), ann()))], final_estimator=RidgeCV(), cv=KFold(5), n_jobs=NJ).fit(Xtr, ytr)
    preds = {"Random Forest": p_rf, "ANN (MLP)": p_nn, f"Hybrid: Blend (w_RF={w:.2f})": w * p_rf + (1 - w) * p_nn,
             "Hybrid: RF + ANN residual": p_rf + res.predict(np.column_stack([Ste, p_rf])), "Hybrid: Stacking → Ridge": stack.predict(Xte)}
    board = pd.DataFrame({k: metrics(y.loc[te], v) for k, v in preds.items()}).T.sort_values("RMSE °C")
    return dict(board=board, preds=preds, y_te=y.loc[te], te=te, corr=float(np.corrcoef(y.loc[te] - p_rf, y.loc[te] - p_nn)[0, 1]))


@st.cache_resource(ttl=6 * 3600, show_spinner="Updating the hybrid with the latest weather…")
def train_live():
    wx, src = load_live()
    y = wx["temperature_2m_mean"]; F = fourier(wx.index, wx.index[0])
    ar = ARIMA(y, exog=F, order=ORDER, trend="c").fit()
    resid = y - ar.fittedvalues
    R = resid_features(wx, resid, y); valid = R.dropna().index[30:]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ).fit(R.loc[valid], resid.loc[valid])
    nxt = wx.index[-1] + pd.Timedelta(days=1)
    a_next = float(np.asarray(ar.forecast(1, exog=fourier(pd.DatetimeIndex([nxt]), wx.index[0])))[0])
    return dict(wx=wx, src=src, resid=resid, rf=rf, nxt=nxt, a_next=a_next)


def next_day_rows(wx, resid, rain=None, rad=None, wind=None):
    """Feature rows for the day after the last observation, optionally with 'what-if' weather for the last day."""
    w = wx.copy()
    last = w.index[-1]
    if rain is not None: w.loc[last, "precipitation_sum"] = rain
    if rad is not None: w.loc[last, "shortwave_radiation_sum"] = rad
    if wind is not None: w.loc[last, "wind_speed_10m_max"] = wind
    nxt = last + pd.Timedelta(days=1)
    w2 = pd.concat([w, pd.DataFrame(index=[nxt], columns=w.columns, dtype=float)])
    y2 = w2["temperature_2m_mean"]
    r2 = pd.concat([resid, pd.Series([np.nan], index=[nxt])])
    return resid_features(w2, r2, y2).loc[[nxt]], make_features(w2).loc[[nxt]]


def show_fig(fig):
    st.pyplot(fig); plt.close(fig)


def fig_barh(series, colors=None, title="", xlabel="", xlim=None):
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.4 * len(series))))
    ax.barh(series.index[::-1], series.values[::-1], color=(colors[::-1] if colors is not None else "#2a9d8f"))
    ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout(); return fig


def fig_lines(df, cols, days, title=""):
    styles = {"Actual": dict(color="black", lw=2.2), "Naive (tomorrow = today)": dict(color="#adb5bd", lw=1.2, ls=":"),
              "ARIMA + Fourier": dict(color="#4c72b0", lw=1.6, ls="--"), "★ Hybrid ARIMA + RF": dict(color="#e76f51", lw=2.0)}
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for c in cols:
        s_ = df[c].iloc[-days:]
        ax.plot(s_.index, s_.values, label=c, **styles.get(c, dict(lw=1.5)))
    ax.set_ylabel("°C (daily mean)"); ax.set_title(title); ax.grid(alpha=.3)
    ax.legend(ncol=min(4, max(1, len(cols))), loc="upper center", fontsize=9, frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); return fig


# =====================================================================================
# UI
# =====================================================================================
PAGES = ["🏠 Home & Story", "⚔️ Ensemble Arena", "🌲 RF vs XGBoost", "★ Hybrid ARIMA + RF",
         "🧠 Hybrid RF + ANN", "💡 Tomorrow in Colombo", "ℹ️ Research & About"]
with st.sidebar:
    st.markdown("## 🌦️ LankaEnsemble Lab")
    st.caption("Ensemble learning on real Colombo weather (Open-Meteo ERA5, 2015–2026)")
    page = st.radio("Navigate", PAGES, index=0)
    st.divider()
    st.caption("B.N.V. Gunathilaka · 224059P\nFaculty of IT, University of Moratuwa")
    st.caption("🤖 UI & parts of code generated with Claude (Anthropic); reviewed by the student.")

if page == PAGES[0]:
    st.markdown("""<div class="hero"><h1>🌦️ LankaEnsemble Lab — Colombo weather</h1>
    <p>Can ensemble learning forecast <b>tomorrow's temperature in Colombo</b>? One real dataset, every major ensemble idea:
    <b>Bagging</b>, <b>Boosting</b>, <b>Stacking</b> and <b>hybrid models</b>.</p></div>""", unsafe_allow_html=True)
    wx = load_history()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Days of real data", f"{len(wx):,}")
    k2.metric("Period", f"{wx.index.min().year} – {wx.index.max().year}")
    k3.metric("Mean temperature", f"{wx['temperature_2m_mean'].mean():.1f} °C")
    k4.metric("Average yearly rain", f"{wx['precipitation_sum'].sum() / (len(wx) / 365.25):,.0f} mm")
    c1, c2 = st.columns(2)
    mon = wx.groupby(wx.index.month).agg(temp=("temperature_2m_mean", "mean"), rain=("precipitation_sum", "sum"))
    mon["rain"] /= wx.index.year.nunique()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.bar(names, mon.rain, color=["#e76f51" if m in (5, 6, 7, 8, 9) else ("#457b9d" if m in (12, 1, 2) else "#adb5bd") for m in mon.index])
        ax.set_title("Rain per month (mm) · orange = SW monsoon · blue = NE monsoon", fontsize=10); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); show_fig(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.plot(names, mon.temp, "o-", color="#e76f51"); ax.set_title("Average daily mean temperature (°C)", fontsize=10); ax.grid(alpha=.3)
        fig.tight_layout(); show_fig(fig)
    st.subheader("How the ensemble families work")
    a, b, c = st.columns(3)
    a.info("**Bagging** 🌳🌳🌳 → average\n\nMany deep trees trained **in parallel** on bootstrap samples; averaging cancels their "
           "mistakes → **reduces variance**. *Random Forest, Extra Trees.*")
    b.warning("**Boosting** 🌱→🌿→🌳 sequential\n\nSmall trees built **one after another**, each fixing the previous errors → "
              "**reduces bias**. *XGBoost, LightGBM, Gradient Boosting, AdaBoost.*")
    c.success("**Stacking & hybrids** 🧠\n\nDifferent models combined: a meta-model learns whom to trust, or **ARIMA + Random Forest** "
              "split the work (linear season vs non-linear weather).")
    st.markdown("**💡 Innovation:** errors by monsoon season, and a live **Tomorrow in Colombo** forecast with an 80% range, "
                "a heat alert and a *rain what-if*.")
    st.caption("Data: real daily weather for Colombo (6.93°N, 79.86°E) from the Open-Meteo historical archive (ERA5 reanalysis). "
               "Target: tomorrow's daily mean temperature, using only information up to today. Test set: the last 365 days (Nov 2025 – Oct 2026).")

elif page == PAGES[1]:
    st.title("⚔️ Ensemble Arena — who forecasts tomorrow best?")
    A = train_arena(); board = A["board"]
    fams = st.multiselect("Show families", ["Baseline", "Bagging", "Boosting", "Stacking"], default=["Baseline", "Bagging", "Boosting", "Stacking"])
    metric = st.selectbox("Rank by", ["RMSE °C", "MAE °C", "R²"], index=0)
    asc = metric != "R²"
    b = board[board.Family.isin(fams)].sort_values(metric, ascending=asc)
    top = b.iloc[0] if len(b) else board.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏆 Best model", top.Model)
    k2.metric(f"Best {metric}", f"{top[metric]:.3f}")
    k3.metric("Skill vs naive", f"{top['Skill vs naive %']:+.1f}%")
    k4.metric("Bagging effect (tree → RF)", f"{board.set_index('Model').loc['Decision Tree', 'RMSE °C']:.3f} → {board.set_index('Model').loc['Random Forest', 'RMSE °C']:.3f}")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        lo, hi = b[metric].min(), b[metric].max(); pad = max(0.005, (hi - lo) * 0.25)
        show_fig(fig_barh(b.set_index("Model")[metric], colors=[FAM_COL[f] for f in b.Family], title=f"{metric} on the 365-day test year",
                          xlabel=metric, xlim=(lo - pad, hi + pad / 2)))
        st.caption("🟩 Bagging · 🟧 Boosting · 🟪 Stacking · ⬜ Baseline")
    with c2:
        sel = st.multiselect("Compare forecasts", list(A["preds"]), default=["Naive (tomorrow = today)", top.Model] if top.Model in A["preds"] else ["Naive (tomorrow = today)"])
        days = st.slider("Last N days of the test year", 30, 365, 90, 15)
        df = pd.DataFrame({"Actual": A["y_te"].values, **{k: A["preds"][k] for k in sel}}, index=A["te"])
        show_fig(fig_lines(df, ["Actual"] + sel, days))
    st.dataframe(b.style.format({"MAE °C": "{:.4f}", "RMSE °C": "{:.4f}", "R²": "{:.3f}", "Train s": "{:.1f}", "Skill vs naive %": "{:+.1f}"})
                 .background_gradient(subset=[metric], cmap="Greens_r" if asc else "Greens"), hide_index=True)
    st.markdown("**Reading the arena:** a single tree over-fits; **bagging** averages many trees and cuts the error. **Boosting** adds small "
                "trees one by one. **Stacking/voting** combine linear and tree models. Tomorrow's temperature is mostly linear in the "
                "last few days, so Ridge is a strong baseline — ensembles win by adding non-linear weather effects.")

elif page == PAGES[2]:
    st.title("🌲 vs 🚀 Random Forest vs XGBoost")
    A = train_arena()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        cur = A["curve"]
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        ax.plot(cur.index, cur["Random Forest"], "o-", color="#2a9d8f", lw=2.5, label="Random Forest (bagging)")
        ax.plot(cur.index, cur["XGBoost"], "s-", color="#e76f51", lw=2.5, label="XGBoost (boosting, lr = 0.03)")
        ax.set_ylim(cur.min().min() * 0.98, min(cur.max().max(), cur.min().min() * 1.35))
        ax.set_xlabel("number of trees"); ax.set_ylabel("test RMSE °C"); ax.set_title("Error vs ensemble size"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); show_fig(fig)
    with c2:
        xg = cur["XGBoost"]
        more = (f"With more trees it slowly gets worse ({xg.iloc[-1]:.3f} °C at {int(xg.index[-1])}): boosting **over-fits if it runs "
                "too long**, so it needs early stopping or a tuned number of trees." if xg.idxmin() < xg.index[-1] else
                "It is still improving at the largest size tried — boosting needs a tuned number of trees.")
        st.markdown(f"""**What this shows**
* **Random Forest** reaches its best error quickly and then stays flat — more trees only average more; they never over-fit.
* **XGBoost** starts far off (learning rate 0.03 = small steps), improves tree by tree — **boosting reduces bias** — and is best at
  **{int(xg.idxmin())} trees** ({xg.min():.3f} °C). {more}""")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        cv = A["cv"]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.plot(range(1, len(cv) + 1), cv["Random Forest"], "o-", color="#2a9d8f", label="Random Forest")
        ax.plot(range(1, len(cv) + 1), cv["XGBoost"], "s-", color="#e76f51", label="XGBoost")
        ax.set_xticks(range(1, len(cv) + 1))
        ax.set_xlabel("time-series CV fold (each tests on the next unseen year)"); ax.set_ylabel("RMSE °C"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); show_fig(fig)
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("RF mean CV RMSE", f"{cv['Random Forest'].mean():.3f} °C")
        m2.metric("XGB mean CV RMSE", f"{cv['XGBoost'].mean():.3f} °C")
        st.caption("Each fold trains on all earlier years and tests on the following year — no future data leaks into training. "
                   "This page uses lighter settings (5 folds, 200 trees) so it loads fast.")
        nb = notebook_results()
        if nb and "rf_vs_xgb" in nb:
            r = nb["rf_vs_xgb"]
            st.info(f"**Notebook (6 folds, full settings):** Random Forest {np.mean(r['RF']):.3f} vs XGBoost {np.mean(r['XGB']):.3f} °C — "
                    f"corrected t-test p = {r['p']:.2f} → " + ("significant at 5%." if r["p"] < 0.05 else "**not significant** at 5%."))
    st.subheader("What drives tomorrow's temperature? (built-in importance)")
    imp = A["imp"].loc[A["imp"].mean(axis=1).sort_values(ascending=False).index[:12]]
    st.bar_chart(imp)

elif page == PAGES[3]:
    st.title("⭐ Hybrid ARIMA + Random Forest")
    H = train_hybrid(); bd = H["board"]
    ar_rmse, hy_rmse = bd.loc["ARIMA + Fourier", "RMSE °C"], bd.loc["★ Hybrid ARIMA + RF", "RMSE °C"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hybrid RMSE", f"{hy_rmse:.3f} °C", f"{100 * (hy_rmse / ar_rmse - 1):+.1f}% vs ARIMA", delta_color="inverse")
    k2.metric("ARIMA alone", f"{ar_rmse:.3f} °C")
    k3.metric("Best ensemble (Part A)", f"{bd.drop(['★ Hybrid ARIMA + RF', 'ARIMA + Fourier', 'Naive (tomorrow = today)']).iloc[0]['RMSE °C']:.3f} °C")
    k4.metric("Naive", f"{bd.loc['Naive (tomorrow = today)', 'RMSE °C']:.3f} °C")
    days = st.slider("Show last N days", 30, 365, 120, 15)
    cols = st.multiselect("Series", list(H["pred"].columns), default=["Actual", "ARIMA + Fourier", "★ Hybrid ARIMA + RF"])
    show_fig(fig_lines(H["pred"], cols, days, "One-day-ahead forecasts on the test year"))
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(bd.style.format("{:.4f}").background_gradient(subset=["RMSE °C"], cmap="Greens_r"))
        st.markdown(f"""**How the hybrid works**
1. **ARIMA{ORDER}** + yearly Fourier terms → the linear part (persistence, season, trend).
2. **Random Forest** learns ARIMA's **errors** from yesterday's rain, wind, solar radiation and monsoon flags.
3. Forecast = ARIMA + RF correction. Only information up to the previous day is used.""")
    with c2:
        show_fig(fig_barh(H["imp"][:12], title="What the forest learned about ARIMA's errors"))
    st.subheader("💡 Which model wins in which monsoon season? (RMSE °C)")
    st.dataframe(H["seas"].style.format("{:.3f}").highlight_min(axis=1, color="#c7f0d8"))

elif page == PAGES[4]:
    st.title("🧠 Hybrid Random Forest + Neural Network")
    N = train_rf_ann(); bd = N["board"]
    k1, k2, k3 = st.columns(3)
    k1.metric("🏆 Best", bd.index[0]); k2.metric("RMSE", f"{bd.iloc[0]['RMSE °C']:.3f} °C"); k3.metric("RF–ANN error correlation", f"{N['corr']:.2f}")
    st.dataframe(bd.style.format("{:.4f}").background_gradient(subset=["RMSE °C"], cmap="Greens_r"))
    pick = st.multiselect("Compare", list(N["preds"]), default=[bd.index[0], "ANN (MLP)"])
    df = pd.DataFrame({"Actual": N["y_te"].values, **{k: N["preds"][k] for k in pick}}, index=N["te"])
    show_fig(fig_lines(df, ["Actual"] + pick, 90, "Last 90 test days"))
    st.markdown("**Why combine them?** The forest captures thresholds (heavy rain, monsoon) but predicts in steps; the network learns smooth "
                "relations. Blending/stacking cancel part of each model's error — the less correlated their errors, the bigger the gain.")

elif page == PAGES[5]:
    st.title("💡 Tomorrow in Colombo")
    L = train_live(); H = train_hybrid(); A = train_arena()
    wx = L["wx"]; last = wx.index[-1]
    st.caption(f"Data source: **{L['src']}** · last observed day: **{last.date()}** (the archive lags ~5 days)")
    with st.expander("🌧️ What-if: change the last day's weather", expanded=True):
        c1, c2, c3 = st.columns(3)
        clip = lambda v, a, b: float(min(b, max(a, round(float(v), 1))))   # keep each slider's default inside its range
        d_rain = clip(wx['precipitation_sum'].iloc[-1], 0.0, 150.0)
        d_rad = clip(wx['shortwave_radiation_sum'].iloc[-1], 2.0, 30.0)
        d_wind = clip(wx['wind_speed_10m_max'].iloc[-1], 3.0, 60.0)
        rain = c1.slider("Rainfall (mm)", 0.0, 150.0, d_rain, 0.5)
        rad = c2.slider("Solar radiation (MJ/m²) — low = cloudy", 2.0, 30.0, d_rad, 0.5)
        wind = c3.slider("Max wind (km/h)", 3.0, 60.0, d_wind, 0.5)
    fr, fx = next_day_rows(wx, L["resid"], rain, rad, wind)
    fr0, fx0 = next_day_rows(wx, L["resid"], d_rain, d_rad, d_wind)
    hyb = L["a_next"] + float(L["rf"].predict(fr)[0]); hyb0 = L["a_next"] + float(L["rf"].predict(fr0)[0])
    lo, hi = hyb + H["q"][0], hyb + H["q"][1]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Forecast for {L['nxt'].date()} (hybrid)", f"{hyb:.1f} °C", f"{hyb - hyb0:+.2f} °C from your what-if" if abs(hyb - hyb0) > 0.005 else None)
    k2.metric("80% range (from last year's errors)", f"{lo:.1f} – {hi:.1f} °C")
    k3.metric("🔥 Heat alert (≥ 28.5 °C)", "YES" if lo >= HEAT_C else ("possible" if hi >= HEAT_C else "no"))
    others = {nm: float(A["models"][nm].predict(fx)[0]) for nm in ["Random Forest", "XGBoost", "LightGBM", "Stacking (RF+XGB+LGBM+Ridge)"]}
    others["★ Hybrid ARIMA + RF"] = hyb
    s_ = pd.Series(others).sort_values()
    c1, c2 = st.columns([1.2, 1])
    with c1:
        show_fig(fig_barh(s_, title="Each model's forecast for tomorrow (°C)", xlim=(s_.min() - 0.6, s_.max() + 0.4)))
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        recent = wx["temperature_2m_mean"].iloc[-30:]
        ax.plot(recent.index, recent.values, "o-", color="black", ms=3, label="observed")
        ax.errorbar([L["nxt"]], [hyb], yerr=[[hyb - lo], [hi - hyb]], fmt="o", color="#e76f51", capsize=6, label="forecast + 80% range")
        ax.set_title("Last 30 days and tomorrow"); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout(); show_fig(fig)
    st.caption("Move the sliders: only the Random Forest part of the hybrid sees rain, sunshine and wind, so the change you see is the "
               "non-linear weather effect that ARIMA alone cannot model.")

else:
    st.title("ℹ️ Research & About")
    st.markdown("""
### Random Forest vs gradient boosting
| | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| Trees built | independent, deep, bootstrap | sequential, depth-wise, regularised | sequential, leaf-wise, histograms | sequential, symmetric trees, ordered boosting |
| Mainly reduces | variance | bias | bias | bias |
| Tuning | minimal | moderate | moderate | minimal |
| Over-fit risk | low | medium | medium–high on small data | low |

**Time series:** trees cannot extrapolate a trend or unseen temperatures — that is why the **ARIMA + RF hybrid** lets ARIMA handle trend and season.

### Tabular vs image / text data
* **Tabular:** tree ensembles remain state-of-the-art for medium tabular data — mixed types, thresholds, interactions, little preprocessing.
* **Images / text:** signal lives in spatial/sequential structure → CNNs / Transformers dominate; ensembles are used on their embeddings or to average networks.

### Data
One real dataset: Colombo daily weather 2015–2026 from the Open-Meteo historical archive (ERA5 reanalysis), saved in `data/colombo_weather_daily_used.csv`.

### Generative AI declaration
The dashboard UI and parts of the code were **generated with Claude (Anthropic)**, then reviewed, run and tested by the student.

**B.N.V. Gunathilaka · 224059P · Faculty of IT, University of Moratuwa**
""")
