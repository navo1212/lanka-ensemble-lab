"""
LankaEnsemble Lab — Ensemble Learning Dashboard on Sri Lankan data
B.N.V. Gunathilaka (224059P) — Faculty of IT, University of Moratuwa

Run locally:   pip install -r requirements.txt  &&  streamlit run app.py
Deploy:        push this folder to GitHub → share.streamlit.io → New app → app.py

AI declaration: the dashboard UI and parts of this code were generated with the help of
Claude (Anthropic) and reviewed/tested by the student.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, StackingClassifier, VotingClassifier,
                              RandomForestRegressor, StackingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, roc_curve, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, mean_absolute_percentage_error)
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA

SEED = 42
NJ = 2   # Streamlit Community Cloud has ~2 CPUs; n_jobs=-1 would start one process per *host* CPU and thrash
DATA = Path(__file__).parent / "data"
FAM_COL = {"Baseline": "#adb5bd", "Bagging": "#2a9d8f", "Boosting": "#e76f51", "Stacking": "#6a4c93"}

st.set_page_config(page_title="LankaEnsemble Lab", page_icon="🇱🇰", layout="wide")
st.markdown("""
<style>
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
@st.cache_data
def load_tabular():
    f1, f2 = DATA / "sl_telecom_churn.csv", DATA / "sl_vehicle_prices.csv"
    if f1.exists() and f2.exists():
        return pd.read_csv(f1), pd.read_csv(f2)
    # first start without CSVs: rebuild the (seeded, reproducible) datasets from the generator
    import sys
    sys.path.insert(0, str(DATA))
    import generate_datasets as g
    c, v = g.make_churn(), g.make_vehicles()
    try:
        c.to_csv(f1, index=False); v.to_csv(f2, index=False)
    except Exception:
        pass
    return c, v


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading Colombo weather from Open-Meteo…")
def load_weather():
    end = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=6.9271&longitude=79.8612"
           f"&start_date=2015-01-01&end_date={end}"
           "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,"
           "wind_speed_10m_max,shortwave_radiation_sum&timezone=Asia%2FColombo")
    try:
        import requests
        r = requests.get(url, timeout=30); r.raise_for_status()
        d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
        src = "REAL · Open-Meteo archive (ERA5), Colombo — downloaded live"
    except Exception:
        if (DATA / "colombo_weather_daily_used.csv").exists():
            d = pd.read_csv(DATA / "colombo_weather_daily_used.csv")
            src = "Saved copy from the notebook run (data/colombo_weather_daily_used.csv)"
        elif (DATA / "colombo_weather_daily.csv").exists():
            d = pd.read_csv(DATA / "colombo_weather_daily.csv")
            src = "SYNTHETIC fallback (no internet)"
        else:
            import sys
            sys.path.insert(0, str(DATA))
            import generate_datasets as g
            d = g.make_weather()
            src = "SYNTHETIC fallback (no internet)"
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").asfreq("D").interpolate(limit_direction="both")
    return d, src


churn_df, veh_df = load_tabular()
CAT_C = ["district", "province", "area_type", "language", "gender", "plan_type", "payment_method"]
NUM_C = [c for c in churn_df.columns if c not in CAT_C + ["customer_id", "churn"]]
CAT_V = ["brand", "model", "fuel_type", "body_type", "transmission", "condition", "district", "seller_type"]
NUM_V = ["engine_cc", "year_of_manufacture", "vehicle_age", "mileage_km", "previous_owners", "on_lease"]


# =====================================================================================
# MODEL TRAINING (cached — runs once per server)
# =====================================================================================
def churn_model_zoo():
    base = [("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=NJ)),
            ("xgb", xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                      colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=NJ)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1)),
            ("lr", LogisticRegression(max_iter=2000))]
    return {
        "Logistic Regression": ("Baseline", LogisticRegression(max_iter=2000)),
        "Decision Tree": ("Baseline", DecisionTreeClassifier(max_depth=6, random_state=SEED)),
        "Bagging (Trees)": ("Bagging", BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=SEED, n_jobs=NJ)),
        "Random Forest": ("Bagging", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=NJ)),
        "Extra Trees": ("Bagging", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=NJ)),
        "AdaBoost": ("Boosting", AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)),
        "Gradient Boosting": ("Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)),
        "XGBoost": ("Boosting", xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                                                  colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=NJ)),
        "LightGBM": ("Boosting", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                                    subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)),
        "Stacking (RF+XGB+LGBM+LR)": ("Stacking", StackingClassifier(base, final_estimator=LogisticRegression(max_iter=2000),
                                                                     cv=3, stack_method="predict_proba", n_jobs=NJ)),
        "Soft Voting (RF+XGB+LGBM+LR)": ("Stacking", VotingClassifier(base, voting="soft", n_jobs=NJ)),
    }


def clf_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, yhat), "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat), "F1": f1_score(y, yhat), "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p)}


@st.cache_resource(show_spinner="Training 11 churn models (bagging, boosting, stacking)… first load ≈ 1 min")
def train_churn():
    X = churn_df[CAT_C + NUM_C]; y = churn_df["churn"].values
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_C),
                             ("num", StandardScaler(), NUM_C)])
    Xtr, Xte = pre.fit_transform(Xtr_raw), pre.transform(Xte_raw)
    feats = [f.split("__", 1)[1] for f in pre.get_feature_names_out()]
    models, probs, rows = {}, {}, []
    for name, (fam, m) in churn_model_zoo().items():
        t0 = time.time(); m.fit(Xtr, ytr); secs = time.time() - t0
        p = m.predict_proba(Xte)[:, 1]
        models[name], probs[name] = m, p
        rows.append({"Model": name, "Family": fam, **clf_metrics(yte, p), "Train s": secs})
    board = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    # ensemble-size curves: RF vs XGBoost
    Ns = [5, 10, 25, 50, 100, 200, 300, 400]
    rf_w = RandomForestClassifier(min_samples_leaf=3, random_state=SEED, n_jobs=NJ, warm_start=True)
    rf_curve = []
    for n in Ns:
        rf_w.set_params(n_estimators=n); rf_w.fit(Xtr, ytr)
        rf_curve.append(roc_auc_score(yte, rf_w.predict_proba(Xte)[:, 1]))
    xg = models["XGBoost"]
    xgb_curve = [roc_auc_score(yte, xg.predict_proba(Xte, iteration_range=(0, n))[:, 1]) for n in Ns]
    curve = pd.DataFrame({"Random Forest": rf_curve, "XGBoost": xgb_curve}, index=pd.Index(Ns, name="trees"))
    test_df = churn_df.loc[Xte_raw.index].copy()
    return dict(pre=pre, feats=feats, models=models, probs=probs, board=board, yte=yte, curve=curve, test_df=test_df)


@st.cache_resource(show_spinner="Training RF, ANN and hybrid RF+ANN price models…")
def train_vehicle():
    X = veh_df[CAT_V + NUM_V]; y = np.log(veh_df["price_lkr_mn"].values)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_V),
                             ("num", StandardScaler(), NUM_V)])
    Xtr, Xte = pre.fit_transform(Xtr_raw), pre.transform(Xte_raw)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=NJ)
    ann = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=2e-3, max_iter=600,
                       early_stopping=True, n_iter_no_change=25, random_state=SEED)
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof_rf = cross_val_predict(rf, Xtr, ytr, cv=kf, n_jobs=NJ)
    oof_ann = cross_val_predict(ann, Xtr, ytr, cv=kf)
    rf.fit(Xtr, ytr); ann.fit(Xtr, ytr)
    ws = np.linspace(0, 1, 101)
    w = float(ws[np.argmin([mean_squared_error(ytr, a * oof_rf + (1 - a) * oof_ann) for a in ws])])
    res_ann = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=600, early_stopping=True,
                           n_iter_no_change=25, random_state=SEED).fit(np.column_stack([Xtr, oof_rf]), ytr - oof_rf)
    stack = StackingRegressor([("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.5,
                                                            random_state=SEED, n_jobs=NJ)),
                               ("ann", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=2e-3,
                                                    max_iter=600, early_stopping=True, n_iter_no_change=25,
                                                    random_state=SEED))],
                              final_estimator=RidgeCV(), cv=5, n_jobs=NJ).fit(Xtr, ytr)
    p_rf, p_ann = rf.predict(Xte), ann.predict(Xte)
    preds = {"Random Forest": p_rf, "ANN (MLP)": p_ann, f"Hybrid: Blend (w_RF={w:.2f})": w * p_rf + (1 - w) * p_ann,
             "Hybrid: RF + ANN residual": p_rf + res_ann.predict(np.column_stack([Xte, p_rf])),
             "Hybrid: Stacking → Ridge": stack.predict(Xte)}

    def m(yl, pl):
        a, b = np.exp(yl), np.exp(pl)
        return {"MAE (LKR mn)": mean_absolute_error(a, b), "RMSE (LKR mn)": np.sqrt(mean_squared_error(a, b)),
                "MAPE %": 100 * mean_absolute_percentage_error(a, b), "R²": r2_score(a, b)}
    board = pd.DataFrame({k: m(yte, v) for k, v in preds.items()}).T.sort_values("MAE (LKR mn)")
    return dict(pre=pre, rf=rf, ann=ann, w=w, board=board, preds=preds, yte=yte)


def ts_features(df, resid, y):
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


def fourier(idx, t0):
    doy = idx.dayofyear.values
    t = (idx - t0).days.values
    return pd.DataFrame({"s1": np.sin(2 * np.pi * doy / 365.25), "c1": np.cos(2 * np.pi * doy / 365.25),
                         "s2": np.sin(4 * np.pi * doy / 365.25), "c2": np.cos(4 * np.pi * doy / 365.25),
                         "trend": t / 3652.5}, index=idx)


@st.cache_resource(show_spinner="Fitting ARIMA and the ARIMA + RF hybrid on Colombo weather…")
def train_weather(test_days=365):
    wx, src = load_weather()
    y = wx["temperature_2m_mean"].astype(float)
    F = fourier(wx.index, wx.index[0])
    ytr, yte = y.iloc[:-test_days], y.iloc[-test_days:]
    order = (2, 0, 1)
    ar = ARIMA(ytr, exog=F.iloc[:-test_days], order=order, trend="c").fit()
    full = ar.append(yte, exog=F.iloc[-test_days:], refit=False)
    fc_all = full.fittedvalues
    resid = y - fc_all
    feats = ts_features(wx, resid, y)
    valid = feats.dropna().index
    tr = valid[valid < yte.index[0]][30:]
    te = yte.index
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)
    rf.fit(feats.loc[tr], resid.loc[tr])
    hybrid = fc_all.loc[te] + rf.predict(feats.loc[te])
    dcols = [c for c in feats.columns if not c.startswith("resid_")]
    rfd = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=NJ)
    rfd.fit(feats.loc[tr, dcols], y.loc[tr])
    preds = pd.DataFrame({"Actual": yte, "Naive (yesterday)": y.shift(1).loc[te], "ARIMA + Fourier": fc_all.loc[te],
                          "Random Forest (direct)": rfd.predict(feats.loc[te, dcols]), "Hybrid ARIMA + RF": hybrid})
    board = pd.DataFrame({c: {"MAE °C": mean_absolute_error(preds.Actual, preds[c]),
                              "RMSE °C": np.sqrt(mean_squared_error(preds.Actual, preds[c])),
                              "R²": r2_score(preds.Actual, preds[c])} for c in preds.columns[1:]}).T.sort_values("RMSE °C")
    # tomorrow's forecast (day after the last observation)
    nxt = wx.index[-1] + pd.Timedelta(days=1)
    wx2 = pd.concat([wx, pd.DataFrame(index=[nxt], columns=wx.columns, dtype=float)])
    wx2.index.name = "date"
    y2 = pd.concat([y, pd.Series([np.nan], index=[nxt])])
    r2_ = pd.concat([resid, pd.Series([np.nan], index=[nxt])])
    f_next = ts_features(wx2, r2_, y2).loc[[nxt]]
    a_next = float(np.asarray(full.forecast(1, exog=fourier(pd.DatetimeIndex([nxt]), wx.index[0])))[0])
    h_next = a_next + float(rf.predict(f_next)[0])
    imp = pd.Series(rf.feature_importances_, index=feats.columns).sort_values(ascending=False)
    return dict(wx=wx, src=src, preds=preds, board=board, imp=imp, nxt=nxt, a_next=a_next, h_next=h_next, order=order)


# =====================================================================================
# UI
# =====================================================================================
PAGES = ["🏠 Home & Story", "⚔️ Ensemble Arena (Churn)", "🎯 Churn Predictor & Retention Optimizer",
         "🚗 ikman Deal Checker (RF + ANN)", "🌦️ Colombo Weather (ARIMA + RF)", "ℹ️ Research & About"]
with st.sidebar:
    st.markdown("## 🇱🇰 LankaEnsemble Lab")
    st.caption("Bagging · Boosting · Stacking · Hybrids — on Sri Lankan data")
    page = st.radio("Navigate", PAGES, index=0)
    st.divider()
    st.caption("B.N.V. Gunathilaka · 224059P\nFaculty of IT, University of Moratuwa")
    st.caption("🤖 UI & parts of code generated with Claude (Anthropic); reviewed by the student.")


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def fig_bar(series, colors=None, title="", xlabel="", xlim=None):
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.38 * len(series))))
    ax.barh(series.index[::-1], series.values[::-1], color=(colors[::-1] if colors is not None else "#2a9d8f"))
    ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- HOME
if page == PAGES[0]:
    st.markdown("""<div class="hero"><h1>🇱🇰 LankaEnsemble Lab</h1>
    <p>One dashboard, three Sri Lankan problems, every major ensemble idea:
    <b>Bagging</b>, <b>Boosting</b>, <b>Stacking</b> and <b>hybrid models</b>.</p></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class="card"><h3>📱 Telecom churn</h3>8,000 mobile customers in 25 districts. Most carry
    <b>two SIMs</b> — one competitor offer and they're gone. <br><br><b>Which ensemble catches them first?</b><br>
    <span class="tag" style="background:#2a9d8f">Bagging</span><span class="tag" style="background:#e76f51">Boosting</span>
    <span class="tag" style="background:#6a4c93">Stacking</span></div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="card"><h3>🚗 ikman.lk car prices</h3>6,000 used-vehicle listings — Aqua, Vitz, Wagon R,
    Vezel, Prado… in LKR. <br><br><b>Is that asking price fair?</b><br>
    <span class="tag" style="background:#264653">Hybrid RF + ANN</span></div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="card"><h3>🌦️ Colombo weather</h3>10+ years of daily Colombo weather, monsoons included.
    <br><br><b>What will tomorrow's temperature be?</b><br>
    <span class="tag" style="background:#457b9d">Hybrid ARIMA + RF</span></div>""", unsafe_allow_html=True)
    st.write("")
    st.subheader("How the three ensemble families work")
    a, b, c = st.columns(3)
    a.info("**Bagging** 🌳🌳🌳 → average\n\nMany deep trees trained **in parallel** on bootstrap samples. "
           "Averaging cancels their individual mistakes → **reduces variance**. *Random Forest, Extra Trees.*")
    b.warning("**Boosting** 🌱→🌿→🌳 sequential\n\nShallow trees built **one after another**, each fixing the errors "
              "of the previous ones → **reduces bias**. *AdaBoost, Gradient Boosting, XGBoost, LightGBM.*")
    c.success("**Stacking** 🧠 meta-learner\n\nDifferent model types (RF, XGB, LGBM, LR) make predictions; a "
              "**meta-model learns how much to trust each one**. *Stacking, soft voting.*")
    st.subheader("💡 Innovation — from predictions to decisions")
    st.markdown("- **Retention Budget Optimizer:** turns churn probabilities into *LKR saved* and picks whom to call within a budget.\n"
                "- **ikman Deal Checker:** uses the spread of 300 Random-Forest trees to give a **fair-price range**, not just one number.\n"
                "- **Tomorrow's Colombo forecast:** ARIMA handles trend & season, RF learns the monsoon-driven non-linear errors.")
    with st.expander("⚠️ Data honesty statement"):
        st.markdown("* **Colombo weather** is **real** (Open-Meteo ERA5 archive, downloaded live; offline fallback is synthetic).\n"
                    "* **Telecom churn** and **vehicle prices** are **synthetic Sri Lankan-context datasets** generated by the "
                    "student (real operator data is confidential; mass-scraping ikman.lk breaches its terms). The generators "
                    "encode local patterns: district populations, language areas, rural network quality, dual-SIM use, eZ Cash "
                    "reloads, popular car models, recondition vs registered, import-ban premium. Code is in `data/generate_datasets.py`.")

# ---------------------------------------------------------------- ARENA
elif page == PAGES[1]:
    st.title("⚔️ Ensemble Arena — who predicts churn best?")
    R = train_churn()
    board = R["board"]
    fams = st.multiselect("Show ensemble families", list(FAM_COL), default=list(FAM_COL))
    metric = st.selectbox("Rank by", ["ROC-AUC", "PR-AUC", "F1", "Accuracy", "Recall", "Precision"], index=0)
    b = board[board.Family.isin(fams)].sort_values(metric, ascending=False)
    top = b.iloc[0] if len(b) else board.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏆 Best model", top.Model)
    k2.metric(f"Best {metric}", f"{top[metric]:.4f}")
    k3.metric("Best bagging AUC", f"{board[board.Family == 'Bagging']['ROC-AUC'].max():.4f}")
    k4.metric("Best boosting AUC", f"{board[board.Family == 'Boosting']['ROC-AUC'].max():.4f}")
    left, right = st.columns([1.15, 1])
    with left:
        lo, hi = b[metric].min(), b[metric].max()
        pad = max(0.01, (hi - lo) * 0.25)
        show_fig(fig_bar(b.set_index("Model")[metric], colors=[FAM_COL[f] for f in b.Family],
                          title=f"{metric} on 1,600 unseen customers", xlabel=metric,
                          xlim=(max(0, lo - pad), min(1, hi + pad / 2))))
        st.caption("🟩 Bagging · 🟧 Boosting · 🟪 Stacking · ⬜ Baseline")
    with right:
        sel = st.multiselect("ROC curves for", list(b.Model), default=list(b.Model[:4]))
        fig, ax = plt.subplots(figsize=(6, 5))
        for n in sel:
            fpr, tpr, _ = roc_curve(R["yte"], R["probs"][n]); ax.plot(fpr, tpr, label=f"{n} ({roc_auc_score(R['yte'], R['probs'][n]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7); ax.set_title("ROC curves"); fig.tight_layout(); show_fig(fig)
    st.dataframe(b.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]} | {"Train s": "{:.1f}"})
                 .background_gradient(subset=[metric], cmap="Greens"), hide_index=True)

    st.subheader("🌲 vs 🚀 Random Forest vs XGBoost — how does accuracy grow with more trees?")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(R["curve"].index, R["curve"]["Random Forest"], "o-", color="#2a9d8f", lw=2.5, label="Random Forest (bagging)")
        ax.plot(R["curve"].index, R["curve"]["XGBoost"], "s-", color="#e76f51", lw=2.5, label="XGBoost (boosting)")
        ax.set_xlabel("number of trees"); ax.set_ylabel("test ROC-AUC"); ax.set_title("Accuracy vs ensemble size")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); show_fig(fig)
    with c2:
        st.markdown("""**What this shows**
* **RF (bagging)** climbs fast then **plateaus** — more trees only average more; they never over-fit.
* **XGBoost (boosting)** climbs **faster** (each tree corrects the last), peaks, then can **over-fit** if it keeps adding trees — boosting needs early stopping.
* Bagging fights **variance**, boosting fights **bias** — visible in one chart.""")

    st.subheader("🎚️ Decision threshold explorer")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        mdl = st.selectbox("Model", list(board.Model), index=0)
        thr = st.slider("Call a customer 'will churn' if probability ≥", 0.05, 0.95, 0.40, 0.05)
        mm = clf_metrics(R["yte"], R["probs"][mdl], thr)
        st.write({k: round(v, 3) for k, v in mm.items()})
        st.caption("Lowering the threshold catches more churners (recall ↑) but calls more loyal customers (precision ↓).")
    with c2:
        cm = confusion_matrix(R["yte"], (R["probs"][mdl] >= thr).astype(int))
        fig, ax = plt.subplots(figsize=(4.5, 3.6)); ax.imshow(cm, cmap="Oranges")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{mdl} @ {thr:.2f}"); fig.tight_layout(); show_fig(fig)

    st.subheader("🔍 What drives churn? (RF impurity importance vs XGBoost gain importance)")
    imp = pd.DataFrame({"Random Forest": R["models"]["Random Forest"].feature_importances_,
                        "XGBoost": R["models"]["XGBoost"].feature_importances_}, index=R["feats"])
    imp = imp / imp.sum()
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index[:12]]
    st.bar_chart(imp)

# ---------------------------------------------------------------- PREDICTOR & OPTIMIZER
elif page == PAGES[2]:
    st.title("🎯 Churn Predictor & 💰 Retention Budget Optimizer")
    R = train_churn()
    tab1, tab2 = st.tabs(["👤 Single customer what-if", "💰 Retention budget optimizer"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        district = c1.selectbox("District", sorted(churn_df.district.unique()), index=sorted(churn_df.district.unique()).index("Kurunegala"))
        prov = churn_df.loc[churn_df.district == district, "province"].iloc[0]
        area = c1.radio("Area", ["Urban", "Rural"], index=1, horizontal=True)
        language = c1.selectbox("Language", ["Sinhala", "Tamil", "English"])
        age = c1.slider("Age", 16, 78, 24)
        gender = c1.radio("Gender", ["Male", "Female"], horizontal=True)
        plan = c2.radio("Plan", ["Prepaid", "Postpaid"], horizontal=True)
        tenure = c2.slider("Tenure (months)", 1, 144, 8)
        spend = c2.number_input("Monthly spend (LKR)", 100, 15000, 900, 50)
        data_gb = c2.slider("Data usage (GB / month)", 0.0, 80.0, 18.0, 0.5)
        voice = c2.slider("Voice minutes / month", 0, 2000, 200, 10)
        pay = c2.selectbox("Payment method", sorted(churn_df.payment_method.unique()))
        netq = c3.slider("Network quality score (1–5)", 1.0, 5.0, 2.8, 0.1)
        drop = c3.slider("Dropped-call rate %", 0.0, 15.0, 2.0, 0.1)
        comp = c3.slider("Complaints in last 6 months", 0, 10, 2)
        dual = c3.toggle("Uses a second SIM", True)
        offer = c3.toggle("Received a competitor offer", True)
        fibre = c3.toggle("Has home fibre / TV bundle", False)
        pts = c3.toggle("Uses loyalty points", False)
        days_reload = c3.slider("Days since last reload", 0, 60, 14) if plan == "Prepaid" else 0
        late = c3.slider("Late bill payments", 0, 6, 0) if plan == "Postpaid" else 0
        contract = c3.slider("Contract months left", 0, 24, 6) if plan == "Postpaid" else 0
        row = pd.DataFrame([{"district": district, "province": prov, "area_type": area, "language": language, "age": age,
                             "gender": gender, "plan_type": plan, "tenure_months": tenure, "monthly_spend_lkr": spend,
                             "data_usage_gb": data_gb, "voice_minutes": voice, "network_quality_score": netq,
                             "dropped_call_rate_pct": drop, "complaints_6m": comp, "dual_sim": int(dual),
                             "has_fibre_bundle": int(fibre), "competitor_offer_received": int(offer),
                             "days_since_last_reload": days_reload, "late_bill_payments": late,
                             "contract_months_left": contract, "payment_method": pay, "uses_loyalty_points": int(pts)}])
        x = R["pre"].transform(row[CAT_C + NUM_C])
        show = ["Random Forest", "Extra Trees", "XGBoost", "LightGBM", "Gradient Boosting", "Stacking (RF+XGB+LGBM+LR)"]
        ps = pd.Series({n: R["models"][n].predict_proba(x)[0, 1] for n in show})
        p_main = ps["Stacking (RF+XGB+LGBM+LR)"]
        risk = "🔴 HIGH" if p_main >= .5 else ("🟠 MEDIUM" if p_main >= .25 else "🟢 LOW")
        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric("Stacking ensemble churn probability", f"{p_main:.1%}")
        k2.metric("Risk level", risk)
        k3.metric("12-month revenue at risk", f"LKR {p_main * spend * 12:,.0f}")
        show_fig(fig_bar(ps.sort_values(ascending=False), title="Each ensemble's opinion (churn probability)"))
        tips = []
        if offer and dual: tips.append("Dual-SIM + competitor offer is the strongest churn trigger → send a matching data offer today.")
        if netq < 3: tips.append("Poor network quality → check the nearest tower; offer Wi-Fi calling / bonus data as compensation.")
        if comp >= 2: tips.append("Repeated complaints → priority callback from the customer-care team.")
        if not fibre: tips.append("No home bundle → bundles strongly reduce churn; cross-sell fibre / PEO-TV style package.")
        if plan == "Prepaid" and days_reload > 20: tips.append("Long reload gap → send a reload-bonus reminder via SMS / eZ Cash.")
        if tips:
            st.markdown("**Suggested retention actions**\n" + "\n".join(f"- {t}" for t in tips))
    with tab2:
        st.markdown("Rank customers by **expected LKR saved** = P(churn) × 12-month value × offer success − offer cost, "
                    "then target until the budget runs out. Evaluated on 1,600 test customers with their **true** outcomes.")
        c1, c2, c3, c4 = st.columns(4)
        budget = c1.number_input("Budget (LKR)", 50_000, 2_000_000, 300_000, 50_000)
        cost = c2.number_input("Cost per offer (LKR)", 100, 5_000, 500, 50)
        succ = c3.slider("Offer success rate", 0.05, 0.9, 0.35, 0.05)
        model_name = c4.selectbox("Probability model", list(R["board"].Model), index=list(R["board"].Model).index("Stacking (RF+XGB+LGBM+LR)"))
        d = R["test_df"].copy(); d["p_churn"] = R["probs"][model_name]; d["value_12m"] = d.monthly_spend_lkr * 12
        d["expected_saving"] = d.p_churn * d.value_12m * succ - cost
        k = int(budget // cost)

        def outcome(dd):
            t = dd.head(k); saved = (t.churn * t.value_12m * succ).sum()
            return {"Customers targeted": len(t), "True churners reached": int(t.churn.sum()),
                    "Revenue saved (LKR)": saved, "Net ROI (LKR)": saved - len(t) * cost}
        res = pd.DataFrame({"Random targeting": outcome(d.sample(frac=1, random_state=SEED)),
                            "Highest churn probability": outcome(d.sort_values("p_churn", ascending=False)),
                            "💡 Value-aware optimizer": outcome(d.sort_values("expected_saving", ascending=False))}).T
        k1, k2 = st.columns(2)
        k1.metric("Optimizer net ROI", f"LKR {res.iloc[2]['Net ROI (LKR)']:,.0f}",
                  f"{res.iloc[2]['Net ROI (LKR)'] - res.iloc[0]['Net ROI (LKR)']:+,.0f} vs random")
        k2.metric("Churners reached by optimizer", int(res.iloc[2]["True churners reached"]),
                  f"{int(res.iloc[2]['True churners reached'] - res.iloc[0]['True churners reached']):+d} vs random")
        st.dataframe(res.style.format({"Customers targeted": "{:,.0f}", "True churners reached": "{:,.0f}",
                                       "Revenue saved (LKR)": "{:,.0f}", "Net ROI (LKR)": "{:,.0f}"}))
        st.markdown("**Call list (top 25)**")
        st.dataframe(d.sort_values("expected_saving", ascending=False).head(25)[
            ["customer_id", "district", "plan_type", "monthly_spend_lkr", "p_churn", "expected_saving"]]
            .style.format({"p_churn": "{:.1%}", "expected_saving": "LKR {:,.0f}", "monthly_spend_lkr": "{:,.0f}"}),
            hide_index=True)

# ---------------------------------------------------------------- VEHICLES
elif page == PAGES[3]:
    st.title("🚗 ikman Deal Checker — Hybrid Random Forest + Neural Network")
    V = train_vehicle()
    best = V["board"].index[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("🏆 Best model", best)
    k2.metric("Mean abs. error", f"LKR {V['board'].iloc[0]['MAE (LKR mn)']:.2f} mn")
    k3.metric("MAPE", f"{V['board'].iloc[0]['MAPE %']:.1f}%")
    tab1, tab2 = st.tabs(["🔎 Check a listing", "📊 Model comparison"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        models = veh_df.groupby("model").size().sort_values(ascending=False).index.tolist()
        model = c1.selectbox("Model", models, index=0)
        info = veh_df[veh_df.model == model].iloc[0]
        c1.caption(f"{info.brand} · {info.fuel_type} · {info.engine_cc} cc · {info.body_type}")
        yrs = veh_df[veh_df.model == model].year_of_manufacture
        yom = c1.slider("Year of manufacture", int(yrs.min()), 2024, int(yrs.median()))
        km = c2.number_input("Mileage (km)", 0, 400_000, int(max(5000, (2025 - yom) * 11500)), 1000)
        cond = c2.radio("Condition", ["Used", "Recondition"], horizontal=True)
        trans = c2.radio("Transmission", ["Automatic", "Manual"], horizontal=True)
        owners = c2.slider("Previous owners", 0, 5, 0 if cond == "Recondition" else 1)
        district = c3.selectbox("District", sorted(veh_df.district.unique()), index=sorted(veh_df.district.unique()).index("Colombo"))
        seller = c3.radio("Seller", ["Owner", "Dealer"], horizontal=True)
        lease = c3.toggle("On lease", False)
        ask = c3.number_input("Asking price on ikman (LKR mn)", 0.5, 120.0, float(round(veh_df[veh_df.model == model].price_lkr_mn.median(), 1)), 0.1)
        row = pd.DataFrame([{"brand": info.brand, "model": model, "fuel_type": info.fuel_type, "body_type": info.body_type,
                             "transmission": trans, "condition": cond, "district": district, "seller_type": seller,
                             "engine_cc": int(info.engine_cc), "year_of_manufacture": yom, "vehicle_age": 2025 - yom,
                             "mileage_km": km, "previous_owners": owners, "on_lease": int(lease)}])
        x = V["pre"].transform(row[CAT_V + NUM_V])
        trees = np.exp([t.predict(x)[0] for t in V["rf"].estimators_])
        point = float(np.exp(V["w"] * V["rf"].predict(x)[0] + (1 - V["w"]) * V["ann"].predict(x)[0]))
        lo, hi = np.percentile(trees, [10, 90])
        verdict = ("✅ GOOD DEAL — below the fair range" if ask < lo else
                   ("⚠️ OVERPRICED — above the fair range" if ask > hi else "👍 FAIR PRICE — within the range"))
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Hybrid RF+ANN estimate", f"LKR {point:.2f} mn")
        m2.metric("Fair range (P10–P90 of 300 trees)", f"{lo:.2f} – {hi:.2f} mn")
        m3.metric("Asking vs estimate", f"LKR {ask:.2f} mn", f"{(ask - point) / point:+.1%}", delta_color="inverse")
        (st.success if "GOOD" in verdict else st.warning if "OVER" in verdict else st.info)(verdict)
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.hist(trees, bins=30, color="#2a9d8f", alpha=.8); ax.axvline(point, color="k", lw=2, label="Hybrid estimate")
        ax.axvline(ask, color="#e76f51", lw=2, ls="--", label="Asking price"); ax.axvspan(lo, hi, color="#2a9d8f", alpha=.12, label="Fair range")
        ax.set_xlabel("LKR mn"); ax.set_title("What each of the 300 Random-Forest trees thinks this car is worth"); ax.legend()
        fig.tight_layout(); show_fig(fig)
    with tab2:
        st.dataframe(V["board"].style.format("{:.3f}").background_gradient(subset=["MAE (LKR mn)"], cmap="Greens_r"))
        pick = st.selectbox("Actual vs predicted for", list(V["preds"]), index=list(V["preds"]).index(best))
        sc = pd.DataFrame({"Actual (LKR mn)": np.exp(V["yte"]), "Predicted (LKR mn)": np.exp(V["preds"][pick])})
        st.scatter_chart(sc, x="Actual (LKR mn)", y="Predicted (LKR mn)")
        st.markdown(f"""**Why hybridise?** RF handles categories and thresholds (import-ban premium, old-hybrid battery risk) but
predicts in *steps* and can't extrapolate. The ANN learns *smooth* depreciation but is weaker on rare models.
Their errors are only partly correlated, so the blend (w_RF = {V['w']:.2f}, chosen on out-of-fold predictions) and stacking
cancel part of each model's error.""")

# ---------------------------------------------------------------- WEATHER
elif page == PAGES[4]:
    st.title("🌦️ Colombo Weather — Hybrid ARIMA + Random Forest")
    W = train_weather()
    st.caption(f"Data source: **{W['src']}** · {W['wx'].index.min().date()} → {W['wx'].index.max().date()}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Forecast for {W['nxt'].date()}", f"{W['h_next']:.1f} °C", f"{W['h_next'] - W['a_next']:+.2f} °C RF correction")
    k2.metric("ARIMA alone", f"{W['a_next']:.1f} °C")
    k3.metric("Hybrid test RMSE", f"{W['board'].loc['Hybrid ARIMA + RF', 'RMSE °C']:.3f} °C")
    k4.metric("ARIMA test RMSE", f"{W['board'].loc['ARIMA + Fourier', 'RMSE °C']:.3f} °C")
    days = st.slider("Show last N test days", 30, 365, 120, 15)
    cols = st.multiselect("Series", list(W["preds"].columns), default=["Actual", "ARIMA + Fourier", "Hybrid ARIMA + RF"])
    styles = {"Actual": dict(color="black", lw=2.2), "Naive (yesterday)": dict(color="#adb5bd", lw=1.2, ls=":"),
              "ARIMA + Fourier": dict(color="#4c72b0", lw=1.6, ls="--"),
              "Random Forest (direct)": dict(color="#2a9d8f", lw=1.4), "Hybrid ARIMA + RF": dict(color="#e76f51", lw=2.0)}
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for c in cols:
        s_ = W["preds"][c].iloc[-days:]
        ax.plot(s_.index, s_.values, label=c, **styles.get(c, {}))
    ax.set_ylabel("°C (daily mean)"); ax.grid(alpha=.3)
    ax.legend(ncol=max(1, len(cols)), loc="upper center", fontsize=9, frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); show_fig(fig)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(W["board"].style.format("{:.4f}").background_gradient(subset=["RMSE °C"], cmap="Greens_r"))
        st.markdown(f"""**How the hybrid works**
1. **ARIMA{W['order']}** + yearly Fourier terms → linear, autocorrelated part (trend + season).
2. **Random Forest** learns ARIMA's **residuals** from lagged rain, wind, solar radiation and monsoon flags.
3. Forecast = ARIMA + RF correction. All inputs use data up to *yesterday* only.""")
    with c2:
        show_fig(fig_bar(W["imp"][:12], title="What the RF learned about ARIMA's errors"))
    with st.expander("Monthly climate of Colombo (from the data)"):
        mon = W["wx"].groupby(W["wx"].index.month).agg(temp=("temperature_2m_mean", "mean"), rain=("precipitation_sum", "sum"))
        mon["rain"] = mon["rain"] / W["wx"].index.year.nunique()
        mon.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        c1, c2 = st.columns(2)
        c1.bar_chart(mon["rain"], y_label="avg rain per month (mm)")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(mon.index, mon["temp"], "o-", color="#e76f51"); ax.set_ylabel("mean temp °C"); ax.grid(alpha=.3)
        fig.tight_layout()
        with c2:
            show_fig(fig)

# ---------------------------------------------------------------- ABOUT
else:
    st.title("ℹ️ Research notes & About")
    st.markdown("""
### Random Forest vs gradient boosting
| | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| Trees built | independent, deep, bootstrap | sequential, depth-wise, regularised | sequential, leaf-wise, histograms | sequential, symmetric trees, ordered boosting |
| Mainly reduces | variance | bias | bias | bias |
| Tuning | minimal | moderate | moderate | minimal |
| Categoricals | need encoding | need encoding | native | native (best) |
| Over-fit risk | low | medium | medium–high on small data | low |

### Tabular vs image / text data
* **Tabular:** tree ensembles remain state-of-the-art for medium tabular data — mixed types, thresholds, interactions, little preprocessing.
* **Images / text:** signal lives in spatial/sequential structure → CNNs / Transformers dominate; ensembles are used on top of their embeddings or to average several networks.

### Data honesty
Colombo weather is **real** (Open-Meteo / ERA5). Telecom churn and vehicle listings are **synthetic, Sri Lankan-context** datasets generated by the student (`data/generate_datasets.py`).

### Generative AI declaration
The dashboard UI and parts of the code were **generated with Claude (Anthropic)**, then reviewed, run and tested by the student.

**B.N.V. Gunathilaka · 224059P · Faculty of IT, University of Moratuwa**
""")
