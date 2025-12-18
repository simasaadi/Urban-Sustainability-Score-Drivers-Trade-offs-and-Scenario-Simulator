# app.py
from __future__ import annotations

import os
import io
import math
import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# Optional Plotly (nice, but not required)
PLOTLY_OK = False
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Urban Sustainability Score — Drivers, Trade-offs, Scenario Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Urban Sustainability Score — Drivers, Trade-offs, and Scenario Simulator")
st.caption("Interactive exploration, modeling, explainability, and what-if scenarios (Streamlit).")


# ----------------------------
# Dataset-specific defaults (based on your uploaded CSV)
# ----------------------------
DEFAULT_CSV_CANDIDATES = [
    "data/urban_planning_dataset.csv",
    "data/raw/urban_planning_dataset.csv",
    "data/processed/urban_planning_dataset.csv",
    "urban_planning_dataset.csv",
]

TARGET_CANDIDATES = ["urban_sustainability_score", "sustainability_score", "target", "y"]


# ----------------------------
# Utilities
# ----------------------------
def _find_default_csv() -> str | None:
    for p in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float32")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="integer")
    return out


def _infer_target(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric columns found to infer a target.")
    return num_cols[-1]


@st.cache_data(show_spinner=False)
def load_data(uploaded: io.BytesIO | None, path: str | None) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if path is None:
            raise FileNotFoundError(
                "No dataset found. Upload a CSV in the sidebar, or add it to your repo under data/."
            )
        df = pd.read_csv(path)
    df = _downcast_numeric(df)
    return df


@dataclass(frozen=True)
class ModelBundle:
    target: str
    feature_cols: List[str]
    ridge: Pipeline
    hgb: HistGradientBoostingRegressor
    best_name: str


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, target: str, seed: int = 42) -> Tuple[ModelBundle, Dict[str, float]]:
    # Keep only numeric columns (your dataset is already numeric + one-hot)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in columns.")

    feature_cols = [c for c in numeric_cols if c != target]
    X = df[feature_cols].copy()
    y = df[target].astype("float32").copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=1.0, random_state=seed)),
        ]
    )

    # Light but strong default model for tabular regression (kept conservative for Cloud limits)
    hgb = HistGradientBoostingRegressor(
        random_state=seed,
        max_depth=6,
        learning_rate=0.06,
        max_iter=250,
        l2_regularization=0.0,
    )

    ridge.fit(X_train, y_train)
    hgb.fit(X_train, y_train)

    # evaluate
    def _eval(m):
        pred = m.predict(X_test)
        return {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
        }

    ridge_metrics = _eval(ridge)
    hgb_metrics = _eval(hgb)

    # choose best by R2 (tie-break by MAE)
    if (hgb_metrics["r2"] > ridge_metrics["r2"]) or (
        math.isclose(hgb_metrics["r2"], ridge_metrics["r2"]) and hgb_metrics["mae"] < ridge_metrics["mae"]
    ):
        best_name = "HistGradientBoosting"
    else:
        best_name = "Ridge"

    bundle = ModelBundle(
        target=target,
        feature_cols=feature_cols,
        ridge=ridge,
        hgb=hgb,
        best_name=best_name,
    )

    metrics = {
        "ridge_r2": ridge_metrics["r2"],
        "ridge_mae": ridge_metrics["mae"],
        "hgb_r2": hgb_metrics["r2"],
        "hgb_mae": hgb_metrics["mae"],
    }
    return bundle, metrics


def get_best_model(bundle: ModelBundle):
    return bundle.hgb if bundle.best_name == "HistGradientBoosting" else bundle.ridge


def safe_predict(model, X1: pd.DataFrame) -> float:
    return float(model.predict(X1)[0])


def baseline_profile(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # 1-row DF, median profile is stable and interpretable
    base = df[feature_cols].median(numeric_only=True).astype("float32")
    # If any NaNs, fallback to first row
    if base.isna().any():
        base = df[feature_cols].iloc[0].astype("float32")
    return pd.DataFrame([base.values], columns=feature_cols)


def scenario_apply(base_df: pd.DataFrame, deltas: Dict[str, float]) -> pd.DataFrame:
    scen = base_df.copy()
    for col, delta in deltas.items():
        if col in scen.columns:
            scen.loc[0, col] = float(scen.loc[0, col]) + float(delta)
    return scen


# ----------------------------
# Sidebar: data + controls
# ----------------------------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    default_path = _find_default_csv()
    st.caption(f"Repo default: {default_path if default_path else 'not found'}")

    seed = st.number_input("Random seed", min_value=1, max_value=10_000, value=42)

    st.divider()
    st.header("Performance / safety")
    sample_for_explain = st.slider(
        "Explainability sample size (lower = faster / less memory)",
        min_value=200, max_value=2000, value=600, step=100
    )


# ----------------------------
# Load + train
# ----------------------------
df = load_data(uploaded, default_path)
target = _infer_target(df)

bundle, metrics = train_models(df, target=target, seed=int(seed))
best_model = get_best_model(bundle)

base_df = baseline_profile(df, bundle.feature_cols)
baseline_pred = safe_predict(best_model, base_df)


# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_explore, tab_model, tab_scen = st.tabs(
    ["Overview", "Explore", "Model + Explain", "Scenario Simulator"]
)


# ----------------------------
# Overview
# ----------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Target", target)
    c4.metric("Best model", bundle.best_name)

    st.subheader("Quick model health")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ridge R²", f"{metrics['ridge_r2']:.3f}")
    m2.metric("Ridge MAE", f"{metrics['ridge_mae']:.3f}")
    m3.metric("HGB R²", f"{metrics['hgb_r2']:.3f}")
    m4.metric("HGB MAE", f"{metrics['hgb_mae']:.3f}")

    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------
# Explore
# ----------------------------
with tab_explore:
    st.subheader("Feature distributions and relationships")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if c != target]

    colA, colB = st.columns([1, 2])

    with colA:
        xcol = st.selectbox("X feature", feature_cols, index=feature_cols.index("public_transport_access") if "public_transport_access" in feature_cols else 0)
        ycol = st.selectbox("Y feature", [target] + feature_cols, index=0)
        chart_kind = st.radio("Chart", ["Scatter", "Histogram"], horizontal=True)

    with colB:
        if PLOTLY_OK:
            if chart_kind == "Scatter":
                fig = px.scatter(df, x=xcol, y=ycol, trendline="ols", opacity=0.55)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(df, x=xcol, nbins=40)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly not available. Add plotly to requirements.txt for richer charts.")

    st.subheader("Correlation heatmap (lightweight)")
    corr = df[[target] + feature_cols].corr(numeric_only=True)
    # Keep it small and cheap: show only top correlations with target
    top = corr[target].abs().sort_values(ascending=False).head(12).index.tolist()
    corr_small = corr.loc[top, top]
    if PLOTLY_OK:
        fig = px.imshow(corr_small, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(corr_small, use_container_width=True)


# ----------------------------
# Model + Explain
# ----------------------------
with tab_model:
    st.subheader("What the model thinks matters (controlled compute)")

    st.write(
        "Explainability is computed on-demand to avoid memory blowups on Streamlit Community Cloud."
    )

    # Sample once for explainability (cached by Streamlit because df is cached; but still keep small)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(df.index.values, size=min(int(sample_for_explain), len(df)), replace=False)
    X_exp = df.loc[idx, bundle.feature_cols].copy()
    y_exp = df.loc[idx, target].copy()

    colL, colR = st.columns([1, 1])

    with colL:
        if st.button("Compute permutation importance (top 15)", type="primary"):
            with st.spinner("Computing permutation importance…"):
                # Permutation importance can be heavy; keep repeats low
                pi = permutation_importance(
                    best_model, X_exp, y_exp,
                    n_repeats=3,
                    random_state=int(seed),
                    n_jobs=1,
                )
                imp = pd.DataFrame({
                    "feature": bundle.feature_cols,
                    "importance_mean": pi.importances_mean.astype("float32"),
                    "importance_std": pi.importances_std.astype("float32"),
                }).sort_values("importance_mean", ascending=False).head(15)

            if PLOTLY_OK:
                fig = px.bar(imp, x="importance_mean", y="feature", orientation="h")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp, use_container_width=True)

    with colR:
        st.write("Partial dependence (single feature, optional)")
        pd_feature = st.selectbox("Choose one feature for PDP", bundle.feature_cols, index=0)
        if st.button("Compute PDP for selected feature"):
            # PDP can be expensive for tree models; we keep it small: 1 feature only, small grid
            # We implement a lightweight PDP approximation by sweeping quantiles and predicting.
            with st.spinner("Computing PDP…"):
                qs = np.linspace(0.05, 0.95, 25).astype("float32")
                x_vals = df[pd_feature].quantile(qs).values.astype("float32")

                base = base_df.copy()
                preds = []
                for xv in x_vals:
                    row = base.copy()
                    row.loc[0, pd_feature] = float(xv)
                    preds.append(safe_predict(best_model, row))

                pdp = pd.DataFrame({pd_feature: x_vals, "pred": np.array(preds, dtype="float32")})

            if PLOTLY_OK:
                fig = px.line(pdp, x=pd_feature, y="pred")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pdp, use_container_width=True)


# ----------------------------
# Scenario Simulator
# ----------------------------
with tab_scen:
    st.subheader("What-if Scenarios (based on trained model)")
    st.caption("Baseline is the median city profile. Sliders apply relative deltas to selected drivers.")

    st.write(f"Baseline predicted score: **{baseline_pred:.4f}** (Model: **{bundle.best_name}**)")

    # Choose drivers
    default_drivers = [
        d for d in [
            "public_transport_access",
            "green_cover_percentage",
            "carbon_footprint",
            "renewable_energy_usage",
            "disaster_risk_index",
        ]
        if d in bundle.feature_cols
    ]
    drivers = st.multiselect(
        "Numeric drivers to adjust",
        options=bundle.feature_cols,
        default=default_drivers if default_drivers else bundle.feature_cols[:5],
    )

    # Step size
    scenario_step = st.slider("Scenario step size", 0.01, 1.00, 0.10, 0.01)

    st.markdown("### Set changes (relative deltas)")
    changes: Dict[str, float] = {}
    cols = st.columns(2)

    for i, d in enumerate(drivers):
        with cols[i % 2]:
            # default 0.0 keeps scenario neutral; tornado plot below will still show sensitivity (±step)
            changes[d] = st.slider(
                f"{d} (Δ)",
                min_value=float(-scenario_step * 5),
                max_value=float(scenario_step * 5),
                value=0.0,
                step=float(scenario_step),
            )

    # Run scenario
    scen_df = scenario_apply(base_df, changes)
    scen_pred = safe_predict(best_model, scen_df)
    delta_pred = scen_pred - baseline_pred

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario predicted score", f"{scen_pred:.4f}")
    c2.metric("Δ vs baseline", f"{delta_pred:+.4f}")
    c3.metric("Drivers adjusted", f"{len(drivers)}")

    # Tornado: one-at-a-time sensitivity around baseline using ±step (never blank)
    st.markdown("### Sensitivity tornado (one-at-a-time)")
    rows = []
    for k in drivers:
        up = base_df.copy()
        dn = base_df.copy()
        up.loc[0, k] = float(up.loc[0, k]) + float(scenario_step)
        dn.loc[0, k] = float(dn.loc[0, k]) - float(scenario_step)

        pred_up = safe_predict(best_model, up)
        pred_dn = safe_predict(best_model, dn)

        # effect size: half-range delta around baseline; keeps sign intuitive
        effect = (pred_up - pred_dn) / 2.0
        rows.append({"driver": k, "effect_per_step": float(effect)})

    sens = pd.DataFrame(rows).sort_values("effect_per_step", ascending=True)

    if PLOTLY_OK and not sens.empty:
        fig = px.bar(sens, x="effect_per_step", y="driver", orientation="h")
        fig.add_vline(x=0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(sens, use_container_width=True)

    st.markdown("### Download scenario outputs")
    out = pd.DataFrame(
        [{
            "baseline_pred": baseline_pred,
            "scenario_pred": scen_pred,
            "delta_vs_baseline": delta_pred,
            **{f"delta__{k}": float(v) for k, v in changes.items()},
        }]
    )
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download scenario CSV", data=csv_bytes, file_name="scenario_output.csv", mime="text/csv")
