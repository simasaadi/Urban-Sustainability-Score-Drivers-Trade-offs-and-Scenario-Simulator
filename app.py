import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Optional: Plotly gives more "portfolio-grade" interactivity
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Urban Sustainability Score — Advanced Analytics + Scenario Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Urban Sustainability Score — Drivers, Trade-offs, and Scenario Simulator")
st.caption("Interactive exploration, modeling, explainability, and what-if scenarios (Streamlit).")


# -----------------------------
# Utilities
# -----------------------------
def _find_default_csv():
    """
    Conservative search for a CSV in common repo locations.
    """
    candidates = [
        "data/urban_planning_dataset.csv",
        "data/raw/urban_planning_dataset.csv",
        "data/processed/urban_planning_dataset.csv",
        "urban_planning_dataset.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file=None, path=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if path is not None and os.path.exists(path):
        return pd.read_csv(path)
    default_path = _find_default_csv()
    if default_path is not None:
        return pd.read_csv(default_path)
    raise FileNotFoundError(
        "No dataset found. Upload a CSV in the sidebar, or add it to your repo under data/..."
    )

# -------------------------
# Scenario engine
# -------------------------
def run_scenario(baseline_df, deltas):
    """
    Apply relative deltas to a 1-row baseline dataframe and return scenario dataframe.
    """
    scen = baseline_df.copy()

    for col, delta in deltas.items():
        if col in scen.columns:
            scen[col] = scen[col] + float(delta)

    return scen

def infer_target_column(df: pd.DataFrame):
    # Prefer exact match, else pick a likely target by name
    preferred = ["urban_sustainability_score", "sustainability_score", "target", "y"]
    for c in preferred:
        if c in df.columns:
            return c
    # fallback: last numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[-1] if num_cols else None


def rmse(y_true, y_pred):
    # Compatible across sklearn versions
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_report(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    if categorical_cols:
        # Basic handling; ideally your dataset is already one-hot encoded
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", cat_pipe, categorical_cols),
            ],
            remainder="drop",
        )
    else:
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
            ],
            remainder="drop",
        )
    return pre, numeric_cols, categorical_cols


def get_feature_names(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    # If categorical columns exist, PDP/importance becomes trickier; we’ll still use X columns for naming
    return X.columns.tolist()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (recommended)", type=["csv"])
default_path = _find_default_csv()
use_default = st.sidebar.checkbox(
    "Use repo default CSV (if available)",
    value=True if (uploaded is None and default_path is not None) else False,
    help="Looks for common paths like data/urban_planning_dataset.csv",
)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)

st.sidebar.divider()
st.sidebar.header("Modeling")
do_modeling = st.sidebar.checkbox("Enable modeling + explainability", value=True)
cv_folds = st.sidebar.slider("CV folds (for model selection)", 3, 10, 5, 1)
n_perm = st.sidebar.slider("Permutation repeats", 5, 40, 20, 5)

st.sidebar.divider()
st.sidebar.header("Scenario Simulator")
scenario_step = st.sidebar.slider("Scenario step (for sliders)", 0.01, 0.20, 0.05, 0.01)
max_features_for_pdp = st.sidebar.slider("Top features for PDP/ICE", 2, 6, 4, 1)


# -----------------------------
# Load data
# -----------------------------
try:
    df = load_csv(uploaded_file=uploaded, path=default_path if use_default else None)
except Exception as e:
    st.error(str(e))
    st.stop()

target_col = infer_target_column(df)
if target_col is None or target_col not in df.columns:
    st.error("Could not infer target column. Please ensure your CSV has a target like 'urban_sustainability_score'.")
    st.stop()

# Basic cleaning
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include=["object"]).columns:
    # normalize whitespace strings
    df[c] = df[c].astype(str).str.strip()

# -----------------------------
# Top-level tabs
# -----------------------------
tab_overview, tab_explore, tab_model, tab_scenarios = st.tabs([
    "Overview",
    "Explore",
    "Model + Explain",
    "Scenario Simulator",
])


# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)

    n_rows, n_cols = df.shape
    missing = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())

    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{n_cols:,}")
    c3.metric("Missing cells", f"{missing:,}")
    c4.metric("Duplicate rows", f"{dupes:,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Target Summary")
    if target_col in df.select_dtypes(include=[np.number]).columns:
        t = df[target_col].dropna()
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Target mean", f"{t.mean():.3f}")
        cc2.metric("Target median", f"{t.median():.3f}")
        cc3.metric("Target std", f"{t.std():.3f}")
        cc4.metric("Target min/max", f"{t.min():.3f} / {t.max():.3f}")

        if PLOTLY_OK:
            fig = px.histogram(df, x=target_col, nbins=30, title="Target Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            plt.figure(figsize=(10, 4))
            plt.hist(t, bins=30)
            plt.title("Target Distribution")
            plt.xlabel(target_col)
            plt.ylabel("count")
            st.pyplot(plt.gcf())
            plt.close()


# -----------------------------
# Explore
# -----------------------------
with tab_explore:
    st.subheader("Filters")
    X_cols = [c for c in df.columns if c != target_col]

    # Light-touch filtering on numeric columns (keeps UI clean)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    filter_cols = st.multiselect("Choose numeric columns to filter", options=num_cols, default=num_cols[:3])
    df_f = df.copy()

    for col in filter_cols:
        if col in df_f.columns:
            lo, hi = float(df_f[col].min()), float(df_f[col].max())
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                r = st.slider(f"{col} range", lo, hi, (lo, hi))
                df_f = df_f[(df_f[col] >= r[0]) & (df_f[col] <= r[1])]

    st.caption(f"Filtered rows: {len(df_f):,} / {len(df):,}")
    st.dataframe(df_f.head(30), use_container_width=True)

    st.subheader("Correlation (numeric)")
    corr_df = df_f.select_dtypes(include=[np.number]).corr(numeric_only=True)
    if PLOTLY_OK and corr_df.shape[0] <= 40:
        fig = px.imshow(corr_df, title="Correlation Heatmap", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 6))
        plt.imshow(corr_df.values)
        plt.title("Correlation Heatmap")
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
        plt.yticks(range(len(corr_df.index)), corr_df.index)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    st.subheader("Feature relationships")
    x1, x2 = st.columns(2)
    with x1:
        x_col = st.selectbox("X feature", options=num_cols, index=0 if num_cols else 0)
    with x2:
        color_col = st.selectbox("Color by (optional)", options=["(none)"] + num_cols, index=0)

    if x_col:
        if PLOTLY_OK:
            fig = px.scatter(
                df_f,
                x=x_col,
                y=target_col,
                color=None if color_col == "(none)" else color_col,
                trendline="ols" if color_col == "(none)" else None,
                title=f"{target_col} vs {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            plt.figure(figsize=(10, 5))
            plt.scatter(df_f[x_col], df_f[target_col], s=10)
            plt.title(f"{target_col} vs {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(target_col)
            st.pyplot(plt.gcf())
            plt.close()


# -----------------------------
# Model + Explain
# -----------------------------
with tab_model:
    if not do_modeling:
        st.info("Modeling is disabled in the sidebar.")
        st.stop()

    st.subheader("Model training + selection")

    # Train/test split
    df_model = df.dropna(subset=[target_col]).copy()
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed)
    )

    pre, numeric_cols, categorical_cols = make_preprocessor(df_model, target_col)

    # Candidate models (fast but strong)
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            random_state=int(seed),
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=int(seed),
            max_depth=None,
            learning_rate=0.06,
            max_iter=400,
        ),
    }

    cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))

    rows = []
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        # Use R2 CV for selection
        scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
        rows.append({
            "model": name,
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
        })

    leaderboard = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)
    st.dataframe(leaderboard, use_container_width=True)

    best_name = leaderboard.iloc[0]["model"]
    best_model = Pipeline(steps=[("pre", pre), ("model", candidates[best_name])])

    with st.spinner(f"Training best model: {best_name}"):
        best_model.fit(X_train, y_train)
        pred_test = best_model.predict(X_test)

    report = regression_report(y_test, pred_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{report['R2']:.3f}")
    c2.metric("Test MAE", f"{report['MAE']:.3f}")
    c3.metric("Test RMSE", f"{report['RMSE']:.3f}")

    st.subheader("Diagnostics")
    y_true = np.array(y_test)
    y_pred = np.array(pred_test)
    resid = y_true - y_pred
    abs_err = np.abs(resid)

    colA, colB = st.columns(2)

    with colA:
        if PLOTLY_OK:
            fig = px.scatter(
                x=y_true, y=y_pred,
                labels={"x": "Actual", "y": "Predicted"},
                title="Actual vs Predicted"
            )
            # 45-degree line
            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            plt.figure(figsize=(8, 5))
            plt.scatter(y_true, y_pred, s=12)
            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            plt.plot([lo, hi], [lo, hi])
            plt.title("Actual vs Predicted")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            st.pyplot(plt.gcf())
            plt.close()

    with colB:
        if PLOTLY_OK:
            fig = px.scatter(
                x=y_pred, y=resid,
                labels={"x": "Predicted", "y": "Residual (actual - predicted)"},
                title="Residuals vs Predicted"
            )
            fig.add_hline(y=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            plt.figure(figsize=(8, 5))
            plt.scatter(y_pred, resid, s=12)
            plt.axhline(0)
            plt.title("Residuals vs Predicted")
            plt.xlabel("Predicted")
            plt.ylabel("Residual (actual - predicted)")
            st.pyplot(plt.gcf())
            plt.close()

    st.subheader("Worst-case errors (top 15)")
    err_df = X_test.copy()
    err_df["y_true"] = y_true
    err_df["y_pred"] = y_pred
    err_df["abs_error"] = abs_err
    st.dataframe(err_df.sort_values("abs_error", ascending=False).head(15), use_container_width=True)

    st.subheader("Explainability")
    st.caption("Permutation importance (with uncertainty) + PDP/ICE for top drivers.")

    # Permutation importance on test set (after preprocessing is inside pipeline)
    with st.spinner("Computing permutation importance..."):
        perm = permutation_importance(
            best_model, X_test, y_test,
            scoring="r2",
            n_repeats=int(n_perm),
            random_state=int(seed),
        )

    feat_names = get_feature_names(df_model, target_col)
    imp = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    top_imp = imp.head(15).iloc[::-1]

    colI1, colI2 = st.columns([1, 1])
    with colI1:
        st.write("Top features (Permutation importance)")
        if PLOTLY_OK:
            fig = px.bar(
                top_imp,
                x="importance_mean",
                y="feature",
                orientation="h",
                error_x="importance_std",
                title="Permutation Importance (mean ± std)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            plt.figure(figsize=(9, 6))
            plt.barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"])
            plt.title("Permutation Importance (mean ± std)")
            plt.xlabel("Decrease in R² when shuffled")
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

    with colI2:
        st.write("PDP + ICE (direction + heterogeneity)")
        top_features = imp["feature"].head(int(max_features_for_pdp)).tolist()

        # PDP/ICE can fail if categorical text columns exist.
        # We'll attempt it; if it errors, user can one-hot encode or drop non-numeric columns.
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                best_model,
                X_test,
                features=top_features,
                kind="both",          # PDP + ICE
                subsample=200,
                grid_resolution=30,
                random_state=int(seed),
                ax=ax
            )
            fig.suptitle("PDP + ICE — Top Drivers", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(
                "PDP/ICE failed (often due to non-numeric columns). "
                "If your dataset has object columns, one-hot encode them first.\n\n"
                f"Error: {e}"
            )


# -----------------------------
# Scenario Simulator
# -----------------------------
with tab_scenarios:
    if not do_modeling:
        st.info("Enable modeling in the sidebar to use scenarios (needs a trained model).")
        st.stop()

    st.subheader("What-if Scenarios (based on trained model)")
    st.caption("Define a baseline city profile and adjust key drivers to see predicted score changes.")

    # Rebuild what we need from model tab (Streamlit tab isolation)
    df_model = df.dropna(subset=[target_col]).copy()
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed)
    )

    pre, numeric_cols, categorical_cols = make_preprocessor(df_model, target_col)

    # Same candidate selection quickly (use the best from earlier logic, deterministic)
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            random_state=int(seed),
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=int(seed),
            max_depth=None,
            learning_rate=0.06,
            max_iter=400,
        ),
    }

    cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))
    rows = []
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
        rows.append({"model": name, "cv_r2_mean": float(scores.mean())})

    best_name = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).iloc[0]["model"]
    best_model = Pipeline(steps=[("pre", pre), ("model", candidates[best_name])])
    best_model.fit(X_train, y_train)

    # Baseline: median profile across numeric columns; keep categorical as mode where possible
    baseline = X.median(numeric_only=True)

    # If categorical columns exist, set them to most frequent
    for c in categorical_cols:
        try:
            baseline[c] = X[c].mode(dropna=True).iloc[0]
        except Exception:
            baseline[c] = X[c].dropna().astype(str).value_counts().index[0] if X[c].notna().any() else ""

    # Keep baseline as a 1-row dataframe aligned to X columns
    baseline_df = pd.DataFrame([baseline], columns=X.columns)
    baseline_pred = float(best_model.predict(baseline_df)[0])

    st.write(f"**Baseline predicted score:** {baseline_pred:.4f}  (Model: {best_name})")

    # Choose scenario features (only numeric, because sliders)
    st.markdown("### Choose scenario drivers")
    candidate_drivers = [c for c in numeric_cols if c in X.columns]
    default_drivers = [c for c in candidate_drivers if any(k in c.lower() for k in ["green", "renew", "public", "carbon", "risk"])][:5]
    drivers = st.multiselect("Numeric drivers to adjust", options=candidate_drivers, default=default_drivers or candidate_drivers[:5])

    if not drivers:
        st.info("Select at least one numeric driver.")
        st.stop()

    # Slider inputs
    st.markdown("### Set changes (relative deltas)")
    changes = {}
    cols = st.columns(2)
    for i, d in enumerate(drivers):
        with cols[i % 2]:
           default_delta = scenario_step if "green" in d.lower() or "renew" in d.lower() else 0.0

changes[d] = st.slider(
    f"{d} (Δ)",
    min_value=float(-scenario_step * 5),
    max_value=float(scenario_step * 5),
    value=float(default_delta),
    step=float(scenario_step),
)



     
scenario_df = run_scenario(baseline_df, deltas)
scenario_pred = float(best_model.predict(scenario_df)[0])
delta_pred = scenario_pred - baseline_pred


    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario predicted score", f"{scen_pred:.4f}")
    c2.metric("Δ vs baseline", f"{delta_pred:+.4f}")
    c3.metric("Drivers adjusted", f"{len(drivers)}")

    st.markdown("### Sensitivity tornado (one-at-a-time)")
    # One-at-a-time sensitivity around baseline for selected drivers
    rows = []
    for k in drivers:
        one = baseline_df.copy()
        one.loc[0, k] = float(one.loc[0, k]) + float(changes[k])
        pred = float(best_model.predict(one)[0])
        rows.append({"driver": k, "delta": pred - baseline_pred})

    sens = pd.DataFrame(rows).sort_values("delta")
    if PLOTLY_OK:
        fig = px.bar(sens, x="delta", y="driver", orientation="h", title="Driver impact on predicted score (one-at-a-time)")
        fig.add_vline(x=0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        plt.barh(sens["driver"], sens["delta"])
        plt.axvline(0)
        plt.title("Driver impact on predicted score (one-at-a-time)")
        plt.xlabel("Δ predicted score")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    st.markdown("### Download scenario outputs")
    out = pd.DataFrame([{
        "baseline_pred": baseline_pred,
        "scenario_pred": scen_pred,
        "delta_pred": delta_pred,
        **{f"delta_{k}": v for k, v in changes.items()}
    }])

    st.download_button(
        "Download scenario results (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="scenario_results.csv",
        mime="text/csv",
    )

