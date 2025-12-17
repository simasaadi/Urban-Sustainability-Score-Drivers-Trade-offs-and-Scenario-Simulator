# ===============================
# Urban Sustainability Score App
# Stable, Guarded, Cloud-Safe
# ===============================

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional Plotly
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Urban Sustainability Score – Drivers & Scenario Simulator",
    layout="wide",
)

st.title("Urban Sustainability Score — Drivers, Trade-offs, and Scenario Simulator")
st.caption("Interactive exploration, modeling, and what-if scenarios (Streamlit)")

# -------------------------------
# Utilities
# -------------------------------
def find_default_csv():
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
def load_csv(path):
    return pd.read_csv(path)

# -------------------------------
# Load data
# -------------------------------
path = find_default_csv()
uploaded = st.file_uploader("Upload dataset (CSV)", type="csv")

if uploaded is not None:
    df = load_csv(uploaded)
elif path is not None:
    df = load_csv(path)
else:
    st.error("No dataset found. Upload a CSV to continue.")
    st.stop()

st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# -------------------------------
# Select target
# -------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

target_col = st.selectbox(
    "Select target variable (sustainability score)",
    numeric_cols,
)

if target_col is None:
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# Train model (lightweight)
# -------------------------------
@st.cache_resource(show_spinner=False)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model, X_test

best_model, X_test = train_model(X, y)

# -------------------------------
# Baseline
# -------------------------------
baseline_df = X_test.iloc[[0]].copy()
baseline_pred = float(best_model.predict(baseline_df)[0])

st.markdown(f"**Baseline predicted score:** `{baseline_pred:.4f}`")

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["Overview", "Scenario Simulator"])

# ===============================
# OVERVIEW TAB
# ===============================
with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(df.head())

# ===============================
# SCENARIO SIMULATOR TAB
# ===============================
with tabs[1]:
    st.subheader("What-if Scenarios")

    drivers = st.multiselect(
        "Choose scenario drivers",
        X.columns.tolist(),
        default=X.columns[:5].tolist(),
    )

    if not drivers:
        st.info("Select at least one driver.")
        st.stop()

    scenario_step = st.slider(
        "Scenario step size (relative change)",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
    )

    # ---------------------------
    # Sliders
    # ---------------------------
    changes = {}
    cols = st.columns(2)

    for i, d in enumerate(drivers):
        with cols[i % 2]:
            changes[d] = st.slider(
                f"{d} (Δ)",
                min_value=-5 * scenario_step,
                max_value=5 * scenario_step,
                value=0.0,
                step=scenario_step,
            )

    # ---------------------------
    # Scenario execution
    # ---------------------------
    scen = baseline_df.copy()
    for k, delta in changes.items():
        if k in scen.columns:
            scen.loc[scen.index[0], k] += delta

    scenario_pred = float(best_model.predict(scen)[0])
    delta_pred = scenario_pred - baseline_pred

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario predicted score", f"{scenario_pred:.4f}")
    c2.metric("Δ vs baseline", f"{delta_pred:+.4f}")
    c3.metric("Drivers adjusted", f"{len(drivers)}")

    # ---------------------------
    # Sensitivity tornado
    # ---------------------------
    st.markdown("### Sensitivity tornado (one-at-a-time)")

    rows = []
    for d in drivers:
        test = baseline_df.copy()
        test.loc[test.index[0], d] += scenario_step
        pred = float(best_model.predict(test)[0])
        rows.append(
            {"driver": d, "delta": pred - baseline_pred}
        )

    sens = pd.DataFrame(rows).sort_values("delta")

    if sens["delta"].abs().sum() == 0:
        st.info("All impacts are ~0 at this step size. Increase step size to see effects.")
    else:
        if PLOTLY_OK:
            fig = px.bar(
                sens,
                x="delta",
                y="driver",
                orientation="h",
                title="Driver impact on predicted score",
            )
            fig.add_vline(x=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(sens.set_index("driver"))

    # ---------------------------
    # Download
    # ---------------------------
    st.markdown("### Download scenario outputs")

    out = pd.DataFrame(
        {
            "baseline_pred": [baseline_pred],
            "scenario_pred": [scenario_pred],
            "delta_pred": [delta_pred],
        }
    )

    st.download_button(
        "Download results (CSV)",
        out.to_csv(index=False),
        file_name="scenario_results.csv",
        mime="text/csv",
    )
