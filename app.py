# =========================================================
# Urban Sustainability Score
# Drivers, Trade-offs, and Scenario Simulator
# =========================================================

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import plotly.express as px

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Urban Sustainability Score – Scenario Simulator",
    layout="wide",
)

st.title("Urban Sustainability Score — Drivers, Trade-offs, and Scenario Simulator")
st.caption("Scenario-based exploration using a trained regression model")

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
DATA_PATH = "urban_planning_dataset.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

target = "urban_sustainability_score"
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

# ---------------------------------------------------------
# Train model (lightweight + stable)
# ---------------------------------------------------------
@st.cache_resource
def train_model(X, y):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

# ---------------------------------------------------------
# Baseline city profile (median)
# ---------------------------------------------------------
baseline = X.median().to_frame().T
baseline_pred = float(model.predict(baseline)[0])

st.subheader("Baseline city profile")
st.write(f"**Baseline predicted score:** `{baseline_pred:.4f}`")

# ---------------------------------------------------------
# Scenario simulator
# ---------------------------------------------------------
st.subheader("Choose scenario drivers")

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

drivers = st.multiselect(
    "Numeric drivers to adjust",
    options=numeric_features,
    default=[
        "public_transport_access",
        "green_cover_percentage",
        "carbon_footprint",
        "renewable_energy_usage",
        "disaster_risk_index",
    ],
)

st.subheader("Set changes (relative deltas)")

changes = {}
cols = st.columns(2)

for i, d in enumerate(drivers):
    with cols[i % 2]:
        changes[d] = st.slider(
            f"{d} (Δ)",
            min_value=-0.5,
            max_value=0.5,
            step=0.05,
            value=0.0,
        )

# ---------------------------------------------------------
# Apply scenario
# ---------------------------------------------------------
scenario = baseline.copy()

for k, delta in changes.items():
    scenario.loc[0, k] = scenario.loc[0, k] + delta

scenario_pred = float(model.predict(scenario)[0])
delta_pred = scenario_pred - baseline_pred

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Scenario predicted score", f"{scenario_pred:.4f}")
c2.metric("Δ vs baseline", f"{delta_pred:+.4f}")
c3.metric("Drivers adjusted", len(drivers))

# ---------------------------------------------------------
# Sensitivity tornado (ONE-AT-A-TIME)
# ---------------------------------------------------------
st.subheader("Sensitivity tornado (one-at-a-time)")

rows = []

for k in drivers:
    temp = baseline.copy()
    temp.loc[0, k] += 0.1
    pred = float(model.predict(temp)[0])
    rows.append(
        {
            "driver": k,
            "delta": pred - baseline_pred,
        }
    )

sens = pd.DataFrame(rows).sort_values("delta")

fig = px.bar(
    sens,
    x="delta",
    y="driver",
    orientation="h",
    title="Driver impact on predicted score (one-at-a-time)",
)
fig.add_vline(x=0)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Download scenario output
# ---------------------------------------------------------
st.subheader("Download scenario output")

out = scenario.copy()
out["baseline_pred"] = baseline_pred
out["scenario_pred"] = scenario_pred
out["delta_pred"] = delta_pred

st.download_button(
    "Download scenario as CSV",
    out.to_csv(index=False),
    file_name="scenario_output.csv",
    mime="text/csv",
)
