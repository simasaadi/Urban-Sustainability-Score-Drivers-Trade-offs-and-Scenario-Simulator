# ============================
# Urban Sustainability Score
# Drivers, Trade-offs & Scenario Simulator
# ============================

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Urban Sustainability Score — Scenario Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Urban Sustainability Score — Drivers, Trade-offs, and Scenario Simulator")
st.caption("Interactive modeling, explainability, and what-if scenarios (Streamlit).")

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    candidates = [
        "data/urban_planning_dataset.csv",
        "data/raw/urban_planning_dataset.csv",
        "urban_planning_dataset.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    raise FileNotFoundError("Dataset not found. Upload or place CSV in data/")

df = load_data()

# ----------------------------
# Feature selection
# ----------------------------
target_col = "sustainability_score"

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != target_col]

X = df[numeric_cols]
y = df[target_col]

# ----------------------------
# Train / test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------
# Model training (ONCE)
# ----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe

best_model = train_model(X_train, y_train)

# ----------------------------
# Baseline city profile
# ----------------------------
@st.cache_data
def compute_baseline(X):
    baseline = X.median(numeric_only=True)
    return pd.DataFrame([baseline], columns=X.columns)

baseline_df = compute_baseline(X)
baseline_pred = float(best_model.predict(baseline_df)[0])

st.markdown("### What-if Scenarios (based on trained model)")
st.write(f"**Baseline predicted score:** `{baseline_pred:.4f}` (Model: Ridge)")

# ----------------------------
# Scenario controls
# ----------------------------
st.markdown("### Choose scenario drivers")

drivers = st.multiselect(
    "Numeric drivers to adjust (max 5)",
    numeric_cols,
    default=numeric_cols[:3],
    max_selections=5,
)

st.markdown("### Set changes (relative deltas)")

changes = {}
cols = st.columns(2)

for i, d in enumerate(drivers):
    with cols[i % 2]:
        changes[d] = st.slider(
            f"{d} (Δ)",
            min_value=-0.25,
            max_value=0.25,
            value=0.0,
            step=0.01,
        )

# ----------------------------
# Run scenario
# ----------------------------
def run_scenario(baseline_df, deltas):
    scen = baseline_df.copy()
    for k, delta in deltas.items():
        scen.loc[0, k] = float(scen.loc[0, k]) + float(delta)
    return scen

scenario_df = run_scenario(baseline_df, changes)
scenario_pred = float(best_model.predict(scenario_df)[0])
delta_pred = scenario_pred - baseline_pred

# ----------------------------
# Metrics
# ----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Scenario predicted score", f"{scenario_pred:.4f}")
c2.metric("Δ vs baseline", f"{delta_pred:+.4f}")
c3.metric("Drivers adjusted", f"{len(drivers)}")

# ----------------------------
# Sensitivity tornado (one-at-a-time)
# ----------------------------
st.markdown("### Sensitivity tornado (one-at-a-time)")

rows = []

for k in drivers:
    temp = baseline_df.copy()
    temp.loc[0, k] = float(temp.loc[0, k]) + float(changes[k])
    pred = float(best_model.predict(temp)[0])
    rows.append({"driver": k, "delta": pred - baseline_pred})

sens = pd.DataFrame(rows).sort_values("delta")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(sens["driver"], sens["delta"])
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Δ predicted score")
ax.set_title("Driver impact on predicted score (one-at-a-time)")
plt.tight_layout()

st.pyplot(fig)
plt.close("all")

# ----------------------------
# Download results
# ----------------------------
st.markdown("### Download scenario outputs")

out = pd.DataFrame({
    "driver": list(changes.keys()),
    "delta_applied": list(changes.values()),
})

out["baseline_pred"] = baseline_pred
out["scenario_pred"] = scenario_pred
out["delta_vs_baseline"] = delta_pred

st.download_button(
    "Download scenario results (CSV)",
    data=out.to_csv(index=False),
    file_name="scenario_results.csv",
    mime="text/csv",
)
