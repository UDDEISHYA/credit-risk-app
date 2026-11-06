from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import plotly.express as px
import streamlit as st

from common import get_bundle, page_setup
from credit_risk.monitoring import monitor_batch, simulate_drift


page_setup("Monitoring", "🚦")
bundle = get_bundle()
reference = bundle["reference_data"]
scenario = st.radio("Incoming batch", ["Stable reference sample", "Simulated adverse drift"], horizontal=True)
current = reference.sample(frac=0.60, random_state=7).copy()
if scenario == "Simulated adverse drift":
    current = simulate_drift(current)
report = monitor_batch(bundle, current)

color = {"GREEN": "green", "AMBER": "orange", "RED": "red"}[report["overall_status"]]
st.markdown(f"## Overall status: :{color}[{report['overall_status']}]")
st.write(report["recommended_action"])
columns = st.columns(3)
columns[0].metric("Score PSI", f"{report['score_psi']:.3f}")
columns[1].metric("Average incoming PD", f"{report['average_pd']:.2%}")
columns[2].metric("Critical quality failures", 0 if report["data_quality"]["passed"] else 1)
drift = report["feature_drift"]
st.plotly_chart(px.bar(drift.head(12), x="feature", y="psi", color="status", color_discrete_map={"GREEN": "#2ca02c", "AMBER": "#ffbf00", "RED": "#d62728"}, title="Largest feature drift indicators"), use_container_width=True)
st.dataframe(drift, use_container_width=True, hide_index=True)
if report["performance"]:
    st.write(f"Labeled-batch ROC-AUC: **{report['performance']['roc_auc']:.3f}** · Brier: **{report['performance']['brier_score']:.3f}**")
st.caption("The adverse batch is simulated because the source has no dates or true later-period sample.")

