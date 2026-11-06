from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import get_bundle, page_setup


page_setup("Model comparison", "📐")
bundle = get_bundle()
rows = []
for name, result in bundle["test_metrics"].items():
    for calibration_state in ("uncalibrated", "calibrated"):
        metrics = result[calibration_state]
        rows.append({"model": name, "state": calibration_state, **{key: metrics[key] for key in ("roc_auc", "gini", "pr_auc", "brier_score", "log_loss", "precision", "recall", "ks")}})
table = pd.DataFrame(rows)
st.dataframe(table.style.format(precision=4), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(table, x="model", y="roc_auc", color="state", barmode="group", range_y=[0.5, 1], title="Discrimination"), use_container_width=True)
with right:
    st.plotly_chart(px.bar(table, x="model", y="brier_score", color="state", barmode="group", title="Calibration error (lower is better)"), use_container_width=True)

figure = go.Figure()
figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash"})
for name, result in bundle["test_metrics"].items():
    calibration = pd.DataFrame(result["calibration_table"])
    figure.add_scatter(x=calibration["mean_predicted_pd"], y=calibration["observed_default_rate"], mode="lines+markers", name=name)
figure.update_layout(title="Observed versus predicted default by PD band", xaxis_title="Mean predicted PD", yaxis_title="Observed default rate")
st.plotly_chart(figure, use_container_width=True)

confusion_columns = st.columns(2)
for column, (name, result) in zip(confusion_columns, bundle["test_metrics"].items()):
    matrix = result["calibrated"]["confusion_matrix"]
    heatmap = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted no default", "Predicted default"],
        y=["Observed no default", "Observed default"],
        color_continuous_scale="Blues",
        title=f"{name.replace('_', ' ').title()} confusion matrix",
    )
    column.plotly_chart(heatmap, use_container_width=True)
st.caption(f"Confusion matrices use the documented classification threshold of {bundle['model_config']['classification_threshold']:.2f}; lending decisions use the separate three-way policy.")
st.success(f"Selected champion: {bundle['champion_name'].replace('_', ' ').title()}")

