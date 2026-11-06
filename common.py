"""Shared helpers for the Streamlit pages (browser build)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_risk.modeling import load_model_bundle  # noqa: E402


@st.cache_resource
def get_bundle():
    path = ROOT / "artifacts/model_bundle.joblib"
    if not path.exists():
        st.error("Model artifact not found.")
        st.stop()
    return load_model_bundle(path)


@st.cache_data
def get_predictions() -> pd.DataFrame:
    path = ROOT / "artifacts/test_predictions.csv"
    if not path.exists():
        st.error("Test predictions not found.")
        st.stop()
    return pd.read_csv(path)


def page_setup(title: str, icon: str = "\N{CREDIT CARD}") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.title(title)


def money(value: float, currency: str = "TWD") -> str:
    return f"{currency} {value:,.0f}"
