"""Streamlit dashboard for the credit card fraud detection project."""

from pathlib import Path

import json

import pandas as pd
import streamlit as st

PLOTS_DIR = Path("plots")
METRICS_PATH = Path("outputs/metrics.json")
EDA_PATH = Path("reports/eda_summary.md")


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    raw = json.loads(METRICS_PATH.read_text())
    records = []
    for model, info in raw.items():
        report = info.get("classification_report", {}).get("1", {})
        records.append(
            {
                "model": model,
                "auprc": info.get("auprc"),
                "avg_precision": info.get("avg_precision"),
                "recall_pos": report.get("recall"),
                "precision_pos": report.get("precision"),
                "support_pos": report.get("support"),
            }
        )
    return pd.DataFrame(records).sort_values("auprc", ascending=False)


def list_plots() -> list[Path]:
    return sorted(PLOTS_DIR.glob("*.png"))


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Explorer")

eda_text = EDA_PATH.read_text() if EDA_PATH.exists() else "Run `src/project_pipeline.py` first to create the EDA summary."
st.markdown("### Dataset Overview")
st.markdown(eda_text)

metrics_df = load_metrics()
if metrics_df.empty:
    st.warning("No metrics to display yet. Run `python src/project_pipeline.py` to generate metrics.json.")
else:
    st.markdown("### Model comparison (AUPRC)")
    st.dataframe(metrics_df.style.format({"auprc": "{:.3f}", "avg_precision": "{:.3f}"}))

st.markdown("### Visualizations")
plot_paths = list_plots()
if plot_paths:
    for path in plot_paths:
        st.image(str(path), caption=path.stem.replace("_", " ").title(), use_column_width=True)
else:
    st.info("Generate plots by running the pipeline.")
