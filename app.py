"""Streamlit dashboard for the credit card fraud detection project."""

from pathlib import Path

import json

import pandas as pd
import streamlit as st

PLOTS_DIR = Path("plots")
METRICS_PATH = Path("outputs/metrics.json")
EDA_PATH = Path("reports/eda_summary.md")

LANG_CHOICES = {"English": "en", "العربية": "ar"}
UI_TEXT = {
    "language_label": {"en": "Language", "ar": "اللغة"},
    "dataset_overview": {"en": "### Dataset Overview", "ar": "### نظرة عامة على البيانات"},
    "model_filters": {"en": "Model filters", "ar": "مرشحات النموذج"},
    "show_models": {"en": "Show models", "ar": "عرض النماذج"},
    "min_auprc": {"en": "Min AUPRC threshold", "ar": "الحد الأدنى لـ AUPRC"},
    "model_comparison": {"en": "### Model comparison (AUPRC)", "ar": "### مقارنة النماذج (AUPRC)"},
    "pr_curves": {"en": "#### Precision-Recall curves", "ar": "#### منحنيات الدقة/الاستدعاء"},
    "shap_summary": {"en": "#### SHAP summary (best model)", "ar": "#### ملخص SHAP (أفضل نموذج)"},
    "no_metrics": {
        "en": "No metrics to display yet. Run `python src/project_pipeline.py` to generate metrics.json.",
        "ar": "لا توجد مقاييس للعرض بعد. شغّل `python src/project_pipeline.py` لإنشاء metrics.json."
    },
    "visualizations": {"en": "### Visualizations", "ar": "### التصويرات"},
    "dataset_warning": {
        "en": "Select a language to toggle the UI between English and Arabic.",
        "ar": "اختر اللغة لتبديل واجهة المستخدم بين الإنجليزية والعربية."
    },
}


def t(key: str, lang_code: str) -> str:
    return UI_TEXT.get(key, {}).get(lang_code, UI_TEXT.get(key, {}).get("en", key))


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


def pr_curve_path(model: str) -> Path:
    return PLOTS_DIR / f"pr_curve_{model}.png"


def shap_path() -> Path | None:
    candidates = sorted(PLOTS_DIR.glob("shap_summary_*.png"))
    return candidates[-1] if candidates else None


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Explorer")

lang_choice = st.sidebar.selectbox(
    f"{t('language_label', 'en')} / {t('language_label', 'ar')}",
    list(LANG_CHOICES.keys()),
    index=0,
)
lang_code = LANG_CHOICES[lang_choice]

st.sidebar.caption(t("dataset_warning", lang_code))

eda_text = EDA_PATH.read_text() if EDA_PATH.exists() else t("no_metrics", lang_code)
st.markdown(t("dataset_overview", lang_code))
st.markdown(eda_text)

metrics_df = load_metrics()
plot_paths = list_plots()

if metrics_df.empty:
    st.warning(t("no_metrics", lang_code))
else:
    st.sidebar.header(t("model_filters", lang_code))
    model_options = metrics_df["model"].tolist()
    selected_models = st.sidebar.multiselect(
        t("show_models", lang_code), model_options, default=model_options
    )
    min_auprc = st.sidebar.slider(
        t("min_auprc", lang_code), 0.0, 1.0, 0.5, 0.01
    )
    filtered = metrics_df[
        (metrics_df["model"].isin(selected_models)) & (metrics_df["auprc"] >= min_auprc)
    ]
    st.markdown(t("model_comparison", lang_code))
    st.dataframe(filtered.style.format({"auprc": "{:.3f}", "avg_precision": "{:.3f}"}))

    st.markdown(t("pr_curves", lang_code))
    pr_images = [
        pr_curve_path(name)
        for name in filtered["model"]
        if pr_curve_path(name).exists()
    ]
    cols = st.columns(max(1, len(pr_images)))
    for col, path in zip(cols, pr_images):
        col.image(str(path), caption=path.stem, use_column_width=True)

    shap_file = shap_path()
    if shap_file is not None:
        st.markdown(t("shap_summary", lang_code))
        st.image(str(shap_file), caption=shap_file.stem, use_column_width=True)

st.markdown(t("visualizations", lang_code))
if plot_paths:
    for path in plot_paths:
        st.image(str(path), caption=path.stem.replace("_", " ").title(), use_column_width=True)
else:
    st.info("Generate plots by running the pipeline.")
