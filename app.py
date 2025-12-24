from pathlib import Path

import json

import pandas as pd
import streamlit as st

PLOTS_DIR = Path("plots")
METRICS_PATH = Path("outputs/metrics.json")
EDA_PATH = Path("reports/eda_summary.md")
IMAGE_WIDTH = 450

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
    "dataset_summary": {"en": "### Dataset at a glance"},
    "dataset_transactions": {"en": "Transactions"},
    "dataset_frauds": {"en": "Fraud cases"},
    "dataset_features": {"en": "Dataset features"},
    "dataset_avg_amount": {"en": "Avg amount ($)"},
    "dataset_insights": {"en": "#### Dataset insights"},
    "class_balance": {"en": "Class count"},
    "top_correlations": {"en": "Top features correlated with fraud"},
    "dataset_preview": {"en": "Dataset preview"},
    "dataset_missing": {
        "en": "Add the full `creditcard.csv` dataset to unlock the richer dataset snapshot and insights."
    },
    "best_model_highlight": {"en": "### Best model overview"},
    "best_model_label": {"en": "Top performer"},
    "best_model_auprc": {"en": "AUPRC"},
    "best_model_precision": {"en": "Precision (positive)"},
    "best_model_recall": {"en": "Recall (positive)"},
    "best_model_caption": {
        "en": "This is the model with the highest average precision recorded in `metrics.json`."
    },
}


def t(key: str, lang_code: str) -> str:
    return UI_TEXT.get(key, {}).get(lang_code, UI_TEXT.get(key, {}).get("en", key))

DATA_PATH = Path("creditcard.csv")
DATA_PREVIEW_ROWS = 120
DATA_RANDOM_STATE = 42


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


@st.cache_data
def load_full_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_dataset(df: pd.DataFrame) -> dict:
    total = len(df)
    fraud_cases = int(df["Class"].sum()) if "Class" in df.columns else 0
    amount_series = df["Amount"] if "Amount" in df.columns else pd.Series(dtype=float)
    top_corr = (
        df.corr()["Class"]
        .abs()
        .sort_values(ascending=False)
        .drop("Class", errors="ignore")
        .head(5)
        if "Class" in df.columns
        else None
    )
    class_counts = df["Class"].value_counts().sort_index() if "Class" in df.columns else None
    return {
        "total": total,
        "fraud": fraud_cases,
        "fraud_ratio": fraud_cases / total if total else 0.0,
        "features": df.shape[1],
        "avg_amount": float(amount_series.mean()) if not amount_series.empty else 0.0,
        "median_amount": float(amount_series.median()) if not amount_series.empty else 0.0,
        "top_corr": top_corr,
        "class_counts": class_counts,
    }


def dataset_preview(df: pd.DataFrame, rows: int = DATA_PREVIEW_ROWS) -> pd.DataFrame:
    sample_size = min(rows, max(len(df), 0))
    if sample_size <= 0:
        return df.head(0)
    return df.sample(n=sample_size, random_state=DATA_RANDOM_STATE).reset_index(drop=True)


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Explorer")

lang_choice = st.sidebar.selectbox(
    f"{t('language_label', 'en')} / {t('language_label', 'ar')}",
    list(LANG_CHOICES.keys()),
    index=0,
)
lang_code = LANG_CHOICES[lang_choice]

st.sidebar.caption(t("dataset_warning", lang_code))

metrics_df = load_metrics()
plot_paths = list_plots()
eda_text = EDA_PATH.read_text() if EDA_PATH.exists() else t("no_metrics", lang_code)

dataset_df = None
dataset_stats = {}
if DATA_PATH.exists():
    with st.spinner("Loading dataset snapshot..."):
        dataset_df = load_full_dataset(DATA_PATH)
        dataset_stats = summarize_dataset(dataset_df)

if dataset_df is not None:
    st.markdown(t("dataset_summary", lang_code))
    summary_cols = st.columns(4)
    summary_cols[0].metric(
        t("dataset_transactions", lang_code),
        f"{dataset_stats['total']:,}",
    )
    summary_cols[1].metric(
        t("dataset_frauds", lang_code),
        f"{dataset_stats['fraud']:,}",
        f"{dataset_stats['fraud_ratio']:.2%} of rows",
    )
    summary_cols[2].metric(
        t("dataset_features", lang_code),
        f"{dataset_stats['features']}",
    )
    summary_cols[3].metric(
        t("dataset_avg_amount", lang_code),
        f"${dataset_stats['avg_amount']:,.2f}",
        f"Median ${dataset_stats['median_amount']:,.2f}",
    )
    st.markdown(t("dataset_insights", lang_code))
    class_counts = dataset_stats.get("class_counts")
    if class_counts is not None and not class_counts.empty:
        st.bar_chart(class_counts)
        st.caption(t("class_balance", lang_code))
    top_corr = dataset_stats.get("top_corr")
    if top_corr is not None and not top_corr.empty:
        st.markdown(t("top_correlations", lang_code))
        corr_df = (
            pd.DataFrame(
                {
                    "feature": top_corr.index,
                    "abs_correlation": top_corr.values,
                }
            )
            .assign(
                abs_correlation=lambda frame: frame["abs_correlation"].map(
                    lambda value: f"{value:.3f}"
                )
            )
        )
        st.table(corr_df)
    with st.expander(t("dataset_preview", lang_code)):
        st.dataframe(dataset_preview(dataset_df), use_container_width=True)
else:
    st.info(t("dataset_missing", lang_code))

st.markdown(t("dataset_overview", lang_code))
st.markdown(eda_text)

if not metrics_df.empty:
    best_row = metrics_df.iloc[0]
    precision_text = (
        "n/a"
        if pd.isna(best_row["precision_pos"])
        else f"{best_row['precision_pos']:.3f}"
    )
    recall_text = (
        "n/a"
        if pd.isna(best_row["recall_pos"])
        else f"{best_row['recall_pos']:.3f}"
    )
    st.markdown(t("best_model_highlight", lang_code))
    highlight_cols = st.columns(4)
    highlight_cols[0].metric(
        t("best_model_label", lang_code),
        best_row["model"].replace("_", " ").title(),
    )
    highlight_cols[1].metric(
        t("best_model_auprc", lang_code),
        f"{best_row['auprc']:.3f}",
    )
    highlight_cols[2].metric(
        t("best_model_precision", lang_code),
        precision_text,
    )
    highlight_cols[3].metric(
        t("best_model_recall", lang_code),
        recall_text,
    )
    st.caption(t("best_model_caption", lang_code))

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
        col.image(str(path), caption=path.stem, width=IMAGE_WIDTH)

    shap_file = shap_path()
    if shap_file is not None:
        st.markdown(t("shap_summary", lang_code))
        st.image(str(shap_file), caption=shap_file.stem, width=IMAGE_WIDTH)

st.markdown(t("visualizations", lang_code))
if plot_paths:
    for path in plot_paths:
        st.image(
            str(path),
            caption=path.stem.replace("_", " ").title(),
            width=IMAGE_WIDTH,
        )
else:
    st.info("Generate plots by running the pipeline.")
