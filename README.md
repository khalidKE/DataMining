# Credit Card Fraud Detection

This project analyzes the anonymized `creditcard.csv` dataset (284,807 transactions with 492 frauds) to establish a reproducible pipeline that covers EDA, preprocessing, and model evaluation with focused metrics such as the Area Under the Precision-Recall Curve (AUPRC).

## Repository layout

- `creditcard.csv` – raw dataset downloaded from the ULB/Worldline research effort.
- `requirements.txt` – dependencies needed to run the analysis end to end.
- `src/project_pipeline.py` – single entry point that runs EDA, preprocessing, training, and evaluation.
- `reports/eda_summary.md` – generated summary computed from the most recent pipeline execution.
- `outputs/` – trained model artifacts (`*.joblib`), scaler, and `metrics.json`.
- `plots/` – saved PNGs for class distribution, transaction amount density, and per-model precision-recall curves.

## Setup

1. Create and activate a Python 3.12 environment (or equivalent).
2. `pip install -r requirements.txt`

## Running the pipeline

```bash
python src/project_pipeline.py
```

By default the script caps the row count at 100,000 (`--max-samples`) to keep the runtime manageable while still preserving the imbalance structure; set this flag higher (or to `0` for no cap) if you have more time. After execution you will find:

- `reports/eda_summary.md` with class balance, select statistics, and the most correlated features.
- `plots/` with `class_distribution.png`, `time_density.png`, `amount_density.png`, `amount_boxplot.png`, `V17_V14_scatter.png`, `feature_correlations.png`, plus the PR curves for each model.
- `outputs/metrics.json` (AUPRC, average precision, classification report) plus the saved scaler and models.

Since the dataset is heavily imbalanced, all classifiers are evaluated using precision-recall curves rather than accuracy.

## Next steps (suggested)

1. Integrate sampling or threshold tuning from `imblearn` to see how synthetic minority oversampling affects recall.
2. Wrap the pipeline in a lightweight UI/notebook to explore high-risk clusters interactively.
3. Add monitoring artifacts such as SHAP explanations or a dashboard that tracks precision at fixed recall.
