"""End-to-end pipeline for the credit card fraud detection project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = Path("creditcard.csv")
RANDOM_STATE = 42


def mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    return df


def summary_text(df: pd.DataFrame) -> str:
    describe = df.describe().transpose()
    class_balance = df["Class"].value_counts().sort_index()
    lines = [
        "# EDA Summary",
        "",
        f"- Rows: {len(df)}, columns: {len(df.columns)}",
        "",
        "## Class balance",
        "",
        *[f"  - `{int(cls)}`: {count} ({count / len(df):.2%})" for cls, count in class_balance.items()],
        "",
        "## Select statistics (median / mean / std)",
        "",
    ]
    for column in ["Time", "Amount"]:
        if column in describe.index:
            row = describe.loc[column]
            lines.append(f"- `{column}`: mean={row['mean']:.2f}, median={row['50%']:.2f}, std={row['std']:.2f}")

    high_corr = (
        df.corr()["Class"]
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .drop("Class")
    )
    lines.extend(["", "## Top features correlated with fraud", ""])
    lines.extend([f"- `{feat}`: {value:.3f}" for feat, value in high_corr.items()])
    return "\n".join(lines)


def create_visuals(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=df)
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df, x="Amount", hue="Class", bins=80, log_scale=(False, True), element="step", stat="density")
    plt.title("Transaction amount distribution (log density)")
    plt.tight_layout()
    plt.savefig(plots_dir / "amount_density.png")
    plt.close()


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    target = df.pop("Class")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
    return scaled_df, target, scaler


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
    label: str,
) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"{label} (AUPRC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"pr_curve_{label}.png")
    plt.close()

    classification = classification_report(y_test, model.predict(X_test), output_dict=True)
    return {
        "auprc": pr_auc,
        "avg_precision": avg_precision,
        "classification_report": classification,
    }


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=16,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE, n_estimators=80, max_depth=4, learning_rate=0.1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit card fraud detection pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the creditcard.csv dataset.")
    parser.add_argument("--max-samples", type=int, default=100_000, help="Cap the number of rows processed.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"), help="Directory for numeric outputs.")
    parser.add_argument("--plots", type=Path, default=Path("plots"), help="Directory for generated plots.")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Directory for textual summaries.")
    args = parser.parse_args()

    mkdirs(args.outputs, args.plots, args.reports)

    df = load_dataset(args.data)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=RANDOM_STATE)
        print(f"Sampling {len(df)} records (max requested {args.max_samples}).")

    eda_path = args.reports / "eda_summary.md"
    eda_path.write_text(summary_text(df))

    create_visuals(df, args.plots)

    X, y, scaler = preprocess(df)
    joblib.dump(scaler, args.outputs / "scaler.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models()
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, args.outputs / f"{name}.joblib")
        metrics[name] = evaluate_model(model, X_test, y_test, args.plots, name)

    metrics_path = args.outputs / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"EDA summary saved to {eda_path}")
    print(f"Plots saved to {args.plots} (class distribution, amount density, PR curves).")
    print(f"Metrics saved to {metrics_path}")
    print("Model artifacts available in the outputs directory.")


if __name__ == "__main__":
    main()
