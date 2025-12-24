import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
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
    sns.histplot(df, x="Time", bins=50, hue="Class", stat="density", element="step")
    plt.title("Transaction time density by class")
    plt.tight_layout()
    plt.savefig(plots_dir / "time_density.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df, x="Amount", hue="Class", bins=80, log_scale=(False, True), element="step", stat="density")
    plt.title("Transaction amount distribution (log density)")
    plt.tight_layout()
    plt.savefig(plots_dir / "amount_density.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Class", y="Amount", data=df)
    plt.title("Amount boxplot split by class")
    plt.tight_layout()
    plt.savefig(plots_dir / "amount_boxplot.png")
    plt.close()

    if "Time" in df.columns:
        df_time = df.assign(hour=(df["Time"] // 3600) % 24)
        hourly = df_time.groupby("hour")["Class"].value_counts().unstack(fill_value=0)
        plt.figure(figsize=(8, 4))
        hourly.plot(kind="bar", stacked=True, color=["#2ca02c", "#d62728"], width=0.8)
        plt.title("Transactions per hour (stacked by class)")
        plt.xlabel("Hour of day")
        plt.ylabel("Transaction count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / "hourly_distribution.png")
        plt.close()

    scatter_x, scatter_y = "V17", "V14"
    if scatter_x in df.columns and scatter_y in df.columns:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=scatter_x, y=scatter_y, hue="Class", data=df, alpha=0.6, s=16)
        plt.title(f"{scatter_x} vs {scatter_y} by class")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{scatter_x}_{scatter_y}_scatter.png")
        plt.close()

    corr_columns = df.corr().abs()["Class"].sort_values(ascending=False).head(10).index.tolist()
    corr = df[corr_columns].corr()
    mask = pd.DataFrame(0, index=corr.index, columns=corr.columns)
    mask.values[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", mask=mask)
    plt.title("Top feature correlations")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlations.png")
    plt.close()


def stratified_sample(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if len(df) <= max_samples:
        return df
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=max_samples, random_state=RANDOM_STATE
    )
    train_idx, _ = next(splitter.split(df, df["Class"]))
    return df.iloc[train_idx].reset_index(drop=True)


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    df_clean = df.drop_duplicates().dropna()
    if "Amount" in df_clean.columns:
        df_clean = df_clean[df_clean["Amount"] >= 0]
    after = len(df_clean)
    return df_clean, {"before": before, "after": after}


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


def generate_shap_summary(model, X: pd.DataFrame, plots_dir: Path, label: str) -> None:
    try:
        sample = X.sample(n=min(800, len(X)), random_state=RANDOM_STATE)
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False)
        plt.title(f"SHAP summary ({label})")
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_summary_{label}.png")
        plt.close()
    except Exception as exc:
        print(f"Unable to generate SHAP summary for {label}: {exc}")


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
        "lightgbm": LGBMClassifier(
            random_state=RANDOM_STATE, n_estimators=120, learning_rate=0.07, class_weight="balanced"
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            random_state=RANDOM_STATE,
            early_stopping=True,
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
    df, clean_stats = clean_data(df)
    print(f"Cleaned data: {clean_stats['before']} rows -> {clean_stats['after']} rows before modeling.")
    if args.max_samples and len(df) > args.max_samples:
        df = stratified_sample(df, args.max_samples)
        print(f"Stratified sampling {len(df)} records (max requested {args.max_samples}).")

    eda_path = args.reports / "eda_summary.md"
    eda_path.write_text(summary_text(df))

    create_visuals(df, args.plots)

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    joblib.dump(scaler, args.outputs / "scaler.joblib")

    models = build_models()
    metrics = {}
    best_score = -1.0
    best_label = None
    best_model = None
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, args.outputs / f"{name}.joblib")
        metrics[name] = evaluate_model(model, X_test_scaled, y_test, args.plots, name)
        if metrics[name]["auprc"] > best_score:
            best_score = metrics[name]["auprc"]
            best_model = model
            best_label = name

    metrics_path = args.outputs / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if best_model is not None and best_label:
        generate_shap_summary(best_model, X_train_scaled, args.plots, best_label)

    print(f"EDA summary saved to {eda_path}")
    print(f"Plots saved to {args.plots} (class distribution, amount density, PR curves).")
    print(f"Metrics saved to {metrics_path}")
    print("Model artifacts available in the outputs directory.")


if __name__ == "__main__":
    main()
