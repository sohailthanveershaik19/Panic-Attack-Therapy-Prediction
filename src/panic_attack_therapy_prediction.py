from pathlib import Path
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# -------------------- Configuration --------------------

DATASET_PATH = Path("Dataset/Panic_Attack.csv")
TESTDATA_PATH = Path("Dataset/testdata.csv")
MODEL_DIR = Path("models")
TARGET_COLUMN = "Therapy"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing based only on the training feature columns."""
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def show_dataset_overview(df: pd.DataFrame) -> None:
    """Print useful dataset checks and show target counts/correlation when possible."""
    print("\n--- Dataset preview ---")
    print(df.head())
    print("\nShape (rows, columns):", df.shape)
    print("Total cells:", df.size)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDataset information:")
    df.info()

    if TARGET_COLUMN in df.columns:
        print("\nUnique target values:")
        print(df[TARGET_COLUMN].dropna().unique())

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=TARGET_COLUMN)
        plt.title(f"Count of {TARGET_COLUMN}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    # Correlation is meaningful only for numeric columns.
    numeric_df = df.select_dtypes(include=["number", "bool"])
    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(max(8, numeric_df.shape[1]), 6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            fmt=".2f",
        )
        plt.title("Correlation heatmap (numeric columns only)")
        plt.tight_layout()
        plt.show()


def calculate_metrics(
    model_name: str,
    y_true,
    y_pred,
    class_labels,
    save_plot: bool = True,
) -> dict:
    """Print classification metrics and display a confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100

    print(f"\n--- {model_name} results ---")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall   : {recall:.2f}%")
    print(f"F1-score : {f1:.2f}%")
    print(
        "\nClassification report:\n",
        classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            target_names=[str(label) for label in class_labels],
            zero_division=0,
        ),
    )

    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        xticklabels=[str(label) for label in class_labels],
        yticklabels=[str(label) for label in class_labels],
        annot=True,
        fmt="g",
        cmap="Blues",
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    if save_plot:
        plot_path = MODEL_DIR / f"{model_name.lower()}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {plot_path}")
    plt.show()

    return {
        "Model": model_name,
        "Accuracy (%)": accuracy,
        "Precision (%)": precision,
        "Recall (%)": recall,
        "F1-score (%)": f1,
    }


def align_test_features(test_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Align new/test data to the exact feature columns used during training."""
    test_df = test_df.copy()

    # If a target column was accidentally included, don't use it as an input.
    if TARGET_COLUMN in test_df.columns:
        test_df = test_df.drop(columns=[TARGET_COLUMN])

    missing = [column for column in feature_columns if column not in test_df.columns]
    extra = [column for column in test_df.columns if column not in feature_columns]

    if missing:
        raise ValueError(
            "testdata.csv is missing these training feature columns: "
            + ", ".join(missing)
        )

    if extra:
        warnings.warn(
            "Ignoring columns in testdata.csv that were not used for training: "
            + ", ".join(extra)
        )

    return test_df.reindex(columns=feature_columns)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {DATASET_PATH.resolve()}\n"
            "Put Panic_Attack.csv inside a folder named Dataset."
        )

    df = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Rows without a known target cannot be used for supervised training.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    if df.empty:
        raise ValueError("No rows with a non-missing target value were found.")

    show_dataset_overview(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target.")

    class_counts = y.value_counts()
    stratify_y = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    if stratify_y is None:
        warnings.warn(
            "Stratified splitting is not possible because at least one class "
            "has fewer than two examples. Using a regular split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_y,
    )

    class_labels = list(pd.unique(y))
    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Target classes:", class_labels)

    models = {
        "SVMClassifier": SVC(probability=True, random_state=RANDOM_STATE),
    }

    # XGBoost is optional. The program will still run the SVM if it isn't installed.
    try:
        from xgboost import XGBClassifier

        # XGBoost classification needs numeric class IDs for string targets.
        # The label encoder is stored with the pipeline for consistent predictions.
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        y_train_for_xgb = label_encoder.fit_transform(y_train.astype(str))
        y_test_for_xgb = label_encoder.transform(y_test.astype(str))

        xgb_model = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        models["XGBoostClassifier"] = xgb_model
    except ImportError:
        print("\nXGBoost is not installed; skipping XGBoost.")
        print("To enable it, run: pip install xgboost")
        label_encoder = None

    results = []
    trained_pipelines = {}

    for model_name, estimator in models.items():
        print(f"\nTraining {model_name}...")

        # Each model gets its own preprocessor, fitted on training data only.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )

        if model_name == "XGBoostClassifier":
            pipeline.fit(X_train, y_train_for_xgb)
            predicted_ids = pipeline.predict(X_test).astype(int)
            predictions = label_encoder.inverse_transform(predicted_ids)
            actual_for_metrics = y_test.astype(str).to_numpy()
            prediction_for_metrics = predictions.astype(str)
            metric_labels = list(label_encoder.classes_)
        else:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            actual_for_metrics = y_test
            prediction_for_metrics = predictions
            metric_labels = class_labels

        metrics = calculate_metrics(
            model_name,
            actual_for_metrics,
            prediction_for_metrics,
            metric_labels,
        )
        results.append(metrics)
        trained_pipelines[model_name] = pipeline

        # Save the full pipeline, including preprocessing, not just the classifier.
        model_path = MODEL_DIR / f"{model_name.lower()}.joblib"
        saved_object = {
            "pipeline": pipeline,
            "feature_columns": X.columns.tolist(),
            "target_column": TARGET_COLUMN,
            "model_name": model_name,
            "xgb_label_encoder": label_encoder if model_name == "XGBoostClassifier" else None,
        }
        joblib.dump(saved_object, model_path)
        print(f"Saved model and preprocessing to: {model_path}")

    print("\n--- Model comparison ---")
    results_df = pd.DataFrame(results).sort_values(
        by="F1-score (%)", ascending=False
    )
    print(results_df.to_string(index=False))

    # Predict new rows, if the testdata file exists.
    if TESTDATA_PATH.exists():
        new_data = pd.read_csv(TESTDATA_PATH)
        if new_data.empty:
            print("\ntestdata.csv is empty; no new predictions were made.")
            return

        feature_columns = X.columns.tolist()
        new_X = align_test_features(new_data, feature_columns)
        output = new_data.copy()

        # Use the best model by macro F1 score for the final prediction column.
        best_model_name = results_df.iloc[0]["Model"]
        best_pipeline = trained_pipelines[best_model_name]

        if best_model_name == "XGBoostClassifier":
            encoded_predictions = best_pipeline.predict(new_X).astype(int)
            best_encoder = label_encoder
            final_predictions = best_encoder.inverse_transform(encoded_predictions)
        else:
            final_predictions = best_pipeline.predict(new_X)

        output["Predicted as"] = final_predictions
        output_path = MODEL_DIR / "testdata_predictions.csv"
        output.to_csv(output_path, index=False)

        print(f"\nBest model by macro F1-score: {best_model_name}")
        print("\nPredictions:")
        print(output.to_string(index=False))
        print(f"\nPredictions saved to: {output_path}")
    else:
        print(
            f"\nNo new-data file found at {TESTDATA_PATH}. "
            "Training and evaluation completed without new-row predictions."
        )


if __name__ == "__main__":
    main()
