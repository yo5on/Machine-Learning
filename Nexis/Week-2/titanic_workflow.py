"""Shared Titanic loading, splitting, preprocessing, and plotting helpers."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TITANIC_CSV = ROOT.parent / "Week-1" / "Assignment" / "Titanic-Dataset.csv"
RANDOM_STATE = 42


def load_titanic():
    if not TITANIC_CSV.exists():
        raise FileNotFoundError(f"Titanic dataset not found: {TITANIC_CSV}")
    data = pd.read_csv(TITANIC_CSV)
    if "Survived" not in data:
        raise ValueError("Expected a 'Survived' target column in the Titanic CSV.")
    y = data["Survived"]
    # These are identifiers or mostly unique text, not useful passenger attributes.
    X = data.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


def make_preprocessor(X_train, scale_numeric=False):
    numeric = X_train.select_dtypes(include="number").columns
    categorical = X_train.select_dtypes(exclude="number").columns
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def save_confusion_matrix(y_test, predictions, output_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Did not survive", "Survived"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
