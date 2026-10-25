"""Linear Regression workflow for a supplied house prices CSV."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "house_prices.csv"
OUTPUT = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not args.csv.exists():
        raise SystemExit(
            f"House prices CSV not found: {args.csv}\n"
            "Download Kaggle House Prices train.csv, place it at "
            f"{DEFAULT_DATA}, or pass its path as an argument. Required target: SalePrice."
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    data = pd.read_csv(args.csv)
    if "SalePrice" not in data:
        raise SystemExit("The CSV must contain a 'SalePrice' target column.")
    y = data["SalePrice"]
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    numeric = X.select_dtypes(include="number").columns
    categorical = X.select_dtypes(exclude="number").columns
    preprocess = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline([("preprocess", preprocess), ("regression", LinearRegression())])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse, r2 = mean_squared_error(y_test, predictions), r2_score(y_test, predictions)
    print(f"Test R²: {r2:.4f}")
    print(f"Test MSE: {mse:.2f}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.65, label="Test homes")
    low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
    plt.xlabel("Actual sale price")
    plt.ylabel("Predicted sale price")
    plt.title("House Prices: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "house_prices_actual_vs_predicted.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
