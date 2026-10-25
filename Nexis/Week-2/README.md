# Nexis Week 2: supervised learning

The assignment PDF asks for linear regression on house prices and logistic regression on Titanic survival. The practice PDF adds logistic regression, confusion matrix/accuracy, Decision Tree versus KNN, and Decision Tree depth tuning. Titanic scripts use the original Week 1 dataset; the supplied Kaggle house prices CSV was not present, so no regression metrics are claimed.

## Files

```text
Nexis/Week-2/
  Assignment/house_price_prediction.py
  Assignment/titanic_survival_prediction.py
  Practice/logistic_regression.py
  Practice/decision_tree_vs_knn.py
  Practice/decision_tree_depth_tuning.py
  results/                         generated plots and CV table
  titanic_workflow.py              shared Titanic preprocessing
  requirements.txt
```

## Setup

From the repository root, create and activate a virtual environment, then install the three direct dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Nexis/Week-2/requirements.txt
```

The Titanic data is read from `Nexis/Week-1/Assignment/Titanic-Dataset.csv`; no copy or Week 1 edit is required. The scripts use a fixed 80/20 stratified split (seed 42). Imputation, one-hot encoding, and scaling are fitted inside training pipelines to prevent test data leakage. `PassengerId`, `Name`, `Ticket`, and `Cabin` are excluded as identifier/high-cardinality or mostly missing columns.

## Run from the repository root

```powershell
python Nexis/Week-2/Assignment/titanic_survival_prediction.py
python Nexis/Week-2/Practice/logistic_regression.py
python Nexis/Week-2/Practice/decision_tree_vs_knn.py
python Nexis/Week-2/Practice/decision_tree_depth_tuning.py
```

The Titanic Logistic Regression reports accuracy, precision, recall, and F1, and writes labeled confusion matrix figures to `Nexis/Week-2/results/`. The tree/KNN practice compares test accuracy on the same split; KNN numeric features are standardized. Depth tuning selects among depths 1, 2, 3, 4, 5, 7, 10, and unlimited using five-fold cross-validation on training data only, then scores the selected model once on the held-out test data. It writes `decision_tree_depth_cv.png` and `decision_tree_depth_cv.csv`. Shallow trees can underfit; unrestricted trees can overfit; CV helps choose a balance.

## House price dataset

The official Kaggle House Prices training CSV (`train.csv`) uses `SalePrice` as its target and is compatible. Put it at `Nexis/Week-2/data/house_prices.csv`, or pass another path:

```powershell
python Nexis/Week-2/Assignment/house_price_prediction.py Nexis/Week-2/data/house_prices.csv
```

The script drops `Id`, imputes numeric and categorical missing values, one-hot encodes categories, makes a reproducible 80/20 split, fits `LinearRegression`, and reports test R² and mean squared error. It saves `Nexis/Week-2/results/house_prices_actual_vs_predicted.png`. A CSV without `SalePrice` is rejected. The supplied repository/archive included no housing dataset, so this workflow could not be trained or scored on real house-price observations yet.

## Metrics and results

- **Accuracy**: fraction of test classifications that are correct.
- **Confusion matrix**: counts of actual versus predicted class labels.
- **Precision / recall / F1**: positive-class correctness / positive cases found / their harmonic mean; the script also reports per-class values.
- **R²**: variance in test prices explained by the model (higher is better; can be negative).
- **MSE**: average squared test prediction error in squared price units (lower is better).

The available Titanic dataset has 891 rows and the expected `Survived` target. Python syntax compilation passed. Runtime training could not be verified in this environment: Windows Application Control blocked Pandas' native `strptime` extension DLL even from a project-local virtual environment. No accuracy, precision, recall, F1, confusion-matrix counts, or model comparison scores are claimed here. On a machine where the installed Python packages are permitted to load, the commands above print the actual scores and create the documented outputs. House price results also remain unavailable until a real CSV is supplied.
