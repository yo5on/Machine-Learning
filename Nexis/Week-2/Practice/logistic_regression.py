"""Practice 1 and 2: Titanic Logistic Regression, confusion matrix, and metrics."""

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from titanic_workflow import ROOT, load_titanic, make_preprocessor, save_confusion_matrix


def main():
    X_train, X_test, y_train, y_test = load_titanic()
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix (rows = actual, columns = predicted; 0=did not survive, 1=survived):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(classification_report(y_test, predictions, target_names=["Did not survive", "Survived"], zero_division=0))
    path = ROOT / "results" / "titanic_practice_confusion_matrix.png"
    save_confusion_matrix(y_test, predictions, path, "Titanic Practice: Logistic Regression")
    print(f"Labeled confusion matrix plot: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
