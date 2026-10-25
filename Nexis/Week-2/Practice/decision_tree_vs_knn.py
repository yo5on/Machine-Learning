"""Practice 3: compare Decision Tree and scaled KNN on the same Titanic split."""

import sys
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from titanic_workflow import load_titanic, make_preprocessor


def main():
    X_train, X_test, y_train, y_test = load_titanic()
    models = {
        "Decision Tree": Pipeline(
            [
                ("preprocess", make_preprocessor(X_train)),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "KNN (k=5)": Pipeline(
            [
                ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }
    print(f"{'Model':<20} Test accuracy")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name:<20} {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print("Trees learn feature thresholds; KNN predicts from nearby scaled observations.")


if __name__ == "__main__":
    main()
