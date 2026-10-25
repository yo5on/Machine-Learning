"""Practice 4: choose Decision Tree depth with training-only cross-validation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from titanic_workflow import ROOT, load_titanic, make_preprocessor


def main():
    X_train, X_test, y_train, y_test = load_titanic()
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(X_train)),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )
    depths = [1, 2, 3, 4, 5, 7, 10, None]
    search = GridSearchCV(
        model,
        {"classifier__max_depth": depths},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Selected max_depth: {search.best_params_['classifier__max_depth']}")
    print(f"Best mean CV accuracy: {search.best_score_:.4f}")
    from sklearn.metrics import accuracy_score

    print(f"Selected model test accuracy: {accuracy_score(y_test, best.predict(X_test)):.4f}")
    scores = pd.DataFrame(search.cv_results_)[["param_classifier__max_depth", "mean_test_score"]]
    scores["depth_label"] = scores["param_classifier__max_depth"].map(lambda d: "unlimited" if pd.isna(d) else str(d))
    print("\nCross-validation results:")
    print(scores[["depth_label", "mean_test_score"]].to_string(index=False, header=["max_depth", "mean CV accuracy"]))

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "decision_tree_depth_cv.csv", index=False)
    x = np.arange(len(scores))
    plt.figure(figsize=(8, 5))
    plt.plot(x, scores["mean_test_score"], marker="o")
    plt.xticks(x, scores["depth_label"])
    plt.xlabel("Tree max_depth (unlimited means no depth cap)")
    plt.ylabel("Mean training cross-validation accuracy")
    plt.title("Decision Tree Depth Tuning (5-fold CV)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "decision_tree_depth_cv.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
