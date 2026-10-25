"""Run K-Means and PCA on Iris, compare clusters with species, and save results."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results"
DATASET = ROOT / "Iris.csv"


def load_iris_data():
    if not DATASET.exists():
        from sklearn.datasets import load_iris

        iris = load_iris()
        return iris.data, iris.target, iris.target_names

    with DATASET.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Iris CSV has no header: {DATASET}")
        species_column = next(
            (name for name in reader.fieldnames if name.casefold() == "species"), None
        )
        feature_columns = [
            name for name in reader.fieldnames
            if name.casefold() not in {"id", "species"}
        ]
        if species_column is None or len(feature_columns) < 2:
            raise ValueError("Iris.csv needs a Species column and at least two measurements.")
        rows = list(reader)

    labels = [row[species_column].strip() for row in rows]
    species_names = np.array(sorted(set(labels)))
    if not rows or len(species_names) != 3:
        raise ValueError("Iris.csv must contain rows for exactly three species.")
    try:
        features = np.array(
            [[float(row[name]) for name in feature_columns] for row in rows]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Iris.csv measurement columns must contain numeric values.") from error
    if not np.isfinite(features).all():
        raise ValueError("Iris.csv measurements cannot be missing or infinite.")
    species_ids = {name: index for index, name in enumerate(species_names)}
    true_species = np.array([species_ids[label] for label in labels])
    return features, true_species, species_names


def assign_clusters(data, centers):
    distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(data)), labels].sum()


def fit_kmeans(data, k, seed=42, n_init=10):
    best_centers, best_labels, best_inertia = None, None, float("inf")
    for restart in range(n_init):
        rng = np.random.default_rng(seed + restart)
        centers = [data[rng.integers(len(data))]]
        for _ in range(1, k):
            nearest = ((data[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            centers.append(data[rng.choice(len(data), p=nearest / nearest.sum())])
        centers = np.array(centers)

        for _ in range(300):
            labels, _ = assign_clusters(data, centers)
            updated = np.array(
                [data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                 for i in range(k)]
            )
            if np.allclose(centers, updated):
                centers = updated
                break
            centers = updated

        labels, inertia = assign_clusters(data, centers)
        if inertia < best_inertia:
            best_centers, best_labels, best_inertia = centers, labels, inertia
    return best_centers, best_labels, best_inertia


def predict(data, saved_model):
    standardized = (data - saved_model["mean"]) / saved_model["scale"]
    return assign_clusters(standardized, saved_model["centers"])[0]


def main():
    features, true_species, species_names = load_iris_data()

    # Scaling gives each of Iris's four measurements comparable influence.
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0] = 1
    standardized = (features - feature_mean) / feature_scale
    centers, clusters, _ = fit_kmeans(standardized, k=3)
    assert len(np.unique(clusters)) == 3

    # PCA uses the two eigenvectors with the largest covariance eigenvalues.
    covariance = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    ratios = eigenvalues[order] / eigenvalues.sum()
    reduced = standardized @ eigenvectors[:, order[:2]]
    assert reduced.shape == (len(features), 2)
    print(f"PCA explained variance ratio: {ratios[:2]}")
    print(f"PCA cumulative explained variance: {ratios[:2].sum():.4f}")
    print("PCA compresses the four scaled measurements into two informative axes for plotting.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for cluster_id, color in enumerate(colors):
        selected = clusters == cluster_id
        axes[0].scatter(
            reduced[selected, 0], reduced[selected, 1],
            color=color, alpha=0.75, label=f"Cluster {cluster_id}",
        )
    axes[0].set_title("K-Means clusters (k=3)")
    for species_id, color in enumerate(colors):
        selected = true_species == species_id
        axes[1].scatter(
            reduced[selected, 0], reduced[selected, 1],
            color=color, alpha=0.75, label=species_names[species_id],
        )
    axes[1].set_title("Actual Iris species")
    for axis in axes:
        axis.set_xlabel("First principal component")
        axis.legend()
    axes[0].set_ylabel("Second principal component")
    fig.suptitle("Iris: K-Means clusters and true labels")
    fig.tight_layout()
    fig.savefig(OUTPUT / "iris_clusters_pca.png", dpi=150)
    plt.close(fig)

    # Cluster IDs have no inherent species meaning; compare counts without remapping.
    comparison = np.zeros((len(species_names), 3), dtype=int)
    for species_id, cluster_id in zip(true_species, clusters):
        comparison[species_id, cluster_id] += 1
    print("\nCounts by actual species (rows) and predicted cluster (columns):")
    print("Species         " + "  ".join(f"Cluster {i}" for i in range(3)))
    for name, counts in zip(species_names, comparison):
        print(f"{name:<15} " + "  ".join(f"{count:>9}" for count in counts))
    print("Cluster IDs are arbitrary and do not directly match species labels.")

    k_values = range(1, 11)
    inertias = [fit_kmeans(standardized, k)[2] for k in k_values]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(list(k_values), inertias, marker="o")
    axis.set_xticks(list(k_values))
    axis.set_xlabel("Number of clusters (K)")
    axis.set_ylabel("Inertia (within-cluster sum of squares)")
    axis.set_title("Iris K-Means Elbow Method")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "iris_elbow.png", dpi=150)
    plt.close(fig)

    model_path = OUTPUT / "iris_kmeans.joblib"
    saved_model = {"mean": feature_mean, "scale": feature_scale, "centers": centers}
    dump(saved_model, model_path)
    reloaded = load(model_path)
    assert np.array_equal(clusters, predict(features, reloaded))
    print(f"Saved and reloaded model; predictions match: {model_path}")
    print(f"Plots saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
