"""Cluster Iris flowers, visualize PCA, and compare clusters with true species."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results"


def main():
    iris = load_iris()
    features = iris.data
    true_species = iris.target
    species_names = iris.target_names

    # Scaling gives each of Iris's four measurements comparable influence.
    model = make_pipeline(
        StandardScaler(), KMeans(n_clusters=3, random_state=42, n_init=10)
    )
    clusters = model.fit_predict(features)
    assert len(np.unique(clusters)) == 3

    standardized = model.named_steps["standardscaler"].transform(features)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(standardized)
    assert reduced.shape == (len(features), 2)
    ratios = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {ratios}")
    print(f"PCA cumulative explained variance: {ratios.sum():.4f}")
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
    inertias = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(standardized).inertia_
        for k in k_values
    ]
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
    dump(model, model_path)
    reloaded = load(model_path)
    assert np.array_equal(clusters, reloaded.predict(features))
    print(f"Saved and reloaded model; predictions match: {model_path}")
    print(f"Plots saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
