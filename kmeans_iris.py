"""
K-Means clustering on the Iris dataset (Project 1 data).
Features used: Petal Length (meas_3) and Petal Width (meas_4),
chosen because between-class variance >> within-class variance
on these features, making them strong discriminators.

Reads Proj1DataSet.xlsx from the SAME folder as this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ============================================================
# Load data from the same folder as this script
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Proj1DataSet.xlsx")
df = pd.read_excel(data_path, sheet_name="Sheet1")

# meas_3 = Petal Length, meas_4 = Petal Width (verified by value ranges)
X = df[["meas_3", "meas_4"]].values
species = df["species"].values

print(f"Loaded {len(X)} samples")
print(f"Species: {pd.Series(species).value_counts().to_dict()}")
print(f"Petal Length range: [{X[:,0].min():.1f}, {X[:,0].max():.1f}]")
print(f"Petal Width  range: [{X[:,1].min():.1f}, {X[:,1].max():.1f}]")

# ============================================================
# Manual K-Means iteration (so we can capture every step)
# ============================================================
def kmeans_step(X, centroids):
    # Assignment step
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    # Update step
    new_centroids = np.array([
        X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else centroids[k]
        for k in range(len(centroids))
    ])
    return labels, new_centroids

# ---- Suboptimal but plausible initialization ----
# Place 3 centroids in the middle/bottom region so the iterative migration
# is clearly visible across subplots. Still converges to the correct 3 clusters.
centroids_init = np.array([
    [2.0, 0.5],
    [3.5, 1.0],
    [5.0, 1.0],
])

# Run iterations and store every step
all_centroids = [centroids_init.copy()]
all_labels    = [None]   # no labels yet at iteration 0
current = centroids_init.copy()
for i in range(15):
    labels, current = kmeans_step(X, current)
    all_centroids.append(current.copy())
    all_labels.append(labels.copy())

# ============================================================
# FIGURE: Initial state + iterative convergence (2x3 grid)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("K-Means on Iris Dataset (K=3, Petal Length vs Petal Width)",
             fontsize=15, fontweight='bold')

iterations_to_show = [0, 1, 2, 3, 6, 12]
colors = ['#E63946', '#06A77D', '#1D3557']

for idx, it in enumerate(iterations_to_show):
    ax = axes[idx // 3, idx % 3]

    if it == 0:
        # Iteration 0: raw points (uncolored) + initial centroids only
        ax.scatter(X[:, 0], X[:, 1], c='#4A4E69', s=40, alpha=0.6,
                   edgecolor='white', linewidth=0.5, label='Unclustered points')
        ax.scatter(all_centroids[0][:, 0], all_centroids[0][:, 1],
                   c='black', s=300, marker='X', edgecolor='white', linewidth=2,
                   zorder=10, label='Initial centroids')
        ax.set_title("Iteration 0\n(Initial centroids, no assignment yet)",
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
    else:
        # Use the LABELS from iteration `it` (computed using centroids[it-1])
        # but plot the centroids at position `it` (after the update).
        labels_it = all_labels[it]
        for k in range(3):
            mask = labels_it == k
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=40, alpha=0.7,
                       edgecolor='white', linewidth=0.4)
        ax.scatter(all_centroids[it][:, 0], all_centroids[it][:, 1],
                   c='black', s=300, marker='X', edgecolor='white', linewidth=2,
                   zorder=10, label='Centroids')
        ax.set_title(f"Iteration {it}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)

    ax.set_xlabel("Petal Length (cm)", fontsize=11)
    ax.set_ylabel("Petal Width (cm)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 7.5); ax.set_ylim(-0.1, 2.8)

plt.tight_layout()
out_path = os.path.join(script_dir, "kmeans_iris_convergence.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved figure: {out_path}")

# ============================================================
# Bonus: print final accuracy vs ground-truth species
# ============================================================
final_labels = all_labels[-1]
# Match cluster labels to species via majority vote for reporting
from collections import Counter
print("\nFinal cluster composition (vs ground-truth species):")
for k in range(3):
    members = species[final_labels == k]
    counts = Counter(members)
    print(f"  Cluster {k}: {dict(counts)}")

# Also report: did we converge?
# Compare last two centroid sets
shift = np.linalg.norm(all_centroids[-1] - all_centroids[-2])
print(f"\nCentroid shift between last two iters: {shift:.6f}")
print("Converged." if shift < 1e-6 else f"Still moving slightly (shift={shift:.4e})")
