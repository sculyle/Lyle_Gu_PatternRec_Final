"""
K-Means clustering demo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

np.random.seed(42)

# ---------- Demo dataset: 4 well-separated Gaussian blobs ----------
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.9,
                       center_box=(-8, 8), random_state=42)

# ============================================================
# FIGURE 0: Raw data (no clustering applied)
# Shows only the input points before K-Means is run.
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(X[:, 0], X[:, 1], c='#4A4E69', s=25, alpha=0.7,
           edgecolor='white', linewidth=0.4)
ax.set_title("Raw Input Data\n(500 points, no clustering applied)",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Feature 1", fontsize=12)
ax.set_ylabel("Feature 2", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_raw_data.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 1: K-Means iteration process (showing convergence)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("K-Means Iterative Convergence (K=4, Random initialization)",
             fontsize=15, fontweight='bold')

# Manually run K-Means step by step
def kmeans_step(X, centroids):
    # Assignment
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    # Update
    new_centroids = np.array([X[labels == k].mean(axis=0) if (labels == k).sum() > 0
                              else centroids[k] for k in range(len(centroids))])
    return labels, new_centroids

# K-Means++ initialization
#km_init = KMeans(n_clusters=4, init='k-means++', n_init=1, max_iter=1, random_state=42)
#km_init.fit(X)
#centroids = km_init.cluster_centers_.copy()

# Random initialization (deliberately less ideal to make the iteration visible)
km_init = KMeans(n_clusters=4, init='random', n_init=1, max_iter=1, random_state=3)
km_init.fit(X)
centroids = km_init.cluster_centers_.copy()

iterations_to_show = [0, 1, 2, 4, 8, 15]
all_centroids = [centroids.copy()]
current_centroids = centroids.copy()
for i in range(20):
    labels, current_centroids = kmeans_step(X, current_centroids)
    all_centroids.append(current_centroids.copy())

colors = ['#E63946', '#06A77D', '#1D3557', '#F4A261']

for idx, it in enumerate(iterations_to_show):
    ax = axes[idx // 3, idx % 3]
    if it == 0:
        # Show initial centroids before any assignment
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=15, alpha=0.6)
    else:
        labels_it, _ = kmeans_step(X, all_centroids[it - 1])
        for k in range(4):
            ax.scatter(X[labels_it == k, 0], X[labels_it == k, 1],
                       c=colors[k], s=15, alpha=0.6)
    ax.scatter(all_centroids[it][:, 0], all_centroids[it][:, 1],
               c='black', s=250, marker='X', edgecolor='white', linewidth=2,
               zorder=10, label='Centroids')
    ax.set_title(f"Iteration {it}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('kmeans_iterations.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Effect of choosing K (initial state + final result)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Effect of Choosing K", fontsize=15, fontweight='bold')

K_values = [2, 3, 4, 8]
titles = ["K=2 (Too Small: Underfitting)", "K=3 (Still Underfitting)",
          "K=4 (Correct)", "K=8 (Too Large: Overfitting)"]
palette = plt.cm.tab10.colors

for col, (k, title) in enumerate(zip(K_values, titles)):
    # Capture initial centroids from K-Means++ (max_iter=1 so no real updates)
    km_init = KMeans(n_clusters=k, init='k-means++', n_init=1,
                     max_iter=1, random_state=42)
    km_init.fit(X)
    init_centers = km_init.cluster_centers_.copy()

    # Top row: initial state (gray points + initial centroids)
    ax = axes[0, col]
    ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=15, alpha=0.6)
    ax.scatter(init_centers[:, 0], init_centers[:, 1],
               c='black', s=200, marker='X', edgecolor='white', linewidth=2,
               zorder=10, label=f'{k} initial centroids')
    ax.set_title(f"{title}\nIteration 0 (initial centroids)",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Bottom row: final converged result
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbl = km.fit_predict(X)
    ax = axes[1, col]
    for i in range(k):
        ax.scatter(X[lbl == i, 0], X[lbl == i, 1],
                   c=[palette[i]], s=15, alpha=0.7)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='black', s=200, marker='X', edgecolor='white', linewidth=2)
    ax.set_title(f"{title}\nFinal result", fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_K_effect.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Elbow Method + Silhouette Score (combined)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

K_range = range(2, 11)
inertias = []
silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbl = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, lbl))

# Elbow plot
ax = axes[0]
ax.plot(list(K_range), inertias, 'o-', color='#E63946', linewidth=2.5, markersize=10)
ax.axvline(4, color='green', linestyle='--', alpha=0.6, label='Elbow at K=4')
ax.set_xlabel("Number of Clusters (K)", fontsize=12)
ax.set_ylabel("Inertia (WCSS)", fontsize=12)
ax.set_title("Elbow Method", fontsize=13, fontweight='bold')
ax.grid(alpha=0.3); ax.legend(fontsize=11)

# Silhouette plot
ax = axes[1]
ax.plot(list(K_range), silhouettes, 's-', color='#1D3557', linewidth=2.5, markersize=10)
best_k = list(K_range)[int(np.argmax(silhouettes))]
ax.axvline(best_k, color='green', linestyle='--', alpha=0.6,
           label=f'Best K={best_k}')
ax.set_xlabel("Number of Clusters (K)", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("Silhouette Analysis", fontsize=13, fontweight='bold')
ax.grid(alpha=0.3); ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('kmeans_elbow_silhouette.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Bad Initialization vs K-Means++ (with iter-0 vs final)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle("Importance of Initialization (K=4)", fontsize=15, fontweight='bold')

# --- Bad random initialization (all centroids clumped together) ---
bad_init = np.array([[6, 6], [6.2, 6.2], [6.4, 6.4], [6.6, 6.6]])

# Top-left: bad init, iteration 0 (raw points + initial centroids)
ax = axes[0, 0]
ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=15, alpha=0.6)
ax.scatter(bad_init[:, 0], bad_init[:, 1],
           c='black', s=250, marker='X', edgecolor='white', linewidth=2,
           zorder=10, label='Initial centroids')
ax.set_title("Poor Init: Iteration 0\n(All 4 centroids clumped together)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=9)

# Bottom-left: bad init, final result
km_bad = KMeans(n_clusters=4, init=bad_init, n_init=1, max_iter=300, random_state=0)
lbl_bad = km_bad.fit_predict(X)
ax = axes[1, 0]
for i in range(4):
    ax.scatter(X[lbl_bad == i, 0], X[lbl_bad == i, 1],
               c=[colors[i]], s=15, alpha=0.7)
ax.scatter(km_bad.cluster_centers_[:, 0], km_bad.cluster_centers_[:, 1],
           c='black', s=250, marker='X', edgecolor='white', linewidth=2)
ax.set_title(f"Poor Init: Final\nLocal Optimum, Inertia = {km_bad.inertia_:.1f}",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.grid(alpha=0.3)

# --- K-Means++ initialization ---
# Get the initial centroids that K-Means++ chose (only 1 iteration to capture them)
km_pp_init = KMeans(n_clusters=4, init='k-means++', n_init=1, max_iter=1, random_state=42)
km_pp_init.fit(X)
pp_init_centers = km_pp_init.cluster_centers_.copy()

# Top-right: k-means++ init, iteration 0
ax = axes[0, 1]
ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=15, alpha=0.6)
ax.scatter(pp_init_centers[:, 0], pp_init_centers[:, 1],
           c='black', s=250, marker='X', edgecolor='white', linewidth=2,
           zorder=10, label='Initial centroids')
ax.set_title("K-Means++ Init: Iteration 0\n(Centroids spread across data)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=9)

# Bottom-right: k-means++ init, final result
km_good = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
lbl_good = km_good.fit_predict(X)
ax = axes[1, 1]
for i in range(4):
    ax.scatter(X[lbl_good == i, 0], X[lbl_good == i, 1],
               c=[colors[i]], s=15, alpha=0.7)
ax.scatter(km_good.cluster_centers_[:, 0], km_good.cluster_centers_[:, 1],
           c='black', s=250, marker='X', edgecolor='white', linewidth=2)
ax.set_title(f"K-Means++ Init: Final\nGlobal Optimum, Inertia = {km_good.inertia_:.1f}",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_init_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: K-Means failure modes (non-spherical, varying density)
# ============================================================
from sklearn.datasets import make_moons, make_circles
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("K-Means Limitations: Assumes Spherical, Equal-Sized Clusters",
             fontsize=14, fontweight='bold')

# Moons
X_moon, _ = make_moons(n_samples=400, noise=0.07, random_state=42)
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_moon)
ax = axes[0]
ax.scatter(X_moon[:, 0], X_moon[:, 1], c=km.labels_, cmap='coolwarm', s=15, alpha=0.7)
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='black', s=200, marker='X', edgecolor='white', linewidth=2)
ax.set_title("Fail: Non-convex (Moons)", fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.grid(alpha=0.3)

# Circles
X_circ, _ = make_circles(n_samples=400, noise=0.05, factor=0.4, random_state=42)
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_circ)
ax = axes[1]
ax.scatter(X_circ[:, 0], X_circ[:, 1], c=km.labels_, cmap='coolwarm', s=15, alpha=0.7)
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='black', s=200, marker='X', edgecolor='white', linewidth=2)
ax.set_title("Fail: Concentric Rings", fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.grid(alpha=0.3)

# Unequal sizes/density
X_uneq, _ = make_blobs(n_samples=[300, 50, 50], centers=[[0, 0], [4, 4], [-4, 4]],
                      cluster_std=[1.5, 0.3, 0.3], random_state=42)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_uneq)
ax = axes[2]
for i in range(3):
    ax.scatter(X_uneq[km.labels_ == i, 0], X_uneq[km.labels_ == i, 1],
               c=[colors[i]], s=15, alpha=0.7)
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c='black', s=200, marker='X', edgecolor='white', linewidth=2)
ax.set_title("Fail: Unequal Cluster Sizes", fontsize=12, fontweight='bold')
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_failures.png', dpi=120, bbox_inches='tight')
plt.close()

print("All figures saved successfully.")
