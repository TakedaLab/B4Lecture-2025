"""
GMM clustering with EM algorithm and visualization tools.

This script loads 1D and 2D datasets, applies GMM clustering using the EM algorithm,
and visualizes the clustering result with gradient coloring and log-likelihood plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

# フォントサイズを全体で統一
plt.rcParams.update({"font.size": 12})


# ------------------------
# GMM（EMアルゴリズム）関数
# ------------------------
def gmm_em(X, K, max_iter=100, tol=1e-4, seed=0):
    """
    Run the EM algorithm for Gaussian Mixture Model clustering.

    Parameters:
        X (ndarray): Input data of shape (N, D).
        K (int): Number of clusters.
        max_iter (int): Maximum number of EM iterations.
        tol (float): Convergence threshold.
        seed (int): Random seed for reproducibility.

    Returns:
        mu (ndarray): Cluster centroids.
        sigma (ndarray): Covariance matrices.
        pi (ndarray): Mixing coefficients.
        gamma (ndarray): Burden rates.
        log_likelihoods (list): Log-likelihood at each iteration.
    """
    np.random.seed(seed)
    N, D = X.shape

    mu = X[np.random.choice(N, K, replace=False)]
    sigma = np.array([np.eye(D) for _ in range(K)])
    pi = np.ones(K) / K
    log_likelihoods = []

    for iteration in range(max_iter):
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        Nk = np.sum(gamma, axis=0)
        pi = Nk / N
        mu = (gamma.T @ X) / Nk[:, np.newaxis]
        for k in range(K):
            X_centered = X - mu[k]
            sigma[k] = (gamma[:, k][:, np.newaxis] * X_centered).T @ X_centered / Nk[k]

        ll = np.sum(
            np.log(
                np.sum(
                    [
                        pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
                        for k in range(K)
                    ],
                    axis=0,
                )
            )
        )
        log_likelihoods.append(ll)

        if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, sigma, pi, gamma, log_likelihoods


# ------------------------
# 1次元データの可視化
# ------------------------
def plot_1d_result(X, mu, sigma, pi, gamma, name):
    """Visualize GMM result for 1D data with gradient coloring."""
    x_grid = np.linspace(X.min() - 1, X.max() + 1, 500)
    total_pdf = np.zeros_like(x_grid)

    for k in range(len(mu)):
        total_pdf += pi[k] * norm.pdf(
            x_grid, loc=mu[k, 0], scale=np.sqrt(sigma[k][0, 0])
        )

    K = gamma.shape[1]
    base_colors = plt.cm.tab10(np.linspace(0, 1, K))
    point_colors = gamma @ base_colors[:, :3]

    plt.figure(figsize=(10, 4))
    plt.plot(x_grid, total_pdf, color="orange", linewidth=2.5, label="GMM (total)")
    plt.scatter(
        X.flatten(),
        np.zeros_like(X.flatten()),
        color=point_colors,
        s=30,
        alpha=0.8,
        edgecolors="k",
        linewidth=0.3,
        label="Data",
    )
    plt.scatter(
        mu[:, 0], np.zeros_like(mu[:, 0]), c="red", s=100, marker="x", label="Centroids"
    )
    plt.title(f"GMM Result ({name})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.ylim(0, None)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"gmm_{name}_result.png")
    plt.show()


# ------------------------
# 2次元データの可視化
# ------------------------
def plot_2d_result(X, mu, sigma, gamma, name):
    """Visualize GMM result for 2D data with gradient coloring and contour lines."""
    K = len(mu)
    base_colors = plt.cm.tab10(np.linspace(0, 1, K))
    point_colors = gamma @ base_colors[:, :3]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X[:, 0],
        X[:, 1],
        color=point_colors,
        s=30,
        alpha=0.8,
        edgecolors="k",
        linewidth=0.3,
        label="Data",
    )
    ax.scatter(mu[:, 0], mu[:, 1], c="black", s=100, marker="x", label="Centroids")

    x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
    y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    pos = np.dstack((X_grid, Y_grid))

    for k in range(K):
        rv = multivariate_normal(mu[k], sigma[k])
        Z = rv.pdf(pos)
        ax.contour(
            X_grid,
            Y_grid,
            Z,
            levels=6,
            linewidths=1.5,
            colors=[base_colors[k][:3]],
            alpha=0.6,
        )

    ax.set_title(f"GMM Result ({name})", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"gmm_{name}_result_clusters.png")
    plt.show()


# ------------------------
# 対数尤度のプロット
# ------------------------
def plot_log_likelihood(log_likelihoods, name):
    """Plot the log-likelihood over EM iterations."""
    plt.figure(figsize=(8, 4))
    plt.plot(log_likelihoods, marker="o", linestyle="-", color="purple")
    plt.title(f"Log-Likelihood during EM iterations ({name})", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Log-Likelihood", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"gmm_{name}_loglikelihood.png")
    plt.show()


# ------------------------
# 元データの散布図（クラスタリング前）
# ------------------------
def plot_raw_data(X, name):
    """Plot raw input data before clustering."""
    plt.figure(figsize=(8, 6))
    if X.shape[1] == 1:
        plt.scatter(
            X[:, 0], np.zeros_like(X[:, 0]), c="blue", s=30, alpha=0.7, label="Raw data"
        )
        plt.ylabel("Y")
        plt.ylim(-0.05, 0.1)
    else:
        plt.scatter(X[:, 0], X[:, 1], c="blue", s=30, alpha=0.7, label="Raw data")
        plt.ylabel("Y")
    plt.title(f"Raw Data Scatter Plot ({name})", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{name}_rawdata.png")
    plt.show()


# ------------------------
# メイン処理
# ------------------------
def main():
    """Run GMM clustering and visualize results."""
    max_iter = 100
    tol = 1e-4
    seed = 0

    datasets = [
        ("data1", pd.read_csv("data1.csv", header=None).values, 2),
        ("data2", pd.read_csv("data2.csv", header=None).values, 3),
        ("data3", pd.read_csv("data3.csv", header=None).values, 4),
    ]

    for name, X, K in datasets:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        print(f"Processing {name} with {K} clusters...")

        plot_raw_data(X, name)
        mu, sigma, pi, gamma, log_likelihoods = gmm_em(
            X, K, max_iter=max_iter, tol=tol, seed=seed
        )
        plot_log_likelihood(log_likelihoods, name)

        if X.shape[1] == 1:
            plot_1d_result(X, mu, sigma, pi, gamma, name)
        else:
            plot_2d_result(X, mu, sigma, gamma, name)


if __name__ == "__main__":
    main()
