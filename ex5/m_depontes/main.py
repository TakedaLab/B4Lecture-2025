"""変分ベイズに基づいたGMMの実装.

変分ベイズに基づいたGMM及びクラスタリングを実装する.
"""

import csv
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import linalg as la
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal


def read_csv(path: str) -> np.ndarray:
    """csvファイルを読み込む関数.

    指定されたパスからcsvファイルを読み込み，numpy配列として返す.
    入力：
      path(str): csvファイルのパス
    出力：
      csv_data(np.ndarray).shape = (行番号, 行データ): csvデータのnumpy配列
    """
    if not os.path.exists(path):
        return np.array([])

    data = csv.reader(open(path, "r"))
    data = [list(map(float, row)) for row in data]  # 各要素をfloatに変換
    data = np.array(data).T

    return data


class GMMVB:
    """Gaussian Mixture Model with Variational Bayes.

    This class implements the Gaussian Mixture Model (GMM) using Variational Bayes for clustering.
    It's cited from beginaid/GMM-EM-VB.
    Original link is https://github.com/beginaid/GMM-EM-VB/blob/main/src/GMMVB.py
    Appreciate the original author for their work.

    Args:
        K (int): The number of clusters.
    Attributes:
        K (int): The number of clusters.
        eps (float): Small amounts to prevent overflow and underflow.
        N (int): The number of data points.
        D (int): The number of dimensions.
        alpha0 (float): Initial value for alpha.
        beta0 (float): Initial value for beta.
        nu0 (float): Initial value for nu.
        m0 (numpy ndarray): Initial value for m.
        W0 (numpy ndarray): Initial value for W.
        r (numpy ndarray): Responsibility matrix.
        alpha (numpy ndarray): Updated alpha values.
        beta (numpy ndarray): Updated beta values.
        nu (numpy ndarray): Updated nu values.
        m (numpy ndarray): Updated mean values.
        W (numpy ndarray): Updated covariance matrices.
        hdi (list): Highest density interval for each cluster.
    Methods:
        init_params(X): Initialize the parameters.
        gmm_pdf(X): Calculate the log-likelihood of the D-dim mixed Gaussian distribution at N data.
        e_step(X): Execute the variational E-step of VB.
        m_step(X): Execute the variational M-step of VB.
        highest_density_interval(X): Calculate the highest density interval.
        visualize(X, hdi=None): Execute the classification and visualize the results.
        execute(X, iter_max, thr): Execute VB.
    """

    def __init__(self, K):
        """Initialize Constructor.

        Args:
            K (int): The number of clusters.

        Returns:
            None.

        Note:
            eps (float): Small amounts to prevent overflow and underflow.
        """
        self.K = K
        self.eps = np.spacing(1)

    def init_params(self, X):
        """Initialize the parameters.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # The size of X is (N, D)
        self.N, self.D = X.shape
        # Set the initial weighting factors
        self.alpha0 = 0.01
        self.beta0 = 1.0
        self.nu0 = float(self.D)
        self.m0 = np.random.randn(self.D)
        self.W0 = np.eye(self.D)
        self.r = np.random.randn(self.N, self.K)
        # Keep the updated parameters in the field
        self.alpha = np.ones(self.K) * self.alpha0
        self.beta = np.ones(self.K) * self.beta0
        self.nu = np.ones(self.K) * self.nu0
        self.m = np.random.randn(self.K, self.D)
        self.W = np.tile(self.W0[None, :, :], (self.K, 1, 1))

    def gmm_pdf(self, X):
        """Calculate the log-likelihood of the D-dimensional mixed Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            Probability density function (numpy ndarray): Values of the mixed D-dim Gaussian
                                                          distribution at N data.
        """
        pi = self.alpha / (np.sum(self.alpha, keepdims=True) + np.spacing(1))  # (K)
        return np.array(
            [
                pi[k]
                * multivariate_normal.pdf(
                    X, mean=self.m[k], cov=la.pinv(self.nu[:, None, None] * self.W)[k]
                )
                for k in range(self.K)
            ]
        ).T  # (N, K)

    def e_step(self, X):
        """Execute the variational E-step of VB.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Calculate and update the optimized Pi
        log_pi_tilde = digamma(self.alpha) - digamma(self.alpha.sum())  # (K)
        # Calculate the optimized Lambda_titlde
        log_sigma_tilde = np.sum(
            [digamma((self.nu + 1 - i) / 2) for i in range(self.D)]
        ) + (
            self.D * np.log(2) + (np.log(la.det(self.W) + np.spacing(1)))
        )  # (K)
        nu_tile = np.tile(self.nu[None, :], (self.N, 1))  # (N, K)
        res_error = np.tile(X[:, None, None, :], (1, self.K, 1, 1)) - np.tile(
            self.m[None, :, None, :], (self.N, 1, 1, 1)
        )  # (N, K, 1, D)
        quadratic = (
            nu_tile
            * (
                (res_error @ np.tile(self.W[None, :, :, :], (self.N, 1, 1, 1)))
                @ res_error.transpose(0, 1, 3, 2)
            )[:, :, 0, 0]
        )  # (N, K)
        log_rho = (
            (log_pi_tilde[None, :])
            + (0.5 * log_sigma_tilde)[None, :]
            - (0.5 * self.D / (self.beta + np.spacing(1)))[None, :]
            - (0.5 * quadratic)
        )  # (N, K)
        log_r = log_rho - logsumexp(log_rho, axis=1, keepdims=True)  # (N, K)
        # Calculate the responsibility
        r = np.exp(log_r)  # (N, K)
        # Replace the element where nan appears
        r[np.isnan(r)] = 1.0 / (self.K)  # (N, K)
        # Update the optimized r
        self.r = r  # (N, K)

    def m_step(self, X):
        """Execute the variational M-step of VB.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Calculate the optimized N_k
        N_k = np.sum(self.r, 0)  # (K)
        r_tile = np.tile(self.r[:, :, None], (1, 1, self.D)).transpose(
            1, 2, 0
        )  # (K, D, N)
        x_bar = np.sum(
            (r_tile * np.tile(X[None, :, :], (self.K, 1, 1)).transpose(0, 2, 1)), 2
        ) / (
            N_k[:, None] + np.spacing(1)
        )  # (K, D)
        res_error = np.tile(X[None, :, :], (self.K, 1, 1)).transpose(0, 2, 1) - np.tile(
            x_bar[:, :, None], (1, 1, self.N)
        )  # (K, D, N)
        S = ((r_tile * res_error) @ res_error.transpose(0, 2, 1)) / (
            N_k[:, None, None] + np.spacing(1)
        )  # (K, D, D)
        res_error_bar = x_bar - np.tile(self.m0[None, :], (self.K, 1))  # (K, D)
        # Update the optimized parameters
        self.alpha = self.alpha0 + N_k  # (K)
        self.beta = self.beta0 + N_k  # (K)
        self.nu = self.nu0 + N_k  # (K)
        self.m = (
            np.tile((self.beta0 * self.m0)[None, :], (self.K, 1))
            + (N_k[:, None] * x_bar)
        ) / (
            self.beta[:, None] + np.spacing(1)
        )  # (K, D)
        W_inv = (
            la.pinv(self.W0)
            + (N_k[:, None, None] * S)
            + (
                (
                    (self.beta0 * N_k)[:, None, None]
                    * res_error_bar[:, :, None]
                    @ res_error_bar[:, None, :]
                )
                / (self.beta0 + N_k)[:, None, None]
                + np.spacing(1)
            )
        )  # (K, D, D)
        self.W = la.pinv(W_inv)  # (K, D, D)

    def highest_density_interval(self, X):
        """Calculate the highest density interval.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        self.hdi = []
        for k in range(self.K):
            # Calculate posterior distribution and take samples
            rv = multivariate_normal(self.m[k, : self.D], self.W[k])
            samples = rv.rvs(size=50000)
            hdi_k = []
            # Calculate the highest density interval using the samples
            for n in range(self.D):
                samples_1D = np.sort(samples[:, n]) if self.D > 1 else np.sort(samples)
                print(samples_1D)
                length = 0.95 * len(samples_1D)
                L = [
                    samples_1D[i + int(length)] - samples_1D[i]
                    for i in range(len(samples_1D) - int(length))
                ]
                i_star = np.argmin(L)
                hdi_k.append((samples_1D[i_star], samples_1D[i_star + int(length)]))

            self.hdi.append(hdi_k)
            if self.D == 1:
                print(
                    f"Cluster {k + 1} (μ={self.m[k]}): HDI = {hdi_k[0][0]:.3f} ～ {hdi_k[0][1]:.3f}"
                )
            elif self.D == 2:
                print(
                    f"Cluster {k + 1} (μ={self.m[k]}): HDI = {hdi_k[0][0]:.3f} ～ {hdi_k[0][1]:.3f}"
                    f",{hdi_k[1][0]:.3f} ～ {hdi_k[1][1]:.3f}"
                )

    def visualize(self, X, hdi=None):
        """Execute the classification.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Execute classification
        labels = np.argmax(self.r, 1)  # (N)
        # Visualize each clusters
        label_frequency_desc = [
            label[0] for label in Counter(labels).most_common()
        ]  # (K)
        # Prepare the visualization
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = Axes3D(fig)
        fig.add_axes(ax)
        # Use the custome color list.
        cm = plt.get_cmap("tab10")
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Change the perspective
        ax.view_init(elev=10, azim=70)
        # Make data 3D
        if X.shape[1] == 2:
            X = np.hstack([X, np.zeros((X.shape[0], 1))])
        elif X.shape[1] == 1:
            X = np.hstack([X, np.zeros((X.shape[0], 2))])

        for k in range(len(label_frequency_desc)):
            color_map = cm(k)
            cluster_indexes = np.where(labels == label_frequency_desc[k])[0]
            ax.plot(
                X[cluster_indexes, 0],
                X[cluster_indexes, 1],
                X[cluster_indexes, 2],
                "o",
                ms=0.5,
                color=color_map,
            )
            # Plot the mean
            mean_x = self.m[k, 0]
            mean_y = self.m[k, 1] if self.D > 1 else 0
            mean_z = self.m[k, 2] if self.D > 2 else 0
            ax.scatter(
                mean_x,
                mean_y,
                mean_z,
                marker="x",
                s=100,
                color=color_map,
                label=f"Cluster {k + 1}",
            )
            if hdi is None:
                # Plot the Probability Density Function
                if self.D == 1:
                    x = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
                    rv = multivariate_normal(
                        mean=self.m[k, : self.D], cov=la.pinv(self.nu[k] * self.W[k])
                    )
                    Z = rv.pdf(x)
                    ax.plot(x, Z, color=color_map, alpha=0.6)
                elif self.D >= 2:
                    x = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
                    y = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
                    X_grid, Y_grid = np.meshgrid(x, y)
                    pos = np.empty(X_grid.shape + (self.D,))
                    pos[:, :, 0] = X_grid
                    pos[:, :, 1] = Y_grid
                    if self.D > 2:
                        pos[:, :, 2] = mean_z

                    rv = multivariate_normal(
                        mean=self.m[k, : self.D], cov=la.pinv(self.nu[k] * self.W[k])
                    )
                    Z = rv.pdf(pos.reshape(-1, self.D)).reshape(X_grid.shape)
                    ax.contour(
                        X_grid,
                        Y_grid,
                        Z,
                        zdir="z",
                        offset=mean_z,
                        levels=3,
                        colors=[color_map],
                        alpha=0.6,
                    )
            else:
                # describe the highest density interval
                if self.D == 1:
                    x = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
                    rv = multivariate_normal(
                        mean=self.m[k, : self.D], cov=la.pinv(self.nu[k] * self.W[k])
                    )
                    Z = rv.pdf(x)
                    ax.plot(x, Z, color=color_map, alpha=0.6)
                    x_min, x_max = hdi[k][0]
                    mask = (x >= x_min) & (x <= x_max)
                    ax.bar(
                        x[mask],
                        Z[mask],
                        zs=0,
                        zdir="z",
                        color=color_map,
                        alpha=0.4,
                        width=(x[1] - x[0]),
                    )
                elif self.D >= 2:
                    x_min, x_max = hdi[k][0]
                    y_min, y_max = hdi[k][1]
                    z = mean_z
                    verts = [
                        [
                            [x_min, y_min, z],
                            [x_max, y_min, z],
                            [x_max, y_max, z],
                            [x_min, y_max, z],
                        ]
                    ]
                    poly = Poly3DCollection(
                        verts,
                        facecolors=color_map,
                        edgecolors=color_map,
                        linewidths=1,
                        linestyles="-",
                        alpha=0.4,
                    )
                    ax.add_collection3d(poly)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_title("GMMVB")
        plt.show()

    def execute(self, X, iter_max, thr):
        """Execute VB.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).
            iter_max (int): Maximum number of updates.
            thr (float): Threshold of convergence condition.

        Returns:
            None.
        """
        # Init the parameters
        self.init_params(X)
        # Prepare the list of the log_likelihood at each iteration
        log_likelihood_list = []
        # Calculate the initial log-likelihood
        log_likelihood_list.append(
            np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
        )
        # Start the iteration
        for i in range(iter_max):
            # Execute E-step
            self.e_step(X)
            # Execute M-step
            self.m_step(X)
            # Add the current log-likelihood
            log_likelihood_list.append(
                np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
            )
            # Print the gap between the previous log-likelihood and current one
            print(
                "Log-likelihood gap: "
                + str(
                    round(
                        np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]), 2
                    )
                )
            )
            # Visualization is performed when the convergence condition is met
            # or when the upper limit is reached
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]) < thr) or (
                i == iter_max - 1
            ):
                print(f"VB has stopped after {i + 1} iteraions.")
                self.highest_density_interval(X)
                self.visualize(X)
                self.visualize(X, hdi=self.hdi)
                break


def parse_arguments():
    """コマンドライン引数を解析する関数.

    入力：
        なし

    出力：
        args: 入力のオブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(description="GMMを利用したクラスタリング")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="データのパス",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="収束条件の閾値",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help="最大イテレーション数",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """メイン関数."""

    args = parse_arguments()

    # データの読み込み
    data = read_csv(args.path)

    # GMMVBの実行
    gmmvb = GMMVB(K=3)

    gmmvb.execute(
        X=data.T,
        iter_max=args.iter,
        thr=args.tol,
    )

    # クラスタリング結果の表示
