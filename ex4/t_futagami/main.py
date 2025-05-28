"""
GMM (Gaussian Mixture Model)によるクラスタリングを実行するプログラム.

対数尤度の変化をプロットし、各クラスタの平均点と混合ガウス分布の確率密度関数を重ねて表示する.
"""

import argparse

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class GMM:
    """
    Gaussian Mixture Model によるクラスタリング.

    Parameters
    ----------
    n_components : int
        ガウス分布の混合数
    max_iter : int, default=100
        EMアルゴリズムの最大反復回数
    tol : float, default=1e-3
        収束判定用の対数尤度の変化閾値

    Attributes
    ----------
    _weights : np.ndarray, shape (n_components,)
        混合係数 π_k
    _means : np.ndarray, shape (n_components, n_features)
        各成分の平均ベクトル μ_k
    _covariances : np.ndarray, shape (n_components, n_features, n_features)
        各成分の共分散行列 Σ_k
    _log_likelihoods : list of float
        各イテレーションの対数尤度
    """

    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-3):
        """
        GMMの初期化.

        Parameters
        ----------
        n_components : int
            ガウス分布の混合数
        max_iter : int, default=100
            EMアルゴリズムの最大反復回数
        tol : float, default=1e-3
            収束判定用の対数尤度の変化閾値
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        # 以下は fit の中で初期化される
        self._weights: np.ndarray
        self._means: np.ndarray
        self._covariances: np.ndarray
        self._log_likelihoods: list

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        パラメータの初期化.

        各成分の平均をデータの範囲に基づいて初期化し、
        共分散行列を単位行列で初期化する.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            入力データ
        """
        n_features = X.shape[1]
        self._weights = np.ones(self.n_components) / self.n_components

        # 各成分の平均をデータの範囲に基づいて初期化
        self._means = np.random.rand(self.n_components, n_features) * (
            np.max(X, axis=0) - np.min(X, axis=0)
        ) + np.min(X, axis=0)

        # 各成分の共分散行列を単位行列で初期化
        self._covariances = np.array(
            [np.eye(n_features) for _ in range(self.n_components)]
        )

        # 対数尤度の初期化
        self._log_likelihoods = []

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Eステップ: responsibilities を計算して返す.

        Returns
        -------
        responsibilities : np.ndarray
        """
        n_samples, n_features = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))

        # 各成分の確率密度関数を計算
        for k in range(self.n_components):
            diff = X - self._means[k]
            cov_inv = np.linalg.inv(self._covariances[k])
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            coeff = 1 / (
                (2 * np.pi) ** (n_features / 2)
                * np.linalg.det(self._covariances[k]) ** 0.5
            )
            responsibilities[:, k] = self._weights[k] * coeff * np.exp(exponent)

        # 各サンプルの責任度を計算
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        Mステップ: _weights, _means, _covariances を更新する.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            入力データ
        responsibilities : np.ndarray, shape (n_samples, n_components)
            各サンプルの責任度
        """
        n_samples = X.shape[0]
        # 混合係数の更新
        self._weights = responsibilities.sum(axis=0) / n_samples

        # 平均ベクトルの更新
        for k in range(self.n_components):
            self._means[k] = (responsibilities[:, k] @ X) / responsibilities[:, k].sum()

        # 共分散行列の更新
        for k in range(self.n_components):
            diff = X - self._means[k]
            self._covariances[k] = (
                (diff * responsibilities[:, k][:, None]).T @ diff
            ) / responsibilities[:, k].sum()

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        対数尤度を計算する.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            入力データ
        Returns
        -------
        log_likelihood : float
            対数尤度
        """
        n_features = X.shape[1]
        log_likelihood = 0.0

        # 各成分の確率密度関数を計算
        for k in range(self.n_components):
            diff = X - self._means[k]
            cov_inv = np.linalg.inv(self._covariances[k])
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            coeff = 1 / (
                (2 * np.pi) ** (n_features / 2)
                * np.linalg.det(self._covariances[k]) ** 0.5
            )
            log_likelihood += self._weights[k] * coeff * np.exp(exponent)

        return np.sum(np.log(log_likelihood))

    def fit(self, X: np.ndarray) -> "GMM":
        """
        EMアルゴリズムで学習を実行.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            入力データ

        Returns
        -------
        self
        """
        self._initialize_parameters(X)

        for i in range(self.max_iter):
            # Eステップ
            responsibilities = self._e_step(X)
            # Mステップ
            self._m_step(X, responsibilities)
            # 対数尤度の計算
            log_likelihood = self._compute_log_likelihood(X)
            self._log_likelihoods.append(log_likelihood)

            # 収束判定
            if i > 0 and abs(log_likelihood - self._log_likelihoods[-2]) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        各サンプルの最尤クラスタラベルを返す.

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
        """
        if not hasattr(self, "_weights"):
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")

        responsibilities = self._e_step(X)
        labels = np.argmax(responsibilities, axis=1)
        return labels


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.

    入力データのパスを取得.
    """
    parser = argparse.ArgumentParser(description="GMMベースのクラスタリングの実行")
    parser.add_argument(
        "input_data",
        type=str,
        help="入力データのパスを指定してください。",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="ガウス分布の混合数を指定してください。デフォルトは3です。",
    )

    args = parser.parse_args()
    # 入力の検証
    if not args.input_data.endswith(".csv"):
        parser.error("入力データはCSVファイルである必要があります。")

    return args


def plot_gmm_results_1D(gmm: GMM, X: np.ndarray) -> None:
    """
    1次元データの場合の GMM の結果をプロットする.

    各サンプルの最尤クラスタラベルを色分けした散布図と,
    各クラスタの平均点および混合ガウス分布の確率密度関数を重ねて表示する.

    Parameters
    ----------
    gmm : GMM
        学習済み GMM モデル
    X : np.ndarray, shape (n_samples, 1)
        入力データ（1次元）
    """
    labels = gmm.predict(X)  # 各サンプルの最尤クラスタラベルを取得
    xs = X.ravel()  # 1次元配列に変換
    zs = np.zeros_like(xs)

    # 散布図
    plt.figure()
    plt.scatter(xs, zs, c=labels, cmap="viridis", alpha=0.6, s=30)

    # クラスタ平均のプロット
    means = gmm._means.ravel()
    plt.scatter(means, np.zeros_like(means), c="red", marker="x", s=100, label="means")

    # 描画領域
    x_min, x_max = xs.min() - 1, xs.max() + 1
    xx = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

    # 確率密度関数
    pdf = np.zeros(xx.shape[0])
    for k in range(gmm.n_components):
        rv = multivariate_normal(mean=gmm._means[k], cov=gmm._covariances[k])
        pdf += gmm._weights[k] * rv.pdf(xx)
    plt.plot(xx.ravel(), pdf, color="black", linewidth=1, label="GMM PDF")

    plt.xlabel("x")
    plt.title("data")
    plt.legend()
    plt.show()


def plot_gmm_results_2D(gmm: GMM, X: np.ndarray) -> None:
    """
    2次元データの場合の GMM の結果をプロットする.

    各サンプルの最尤クラスタラベルを色分けした散布図と,
    各クラスタの平均点および混合ガウス分布の確率密度関数を重ねて表示する.

    Parameters
    ----------
    gmm : GMM
        学習済み GMM モデル
    X : np.ndarray, shape (n_samples, 2)
        入力データ（2次元）
    """
    # ラベルと責任度を取得
    labels = gmm.predict(X)

    fig, ax = plt.subplots()

    # 散布図
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6, s=30)

    # クラスタ平均のプロット
    means = gmm._means
    ax.scatter(means[:, 0], means[:, 1], c="red", marker="x", s=100, label="means")

    # 描画領域
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # 混合ガウス分布の確率密度関数を計算
    pdf = np.zeros_like(xx)
    for k in range(gmm.n_components):
        rv = multivariate_normal(mean=gmm._means[k], cov=gmm._covariances[k])
        pdf += gmm._weights[k] * rv.pdf(grid).reshape(xx.shape)

    # 等高線のプロット
    cs = ax.contour(xx, yy, pdf, levels=10, cmap="viridis", linewidths=1.0)
    norm = colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
    # カラーバー用の Normalize と ScalarMappable を作成
    sm = plt.cm.ScalarMappable(cmap=cs.cmap, norm=norm)
    sm.set_array([])  # 空の配列を設定
    fig.colorbar(sm, ticks=cs.levels, ax=ax, label="GMM PDF")  # カラーバーを追加
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("data3")
    ax.legend()
    plt.show()


def plot_gmm_results(gmm: GMM, X: np.ndarray) -> None:
    """
    GMM の結果をプロットする.

    1次元データの場合は散布図と確率密度関数を,
    2次元データの場合は散布図と等高線を描画する.

    Parameters
    ----------
    gmm : GMM
        学習済み GMM モデル
    X : np.ndarray, shape (n_samples, 2)
        入力データ（1次元または2次元）
    """
    if X.shape[1] == 1:
        plot_gmm_results_1D(gmm, X)
    elif X.shape[1] == 2:
        plot_gmm_results_2D(gmm, X)
    else:
        raise ValueError("X must be either 1D or 2D array.")


def plot_log_likelihood(gmm: GMM) -> None:
    """
    各イテレーションの対数尤度をプロットする.

    Parameters
    ----------
    gmm : GMM
        学習済み GMM モデル
    """
    plt.figure()
    plt.plot(gmm._log_likelihoods)
    plt.title("data3")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.grid()
    plt.show()


def main():
    """
    メイン関数.

    コマンドライン引数を解析し、GMMを学習させ、
    対数尤度の変化とクラスタリング結果をプロットする.
    """
    args = parse_arguments()
    X = np.genfromtxt(args.input_data, delimiter=",")  # CSVファイルからデータを読み込む
    n_components = args.n_components  # ガウス分布の混合数を取得
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # 1次元データを2次元に変換
    gmm = GMM(n_components=n_components, max_iter=100, tol=1e-3)
    gmm.fit(X)
    plot_log_likelihood(gmm)
    plot_gmm_results(gmm, X)


if __name__ == "__main__":
    main()
