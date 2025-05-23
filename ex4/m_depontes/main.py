"""GMMの実装.

GMM及びそれを用いたクラスタリングを行うプログラムを実装する.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


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


def make_scatter(
    data: np.ndarray, title: str, cluster: np.ndarray = None, mean: np.ndarray = None
) -> None:
    """散布図を作成する関数.

    入力：
        data(np.ndarray): 散布図のデータ, shape = (2, data) || (3, data)
        title(str): グラフのタイトル
    出力：
        None(散布図を出力)
    """
    fig = plt.figure()

    if cluster is None:
        # 散布図を描画する場合の分岐
        if data.shape[0] == 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[0], data[1])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        elif data.shape[0] == 1:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[0], np.zeros(np.shape(data[0])))
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    else:
        # クラスタリング結果を描画する場合の分岐
        from scipy.stats import multivariate_normal

        if data.shape[0] == 1:
            ax = fig.add_subplot(1, 1, 1)
            for i in range(gmm.cluster):
                x = data[0][cluster == i]
                y = np.zeros_like(x)
                ax.scatter(x, y, label=f"Cluster {i}")

            # 確率密度関数
            x_min, x_max = np.min(data[0]), np.max(data[0])
            xx = np.linspace(x_min, x_max, 200).reshape(-1, 1)
            pdf = np.zeros(xx.shape[0])
            for k in range(gmm.cluster):
                rv = multivariate_normal(mean=gmm.mean[k], cov=gmm.cov[k])
                pdf += gmm.mixture_ratio[k] * rv.pdf(xx)
            plt.plot(xx.ravel(), pdf, color="black", linewidth=1, label="GMM PDF")

            ax.scatter(
                mean[:, 0],
                [0 for _ in range(len(mean))],
                c="red",
                marker="x",
                s=100,
                label="Mean",
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
        elif data.shape[0] == 2:
            ax = fig.add_subplot(1, 1, 1)
            x = data[0]
            y = data[1]
            for i in range(gmm.cluster):
                ax.scatter(
                    x[cluster == i],
                    y[cluster == i],
                    label=f"Cluster {i}",
                    cmap="coolwarm",
                )

            # 確率密度関数
            x = np.linspace(np.min(data[0]), np.max(data[0]), 100)
            y = np.linspace(np.min(data[1]), np.max(data[1]), 100)
            xx, yy = np.meshgrid(x, y)
            grid = np.stack([xx.ravel(), yy.ravel()]).T
            pdf = np.zeros(grid.shape[0])
            for k in range(gmm.cluster):
                rv = multivariate_normal(mean=gmm.mean[k], cov=gmm.cov[k])
                pdf += gmm.mixture_ratio[k] * rv.pdf(grid)
            pdf = pdf.reshape(xx.shape)
            ax.contour(xx, yy, pdf, levels=10, cmap="viridis")

            plt.scatter(
                mean[:, 0],
                mean[:, 1],
                c="red",
                marker="x",
                s=100,
                label="Mean",
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()

    plt.show()


class GMMClustering:
    """GMMを利用したクラスタリングの実装を行うクラス.

    メンバ関数：
        _gaussianModel: ガウス分布の確率密度関数を計算する内生関数
        _logLikelihood: 対数尤度を計算する内生関数
        emAlgorithm: EMアルゴリズムを実装する関数
        _e_Step: EMアルゴリズムのEステップを実装する内生関数
        _m_Step: EMアルゴリズムのMステップを実装する内生関数
        clustering: クラスタリングを行う関数
        _define_cluster: AICまたはBICを用いてクラスター数を決定する関数

    属性：
        data(np.ndarray): 入力値
        cluster(int): クラスター数
        n_samples(int): サンプル数
        n_dimensions(int): 次数
        tol(float): 収束条件の閾値
        mixture_ratio(float): 混合比率
        mean(np.ndarray): 平均, shape = (クラスター数, 次数)
        cov(np.ndarray): 共分散行列, shape = (クラスター数, 次数, 次数)
        responsibility(np.ndarray): 責任度, shape = (クラスター数, サンプル数)
        _logLikelihoods(List<float>): 対数尤度のリスト, shape = (イテレーション数, )
    """

    def __init__(
        self, data: np.ndarray, tol: float, max_iteration: int, model: str = None
    ) -> None:
        """初期化関数.

        クラスタリングの初期化を行う関数. 変数の受領及び初期化を行う.
        入力：
            data(np.ndarray): 入力値
            tol(float): 収束条件の閾値
            model(str): クラスター数指定モデルの種類, "AIC" or "BIC" or None
            max_iteration(int): 最大イテレーション数
        """
        self.data = data
        self.n_samples = data.shape[1]
        self.n_dimensions = data.shape[0]
        self.cluster = self._define_cluster(self.data, model)
        print(f"cluster:{self.cluster}")
        self.tol = tol
        self.max_iteration = max_iteration

        self.mixture_ratio = np.ones(self.cluster) / self.cluster
        self.mean = self.data[
            :, np.random.choice(self.n_samples, self.cluster, replace=False)
        ].T
        self.cov = np.array([np.eye(self.n_dimensions) for _ in range(self.cluster)])

        self.responsibility = np.zeros((self.cluster, self.n_samples))
        self._logLikelihoods = []

    def _gaussianModel(self, mean, cov):
        """ガウス分布の確率密度関数を計算する関数.

        入力：
            mean(np.ndarray): 平均, shape = (次数)
            cov(np.ndarray): 共分散行列, shape = (次数, 次数)
        出力：
            likelihood(np.ndarray): ガウス分布の確率密度関数, shape = (クラスター数, サンプル数)
        """
        inv_cov = np.linalg.pinv(cov)  # shape = (次元数, 次元数)
        det_cov = np.linalg.det(cov)

        diff = self.data.T - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)  # shape = (サンプル数,)
        normalization = np.sqrt((2 * np.pi) ** self.n_dimensions * det_cov)
        return (1 / normalization) * np.exp(exponent)  # shape = (サンプル数,)

    def _logLikelihood(self) -> float:
        """対数尤度を計算する関数.

        入力：
            mixture_ratio(float): 混合比率
            mean(np.ndarray): 平均, shape = (クラスター数, 次数)
            cov(np.ndarray): 共分散行列, shape = (クラスター数, 次数, 次数)
        出力：
            L(float): 対数尤度
        """
        # ガウス分布の確率密度関数を計算
        likelihood = np.zeros(
            (self.cluster, self.n_samples)
        )  # shape = (クラスター数, サンプル数)
        for k in range(self.cluster):
            likelihood[k] = self.mixture_ratio[k] * self._gaussianModel(
                self.mean[k], self.cov[k]
            )  # shape = (クラスター数, サンプル数)
        total_likelihood = np.sum(likelihood, axis=0)  # shape = (サンプル数,)
        # 対数尤度を計算
        L = np.sum(np.log(total_likelihood + 1e-10))
        return L

    def emAlgorithm(self):
        """EMアルゴリズムの実装を行う関数.

        入力：
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
            cluster(int): クラスター数
        出力：
            responsibility(np.ndarray): 責任度, shape = (データ数, クラスター数)
        """
        L = 0
        L_new = self._logLikelihood()
        self._logLikelihoods.append(L_new)
        responsibility = np.zeros((self.cluster, self.n_samples))  # 責任度
        for i in range(self.max_iteration):
            L = L_new
            responsibility = self._e_Step()
            self._m_Step(responsibility)
            L_new = self._logLikelihood()
            self._logLikelihoods.append(L_new)
            if abs(L_new - L) < 1e-3:
                break

        self.responsibility = responsibility

    def _e_Step(self):
        """EMアルゴリズムのEステップを実装する関数.

        入力：
            mixture_ratio(float): 混合比率
            mean(np.ndarray): 平均, shape = (次数)
            cov(np.ndarray): 共分散行列, shape = (次数, 次数)
        出力：
            yupsilon(np.ndarray): 責任度, shape = (データ数, クラスター数)
        """
        # ガウス分布の確率密度関数を計算
        gaussians = np.zeros(
            (self.cluster, self.data.shape[1])
        )  # shape = (クラスター数, サンプル数)
        for k in range(self.cluster):
            gaussians[k] = self._gaussianModel(
                self.mean[k], self.cov[k]
            )  # shape = (クラスター数, サンプル数)
        # 責任度 (yupsilon) を計算
        yupsilon = np.zeros(
            (self.cluster, self.n_samples)
        )  # shape = (クラスター数, サンプル数)
        for k in range(self.cluster):
            yupsilon[k] = (
                self.mixture_ratio[k]
                * gaussians[k]
                / np.sum(self.mixture_ratio[:, np.newaxis] * gaussians, axis=0)
            )  # shape = (クラスター数, サンプル数)
        return yupsilon

    def _m_Step(self, responsibility) -> None:
        """EMアルゴリズムのMステップを実装する関数.

        入力：
            responsibility(np.ndarray): 責任度, shape = (クラスター数, サンプル数)
        出力：
            mixture_ratio(np.ndarray): 混合比率, shape = (クラスター数)
            mean(np.ndarray): 平均, shape = (クラスター数, 次数)
            cov(np.ndarray): 共分散行列, shape = (クラスター数, 次数, 次数)
        """
        Nk = np.sum(responsibility, axis=1)
        for k in range(self.cluster):
            self.mean[k] = np.sum(self.data * responsibility[k], axis=1) / Nk[k]

            diff = self.data - self.mean[k][:, np.newaxis]
            cov_k = (responsibility[k] * diff) @ diff.T / Nk[k]
            self.cov[k] = cov_k + np.eye(self.n_dimensions) * 1e-6

            self.mixture_ratio[k] = Nk[k] / self.n_samples

    def clustering(self) -> np.ndarray:
        """クラスタリングを行う関数.

        入力：
            responsibility(np.ndarray): 責任度, shape = (クラスター数, サンプル数)
        出力：
            cluster(np.ndarray): クラスタリング結果, shape = (サンプル数)
        """
        # クラスタリング結果を取得
        self.emAlgorithm()
        cluster = np.argmax(self.responsibility, axis=0)
        return cluster

    def _define_cluster(self, data: np.ndarray, model: str = None) -> int:
        """AICまたはBICを用いてクラスター数を決定する関数.

        入力：
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
            model(str): モデルの種類, "AIC" or "BIC" or None
        出力：
            cluster(int): クラスター数
        """
        mean = np.mean(data, axis=1)
        cov = np.cov(data)
        # AIC
        if model == "AIC":
            AIC = []
            for k in range(2, 11):
                AIC.append(2 * k - 2 * np.log(np.sum(self._gaussianModel(mean, cov))))
            return np.argmin(AIC) + 2

        # BIC
        elif model == "BIC":
            BIC = []
            for k in range(2, 11):
                BIC.append(
                    k * np.log(self.n_samples)
                    - 2 * np.log(np.sum(self._gaussianModel(mean, cov)))
                )
            return np.argmin(BIC) + 2

        else:
            return 3

    def display_log_likelihood(self) -> None:
        """対数尤度を表示する関数.

        入力：
            _logLikelihoods(List<float>): 対数尤度のリスト
        出力：
            None
        """
        plt.plot(self._logLikelihoods)
        plt.title("Log Likelihood")
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.show()


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
        "--model",
        type=str,
        default=None,
        help="クラスター数指定モデルの種類, 'AIC' or 'BIC'",
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

    # 散布図の表示
    make_scatter(data, f"Scatter of {args.path.split('/')[-1].split('.')[0]}")

    # GMMクラスタリングの実行
    gmm = GMMClustering(data, args.tol, args.model)
    cluster = gmm.clustering()

    # クラスタリング結果の表示
    gmm.display_log_likelihood()
    make_scatter(
        data,
        f"Clustering of {args.path.split('/')[-1].split('.')[0]}",
        cluster,
        gmm.mean,
    )
