#!/usr/bin/env python
# coding: utf-8

"""kadai4.py.

- EMアルゴリズムを用いたGMMによる分類の実装
- BIC(AIC)によるクラスター数の決定
- 実行コマンド
 `$ python3 kadai4.py`

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgError
from scipy.stats import multivariate_normal as mn

PLOT_SIZE = 128


def MGD(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, rate: np.ndarray):
    """混合ガウス分布を返します。.

    x:
      入力([N,dim] or [N,M,dim] ...)
    mean:
      平均ベクトル([K,dim])
    cov:
      分散共分散行列([K,dim,dim])
    rate:
      重み([K])
    """
    MGD_map = np.zeros(x.shape[:-1])
    for k in range(len(rate)):
        MGD_map += rate[k] * mn.pdf(x, mean[k], cov[k])
    return MGD_map


class GMM:
    """混合ガウスモデル(EMアルゴリズム)."""

    def __init__(self, data: np.ndarray):
        """入力データを保存します。.

        data:
          入力データ
        """
        if len(data.shape) == 1:
            self.data = data.reshape((-1, 1))
        else:
            self.data = data
        # クラス数Kが不明なため、ここでは何も入力しない
        self.rate = None  # k個のスカラー
        self.mean = None  # k個のベクトル
        self.cov = None  # k個の行列
        self.judo_list = []  # 対数尤度保存用
        self.gosa = []  # 誤差保存用

    def EStep(self) -> np.ndarray:
        """負担率を計算します。."""
        # N:dataの長さ K:クラスター数=重みの数
        burden = np.zeros((self.data.shape[0], len(self.rate)))  # 縦N×横K
        for k in range(len(self.rate)):  # π_k*N(X|μ_k,Σ_k)を計算(縦に保存)
            burden[:, k] = self.rate[k] * mn.pdf(self.data, self.mean[k], self.cov[k])
        return burden / np.sum(burden, axis=1).reshape(-1, 1)

    def MStep(self, brate: np.ndarray):
        """パラメータを更新します。.

        brate:
          負担率
        """
        N_k = np.sum(brate, axis=0)
        # Σ_kの更新
        for k in range(len(self.rate)):
            # np.einsumをchatGPTに教えてもらいました！
            dif = self.data - self.mean[k]
            covs = (
                brate[:, k].reshape(-1, 1, 1) * np.einsum("ni,nj->nij", dif, dif)
            ).sum(axis=0)
            self.cov[k] = covs / N_k[k]
        # μ_kの更新
        for k in range(len(self.rate)):
            self.mean[k] = brate[:, k] @ self.data / N_k[k]
        # πの更新
        self.rate = N_k / self.data.shape[0]

    def EM(self, clst: int = 3):
        """EMアルゴリズムを行います。完了時に各パラメータを返します。.

        clst:
          クラスター数。必ず2以上を指定してください(default:3)
        """
        self.judo_list = []  # 対数尤度保存用
        self.gosa = []  # 誤差保存用
        # 初期値設定
        self.mean = np.random.randn(clst, self.data.shape[1])
        self.cov = np.array([np.eye(self.data.shape[1])] * clst)
        self.rate = np.array([1 / clst] * clst)

        log_judo_past = -np.inf
        while True:
            # Eステップ
            brate = self.EStep()
            # Mステップ
            self.MStep(brate)
            # 収束確認
            log_judo = np.sum(np.log(MGD(self.data, self.mean, self.cov, self.rate)))
            self.judo_list.append(log_judo)  # 対数尤度保存用
            self.gosa.append(log_judo - log_judo_past)  # 誤差保存用
            if log_judo - log_judo_past < 1.0e-6:
                return self.mean, self.cov, self.rate
            log_judo_past = log_judo

    def predict(self) -> np.ndarray:
        """各データをクラスタリングします。."""
        brate = self.EStep()
        data_clst = np.argmax(brate, axis=1)
        return data_clst

    def plot_likelihood_GMMpredict(
        self,
        data_clst: np.ndarray,
        dataname: str = "data",
        filename: str = "result.png",
    ):
        """対数尤度と分類結果を出力する専門関数です。.

        data_clst:
          データの分類結果(配列)
        dataname:
          データ名
        filename:
          保存ファイル名
        """
        fig = plt.figure(figsize=(9, 4))
        # 対数尤度表示
        ax1 = fig.add_subplot(121)  # 左に
        ax1.set_title(dataname + " 対数尤度", fontname="MS Gothic")
        ax1.set_xlabel("iter")
        ax1.set_ylabel("ln likelihood")
        xlist = np.linspace(1, len(self.judo_list) + 1, len(self.judo_list))
        ax1.plot(xlist, self.judo_list)
        # 誤差表示
        ax1_copy = ax1.twinx()
        ax1_copy.set_ylabel("error(log10)")
        ax1_copy.plot(xlist, np.log10(np.array(self.gosa)), color="orange")
        # 分類結果表示
        ax2 = fig.add_subplot(122)  # 右に
        ax2.set_title(dataname)
        ax2.set_xlabel("x")
        plot_max = 1.05 * self.data.max(axis=0) - 0.05 * self.data.min(
            axis=0
        )  # 分布の右端
        plot_min = 1.05 * self.data.min(axis=0) - 0.05 * self.data.max(
            axis=0
        )  # 分布の左端
        if self.data.shape[1] == 1:  # 一次元
            ax2.set_ylabel("GM Value")
            ax2.scatter(
                self.data,
                np.zeros(data1.shape[0]),
                c=data_clst,
                s=10,
                alpha=0.2,
                zorder=2,
            )  # 分類後データ
            ax2.scatter(
                self.mean, np.zeros(len(self.mean)), s=10, marker="x", zorder=3
            )  # 平均
            xlist = np.linspace(plot_min, plot_max, PLOT_SIZE)
            ylist = MGD(xlist, self.mean, self.cov, self.rate)
            ax2.plot(xlist, ylist, zorder=1)  # 混合ガウス分布
        elif self.data.shape[1] == 2:  # 二次元
            ax2.set_ylabel("y")
            ax2.scatter(
                self.data[:, 0], self.data[:, 1], c=data_clst, s=10, alpha=0.2, zorder=2
            )  # 分類後データ
            ax2.scatter(
                self.mean[:, 0], self.mean[:, 1], s=10, marker="x", zorder=3
            )  # 平均
            xlist, ylist = np.mgrid[
                plot_min[0] : plot_max[0] : (plot_max[0] - plot_min[0]) / PLOT_SIZE,
                plot_min[1] : plot_max[1] : (plot_max[1] - plot_min[1]) / PLOT_SIZE,
            ]
            xylist = np.dstack((xlist, ylist))
            zlist = MGD(xylist, self.mean, self.cov, self.rate)
            qcs = ax2.contour(xlist, ylist, zlist, zorder=1)  # 混合ガウス分布
            fig.colorbar(qcs)
        else:
            print(f"dim-{self.data.shape[1]}-data is unsupported.")
        fig.tight_layout()
        plt.savefig(filename)
        plt.show()


if __name__ == "__main__":
    # ファイル読み込み
    data1 = np.loadtxt("data1.csv", delimiter=",")
    data2 = np.loadtxt("data2.csv", delimiter=",")
    data3 = np.loadtxt("data3.csv", delimiter=",")

    # 散布図表示
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)  # 左に
    ax1.set_title("data1")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_ylim(-0.1)
    ax1.scatter(data1, np.zeros(data1.shape[0]), s=10, alpha=0.2)
    ax2 = fig.add_subplot(132)  # 真ん中に
    ax2.set_title("data2")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.scatter(data2[:, 0], data2[:, 1], s=10, alpha=0.2)
    ax3 = fig.add_subplot(133)  # 右に
    ax3.set_title("data3")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.scatter(data3[:, 0], data3[:, 1], s=10, alpha=0.2)
    fig.tight_layout()
    plt.savefig("scatters.png")
    plt.show()

    # GMMで分類
    gmm1 = GMM(data1)
    gmm2 = GMM(data2)
    gmm3 = GMM(data3)
    gmm1.EM(clst=2)
    gmm2.EM(clst=3)
    gmm3.EM()
    predict1 = gmm1.predict()
    predict2 = gmm2.predict()
    predict3 = gmm3.predict()

    # 表示
    gmm1.plot_likelihood_GMMpredict(predict1, "data1", "result1.png")
    gmm2.plot_likelihood_GMMpredict(predict2, "data2", "result2.png")
    gmm3.plot_likelihood_GMMpredict(predict3, "data3", "result3.png")

    # おまけ
    # data3を、AIC/BICを用いてクラスター数決定
    MAX = 8
    AIC_score = []
    BIC_score = []
    for k in range(2, MAX + 1):
        t = 0
        while True:
            try:
                meanic, covic, ratic = gmm3.EM(k)
                break
            except LinAlgError:  # 正則でない場合はやり直す
                t += 1
                if t > 10:
                    print("Oh no...")
                    break
        log_judo = np.sum(np.log(MGD(data3, meanic, covic, ratic)))
        param_num = k * (1 + 3 / 2 * data3.shape[1] + data3.shape[1] ** 2 / 2)
        aic = -2 * log_judo + 2 * param_num
        bic = -2 * log_judo + np.log(len(data3)) * param_num
        AIC_score.append(aic)
        BIC_score.append(bic)

    # スコア表示
    plt.plot(np.arange(2, MAX + 1, 1), AIC_score, label="AIC")
    plt.plot(np.arange(2, MAX + 1, 1), BIC_score, label="BIC")
    plt.title("AIC & BIC")
    plt.xlabel("clst")
    plt.ylabel("score")
    plt.legend()
    plt.savefig("AIC&BIC.png")
    plt.show()

    # BICの最も低いスコアの分類結果表示
    min_k = np.argmin(BIC_score) + 2
    gmm3.EM(min_k)
    predict_bic = gmm3.predict()
    gmm3.plot_likelihood_GMMpredict(predict_bic, f"data3(K={min_k})", "result_bic.png")
