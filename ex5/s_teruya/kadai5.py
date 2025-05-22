#!/usr/bin/env python
# coding: utf-8

"""kadai5.py.

- GMMVBを用いたパラメータ推定、及び可視化の実装
- ベイズ信用区間の実装
- 実行コマンド
 `$ python kadai5.py`

"""

from collections import Counter  # 頻度カウント

import matplotlib.pyplot as plt  # 可視化
import numpy as np  # 数値計算
from mpl_toolkits.mplot3d import Axes3D  # 可視化
from numpy import linalg as la  # 行列計算
from scipy.special import digamma, logsumexp  # 数値計算
from scipy.stats import multivariate_normal  # 多次元ガウス分布の確率密度関数の計算

PLOT_SIZE = 128


class GMMVB:
    """GMMVB."""

    def __init__(self, K):
        """コンストラクタ.

        Args:
            K (int): クラスタ数

        Returns:
            None.

        Note:
            eps (float): オーバーフローとアンダーフローを防ぐための微小量
        """
        self.K = K
        self.eps = np.spacing(1)

    def init_params(self, X):
        """パラメータ初期化メソッド.

        Args:
            X (numpy ndarray): (N, D)サイズの入力データ

        Returns:
            None.
        """
        # 入力データ X のサイズは (N, D)
        self.N, self.D = X.shape
        # スカラーのパラメータセット
        self.alpha0 = 0.01
        self.beta0 = 1.0
        self.nu0 = float(self.D)
        # 平均は標準ガウス分布から生成
        self.m0 = np.random.randn(self.D)
        # 分散共分散行列は単位行列
        self.W0 = np.eye(self.D)
        # 負担率は標準正規分布から生成するがEステップですぐ更新するので初期値自体には意味がない
        self.r = np.random.randn(self.N, self.K)
        # 更新対象のパラメータを初期化
        self.alpha = np.ones(self.K) * self.alpha0
        self.beta = np.ones(self.K) * self.beta0
        self.nu = np.ones(self.K) * self.nu0
        self.m = np.random.randn(self.K, self.D)
        self.W = np.tile(self.W0[None, :, :], (self.K, 1, 1))

    def gmm_pdf(self, X):
        """N個のD次元データに対してGMMの確率密度関数を計算するメソッド.

        Args:
            X (numpy ndarray): (N, D)サイズの入力データ

        Returns:
            Probability density function (numpy ndarray): 各クラスタにおけるN個のデータに関して計算するため出力サイズは (N, K) となる
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
        """変分Eステップを実行するメソッド.

        Args:
            X (numpy ndarray): (N, D)サイズの入力データ

        Returns:
            None.

        Note:
            以下のフィールドが更新される
                self.r (numpy ndarray): (N, K)サイズの負担率
        """
        # rhoを求めるために必要な要素の計算
        log_pi_tilde = (digamma(self.alpha) - digamma(self.alpha.sum()))[
            None, :
        ]  # (1, K)
        log_sigma_tilde = (
            np.sum([digamma((self.nu + 1 - i) / 2) for i in range(self.D)])
            + (self.D * np.log(2) + (np.log(la.det(self.W) + np.spacing(1))))
        )[
            None, :
        ]  # (1, K)
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
        # 対数領域でrhoを計算
        log_rho = (
            log_pi_tilde
            + (0.5 * log_sigma_tilde)
            - (0.5 * self.D / (self.beta + np.spacing(1)))[None, :]
            - (0.5 * quadratic)
        )  # (N, K)
        # logsumexp関数を利用して対数領域で負担率を計算
        log_r = log_rho - logsumexp(log_rho, axis=1, keepdims=True)  # (N, K)
        # 対数領域から元に戻す
        r = np.exp(log_r)  # (N, K)
        # np.expでオーバーフローを起こしている可能性があるためnanを置換しておく
        r[np.isnan(r)] = 1.0 / (self.K)  # (N, K)
        self.r = r  # (N, K)

    def m_step(self, X):
        """変分Mステップを実行するメソッド.

        Args:
            X (numpy ndarray): (N, D)サイズの入力データ

        Returns:
            None.

        Note:
            以下のフィールドが更新される
                self.alpha (numpy ndarray): (K) サイズのディリクレ分布のパラメータ
                self.beta (numpy ndarray): (K) ガウス分布の分散共分散行列の係数
                self.nu (numpy ndarray): (K) サイズのウィシャート分布のパラメータ
                self.m (numpy ndarray): (K, D) サイズの混合ガウス分布の平均
                self.W (numpy ndarray): (K, D, D) サイズのウィシャート分布のパラメータ
        """
        # 各パラメータを求めるために必要な要素の計算
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
        # 各パラメータを更新
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

    def HDI(self, z=1.96):
        """各μₖの95%ベイズ信用区間を計算するメソッド.

        Returns:
            HDI (numpy ndarray): 下限値と上限値で構成される二組の (K, D) 配列
        """
        sigma_k = np.array([np.diag(self.W[x]) for x in range(self.K)])
        return (self.m - z * np.sqrt(sigma_k), self.m + z * np.sqrt(sigma_k))

    def visualize(self, X, name, call_HDI=False):
        """可視化を実行するメソッド.

        Args:
            X (numpy ndarray): (N, D)サイズの入力データ
            name (str): 保存ファイル名
            call_HDI (bool): HDI描画指定

        Returns:
            None.

        Note:
            このメソッドでは plt.show が実行されるが plt.close() は実行されない
        """
        # クラスタリングを実行
        labels = np.argmax(self.r, 1)  # (N)
        # 利用するカラーを極力揃えるためクラスタを出現頻度の降順に並び替える
        label_frequency_desc = [x[0] for x in Counter(labels).most_common()]
        # tab10 カラーマップを利用
        cm = plt.get_cmap("tab10")
        # グラフの範囲
        plot_max = 1.05 * X.max(axis=0) - 0.05 * X.min(axis=0)  # 分布の右端
        plot_min = 1.05 * X.min(axis=0) - 0.05 * X.max(axis=0)  # 分布の左端
        # 描画準備
        fig = plt.figure(figsize=(6, 5))
        if self.D == 1:
            ax = fig.add_subplot()
            # メモリを除去
            ax.set_xticks([])
            ax.set_yticks([])
            # 各クラスタごとに可視化を実行する
            for k in range(len(label_frequency_desc)):
                cluster_indexes = np.where(labels == label_frequency_desc[k])[0]
                y0 = np.zeros(len(X[cluster_indexes, 0]))
                ax.plot(
                    X[cluster_indexes, 0], y0, "o", alpha=0.2, color=cm(k), zorder=2
                )
            # グラフ or HDI表示
            if call_HDI:
                fill_mins, fill_maxs = self.HDI()
                for fill_min, fill_max in zip(fill_mins, fill_maxs):
                    ax.axvspan(fill_min[0], fill_max[0], alpha=0.5, zorder=1)
            else:
                xlist = np.linspace(plot_min, plot_max, PLOT_SIZE)
                ylist = self.gmm_pdf(xlist).sum(axis=1)
                ax.plot(xlist, ylist, zorder=1)
            # ラベル
            ax.set_xlabel("x")
            ax.set_ylabel("GM Value")
        elif self.D == 2:
            ax = fig.add_subplot()
            # メモリを除去
            ax.set_xticks([])
            ax.set_yticks([])
            # 各クラスタごとに可視化を実行する
            for k in range(len(label_frequency_desc)):
                cluster_indexes = np.where(labels == label_frequency_desc[k])[0]
                ax.plot(
                    X[cluster_indexes, 0],
                    X[cluster_indexes, 1],
                    "o",
                    alpha=0.5,
                    color=cm(k),
                    zorder=2,
                )
            # 等高線 or HDI表示
            if call_HDI:
                fill_mins, fill_maxs = self.HDI()
                for fill_min, fill_max in zip(fill_mins, fill_maxs):
                    fill_range = np.linspace(fill_min[0], fill_max[0], PLOT_SIZE)
                    ax.fill_between(
                        fill_range, fill_min[1], fill_max[1], alpha=0.5, zorder=1
                    )
            else:
                xlist, ylist = np.mgrid[
                    plot_min[0] : plot_max[0] : (plot_max[0] - plot_min[0]) / PLOT_SIZE,
                    plot_min[1] : plot_max[1] : (plot_max[1] - plot_min[1]) / PLOT_SIZE,
                ]
                xylist = np.dstack((xlist, ylist))
                zlist = self.gmm_pdf(xylist).sum(axis=2).T
                qcs = ax.contour(xlist, ylist, zlist, zorder=1)
                # カラーバー...
                qcf = ax.contourf(xlist, ylist, zlist, levels=qcs.levels)
                fig.colorbar(qcf)
                for c in qcf.collections:
                    c.remove()
            # ラベル
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif self.D == 3:
            ax = Axes3D(fig)
            fig.add_axes(ax)
            # メモリを除去
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # 少し回転させて見やすくする
            ax.view_init(elev=10, azim=70)
            # 各クラスタごとに可視化を実行する
            for k in range(len(label_frequency_desc)):
                cluster_indexes = np.where(labels == label_frequency_desc[k])[0]
                ax.plot(
                    X[cluster_indexes, 0],
                    X[cluster_indexes, 1],
                    X[cluster_indexes, 2],
                    "o",
                    ms=1.0,
                    alpha=0.5,
                    color=cm(k),
                )
        plt.title(name)
        plt.savefig(f"picture/result_{name}.png")
        plt.show()

    def execute(self, X, iter_max, thr, name="data"):
        """VBを実行するメソッド.

        Args:
            X (numpy ndarray): (N, D)サイズの入力データ
            iter_max (int): 最大更新回数
            thr (float): 更新停止の閾値 (対数尤度の増加幅)
            name (str): 保存ファイル名

        Returns:
            None.
        """
        # パラメータ初期化
        self.init_params(X)
        # 各イテレーションの対数尤度を記録するためのリスト
        log_likelihood_list = []
        # 対数尤度の初期値を計算
        log_likelihood_list.append(
            np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
        )
        # 更新開始
        for i in range(iter_max):
            # Eステップの実行
            self.e_step(X)
            # Mステップの実行
            self.m_step(X)
            # 今回のイテレーションの対数尤度を記録する
            log_likelihood_list.append(
                np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
            )
            # 前回の対数尤度からの増加幅を出力する
            print(
                "Log-likelihood gap: "
                + str(
                    round(
                        np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]), 4
                    )
                )
            )
            # もし収束条件を満たした場合，もしくは最大更新回数に到達した場合は更新停止して可視化を行う
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]) < thr) or (
                i == iter_max - 1
            ):
                print(f"VB has stopped after {i + 1} iteraions.")
                self.visualize(X, name)
                self.visualize(X, f"{name}_HDI", True)  # HDIを表示
                break


if __name__ == "__main__":
    # ファイル読み込み
    data1 = np.loadtxt("data1.csv", delimiter=",").reshape(-1, 1)
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
    plt.savefig("picture/scatters.png")
    plt.show()

    # パラメータ推定
    model2 = GMMVB(K=2)
    model2.execute(data1, iter_max=100, thr=0.0001, name="data1")
    model3 = GMMVB(K=3)
    model3.execute(data2, iter_max=100, thr=0.0001, name="data2")
    model3.execute(data3, iter_max=100, thr=0.0001, name="data3")
