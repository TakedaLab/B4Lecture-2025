"""
Elastic Net回帰モジュール.

機能:
  - Elastic Net (L1/L2) 正則化回帰モデルの学習および予測.
  - 散布図、超平面のプロット.
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt


class ElasticNet:
    """
    Elastic Net回帰モデル.

    Attributes:
        alpha (float): 正則化パラメータ
        l1_ratio (float): L1正則化の比率 (0 <= l1_ratio <= 1)
        max_iter (int): 最大反復回数
        tol (float): 収束判定の許容誤差
        coef_ (np.ndarray): 重みベクトル
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        """
        Elastic Net回帰モデルの初期化.

        Parameters:
            alpha (float): 正則化パラメータ
            l1_ratio (float): L1正則化の比率 (0 <= l1_ratio <= 1)
            max_iter (int): 最大反復回数
            tol (float): 収束判定の許容誤差
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        特徴量行列に切片項を追加.

        Parameters:
            X (np.ndarray): 特徴量行列

        Returns:
            np.ndarray: 切片項を追加した特徴量行列
        """
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Elastic Netのコスト関数を計算.

        Parameters:
            X (np.ndarray): 特徴量行列
            y (np.ndarray): 目的変数ベクトル
            weights (np.ndarray): 重みベクトル

        Returns:
            float: コスト関数の値
        """
        n_sample = X.shape[0]
        residual = y - X @ weights
        l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(weights[1:] ** 2)
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(weights[1:]))
        return 1 / (2 * n_sample) * np.sum(residual**2) + l2_penalty + l1_penalty

    def _update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        normalization_factors: np.ndarray,
    ) -> np.ndarray:
        """
        座標降下法を用いて重みを更新.

        Parameters:
            X (np.ndarray): 特徴量行列
            y (np.ndarray): 目的変数ベクトル
            weights (np.ndarray): 現在の重みベクトル
            normalization_factors (np.ndarray): 正規化係数

        Returns:
            np.ndarray: 更新された重みベクトル
        """
        n_samples = X.shape[0]
        weights[0] = np.sum(y - X[:, 1:] @ weights[1:]) / n_samples  # 切片項の更新

        for j in range(1, X.shape[1]):
            tmp = (y - np.delete(X, j, 1) @ np.delete(weights, j)) @ X[:, j] / n_samples

            # ソフト閾値処理
            if tmp > self.alpha * self.l1_ratio:
                weights[j] = (tmp - self.alpha * self.l1_ratio) / normalization_factors[
                    j
                ]
            elif tmp < -self.alpha * self.l1_ratio:
                weights[j] = (tmp + self.alpha * self.l1_ratio) / normalization_factors[
                    j
                ]
            else:
                weights[j] = 0.0

        return weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Elastic Netのモデルを訓練.

        Parameters:
            X (np.ndarray): 特徴量行列
            y (np.ndarray): 目的変数ベクトル
        """
        # データ前処理
        X = self._add_intercept(X)
        n_samples, n_features = X.shape

        # 重みの初期化
        self.coef_ = np.zeros(n_features)

        # 正規化係数の計算
        normalization_factors = np.sum(X**2, axis=0) / n_samples + self.alpha * (
            1 - self.l1_ratio
        )
        normalization_factors += 1e-8  # ゼロ除算を避けるために小さな値を加える

        # 座標降下法による最適化
        prev_cost = float("inf")
        for iteration in range(self.max_iter):
            self.coef_ = self._update_weights(X, y, self.coef_, normalization_factors)
            # コスト関数の計算
            current_cost = self._compute_cost(X, y, self.coef_)

            # 収束判定
            if np.abs(prev_cost - current_cost) < self.tol:
                print(f"収束しました (反復回数: {iteration})")
                break
            prev_cost = current_cost

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        訓練されたモデルを使用して予測を行う.

        Parameters:
            X (np.ndarray): 特徴量行列

        Returns:
            np.ndarray: 予測値
        """
        # データ前処理
        X = self._add_intercept(X)
        return X @ self.coef_


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.

    入力データのパスを取得.
    """
    parser = argparse.ArgumentParser(description="Elastic Net回帰の実行")
    parser.add_argument(
        "input_data",
        type=str,
        help="入力データのパスを指定してください。",
    )

    args = parser.parse_args()
    # 入力の検証
    if not args.input_data.endswith(".csv"):
        parser.error("入力データはCSVファイルである必要があります。")

    return args


def plot_results(X, y):
    """
    結果をプロットする関数.

    Parameters:
        X (np.ndarray): 特徴量行列
        y (np.ndarray): 目的変数ベクトル
        model (ElasticNet): 訓練されたモデル
    """
    n_features = X.shape[1]
    model = ElasticNet(
        alpha=0.0, l1_ratio=0.5, max_iter=10000, tol=1e-7
    )  # モデルの初期化
    if n_features == 1:
        plt.scatter(X, y)
        plt.xlabel("x")
        plt.ylabel("y")

        X = np.hstack((X, X**2, X**3))  # 2次と3次の特徴量を追加
        model.fit(X, y)  # モデルの訓練
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        X_grid = np.hstack(
            (
                x_vals.reshape(-1, 1),
                x_vals.reshape(-1, 1) ** 2,
                x_vals.reshape(-1, 1) ** 3,
            )
        )
        y_grid = model.predict(X_grid)
        # 回帰式を文字列にフォーマット
        equation = (
            f"y = {model.coef_[0]:.2f}"
            f" + {model.coef_[1]:.2f}x"
            f" + {model.coef_[2]:.2f}x^2"
            f" + {model.coef_[3]:.2f}x^3"
        )
        plt.plot(x_vals, y_grid, color="red", label=equation)
        plt.legend()

    elif n_features == 2:
        ax = plt.axes(projection="3d")
        ax.scatter3D(X[:, 0], X[:, 1], y)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        X = np.hstack((X, X**2))  # 2次の特徴量を追加
        model.fit(X, y)  # モデルの訓練
        x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
        x2_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
        X1g, X2g = np.meshgrid(x1_vals, x2_vals)
        X_grid = np.hstack(
            (
                X1g.reshape(-1, 1),
                X2g.reshape(-1, 1),
                (X1g**2).reshape(-1, 1),
                (X2g**2).reshape(-1, 1),
            )
        )
        y_grid = model.predict(X_grid).reshape(X1g.shape)

        # サーフェスプロット
        ax.plot_surface(X1g, X2g, y_grid, color="red", alpha=0.5)

        # 回帰式を表示
        equation = (
            f"y = {model.coef_[0]:.2f} + {model.coef_[1]:.2f}x1 + {model.coef_[2]:.2f}x2 "
            f"+ {model.coef_[3]:.2f}x1^2 + {model.coef_[4]:.2f}x2^2"
        )
        ax.text2D(
            0.05,
            0.95,
            equation,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )
    else:
        raise ValueError("2次元または3次元のデータをプロットする必要があります。")

    plt.show()


def main():
    """
    メイン関数.

    コマンドライン引数を解析し、Elastic Net回帰を実行.
    """
    args = parse_arguments()  # 引数を解析
    data = np.genfromtxt(
        args.input_data, delimiter=",", skip_header=1
    )  # CSVファイルを読み込み
    X = data[:, :-1]  # 特徴量
    y = data[:, -1]  # 目的変数
    plot_results(X, y)  # 結果をプロット


if __name__ == "__main__":
    main()
