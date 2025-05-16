"""線形回帰・主成分分析の実装.

線形回帰・主成分分析を行うプログラムを実装する.
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
    data = [row for row in data][1:]  # ヘッダーを除外
    data = [list(map(float, row)) for row in data]  # 各要素をfloatに変換
    data = np.array(data).T

    return data


def make_scatter(
    data: np.ndarray, title: str, pred: np.ndarray = None, pca=None
) -> None:
    """散布図を作成する関数.

    入力：
        data(np.ndarray): 散布図のデータ, shape = (2, data) || (3, data)
        title(str): グラフのタイトル
    出力：
        None(散布図を出力)
    """
    fig = plt.figure()

    if pred is None:
        # 回帰曲線を描画しない場合の分岐
        if pca is None:
            # 主成分分析を行う場合の分岐
            if data.shape[0] == 2:

                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(data[0], data[1])
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            elif data.shape[0] == 3:
                ax = fig.add_subplot(projection="3d")
                ax.scatter(data[0], data[1], data[2])
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

        else:
            # 主成分分析を行わない場合の分岐
            if data.shape[0] == 2:
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(data[0], data[1])
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

                # 主成分軸の描画
                origin = pca.means  # データの平均点
                for vec in pca.eigenvectors[:2].T:
                    line_x = np.linspace(min(data[0]), max(data[0]), 100)
                    line_y = origin[1] + (vec[1] / vec[0]) * (line_x - origin[0])
                    ax.plot(line_x, line_y, color="r", linestyle="--")
                ax.legend()

            elif data.shape[0] == 3:
                ax = fig.add_subplot(projection="3d")
                ax.scatter(data[0], data[1], data[2])
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

                origin = pca.means  # データの平均点
                for vec in pca.eigenvectors[:, :3].T:
                    line_x = np.linspace(
                        min(data[2]),
                        max(data[2]),
                        100,
                    )  # 主成分軸の範囲を調整
                    line_y = origin[1] + vec[1] * line_x
                    line_z = origin[2] + vec[2] * line_x
                    ax.plot(
                        origin[0] + vec[0] * line_x,
                        line_y,
                        line_z,
                        color="r",
                        linestyle="--",
                        label="Principal Component",
                    )
                ax.legend()

    else:
        # 回帰曲線を描画する場合の分岐
        if data.shape[0] == 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[0], data[1])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            sorted_indices = np.argsort(data[0])
            plt.plot(data[0][sorted_indices], pred[sorted_indices], color="red")
            plt.legend(["Actual", "Predicted"])

        elif data.shape[0] == 3:
            ax = fig.add_subplot(projection="3d")
            ax.scatter(data[0], data[1], data[2])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            # 回帰平面の描画
            x = np.linspace(min(data[0]), max(data[0]), 50)
            y = np.linspace(min(data[1]), max(data[1]), 50)
            X, Y = np.meshgrid(x, y)
            Z = pred[0] + pred[1] * X + pred[2] * Y  # 平面の方程式
            ax.plot_surface(X, Y, Z, color="red", alpha=0.5)
    plt.show()


class LinearRegression:
    """線形回帰の実装を行うクラス.

    メンバ関数：
        OLS: 最小二乗法の計算を行う関数
        fit: データへの適応を行う関数
        predict: 予測を行う関数
        elastic_net_coordinate_descent: Elastic Net回帰及びLasso回帰を座標降下法で解く関数

    属性：
        M(int): 次数
        w(np.ndarray): 重み, shape = (次数)
        path(int): データファイルのパス.
        data(np.ndarray): データ, shape = (データ数, データの次元数)
        model(str): モデルの種類
    """

    def __init__(self, M: int, path: str, model: str) -> None:
        """初期化関数.

        入力：
            M(int): 次数
            path(str): データファイルのパス
            model(str): モデルの種類
        """
        self.M = M
        self.w = None  # 重み
        self.path = path
        self.data = np.array([])
        self.model = model

    def OLS(self):
        """OLS計算を行う関数.

        最小二乗法を用いて重みを計算する.
        モデルを指定することで，Ridge回帰，Lasso回帰，Elastic Net回帰を行うことができる.

        入力：
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
            M(int): 次元数

        出力：
            w: 重み
        """
        x = self.data[0]  # 入力値
        y = self.data[1]  # 出力値

        # 入力行列 X の構築 (バイアス項を含む)
        X = np.vstack([x**i for i in range(self.M)]).T  # shape = (サンプル数, M)
        # 正規方程式を用いて重みを計算
        match self.model:
            case "normal":
                w = np.linalg.inv(X.T @ X) @ X.T @ y
                return w
            case "Ridge":
                alpha = 0.1  # 正則化パラメータ
                w = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
                return w
            case "Lasso":
                alpha = 0.1  # 正則化パラメータ
                w = self.elastic_net_coordinate_descent(X, y, alpha)
                return w
            case "Elastic":
                alpha = 0.1  # 正則化パラメータ
                l1_ratio = 0.5  # L1正則化の比率（0～1）
                w = self.elastic_net_coordinate_descent(X, y, alpha, l1_ratio)
                return w

    def elastic_net_coordinate_descent(
        self, X, y, alpha, l1_ratio=1.0, max_iter=1000, tol=1e-4
    ):
        """Elastic Net回帰及びLasso回帰を座標降下法で解く関数.

        入力：
            X(np.ndarray): 入力行列, shape = (サンプル数, 特徴量数)
            y(np.ndarray): 出力値, shape = (サンプル数,)
            alpha(float): 正則化パラメータ
            l1_ratio(float): L1正則化の比率（0～1, デフォルトは1.0でLasso回帰）
            max_iter(int): 最大反復回数
            tol(float): 収束判定の閾値

        出力：
            w(np.ndarray): 重みベクトル, shape = (特徴量数,)
        """
        m, n = X.shape
        w = np.zeros(n)  # 初期重みをゼロで初期化

        for iteration in range(max_iter):
            w_old = w.copy()
            for j in range(n):
                # 残差を計算
                residual = y - (X @ w - X[:, j] * w[j])
                # 重みの更新
                rho = X[:, j].T @ residual
                l1_term = alpha * l1_ratio
                l2_term = alpha * (1 - l1_ratio)
                if rho < -l1_term:
                    w[j] = (rho + l1_term) / (X[:, j].T @ X[:, j] + l2_term)
                elif rho > l1_term:
                    w[j] = (rho - l1_term) / (X[:, j].T @ X[:, j] + l2_term)
                else:
                    w[j] = 0

            # 収束判定
            if np.linalg.norm(w - w_old, ord=2) < tol:
                break

        return w

    def fit(self) -> None:
        """データを読み込み、線形回帰を行う関数.

        入力：
            path(str): データファイルのパス
            M(int): 次数

        出力：
            w(np.ndarray): 重み, shape = (次数)
        """
        self.data = read_csv(self.path)

        self.w = self.OLS(self.M, self.data)

    def predict(self) -> np.ndarray:
        """予測を行う関数.

        入力：
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
            M(int): 次元数

        出力：
            y_pred(np.ndarray): 予測値, shape = (次数)
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
        """
        x = self.data[0]
        X = np.vstack([x**i for i in range(self.w.shape[0])]).T
        y_pred = (X @ self.w).flatten()  # 予測値

        return self.data, y_pred


class PCA:
    """主成分分析の実装を行うクラス.

    メンバ関数：
        compute_S: 観測値ベクトルの集合およびその期待値ベクトルから変動行列を計算する関数
        compute_sorted_eigen: 分散共分散行列から固有値および固有ベクトルを計算し，それぞれ固有値の降順に並べて返す関数
        fit: 分散最大基準に基づき，変換行列を学習する関数
        transform: 学習した変換行列によって特徴量ベクトルを射影する関数
        explained_variance_ratio: 累積寄与率を計算する関数
    属性：
        n_components(int): 主成分の数
        A(np.ndarray): 変換行列, shape = (次元数, 次元数)
        eigenvalues(np.ndarray): 固有値, shape = (次元数,)
        eigenvectors(np.ndarray): 固有ベクトル, shape = (次元数, 次元数)
        means(np.ndarray): 期待値ベクトル, shape = (次元数)
    """

    def __init__(
        self,
        n_components: int = 2,
    ):
        """初期化関数.

        入力：
            n_components(int): 主成分の数
        出力：
            None
        """
        self.n_components = n_components
        self.A = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.means = None

    def compute_S(
        self,
        X: np.ndarray,
        means: np.ndarray,
    ) -> np.ndarray:
        """観測値ベクトルの集合およびその期待値ベクトルから変動行列を計算する関数.

        入力：
            X(np.ndarray): 観測値ベクトルの集合, shape = (観測値ベクトルの数, 次元数)
            means(np.ndarray): 期待値ベクトル, shape = (次元数)

        出力：
            S(np.ndarray): 変動行列, shape = (次元数, 次元数)
        """
        S = 0
        for i in range(len(X)):
            S += np.outer((X[i] - means), (X[i] - means).T)
        return S

    def compute_sorted_eigen(
        self,
        cov_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """分散共分散行列から固有値および固有ベクトルを計算し，それぞれ固有値の降順に並べて返す関数.

        入力：
            cov_matrix(np.ndarray): 分散共分散行列, shape = (次元数, 次元数)
        出力：
            eigenvalues(np.ndarray): 固有値, shape = (次元数,)
            eigenvectors(np.ndarray): 固有ベクトル, shape = (次元数, 次元数)
        """
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        index = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[index]
        self.eigenvectors = eigenvectors[:, index]

        return self.eigenvalues, self.eigenvectors

    def fit(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """分散最大基準に基づき，変換行列を学習する関数.

        入力：
            X(np.ndarray): 観測値ベクトルの集合, shape = (観測値ベクトルの数, 次元数)
        出力：
            A(np.ndarray): 変換行列, shape = (次元数, 次元数)
        """
        # 特徴ベクトルの総数を取得
        n = len(X)

        # 特徴ベクトルの分散共分散行列を計算
        self.means = X.mean(axis=1)
        centered_data = X - self.means[:, np.newaxis]
        S = self.compute_S(centered_data.T, np.zeros(centered_data.shape[0])) / n

        # 分散共分散行列から固有値及び固有値ベクトルを計算し，固有値の大きい順に整列
        eigenvalues, eigenvectors = self.compute_sorted_eigen(S)

        # 上位n_components個の固有値ベクトルを変換行列として抽出
        self.A = eigenvectors[: self.n_components]
        return self.A

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """学習した変換行列によって特徴量ベクトルを射影する関数.

        入力：
            X(np.ndarray): 観測値ベクトルの集合, shape = (観測値ベクトルの数, 次元数)
        出力：
            _(np.ndarray): 射影された特徴量ベクトル, shape = (観測値ベクトルの数, 次元数)
        """
        return np.dot(self.A, X)

    def explained_variance_ratio(self) -> np.ndarray:
        """累積寄与率を計算する関数.

        入力：
            なし
        出力：
            cumulative_ratio(np.ndarray): 各主成分の累積寄与率
        """
        total_variance = np.sum(self.eigenvalues)
        explained_variance = self.eigenvalues / total_variance
        cumulative_ratio = np.cumsum(explained_variance)
        return cumulative_ratio


def parse_arguments():
    """コマンドライン引数を解析する関数.

    入力：
        なし

    出力：
        args: 入力のオブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(description="線形回帰・主成分分析の実装")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["linear", "pca"],
        help="実行する処理のモードを指定 (linear: 線形回帰, pca: 主成分分析)",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="データのパス",
    )
    parser.add_argument(
        "--M",
        type=int,
        required=True,
        help="(linearの場合)投影する次元数、(PCAの場合)削減後の次元数を指定",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="normal",
        choices=["normal", "Ridge", "Lasso", "Elastic"],
        help="(線形回帰の場合)モデルを指定（標準：normal）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """メイン関数."""

    args = parse_arguments()

    if args.mode == "pca":
        pca = PCA(n_components=args.M)
        data = read_csv(args.path)
        pca.fit(data)
        if data.shape[0] == args.M:
            make_scatter(
                data,
                f"PCA of {args.path.split('/')[-1].split('.')[0]}",
                pca=pca,
            )
        else:
            transformed_data = pca.transform(data)
            make_scatter(
                transformed_data,
                f"Dimension Division of {args.path.split('/')[-1].split('.')[0]}",
            )

    elif args.mode == "linear":
        linearRegression = LinearRegression(path=args.path, M=args.M, model=args.model)
        linearRegression.fit(args.path)
        data, pred = linearRegression.predict()
        make_scatter(data, "scatter")
        make_scatter(data, "linearRegression", pred=pred)
