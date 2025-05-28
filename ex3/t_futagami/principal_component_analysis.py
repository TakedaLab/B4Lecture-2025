"""
主成分分析 (PCA) モジュール.

機能:
  - 入力データに対する PCA 実行
  - 累積寄与率が指定閾値（デフォルト 90%）を超える主成分数の算出
  - 2D/3D データへの主成分軸描画
  - 主成分空間への射影結果の散布図表示
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt


class PCA:
    """
    主成分分析 (PCA) クラス.

    Attributes:
        n_components (int): 主成分の数
        mean_ (np.ndarray): 平均ベクトル
        components_ (np.ndarray): 主成分ベクトル
        explained_variance_ (np.ndarray): 各主成分の分散
        explained_variance_ratio_ (np.ndarray): 各主成分の寄与率
        cumulative_variance_ratio_ (np.ndarray): 累積寄与率
    """

    def __init__(self, n_components=None):
        """
        PCA クラスの初期化.

        Parameters:
            n_components (int): 主成分の数 (None の場合は全成分)
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        主成分分析を実行し、主成分ベクトルと寄与率を計算する.

        Parameters:
            X (np.ndarray): 入力データ (shape = [n_samples, n_features])
        Returns:
            self (PCA): 学習済みの PCA オブジェクト
        """
        X = np.asarray(X)
        # サンプル数と特徴量数
        n_samples, n_features = X.shape

        # 平均ベクトルを計算・保存
        self.mean_ = X.mean(axis=0)

        # 中心化
        X_centered = X - self.mean_

        # 共分散行列を計算
        cov = np.cov(X_centered, rowvar=False)

        # 固有値・固有ベクトルを計算（昇順で返るので降順にソート）
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 主成分数の設定（None の場合は全成分を使用）
        if self.n_components is None or self.n_components > n_features:
            self.n_components = n_features

        # 主成分ベクトル
        # 各行が1成分分のベクトル（shape = [n_components, n_features]）
        self.components_ = eigvecs[:, : self.n_components].T

        # 各主成分の分散（固有値）
        self.explained_variance_ = eigvals[: self.n_components]

        # 分散寄与率
        total_var = eigvals.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # 累積寄与率
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        データを学習済みのPCAで主成分空間に射影.

        Returns:
            Z (np.ndarray): 圧縮後のデータ (shape=[n_samples, n_components])
        """
        X = np.asarray(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        PCAを実行し、データを主成分空間に射影.

        Parameters:
            X (np.ndarray): 入力データ (shape = [n_samples, n_features])
        Returns:
            Z (np.ndarray): 圧縮後のデータ (shape=[n_samples, n_components])
        """
        return self.fit(X).transform(X)

    def plot_pca_2d(self, X: np.ndarray):
        """
        n次元データを上位2主成分に圧縮して2次元散布図を表示します.

        Parameters:
            X (np.ndarray): 入力データ (shape = [n_samples, n_features])
        """
        # PCA 未実行なら先に fit
        Z = self.fit_transform(X) if self.mean_ is None else self.transform(X)
        # 上位2成分を散布図表示
        plt.scatter(Z[:, 0], Z[:, 1], s=20, alpha=0.7, label="Projected data")
        plt.xlabel(f"PC1 ({self.explained_variance_ratio_[0] * 100:.1f}%)")
        plt.ylabel(f"PC2 ({self.explained_variance_ratio_[1] * 100:.1f}%)")
        plt.legend()

    def plot_components(self, X: np.ndarray, pcvecs=None):
        """
        元のデータ散布図に，主成分軸をプロット.

        Parameters:
            X (np.ndarray): 入力データ (shape = [n_samples, n_features])
            pc (tuple[int,int]): 描画する主成分のインデックス（1始まり）
        """
        n_samples, n_features = X.shape
        if self.mean_ is None:
            self.fit(X)

        if pcvecs is None:
            pcvecs = self.components_.T

        n_samples, n_features = X.shape
        center = self.mean_
        if n_features == 2:
            plt.scatter(X[:, 0], X[:, 1])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim(min(X[:, 0]), max(X[:, 0]))
            plt.ylim(min(X[:, 1]), max(X[:, 1]))

            span = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]))
            t = np.linspace(-span, span, 100)
            # 上位2主成分軸を点線で描画
            for k in range(min(2, self.n_components)):
                v = self.components_[k]  # 第 k+1 主成分ベクトル
                x_line = center[0] + v[0] * t
                y_line = center[1] + v[1] * t
                ratio = self.explained_variance_ratio_[k] * 100
                plt.plot(x_line, y_line, "--", label=f"PC{k + 1} ({ratio:.1f}%)")
            plt.legend()

        elif n_features == 3:
            ax = plt.axes(projection="3d")
            ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], label="data")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(min(X[:, 0]), max(X[:, 0]))
            ax.set_ylim(min(X[:, 1]), max(X[:, 1]))
            ax.set_zlim(min(X[:, 2]), max(X[:, 2]))
            span = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2]))
            t = np.linspace(-span, span, 100)
            # 上位3主成分軸を点線で描画
            for k in range(min(3, self.n_components)):
                v = self.components_[k]
                x_line = center[0] + v[0] * t
                y_line = center[1] + v[1] * t
                z_line = center[2] + v[2] * t
                ratio = self.explained_variance_ratio_[k] * 100
                ax.plot(x_line, y_line, z_line, "--", label=f"PC{k + 1} ({ratio:.1f}%)")
            ax.legend()
        else:
            raise ValueError("2次元または3次元のデータをプロットする必要があります。")

    def select_components_by_threshold(self, threshold):
        """
        寄与率の閾値を指定して主成分を選択する.

        Parameters:
            threshold (float): 寄与率の閾値
        Returns:
            selected_components (np.ndarray): 選択された主成分ベクトル
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCAを実行していません。fitメソッドを呼び出してください。")

        # 寄与率が閾値以上の主成分を選択
        selected_components = self.components_[
            self.explained_variance_ratio_ >= threshold
        ]
        return selected_components


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.

    入力データのパスを取得.
    """
    parser = argparse.ArgumentParser(description="主成分分析を実行します。")
    parser.add_argument(
        "input_data",
        type=str,
        help="入力データのパスを指定してください。",
    )
    parser.add_argument(
        "n_components",
        type=int,
        default=None,
        help="主成分の数を指定します。デフォルトは全成分です。",
    )

    args = parser.parse_args()
    # 入力の検証
    if not args.input_data.endswith(".csv"):
        parser.error("入力データはCSVファイルである必要があります。")

    return args


def main():
    """
    メイン関数.

    コマンドライン引数を解析し、データを読み込み、PCAを実行します.
    """
    args = parse_arguments()
    X = np.genfromtxt(
        args.input_data, delimiter=",", skip_header=1
    )  # CSVファイルを読み込み
    pca = PCA(n_components=args.n_components)
    pca.fit(X)  # PCAを実行

    n_components = np.argmax(pca.cumulative_variance_ratio_ >= 0.9) + 1
    print(f"累積寄与率が90%以上となる主成分の数: {n_components}")

    pca.plot_components(X)  # 主成分をプロット
    plt.title("data")
    plt.show()  # プロットを表示

    pca.plot_pca_2d(X)  # 2次元散布図を表示
    plt.title("Dimensionality Reduction of data")
    plt.show()  # プロットを表示


if __name__ == "__main__":
    main()
