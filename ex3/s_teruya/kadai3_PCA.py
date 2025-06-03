#!/usr/bin/env python
# coding: utf-8

"""kadai3_PCA.py.

- 主成分分析の実装(kadai3_PCA.py)
- 実行コマンド
 `$ python3 kadai3_PCA.py`

"""

import matplotlib.pyplot as plt
import numpy as np

PLOT_SIZE = 128


def PCA(data: np.ndarray, reduce: int = -1):
    """主成分分析を行い、変換行列と変換後のデータを返します。.

    data:
      入力データ
    reduce:
      削減する次元数 (指定がない場合は累積寄与率が90%以上になるまで削減されます。)
    """
    # 共分散行列を求める
    m = np.mean(data, axis=0)
    data_dif = data - m
    cov = data_dif.T @ data_dif / len(data)
    # QR法で計算:
    eig_mat = cov
    eig_vec = np.identity(len(cov))
    for repeat in range(10000):
        q, r = np.linalg.qr(eig_mat)  # QR分解
        eig_mat_new = r @ q
        eig_vec = eig_vec @ q  # 固有ベクトルに近づく
        if np.linalg.norm(np.diag(eig_mat_new) - np.diag(eig_mat)) < 1e-10:
            break
        eig_mat = eig_mat_new
    eig_val = np.diag(eig_mat)
    # 変換行列を生成
    eig_val.flags.writeable = True
    transform_arg = []
    if reduce >= 0:
        for i in range(min(len(eig_vec), len(eig_vec) - reduce)):
            arg = np.argmax(eig_val)
            transform_arg.append(arg)  # 変換行列に追加
            eig_val[arg] = np.min(eig_val) - 1  # 取り出したら選ばれないようにする
    else:
        # reduceの指定がない場合は累積寄与率90%以上で
        ev_sum = np.sum(eig_val)
        ev_sum_i = 0
        for i in range(len(eig_vec)):
            arg = np.argmax(eig_val)
            transform_arg.append(arg)  # 変換行列に追加
            ev_sum_i += np.max(eig_val)
            eig_val[arg] = np.min(eig_val) - 1  # 取り出したら選ばれないようにする
            if ev_sum_i / ev_sum >= 0.9:  # 累積寄与率90%以上
                break
    transform = eig_vec[:, transform_arg]
    # 変換
    data_reduce = transform.T @ data.T
    return transform, data_reduce


def prepare_scatters(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray):
    """data1~data3の散布図を用意する専用関数です。3つのAxesを返します。.

    data1, data2:
      2次元入力データ
    data3:
      3次元入力データ
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 5)  # 表示場所を用意
    ax1 = fig.add_subplot(gs[0:2])  # 左上に
    ax1.set_title("data1")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(data1[:, 0], data1[:, 1], s=10)
    ax2 = fig.add_subplot(gs[5:7])  # 左下に
    ax2.set_title("data2")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.scatter(data2[:, 0], data2[:, 1], s=10)
    ax3 = fig.add_subplot(gs[:, 2:], projection="3d")  # 右に3D散布図
    ax3.set_title("data3")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("y")
    ax3.scatter(data3[:, 0], data3[:, 1], data3[:, 2])
    fig.tight_layout()
    return ax1, ax2, ax3


def plotPC(ax: plt.Axes, transform: np.ndarray, data: np.ndarray):
    """変換行列に応じて、主成分軸(主成分平面)をプロットします。.

    ax:
      描画する対象Axes
    tramsform:
      変換行列
    data:
      入力データ (プロットの中心決定や次元参照に用いられます)
    """
    # プロットの中心
    center = [
        (np.max(data[:, x]) + np.min(data[:, x])) / 2 for x in range(len(data[0]))
    ]
    if len(data[0]) == 2:
        # 二次元データ
        xlist = np.linspace(data[:, 0].min(), data[:, 0].max(), PLOT_SIZE)  # x軸
        if len(transform[0]) == 1:
            # 軸が一つの場合
            ylist = (
                transform[1] / transform[0] * (xlist - (center[0])) + center[1]
            )  # y軸
            ax.plot(xlist, ylist)
        else:
            # 軸が二つ以上の場合(一応)
            for i in range(len(transform[0])):
                ylist = (
                    transform[i, 1] / transform[i, 0] * (xlist - (center[0]))
                    + center[1]
                )
                ax.plot(xlist, ylist)
    elif len(data[0]) == 3:
        # 三次元データ
        if len(transform[0]) == 1:
            # 軸が一つの場合(スケールを考慮してないので表示がおかしくなるときがあります)
            zlist = np.linspace(data[:, 2].min(), data[:, 2].max(), PLOT_SIZE)  # z軸
            ylist = (
                transform[1] / transform[2] * (zlist - (center[2])) + center[1]
            )  # y軸
            xlist = (
                transform[0] / transform[2] * (zlist - (center[2])) + center[0]
            )  # x軸
            ax.plot(xlist, ylist, zlist)
        elif len(data[0]) == 3 and len(transform[0]) == 2:
            # 軸が二つの場合(平面)
            nm_vec = np.cross(transform[:, 0], transform[:, 1])  # 法線ベクトル
            xlist = np.linspace(data[:, 0].min(), data[:, 0].max(), PLOT_SIZE)  # x軸
            ylist = np.linspace(data[:, 1].min(), data[:, 1].max(), PLOT_SIZE)  # y軸
            x_map, y_map = np.meshgrid(xlist, ylist)
            z_map = (
                nm_vec[0] * (x_map - center[0]) + nm_vec[1] * (y_map - center[1])
            ) / abs(nm_vec[2]) + center[
                2
            ]  # z軸
            ax.plot_surface(x_map, y_map, z_map, alpha=0.5)
        else:
            # 軸が三つ以上の場合(一応)
            zlist = np.linspace(data[:, 2].min(), data[:, 2].max(), PLOT_SIZE)
            for i in range(len(transform[0])):
                ylist = (
                    transform[i, 1] / transform[i, 2] * (zlist - (center[2]))
                    + center[1]
                )
                xlist = (
                    transform[i, 0] / transform[i, 2] * (zlist - (center[2]))
                    + center[0]
                )
                ax.plot(xlist, ylist, zlist)
    else:
        print("too many axes")


if __name__ == "__main__":
    # ファイル読み込み
    data1 = np.loadtxt("data/data1.csv", delimiter=",", skiprows=1)
    data2 = np.loadtxt("data/data2.csv", delimiter=",", skiprows=1)
    data3 = np.loadtxt("data/data3.csv", delimiter=",", skiprows=1)
    data4 = np.loadtxt("data/data4.csv", delimiter=",")

    # 変換行列を生成し変換
    transform1, _ = PCA(data1, 1)
    transform2, _ = PCA(data2, 1)
    transform3, data3_reduce = PCA(data3, 1)

    # 軸を表示
    ax1, ax2, ax3 = prepare_scatters(data1, data2, data3)
    plotPC(ax1, transform1, data1)
    plotPC(ax2, transform2, data2)
    plotPC(ax3, transform3, data3)
    plt.savefig("PCA_ax.png")
    plt.show()

    # 2次元散布図を表示
    plt.scatter(data3_reduce[0], data3_reduce[1])
    plt.title("次元削減後のdata3", fontname="MS Gothic")
    plt.xlabel("x_reduce")
    plt.ylabel("y_reduce")
    plt.savefig("PCA_scatter.png")
    plt.show()

    # 削減後の次元を求める
    _, data4_reduce = PCA(data4)
    print(f"累積寄与率90%以上: {len(data4_reduce)}次元")
