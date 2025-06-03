#!/usr/bin/env python
# coding: utf-8

"""kadai3_LR.py.

- 最小二乗法による線形回帰の実装(kadai3_LR.py)
- 実行コマンド
 `$ python3 kadai3_LR.py`

"""

import matplotlib.pyplot as plt
import numpy as np

PLOT_SIZE = 128


class LR:
    """線形回帰に関する操作を取り扱うクラスです。."""

    def __init__(self, inputs: np.ndarray, deg: int | tuple = 1):
        """入力変数を用意します。拡張入力ベクトルの他に、グラフ出力用の配列も用意されます。.

        inputs:
          入力データ
        deg:
          回帰関数の次数 (入力データは(1+degの合計)次元に拡張されます。)
        """
        if type(deg) is int:  # 一次元
            # (入力数) × d+1 に拡張
            input_exts = np.zeros((len(inputs), deg + 1))
            # 1+x+x^2+...+x^deg
            for i in range(deg + 1):
                input_exts[:, i] = inputs**i  # x^i(i>=0)
            # グラフ用に入力軸の数字を生成
            input_field = np.linspace(
                1.1 * np.min(inputs), 1.1 * np.max(inputs), PLOT_SIZE
            )
        else:  # 多次元
            # (入力数) × Σ(d)+1 に拡張
            input_exts = np.zeros((len(inputs[:, 0]), sum(deg) + 1))
            # 1+x+...+x^deg+y+...+y^deg+...
            input_exts[:, 0] = np.ones((len(inputs[:, 0])))  # 1
            input_exts_index = 1
            for i, deg_i in enumerate(deg):  # a ← x, y,...
                for j in range(1, deg_i + 1):
                    input_exts[:, input_exts_index] = inputs[:, i] ** j  # a^j(j>=1)
                    input_exts_index += 1
            # グラフ用に入力軸の数字を生成
            input_dim = np.zeros((len(inputs[0]), PLOT_SIZE))
            for i in range(len(inputs[0])):
                input_dim[i, :] = np.linspace(
                    1.1 * np.min(inputs[:, i]), 1.1 * np.max(inputs[:, i]), PLOT_SIZE
                )
            input_field = np.meshgrid(*input_dim)

        self.inputs = input_exts  # 拡張入力ベクトル
        self.input_field = input_field  # 回帰関数出力用の配列
        self.deg = deg  # 次元
        self.weight = None  # 重み(OLSで計算)

    def OLS(self, outputs: np.ndarray, reg_name: str = "", param: float | tuple = 1):
        """最小二乗法の式に代入し、重みを求めます。.

        outputs:
          出力データ
        reg_name:
          回帰手法(ridge, lasso, elastic net)
          正則化の手法として、Ridge回帰、Lasso回帰、Elastic Netを指定できます。
          何も指定しない場合は通常の最小二乗法による回帰が行われます。
        param:
          回帰手法を指定した場合のパラメータを設定します。
          Elastic Netでは2つのパラメータが必要となります。
        """
        if reg_name.lower() == "ridge":
            if type(param) is tuple:
                param = param[0]  # 最初のみ参照
            # Ridge回帰
            w = (
                np.linalg.inv(
                    self.inputs.T @ self.inputs
                    + param / 2 * np.identity(len(self.inputs[0]))
                )
                @ self.inputs.T
                @ outputs
            )
            self.weight = w
            return w
        elif reg_name.lower() == "lasso":
            if type(param) is tuple:
                param = param[0]  # 最初のみ参照
            # Lasso回帰(座標降下法)
            w = np.zeros(len(self.inputs[0]))
            wpast = np.ones(len(w))
            while np.mean((w - wpast) ** 2) > 10**-10:  # 更新規則
                wpast = np.copy(w)
                w[0] = np.sum(outputs - self.inputs[:, 1:] @ w[1:]) / len(
                    outputs
                )  # バイアスの更新
                for n in range(1, len(w)):  # 重みの更新
                    cond = (
                        outputs - np.delete(self.inputs, n, 1) @ np.delete(w, n)
                    ) @ self.inputs[
                        :, n
                    ]  # 更新の方向を判定
                    if cond > 2 * param:
                        w[n] = (cond - 2 * param) / np.sum(self.inputs[:, n] ** 2)
                    elif cond < -2 * param:
                        w[n] = (cond + 2 * param) / np.sum(self.inputs[:, n] ** 2)
                    else:
                        w[n] = 0
            self.weight = w
            return w
        elif "elastic" in reg_name.lower():
            if type(param) is float:
                param = (param, 0)  # Lasso回帰として処理
            # Elastic Net(座標降下法)
            w = (np.random.random(len(self.inputs[0])) + 0.1) * 10 * max(param)
            wpast = np.zeros(len(w))
            while np.mean((w - wpast) ** 2) > 10**-10:  # 更新規則
                wpast = np.copy(w)
                w[0] = np.sum(outputs - self.inputs[:, 1:] @ w[1:]) / (
                    len(outputs) + param[1]
                )  # バイアスの更新
                for n in range(1, len(w)):  # 重みの更新
                    cond = (
                        outputs - np.delete(self.inputs, n, 1) @ np.delete(w, n)
                    ) @ self.inputs[
                        :, n
                    ]  # 更新の方向を判定
                    if cond > 2 * param[0]:
                        w[n] = (cond - 2 * param[0]) / (
                            np.sum(self.inputs[:, n] ** 2) + param[1]
                        )
                    elif cond < -2 * param[0]:
                        w[n] = (cond + 2 * param[0]) / (
                            np.sum(self.inputs[:, n] ** 2) + param[1]
                        )
                    else:
                        w[n] = 0
            self.weight = w
            return w
        else:
            # 通常
            w = np.linalg.inv(self.inputs.T @ self.inputs) @ self.inputs.T @ outputs
            self.weight = w
            return w

    def output(self, weight: np.ndarray = None):
        """関数出力用の配列を生成します。出力軸の配列と入力軸の配列が出力されます。.

        weight:
          重み (内部で保持されているため、通常は指定する必要はありません。)
        """
        # 出力を生成
        if weight is None:
            weight = self.weight  # デフォルトは内部重み
        if type(self.deg) is int:  # 一次元
            outputs = np.zeros(PLOT_SIZE)
            for i in range(self.deg + 1):
                outputs += weight[i] * self.input_field**i
            return outputs, self.input_field
        else:  # 多次元
            outputs = np.full(self.input_field[0].shape, weight[0])
            weight_index = 1
            for i, deg_i in enumerate(self.deg):
                for j in range(1, deg_i + 1):
                    outputs += weight[weight_index] * self.input_field[i] ** j
                    weight_index += 1
            return outputs, *self.input_field

    def __str__(self):
        """文字列として出力する場合は、回帰関数の文字列を出力します。."""
        if self.weight is None:
            return "Unsolved"
        else:
            if type(self.deg) is int:
                deg = (self.deg,)
            else:
                deg = self.deg
            formula = f"$y={self.weight[0]:.3g}"
            weight_index = 1
            for i, deg_i in enumerate(deg):
                if len(deg) == 1:
                    x_str = "x"
                else:
                    x_str = f"x_{{{i}}}"
                for j in range(1, deg_i + 1):
                    if self.weight[weight_index] > 0:
                        formula += f"+{self.weight[weight_index]:.3g}{x_str}^{{{j}}}"
                    elif self.weight[weight_index] < 0:
                        formula += f"-{-self.weight[weight_index]:.3g}{x_str}^{{{j}}}"
                    weight_index += 1
            formula += "$"
            return formula

    def __repr__(self):
        """formatter対策."""
        return str(self)


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


if __name__ == "__main__":
    # ファイル読み込み
    data1 = np.loadtxt("data/data1.csv", delimiter=",", skiprows=1)
    data2 = np.loadtxt("data/data2.csv", delimiter=",", skiprows=1)
    data3 = np.loadtxt("data/data3.csv", delimiter=",", skiprows=1)

    # 散布図表示
    prepare_scatters(data1, data2, data3)
    plt.savefig("scatters.png")
    plt.show()

    # OLS適用
    deg1 = 1
    deg2 = 3
    deg3 = (2, 2)
    lr1 = LR(data1[:, 0], deg1)
    lr2 = LR(data2[:, 0], deg2)
    lr3 = LR(data3[:, 0:2], deg3)
    lr1.OLS(data1[:, 1])
    lr2.OLS(data2[:, 1])
    lr3.OLS(data3[:, 2])
    # 表示
    ax1, ax2, ax3 = prepare_scatters(data1, data2, data3)
    outputs1, field1 = lr1.output()
    ax1.plot(field1, outputs1, label=lr1)
    ax1.legend(fontsize=6)
    outputs2, field2 = lr2.output()
    ax2.plot(field2, outputs2, label=lr2)
    ax2.legend(fontsize=6)
    outputs3, field3_1, field3_2 = lr3.output()
    surf = ax3.plot_surface(field3_1, field3_2, outputs3, label=lr3, alpha=0.5)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax3.legend(loc="lower right", fontsize=6)
    plt.savefig("OLS.png")
    plt.show()

    # 正則化OLS適用
    reg_name1 = "ridge"
    reg_name2 = "lasso"
    reg_name3 = "elastic net"
    lr1.OLS(data1[:, 1], reg_name1)
    lr2.OLS(data2[:, 1], reg_name2)
    lr3.OLS(data3[:, 2], reg_name3, (1, 1))
    # 表示
    ax1, ax2, ax3 = prepare_scatters(data1, data2, data3)
    outputs1, field1 = lr1.output()
    ax1.plot(field1, outputs1, label=lr1)
    ax1.legend(fontsize=6, title=reg_name1)
    outputs2, field2 = lr2.output()
    ax2.plot(field2, outputs2, label=lr2)
    ax2.legend(fontsize=6, title=reg_name2)
    outputs3, field3_1, field3_2 = lr3.output()
    surf = ax3.plot_surface(field3_1, field3_2, outputs3, label=lr3, alpha=0.5)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax3.legend(loc="lower right", fontsize=6, title=reg_name3)
    plt.savefig("OLSreg.png")
    plt.show()


# ---------------------------------未使用---------------------------------#

# def OLS(inputs:np.ndarray, outputs:np.ndarray, deg:int|tuple=1,
#         reg_name:str="", param:float|tuple=0.1):
#     # 入力変数を用意
#     if type(deg) is int:    # 一次元
#         input_exts=np.zeros((len(inputs),deg+1))    # (入力数) × d+1 に拡張
#         for i in range(deg+1):
#             input_exts[:,i]=inputs**i   # 1+x+x^2+...+x^deg
#     else:   # 多次元
#         input_exts=np.zeros((len(inputs[:,0]),sum(deg)+1))    # (入力数) × Σ(d)+1 に拡張
#         input_exts[:,0]=np.ones((len(inputs[:,0])))
#         input_exts_num=1
#         for i,deg_i in enumerate(deg):
#             for j in range(1,deg_i+1):
#                 input_exts[:,input_exts_num]=inputs[:,i]**j # 1+x+...+x^deg+y+...+y^deg+...
#                 input_exts_num+=1
#     # 式に代入
#     if reg_name.lower()=="ridge":
#         return np.linalg.inv(input_exts.T @ input_exts + param/2*np.identity(len(input_exts[0])))
#                @input_exts.T@outputs
#     elif reg_name.lower()=="lasso":
#         w=(np.random.random(len(input_exts[0]))+0.1)*10*param
#         for repeat in range(10000):
#             w[0]=np.sum(outputs-input_exts@w)/len(input_exts)
#             for n in range(1,len(w)):
#                 cond=(outputs-w[0]-np.delete(input_exts, n, 1)@np.delete(w,n))@input_exts[:,n]
#                 if cond > 2*param:
#                     w[n]=(cond-2*param)/sum(input_exts[:,n]**2)
#                 elif cond < -2*param:
#                     w[n]=(cond+2*param)/sum(input_exts[:,n]**2)
#                 else:
#                     w[n]=0
#         return w
#     elif "elastic" in reg_name.lower():
#         if type(param) is float:
#             print("param must be tuple!")
#         w=(np.random.random(len(input_exts[0]))+0.1)*10*max(param)
#         for repeat in range(10000):
#             w[0]=np.sum(outputs-input_exts@w)/(len(input_exts)+param[1])
#             for n in range(1,len(w)):
#                 cond=(outputs-w[0]-np.delete(input_exts, n, 1)@np.delete(w,n))@input_exts[:,n]
#                 if cond > 2*param[0]:
#                     w[n]=(cond-2*param[0])/(sum(input_exts[:,n]**2)+param[1])
#                 elif cond < -2*param[0]:
#                     w[n]=(cond+2*param[0])/(sum(input_exts[:,n]**2)+param[1])
#                 else:
#                     w[n]=0
#         return w
#     else:
#         return np.linalg.inv(input_exts.T @ input_exts)@input_exts.T@outputs

# def output(inputs:np.ndarray, weight:np.ndarray, deg:int|tuple=1, col:int=100):
#     if type(deg) is int:    # 一次元
#         # 入力軸の数字を生成
#         input_field=np.linspace(1.1*np.min(inputs),1.1*np.max(inputs),col)
#         # 出力を生成
#         outputs=np.zeros(col)
#         for i in range(deg+1):
#             outputs+=weight[i]*input_field**i
#         return outputs, input_field
#     else:   # 多次元
#         # 入力軸の数字を生成
#         input_dim=np.zeros((len(inputs[0]),col))
#         for i in range(len(inputs[0])):
#             input_dim[i,:]=np.linspace(1.1*np.min(inputs[:,i]),1.1*np.max(inputs[:,i]),col)
#         input_field=np.meshgrid(*input_dim)
#         # 出力を生成
#         outputs=np.full(input_field[0].shape,weight[0])
#         weight_num=1
#         for i,deg_i in enumerate(deg):
#             for j in range(1,deg_i+1):
#                 outputs+=weight[weight_num]*input_field[i]**j
#                 weight_num+=1
#         return outputs, *input_field
