#!/usr/bin/env python
# coding: utf-8

"""kadai6.py.

-
- 実行コマンド
 `$ python kadai6.py`

"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn
from sklearn.metrics import confusion_matrix


class HMM:
    """隠れマルコフモデルによるモデル予測."""

    def __init__(self, data: dict, dataname: str = "data"):
        """HMMインスタンスを生成します。.

        Args:
            data(dict):pickleから読み込まれたデータ
            dataname(str):データ名(default:data)

        Note:
            dataの構造は以下のようになっています。
            data #[次元数, 次元数, ...]
            ├─answer_models # 出力系列を生成したモデル（正解ラベル）[p,]
            ├─output # 出力系列 [p, t]
            └─models # 定義済みHMM
              ├─PI # 初期確率 [k, l, 1]
              ├─A # 状態遷移確率行列 [k, l, l]
              └─B # 出力確率 [k, l, n]
        """
        try:
            # データ
            self.dataname = dataname
            self.answer_models = np.asarray(data["answer_models"])
            self.output = np.asarray(data["output"])
            self.PI = np.asarray(data["models"]["PI"])
            self.A = np.asarray(data["models"]["A"])
            self.B = np.asarray(data["models"]["B"])
            # データの長さ情報
            self.len_output = len(data["output"][0])  # 出力1個あたりの長さ
            self.model_total = len(data["models"]["B"])  # モデル数
            self.state_total = len(data["models"]["B"][0])  # 状態の数
            self.output_total = len(data["models"]["B"][0][0])  # 出力記号の数
        except Exception:
            print(f"{dataname}はHMMに対応していません")
            raise

    def forward(self, output: np.ndarray):
        """forwardアルゴリズムによって、P(O|M)を求めます。.

        Args:
            output(NDArray):出力系列

        Returns:
            percents(NDArray):各モデルのP(O|M)
        """
        # 初期化
        forwardP = self.PI[:, :, 0] * self.B[:, :, output[0]]
        # 再帰計算
        for t in range(1, self.len_output):
            forwardP = (
                np.einsum("mi, mij -> mj", forwardP, self.A) * self.B[:, :, output[t]]
            )
        # 確率計算
        return forwardP.sum(axis=1)

    def viterbi(self, output: np.ndarray):
        """viterbiアルゴリズムによって、尤もらしい状態系列とその確率を求めます。.

        Args:
            output(NDArray):出力系列

        Returns:
            Pmax(NDArray):各モデルの、尤もらしい状態系列の確率
            state_max(NDArray):各モデルの尤もらしい状態系列
        """
        # 初期化
        Pmax_state = self.PI[:, :, 0] * self.B[:, :, output[0]]
        # 再帰計算
        past_arg = np.zeros((self.model_total, self.len_output, self.state_total))
        for t in range(1, self.len_output):
            past_val = Pmax_state[:, :, None] * self.A
            Pmax_state = np.max(past_val, axis=1) * self.B[:, :, output[t]]
            past_arg[:, t] = np.argmax(past_val, axis=1)
        # 再帰計算の終了
        Pmax = np.max(Pmax_state, axis=1)
        state_max = np.zeros((self.model_total, self.len_output), dtype=int)
        state_max[:, -1] = np.argmax(Pmax_state, axis=1)
        # 最適状態遷移系列の復元
        for t in range(self.len_output - 1, 0, -1):
            for i in range(self.model_total):
                state_max[i, t - 1] = past_arg[i, t, state_max[i, t]]
        return Pmax, state_max

    def predict(self, mode="F"):
        """各出力系列がどのモデルから生成されたかを予測します。.

        Args:
            mode(str):適用するアルゴリズムを選択します。
                文字列内にFがあればforwardアルゴリズム、
                Bがあればbackwardアルゴリズム、
                Vがあればviterbiアルゴリズムが実行されます。

        Returns:
            pred_dict(dict):各アルゴリズムの予測結果
                (key : forward, backward, viterbi)

        Note:
            backwardアルゴリズムは実装されていません。
        """
        pred_dict = {}
        if "F" in mode:
            pred_dict["forward"] = []
            start = time.time()
            # 各出力ごとにforwardを実行
            for output in self.output:
                percents = self.forward(output)
                # P(M)が一定の時、argmaxP(M|O)=argmaxP(O|M)
                pred_dict["forward"].append(np.argmax(percents))
            end = time.time()
            print(f"{self.dataname}: Forward algorithm finished in {end - start:.4g}")
        if "B" in mode:
            print("Not available backward algorithm yet!")
        if "V" in mode:
            pred_dict["viterbi"] = []
            start = time.time()
            # 各出力ごとにviterbiを実行
            for output in self.output:
                percents, _ = self.viterbi(output)
                # P(M)が一定の時、argmax(max_Q P(M,Q|O))=argmax(max_Q P(O,Q|M))
                pred_dict["viterbi"].append(np.argmax(percents))
            end = time.time()
            print(f"{self.dataname}: Viterbi algorithm finished in {end - start:.4g}")
        return pred_dict

    def plot_matrix(self, pred_dict: dict):
        """各アルゴリズムの予測結果をもとに混同行列を作成します。.

        Args:
            pred_dict(dict):各アルゴリズムの予測結果
                (key : forward, backward, viterbi)
        """
        if len(pred_dict) >= 4:  # 最大3つ
            print("plot_matrix error: pred_dict has too many keys")
            return
        fig = plt.figure(figsize=(4 * len(pred_dict), 4.5))
        # 各アルゴリズムに対して混同行列を作成
        for n, (name, pred) in enumerate(pred_dict.items(), 1):
            ax = fig.add_subplot(1, len(pred_dict), n)  # 左から順に
            # 混同行列表示
            cm = confusion_matrix(self.answer_models, pred)
            seaborn.heatmap(cm, ax=ax, annot=True, cbar=False)
            # 行列周りの表示
            ax.set_xlabel("Predict model")
            ax.set_ylabel("Actual model")
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            acc = (
                100
                * np.count_nonzero(self.answer_models == np.asarray(pred))
                / len(pred)
            )  # 正解率
            ax.set_title(f"{name} algorithm\n(acc: {acc}%)")
        plt.tight_layout()
        os.makedirs("picture", exist_ok=True)
        plt.savefig(f"picture/{self.dataname}.png")
        plt.show()


if __name__ == "__main__":
    # 読み込み
    if len(sys.argv) == 1:  # 指定がない場合
        data1 = pickle.load(open("data1.pickle", "rb"))
        data2 = pickle.load(open("data2.pickle", "rb"))
        data3 = pickle.load(open("data3.pickle", "rb"))
        data4 = pickle.load(open("data4.pickle", "rb"))
        dataset = {"data1": data1, "data2": data2, "data3": data3, "data4": data4}
    else:
        dataset = {}
        for filename in sys.argv[1:]:
            try:
                data = pickle.load(open(filename, "rb"))
                dataset[filename.split("\\")[-1].split(".")[0]] = (
                    data  # .pickle以前をkeyに
                )
            except FileNotFoundError:
                print(f"{filename} not found")

    for name, data in dataset.items():
        # 予測
        hmm = HMM(data, dataname=name)
        pred = hmm.predict("FV")
        # 表示
        hmm.plot_matrix(pred)
