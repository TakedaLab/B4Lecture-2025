"""HMMのアルゴリズム比較

隠れマルコフモデルで出力された結果をもとに，forwardアルゴリズムおよび
Viterbiアルゴリズムの精度を比較する.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle


class HMM:
    """HMMを利用したアルゴリズムの比較を行うクラス.

    メンバ関数：
        - _load_pickle: pickleファイルを読み込み，HMMのパラメータを初期化する.
        - _forward_algorithm: Forward Algorithmを実装する関数.
        - _viterbi_algorithm: Viterbi Algorithmを実装する関数.
        - compare: Forward AlgorithmとViterbi Algorithmの結果を比較する関数.
        - visualize: 混合行列を可視化する関数.

    属性：
        - file_path: pickleファイルのパス
        - pi: 初期状態分布
        - A: 状態遷移行列
        - B: 観測確率行列
        - answer_models: 正解モデル
        - output: 観測された出力系列
        - confusion_matrix: 混合行列
        - accuracy: 精度
        - time: 実行時間
    """

    def __init__(self, file_path):
        """初期化関数.

        HMMの初期化を行う関数. 変数の受領及び初期化を行う.
        入力：
            file_pat(str): pickleファイルのパス
        """
        self.file_path = file_path
        self.pi = None
        self.A = None
        self.B = None
        self.answer_models = None
        self.output = None

        self.confusion_matrix = [0, 0]
        self.accuracy = [0, 0]
        self.time = [0, 0]

    def _load_pickle(self):
        """pickleファイルを読み取る関数.

        pickleファイルからHMMのパラメータを読み込み，初期化する．
        入力：
            - self.file_path(str): pickleファイルのパス
        出力：
            - self.pi(np.array): 初期状態分布 shape=(k, l, 1)
            - self.A(np.array): 状態遷移行列 shape=(k, k, l)
            - self.B(np.array): 観測確率行列 shape=(k, l, n)
            - self.answer_models(np.array): 正解モデル shape=(p,)
            - self.output(np.array): 観測された出力系列 shape=(p, T) 
        """
        with open(self.file_path, "rb") as file:
            data = pickle.load(file)

        self.pi = np.array(data["models"]["PI"])
        self.A = np.array(data["models"]["A"])
        self.B = np.array(data["models"]["B"])
        self.answer_models = np.array(data["answer_models"])
        self.output = np.array(data["output"])

    def _forward_algorithm(self):
        """Forward Algorithmを実装する関数
        Forward Algorithmを用いて、観測された出力系列の確率を計算する。

        入力：
            - self.pi: 初期状態分布 shape=(k, l, 1)
            - self.A: 状態遷移行列 shape=(k, k, l)
            - self.B: 観測確率行列 shape=(k, l, n)
            - self.output: 観測された出力系列 shape=(p, T)
        出力：
            - probability: 観測された出力系列の確率 shape=(p, k)
        """
        k, s, _ = self.B.shape
        _, T = self.output.shape
        probability = []

        for output in self.output:
            alpha = np.zeros((k, T, s))
            for i in range(s):
                alpha[:, 0, i] = self.pi[:, i, 0] * self.B[:, i, output[0]]

            for t in range(1, T):
                for j in range(s):
                    alpha[:, t, j] = (
                        np.sum(alpha[:, t-1, :] * self.A[:, :, j], axis=1)
                        * self.B[:, j, output[t]]
                    )
            probability.append(np.sum(alpha[:, -1, :], axis=1))

    def _viterbi_algorithm(self):
        """Viterbi Algorithmを実装する関数
        Viterbi Algorithmを用いて、最も確率の高い隠れ状態系列を推定する。

        入力：
            - self.pi: 初期状態分布 shape=(k, l, 1)
            - self.A: 状態遷移行列 shape=(k, k, l)
            - self.B: 観測確率行列 shape=(k, l, n)
            - self.output: 観測された出力系列 shape=(p, T)
        出力：
            - p_hat: 最も確率の高い隠れ状態系列の確率 shape=(p, k)
            - q_hat: 最も確率の高い隠れ状態系列 shape=(p, k, T)
        """
        k, s, _ = self.B.shape
        _, T = self.output.shape
        p_hats = []
        q_hats = []

        for output in self.output:
            delta = np.zeros((k, T, s))
            psi = np.zeros((k, T, s), dtype=int)
            for i in range(s):
                delta[:, 0, i] = self.pi[:, i, 0] * self.B[:, i, output[0]]

            for t in range(1, T):
                for j in range(s):
                    temp = delta[:, t - 1, :] * self.A[:, :, j]
                    delta[:, t, j] = np.max(temp, axis=1) * self.B[:, j, output[t]]
                    psi[:, t, j] = np.argmax(temp, axis=1)

            p_hat = np.max(delta[:, -1, :], axis=1)
            q_hat = np.zeros((k, T), dtype=int)
            q_hat[:, -1] = np.argmax(delta[:, -1, :], axis=1)

            for t in range(T - 2, -1, -1):
                q_hat[:, t] = psi[np.arange(k), t + 1, q_hat[:, t + 1]]

            p_hats.append(p_hat)
            q_hats.append(q_hat)

    def compare(self):
        """アルゴリズム比較を行う関数

        混合行列と精度を計算し，Forward AlgorithmとViterbi Algorithmの結果を比較する．
        入力：
            なし
        出力：
            - self.confusion_matrix: 混合行列 shape=(2, k, k)
            - self.accuracy: 精度 shape=(2,)
            - self.time: 実行時間 shape=(2,)
        """

        start_forward = time.time()
        forward_prob = self._forward_algorithm()
        end_forward = time.time()

        start_viterbi = time.time()
        viterbi_prob, _ = self._viterbi_algorithm()
        end_viterbi = time.time()

        p = self.output.shape[0]
        pred_forward = np.argmax(forward_prob, axis=1)
        pred_viterbi = np.argmax(viterbi_prob, axis=1)

        confusion_matrix_viterbi = np.zeros(
            (self.pi.shape[0], self.pi.shape[0]), dtype=int
        )
        confusion_matrix_forward = np.zeros_like(confusion_matrix_viterbi)

        for i in range(p):
            true = self.answer_models[i]
            confusion_matrix_viterbi[true, pred_viterbi[i]] += 1
            confusion_matrix_forward[true, pred_forward[i]] += 1

        accuracy_viterbi = (
            np.trace(confusion_matrix_viterbi) / np.sum(confusion_matrix_viterbi)
        )
        accuracy_forward = (
            np.trace(confusion_matrix_forward) / np.sum(confusion_matrix_forward)
        )

        print("Confusion Matrix (Viterbi):\n", confusion_matrix_viterbi)
        print("Accuracy (Viterbi):", accuracy_viterbi)
        print("Confusion Matrix (Forward):\n", confusion_matrix_forward)
        print("Accuracy (Forward):", accuracy_forward)

        self.confusion_matrix[0] = confusion_matrix_forward
        self.confusion_matrix[1] = confusion_matrix_viterbi
        self.accuracy[0] = accuracy_forward
        self.accuracy[1] = accuracy_viterbi
        self.time[0] = end_forward - start_forward
        self.time[1] = end_viterbi - start_viterbi

    def visualize(self):
        """混合行列を可視化する関数.
        混合行列を可視化し，精度と実行時間を表示する．

        入力：
            - self.confusion_matrix: 混合行列 shape=(2, k, k)
            - self.accuracy: 精度 shape=(2,)
            - self.time: 実行時間 shape=(2,)
        出力：
            - None
        """

        for confusion_matrix, accuracy, cost_time, title in zip(
            self.confusion_matrix,
            self.accuracy,
            self.time,
            ["Forward", "Viterbi"],
        ):
            k = confusion_matrix.shape[0]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.matshow(confusion_matrix, cmap="Blues")
            ax.set_xticks(np.arange(k))
            ax.set_yticks(np.arange(k))
            for (i, j), val in np.ndenumerate(confusion_matrix):
                text_color = "white" if val > confusion_matrix.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    int(val),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )
            ax.set_xticklabels([i + 1 for i in range(k)], fontsize=12)
            ax.set_yticklabels([i + 1 for i in range(k)], fontsize=12)
            ax.set_xlabel("Predicted model", fontsize=12)
            ax.xaxis.set_label_position("top")
            ax.set_ylabel("Actual model", fontsize=12)
            ax.set_title(
                f"{title} Confusion Matrix\n(Acc. {accuracy * 100:.0f}%, Time. {cost_time:.3f}sec)",
                fontsize=12,
            )
            fig.tight_layout()
            plt.savefig(
                f"{self.file_path.split('.')[0]}_{title}.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.show()


def parse_args():
    """コマンドライン引数を解析する関数.

    入力：
        なし
    出力：
        args: 入力のオブジェクト
    """
    parser = argparse.ArgumentParser(description="HMM Implementation")
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the pickle file containing HMM data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    hmm = HMM(args.file_path)
    hmm.compare()
    hmm.visualize()
