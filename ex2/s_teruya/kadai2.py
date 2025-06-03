#!/usr/bin/env python
# coding: utf-8

""".

- 時間領域におけるデジタルフィルターの実現
- 実行コマンド
`$ python3 kadai2.py` (サンプル音声blank.wavが適用されます)
  または `$ python3 kadai2.py (任意のwavファイル)`

"""

import cmath
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, spectrogram


def readwav(filename: str = "blank.wav"):
    """wavファイル読み込み.

    filename:
      wavファイル名(default: Disfigure - Blank [NCS Release] Music provided by NoCopyrightSounds)
    """
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    try:
        return wavfile.read(filename)
    except Exception:
        print("filename is invalid!")
        sys.exit()


def writewav(
    rate: int,
    data: np.ndarray,
    datasub: np.ndarray = None,
    filename: str = "result2.wav",
):
    """wavファイル書き出し.

    rate:
      1秒あたりのサンプル数
    data:
      音声信号
    datasub:
      二つ目の音声信号(ステレオ用、指定無しの場合はモノラルになります)
    filename:
      出力ファイル名(default: "result2.wav")
    """
    if datasub is None:
        data_int16 = np.int16(data / np.max(np.abs(data)) * 32767)
        wavfile.write(filename, rate, data_int16)
    else:
        data_int16 = np.int16(data / np.max(np.abs(data)) * 32767)
        datasub_int16 = np.int16(datasub / np.max(np.abs(datasub)) * 32767)
        multidata = np.array([data_int16, datasub_int16])
        wavfile.write(filename, rate, multidata.T)


class DigitalFilter:
    """デジタルフィルター(LPF, HPF, BPF, BEF)."""

    def __init__(
        self,
        filter_name: str,
        fvals: float | tuple[float],
        window_name: str = "boxcar",
        size: int = 32,
    ):
        """フィルターを時間領域で作成します.

        filiter_name:
          フィルターの名前(LPF, HPF, BPF, BEF)
        fvals:
          周波数の範囲(LPF, HPF → float | BPF, BEF → (float, float))
          範囲は0~np.piの間で、入力数値/np.piの割合で効果を発揮します
        window_name:
          窓関数の名前(default: boxcar)
        size:
          フィルターの大きさ(入力された値の2倍になります、default: 32)
        """
        size_list = np.arange(-size, size)  # フィルターの大きさ(M=2*size)
        window = get_window(window_name, 2 * size, False)  # 窓関数
        # フィルター(窓無し)
        # filter_valsの式はプログラム内で離散時間フーリエ逆変換を行わず、手計算で得られた式に直接代入しています。
        if filter_name in ["LPF", "lpf"]:
            if type(fvals) is tuple:  # 最初の値を適用
                fvals = fvals[0]
            filter_vals = fvals / np.pi * np.sinc(fvals * size_list / np.pi)
        elif filter_name in ["HPF", "hpf"]:
            if type(fvals) is tuple:  # 最初の値を適用
                fvals = fvals[0]
            filter_vals = np.sinc(size_list) - fvals / np.pi * np.sinc(
                fvals * size_list / np.pi
            )
        elif filter_name in ["BPF", "bpf"]:
            if type(fvals) is float:  # LPFとして処理
                print("Warning!: filter was made as LPF")
                filter_name = "LPF"
                filter_vals = fvals / np.pi * np.sinc(fvals * size_list / np.pi)
            else:
                filter_vals = fvals[1] / np.pi * np.sinc(
                    fvals[1] * size_list / np.pi
                ) - fvals[0] / np.pi * np.sinc(fvals[0] * size_list / np.pi)
        elif filter_name in ["BEF", "bef"]:
            if type(fvals) is float:  # HPFとして処理
                print("Warning!: filter was made as HPF")
                filter_name = "HPF"
                filter_vals = np.sinc(size_list) - fvals / np.pi * np.sinc(
                    fvals * size_list / np.pi
                )
            else:
                filter_vals = np.sinc(size_list) - (
                    fvals[1] / np.pi * np.sinc(fvals[1] * size_list / np.pi)
                    - fvals[0] / np.pi * np.sinc(fvals[0] * size_list / np.pi)
                )
        else:
            print("filter name is invalid")
            sys.exit()

        self.filter = filter_vals * window  # フィルター(窓有り)
        self.filter_name = filter_name  # フィルターの名前
        self.filter_size = size  # フィルターの大きさ(正方向)
        self.fvals = fvals  # 周波数の範囲

    def convolution(self, data: np.ndarray):
        """入力された音声信号にフィルターを畳み込みます.

        data:
          音声信号
        """
        result = np.array([0] * len(data))
        zero_ext = np.zeros(len(self.filter))
        data_ext = np.append(zero_ext, data)  # IndexError対策に負領域を0埋め
        for i in range(len(data)):
            # 各iについて畳み込み (result[i] = h[i] * x[i])
            result[i] += self.filter @ np.flipud(data_ext[i : i + len(self.filter)])
        return result

    def plot(self, start: float = 0, stop: float = np.pi, accuracy: float = 0.01):
        """フィルターを周波数領域に復元します.

        start, stop:
          作成するグラフの左端と右端
        accuracy:
          グラフの精度(値が小さいほど細かいです)
        """
        # 離散時間フーリエ変換
        flist = np.arange(start, stop, accuracy)
        transfer = np.zeros(len(flist), dtype=np.complex64)
        for i in range(len(flist)):
            for n in range(-self.filter_size, self.filter_size):
                transfer[i] += self.filter[n + self.filter_size] * cmath.exp(
                    -1j * flist[i] * n
                )
        # プロット
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.abs(transfer))  # 振幅
        ax[0].set_xlabel("f")
        ax[0].set_ylabel("abs")
        ax[0].set_title("振幅特性", fontname="MS Gothic")
        ax[1].plot(np.angle(transfer))  # 位相
        ax[1].set_xlabel("f")
        ax[1].set_ylabel("angle")
        ax[1].set_title("位相特性", fontname="MS Gothic")
        plt.tight_layout()
        plt.savefig(f"{self.filter_name}.png")  # 保存
        plt.show()


if __name__ == "__main__":
    # フィルター作成
    filter_name = "LPF"
    dig_filter = DigitalFilter(filter_name, (1.0, 1.5), window_name="hann")
    dig_filter.plot()

    # フィルターで畳み込み
    r, data = readwav()  # 読み込み
    filtered_datasub = None
    if len(data.shape) == 2:  # ステレオの場合は分割
        datasub = data[:, 1]
        data = data[:, 0]
        filtered_datasub = dig_filter.convolution(datasub)  # 変換
    filtered_data = dig_filter.convolution(data)  # 変換
    writewav(r, filtered_data, filtered_datasub)  # wavファイル保存

    # スペクトログラム表示(ステレオの場合は片方のみ)
    fig, ax = plt.subplots(2, 1)
    # 変換前
    fB, tB, spektremB = spectrogram(data, r)
    ax[0].pcolormesh(tB, fB, np.log1p(np.abs(spektremB)), shading="auto")
    ax[0].set_xlabel("s")
    ax[0].set_ylabel("Hz")
    ax[0].set_title("変換前", fontname="MS Gothic")
    # 変換後
    fA, tA, spektremA = spectrogram(filtered_data, r)
    ax[1].pcolormesh(tA, fA, np.log1p(np.abs(spektremA)), shading="auto")
    ax[1].set_xlabel("s")
    ax[1].set_ylabel("Hz")
    ax[1].set_title("変換後", fontname="MS Gothic")
    plt.tight_layout()
    plt.savefig("result2.png")  # 保存
    plt.show()


# --------------------------------未使用-------------------------------- #


# def convolution(h: np.ndarray, x: np.ndarray):
#     """
#     畳み込み演算

#     h:
#       畳み込む配列
#     x:
#       畳み込まれる配列
#     """
#     result = np.array([0] * len(x))
#     zero_ext = np.zeros(len(h))
#     x_ext = np.append(zero_ext, x)  # IndexError対策に負領域を0埋め
#     for i in range(len(x)):
#         # 各iについて畳み込み (result[i] = h[i] * x[i])
#         result[i] += h @ np.flipud(x_ext[i : i + len(h)])
#     return result
