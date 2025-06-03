#!/usr/bin/env python
# coding: utf-8

"""
    -STFT変換によるスペクトログラム表示とISTFT逆変換による音声復元の実装
    -実行コマンド
     $ python3 kadai1.py (サンプル音声blank.wavが適用されます)
     または $ python3 kadai1.py (任意のwavファイル)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from scipy.signal import get_window

onlyOne = False
TRANGE = 0.2
OVERRATE = 0.5  # 0.1にすると雑音が入りやすくなりました


def stft(
    data: np.ndarray, rate: float, trange: float = TRANGE, overrate: float = OVERRATE
):
    """スペクトログラム変換 trangeは区切り間隔、overrateは左右のオーバーラップ率."""
    spektrem = np.array([[]])
    for i in range(int(len(data) / rate / trange)):
        # 読み出し
        try:
            if i == 0:
                shortWave = data[: int((1 + overrate) * trange * rate)]
            else:
                shortWave = data[
                    int((i - overrate) * trange * rate) : int(
                        (i + overrate + 1) * trange * rate
                    )
                ]
        except IndexError:
            shortWave = data[int((i - 0.01) * trange * rate) :]
        # 変換
        # window = np.sin(np.linspace(0, 2*np.pi, len(shortwave)+2)[1:-1])    # sin窓
        window = get_window("hann", len(shortWave) + 1)[1:]  # hann窓
        windoWave = shortWave * window
        spek = fft(windoWave)  # FFTで変換
        # 結合
        if spektrem.shape == (0, 0):
            spektrem = np.array([spek])
        else:
            dif = len(spektrem[-1]) - len(spek)
            if dif > 0:
                zeros = np.zeros(dif)
                spektrem = np.block([[spektrem], [spek, zeros]])
            elif dif < 0:
                zeros = np.zeros((len(spektrem), -dif))
                spektrem = np.block([[spektrem, zeros], [spek]])
            else:
                spektrem = np.block([[spektrem], [spek]])
    # スペクトログラム表示用
    tnum, fnum = spektrem.shape
    tlist = np.linspace(0, (tnum - 1) * trange, tnum)
    flist = np.linspace(0, fnum - 1, fnum)
    return tlist, flist, spektrem


def istft(
    spektrem: np.ndarray[np.ndarray],
    rate: float,
    overrate: float = OVERRATE,
    limit=None,
):
    """逆変換 overrateは左右のオーバーラップ率."""
    # window = np.sin(np.linspace(0, 2*np.pi, len(spektrem[0])+2)[1:-1])  # sin窓
    window = get_window("hann", len(spektrem[0]) + 1)[1:]  # hann窓
    data = np.array([])
    dist = np.array([])  # 正規化用
    # メイン処理
    for i in range(len(spektrem)):
        if i == 0:
            fixPart = np.real(ifft(spektrem[i])) * window  # IFFTで逆変換
            data = np.append(data, fixPart)
            dist = np.append(dist, window**2)
        else:
            fixPart = np.real(ifft(spektrem[i])) * window  # IFFTで逆変換
            overline = int(
                len(fixPart) * 2 * overrate / (1 + 2 * overrate)
            )  # 重なり部分は足す
            data[-overline:] += fixPart[:overline]
            dist[-overline:] += window[:overline] ** 2
            data = np.append(data, fixPart[overline:])
            dist = np.append(dist, window[overline:] ** 2)
    data /= dist  # 正規化
    if limit is not None:
        data = np.clip(data, -limit, limit)  # 雑音を抑える(除去はされない)
    tlist = np.linspace(0, len(data) / rate, len(data))  # 波形表示用
    return tlist, data


if __name__ == "__main__":
    # 読み込み
    try:
        filename = sys.argv[1]  # python3 kadai1.py ???.wav
        r, data = wavfile.read(filename)
    except IndexError:
        # Song: Disfigure - Blank [NCS Release] Music provided by NoCopyrightSounds
        filename = "blank.wav"
        r, data = wavfile.read(filename)
    except Exception:
        print("No such wavfile!")
        sys.exit()

    # 出力 解析結果の画像とwavファイルが保存される
    if len(data.shape) == 2:
        # ステレオの場合は分割
        data1 = data[:, 0]
        data2 = data[:, 1]
        t = np.linspace(0, len(data1) / r, len(data1))
        if onlyOne:
            # 片方のみ
            plt.figure(figsize=(5, 9))
            plt.subplot(311)
            plt.plot(t, data1, label="data1")
            plt.subplot(312)
            spek_t, f, spektrem1 = stft(data1, r)
            plt.pcolormesh(spek_t, f, np.log1p(np.abs(spektrem1.T)), shading="auto")
            plt.xlabel("s")
            plt.ylabel("Hz")
            plt.subplot(313)
            fix_t, fixdata1 = istft(spektrem1, r, limit=1.1 * np.abs(data1).max())
            plt.plot(fix_t, fixdata1)
            fixdata1_int16 = np.int16(fixdata1 / np.max(np.abs(fixdata1)) * 32767)
            wavfile.write("result.wav", r, fixdata1_int16)
        else:
            # 両方
            plt.figure(figsize=(10, 9))
            plt.subplot(321)
            plt.title("data1")
            plt.plot(t, data1, label="data1")
            plt.subplot(323)
            spek_t, f, spektrem1 = stft(data1, r)
            plt.pcolormesh(spek_t, f, np.log1p(np.abs(spektrem1.T)), shading="auto")
            plt.xlabel("s")
            plt.ylabel("Hz")
            plt.subplot(325)
            fix_t, fixdata1 = istft(spektrem1, r, limit=1.1 * np.abs(data1).max())
            plt.plot(fix_t, fixdata1)
            plt.subplot(322)
            plt.title("data2")
            plt.plot(t, data2, label="data2")
            plt.subplot(324)
            spek_t, f, spektrem2 = stft(data2, r)
            plt.pcolormesh(spek_t, f, np.log1p(np.abs(spektrem2.T)), shading="auto")
            plt.xlabel("s")
            plt.ylabel("Hz")
            plt.subplot(326)
            fix_t, fixdata2 = istft(spektrem2, r, limit=1.1 * np.abs(data1).max())
            plt.plot(fix_t, fixdata2)
            if len(fixdata1) == len(fixdata2):
                fixdata1_int16 = np.int16(fixdata1 / np.max(np.abs(fixdata1)) * 32767)
                fixdata2_int16 = np.int16(fixdata2 / np.max(np.abs(fixdata2)) * 32767)
                multi_fixdata = np.array([fixdata1_int16, fixdata2_int16])
                wavfile.write("result.wav", r, multi_fixdata.T)
        plt.savefig("result.png")
        plt.show()
    else:
        # 通常
        t = np.linspace(0, len(data) / r, len(data))
        plt.figure(figsize=(5, 9))
        plt.subplot(311)
        plt.plot(t, data, label="data1")
        plt.subplot(312)
        spek_t, f, spektrem = stft(data, r)
        plt.pcolormesh(spek_t, f, np.log1p(np.abs(spektrem.T)), shading="auto")
        plt.xlabel("s")
        plt.ylabel("Hz")
        plt.subplot(313)
        fix_t, fixdata = istft(spektrem, r, limit=1.1 * np.abs(data).max())
        plt.plot(fix_t, fixdata)
        # int16じゃないと動作しないようです。ありがとうchatGPT!
        fixdata_int16 = np.int16(fixdata / np.max(np.abs(fixdata)) * 32767)
        wavfile.write("result.wav", r, fixdata_int16)
        plt.savefig("result.png")
        plt.show()
