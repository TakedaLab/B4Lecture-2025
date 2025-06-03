"""
このモジュールは、音声信号に対してフィルタを適用し、その結果をプロットするための関数を定義しています.

フィルタの種類としては、ローパスフィルタ (LPF)、ハイパスフィルタ (HPF)、
バンドパスフィルタ (BPF)、バンドエリミネーションフィルタ (BEF) があり、
それぞれのフィルタに対して遮断周波数を指定することができます.
"""

import argparse

import numpy as np
import soundfile as sf
import spectrogram as sp
from matplotlib import pyplot as plt

# 定数の定義
FFT_SIZE = 1024  # FFTのサイズ
FILTER_ORDER = 128  # フィルタの次数


def impulse_response(filter_type: str, cutoff: list, freqs: np.ndarray) -> np.ndarray:
    """
    フィルタのインパルス応答を計算する関数.

    フィルタの種類と遮断周波数を指定して、インパルス応答を計算します.

    Parameters:
        filter_type (str): フィルタの種類 (LPF, HPF, BPF, BEF)
        cutoff (list): 遮断周波数
        freqs (np.ndarray): 周波数軸

    Returns:
        np.ndarray: フィルタのインパルス応答
    """
    H = np.zeros(FFT_SIZE)  # フィルタの周波数応答
    if filter_type == "LPF":
        H[np.abs(freqs) <= cutoff[0]] = 1
    elif filter_type == "HPF":
        H[np.abs(freqs) >= cutoff[0]] = 1
    elif filter_type == "BPF":
        H[(cutoff[0] <= np.abs(freqs)) & (np.abs(freqs) <= cutoff[1])] = 1
    elif filter_type == "BEF":
        H[(np.abs(freqs) < cutoff[0]) | (np.abs(freqs) > cutoff[1])] = 1

    h = np.fft.ifft(H)  # 周波数応答からインパルス応答を計算
    h = np.roll(h, FILTER_ORDER // 2)  # 右にM / 2だけシフト
    window = np.hamming(FILTER_ORDER + 1)  # ハミング窓を作成
    for i in range(FILTER_ORDER + 1):
        h[i] *= window[i]  # 窓関数を適用
    for i in range(FILTER_ORDER + 1, FFT_SIZE):
        h[i] = 0  # フィルタの長さをMに制限
    return h.real  # 実部を返す


def convolve(input: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    音声信号にフィルタを適用する関数.

    Parameters:
        input (np.ndarray): 音声データ
        h (np.ndarray): フィルタのインパルス応答

    Returns:
        np.ndarray: フィルタリングされた音声データ
    """
    N = len(input)  # 音声データの長さ
    output = np.zeros(N + FILTER_ORDER)  # 出力信号のサイズ
    for i in range(N + FILTER_ORDER):
        for j in range(FILTER_ORDER + 1):
            if 0 <= i - j < N:  # インデックスが範囲内であることを確認
                output[i] += h[j] * input[i - j]  # 畳み込み演算

    # 出力信号の中央部分を切り出す
    output = output[FILTER_ORDER // 2 :]
    output = output[:N]
    return output


def plot_results(
    Filter: np.ndarray,
    freqs: np.ndarray,
    spec_origin: np.ndarray,
    filtered_spec: np.ndarray,
    sr: int,
    filter_type: str,
):
    """
    フィルタの振幅特性、位相特性、フィルタ前後のスペクトログラムを1枚の画像にプロット.

    Parameters:
        Filter (np.ndarray): フィルタの周波数応答
        freqs (np.ndarray): 周波数軸
        spec_origin (np.ndarray): フィルタ前のスペクトログラム
        filtered_spec (np.ndarray): フィルタ後のスペクトログラム
        sr (int): サンプリング周波数
        filter_type (str): フィルタの種類
    """
    amp_dB = 20 * np.log10(np.abs(Filter))  # 振幅特性をdBに変換
    deg = np.unwrap(np.angle(Filter))  # 位相特性

    # プロットの準備
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(
        4, 2, width_ratios=[45, 1], height_ratios=[1, 1, 1, 1], wspace=0.02, hspace=0.1
    )

    # 振幅特性のプロット
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(
        freqs[: FFT_SIZE // 2], amp_dB[: FFT_SIZE // 2]
    )  # 周波数軸は[0, f_max]にする
    ax0.set_xlim(0, sr / 2)
    ax0.set_title("Amplitude Characteristic")
    ax0.set_xlabel("Frequency [Hz]", fontsize=12)
    ax0.set_ylabel("Amplitude [dB]", fontsize=12)

    # 位相特性のプロット
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(freqs[: FFT_SIZE // 2], deg[: FFT_SIZE // 2])  # 位相特性
    ax1.set_xlim(0, sr / 2)
    ax1.set_title("Phase Characteristic")
    ax1.set_xlabel("Frequency [Hz]", fontsize=12)
    ax1.set_ylabel("Phase [rad]", fontsize=12)

    vmin = -60
    vmax = 30

    # フィルタ前のスペクトログラム
    ax2 = fig.add_subplot(gs[2, 0])
    img1 = ax2.imshow(
        spec_origin.T,
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=[0, spec_origin.shape[1], 0, sr / 2],
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("Original Spectrogram")
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Frequency [Hz]", fontsize=12)

    # カラーバー (フィルタ前)
    cbar_ax1 = fig.add_subplot(gs[2, 1])
    plt.colorbar(img1, cax=cbar_ax1, format="%+2.0f dB")

    # フィルタ後のスペクトログラム
    ax3 = fig.add_subplot(gs[3, 0])
    img2 = ax3.imshow(
        filtered_spec.T,
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=[0, filtered_spec.shape[1], 0, sr / 2],
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_title("Filtered Spectrogram")
    ax3.set_xlabel("Time [s]", fontsize=12)
    ax3.set_ylabel("Frequency [Hz]", fontsize=12)

    # カラーバー (フィルタ後)
    cbar_ax2 = fig.add_subplot(gs[3, 1])
    plt.colorbar(img2, cax=cbar_ax2, format="%+2.0f dB")

    fig.set_constrained_layout(True)
    plt.show()
    # figsディレクトリの中に画像を保存
    fig.savefig("figs/" + filter_type + ".png", dpi=300)


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.

    フィルターの種類と遮断周波数を指定します.
    """
    parser = argparse.ArgumentParser(
        description="フィルターの種類と遮断周波数を指定します。"
    )

    # フィルターの種類を指定
    parser.add_argument(
        "filter_type",
        choices=["LPF", "HPF", "BPF", "BEF"],
        help="フィルターの種類を指定します (LPF, HPF, BPF, BEF)",
    )

    # 遮断周波数を指定
    parser.add_argument(
        "cutoff",
        nargs="+",  # 1つ以上の引数を受け取る
        type=float,
        help="遮断周波数を指定します。LPF, HPFは1つ、BPF, BEFは2つの周波数を指定してください。",
    )

    args = parser.parse_args()

    # 入力の検証
    if args.filter_type in ["LPF", "HPF"] and len(args.cutoff) != 1:
        parser.error(f"{args.filter_type}では遮断周波数を1つ指定してください。")
    elif args.filter_type in ["BPF", "BEF"] and len(args.cutoff) != 2:
        parser.error(f"{args.filter_type}では遮断周波数を2つ指定してください。")

    return args


def main():
    """
    メイン関数.

    音声データを読み込み、フィルタを適用し、結果をプロットします.
    """
    args = parse_arguments()  # 引数を解析
    data, sr = sf.read("audio/audio.wav")  # 音声データを読み込み
    freqs = np.fft.fftfreq(FFT_SIZE, d=1 / sr)  # 周波数軸 [0, f_max] → [-f_max, 0]

    h = impulse_response(args.filter_type, args.cutoff, freqs)
    H = np.fft.fft(h)  # フィルタの周波数応答
    filtered_data = convolve(data, h[: FILTER_ORDER + 1])  # フィルタリング
    spec_origin = sp.spectrogram_dB(data, FFT_SIZE)  # 元のスペクトログラム
    filtered_spec = sp.spectrogram_dB(
        filtered_data, FFT_SIZE
    )  # フィルタ後のスペクトログラム
    plot_results(
        H, freqs, spec_origin, filtered_spec, sr, args.filter_type
    )  # フィルタの特性とスペクトログラムをプロット
    sf.write(
        "audio/audio" + "_" + args.filter_type + ".wav", filtered_data, sr
    )  # フィルタ後の音声データを保存


if __name__ == "__main__":
    main()
