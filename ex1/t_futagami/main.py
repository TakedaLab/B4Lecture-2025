"""
このモジュールは、音声信号に対して短時間フーリエ変換 (STFT) とその逆変換 (iSTFT) を適用する機能を提供します.

また、スペクトログラムを生成し、元の音声信号を再構成する機能や、その結果をプロットする機能も含まれています.
"""

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt


def stft(signal: np.ndarray, win_len: int) -> np.array:
    """
    短時間フーリエ変換 (STFT).

    Parameters:
        signal (np.ndarray): 音声データ
        win_len (int): 窓関数の長さ

    Returns:
        np.ndarray: スペクトログラム
    """
    spectrogram = []
    window = np.hamming(win_len)  # ハミング窓を作成
    overlap = 0.5  # オーバラップ率
    hop_size = int(win_len * (1 - overlap))  # ホップサイズを計算
    for i in range(0, len(signal) - win_len, hop_size):
        frame = signal[i : i + win_len] * window  # 窓関数を適用
        spectrum = np.fft.fft(frame)  # FFTを計算
        spectrogram.append(spectrum)  # スペクトルを保存
    return np.array(spectrogram)


def istft(spectrogram: np.ndarray, win_len: int, orig_len: int) -> np.array:
    """
    逆短時間フーリエ変換 (iSTFT).

    Parameters:
        spectrogram (np.ndarray): スペクトログラム
        win_len (int): 窓関数の長さ
        orig_len (int): 元の音声データの長さ

    Returns:
        np.ndarray: 再構成された音声データ
    """
    window = np.hamming(win_len)  # ハミング窓を作成
    overlap = 0.5  # オーバラップ率
    hop_size = int(win_len * (1 - overlap))  # ホップサイズを計算
    signal = np.zeros(
        int(spectrogram.shape[0] * hop_size + win_len)
    )  # 出力信号を初期化
    for i, spectrum in enumerate(spectrogram):
        spectrum = np.concatenate(
            (spectrum, np.conj(spectrum[-2:0:-1]))
        )  # スペクトルを複素共役対称にする
        frame = np.fft.ifft(spectrum).real / window  # 逆FFTを計算し、窓関数で正規化
        start = i * hop_size
        signal[start : start + win_len] += frame / 2  # オーバラップを考慮して加算
    return signal[:orig_len]  # 元の長さに切り詰めて返す


def plot_results(data, sr, spectrogram, reconstructed):
    """
    結果をプロット.

    Parameters:
        data (np.ndarray): 元の音声データ
        sr (int): サンプリング周波数
        spectrogram (np.ndarray): スペクトログラム
        reconstructed (np.ndarray): 再構成された音声データ
    """
    nyquist = sr // 2  # ナイキスト周波数
    spec_dB = 20 * np.log10(np.abs(spectrogram))  # dBスケールに変換
    time = np.linspace(0, len(data) / sr, len(data))  # 時間軸を計算
    time_spec = np.linspace(
        0, len(data) / sr, spectrogram.shape[0]
    )  # スペクトログラムの時間軸を計算

    # プロット
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[30, 1],
        height_ratios=[1, 1, 1],
        wspace=0.1,
        hspace=0.5,
    )

    # 元の信号
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time, data)
    ax0.set_xlim(time[0], time[-1])
    ax0.set_title("Original Signal")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Amplitude")

    # スペクトログラム
    ax1 = fig.add_subplot(gs[1, 0])
    img = ax1.imshow(
        spec_dB.T,
        cmap="jet",
        aspect="auto",
        origin="lower",
        vmin=-60,
        vmax=30,
        extent=[time_spec[0], time_spec[-1], 0, nyquist],
    )
    ax1.set_xlim(time_spec[-0], time_spec[-1])
    ax1.set_title("Spectrogram")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")

    # カラーバー
    cbar_ax = fig.add_subplot(gs[1, 1])
    plt.colorbar(img, cax=cbar_ax, format="%+2.0f dB")

    # 再構成された信号
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(time, reconstructed)
    ax2.set_xlim(time[0], time[-1])
    ax2.set_title("Reconstructed Signal")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")

    plt.show()


def main():
    """メイン処理: STFTとiSTFTを実行し、結果をプロット・保存."""
    data, sr = sf.read("audio.wav")  # 音声データを読み込み
    win_len = 1024  # 窓関数の長さ

    # STFTを実行
    fft_results = stft(data, win_len)
    # スペクトログラムを計算
    spectrogram = fft_results[:, : win_len // 2 + 1]  # ナイキスト周波数までの部分を取得
    # iSTFTを実行
    reconstructed = istft(spectrogram, win_len, len(data))
    # 結果を表示
    plot_results(data, sr, spectrogram, reconstructed)


if __name__ == "__main__":
    main()
