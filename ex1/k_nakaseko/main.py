"""
このモジュールは、音声信号の短時間フーリエ変換（STFT）とその逆変換（iSTFT）を行います.

また、スペクトログラムの可視化や波形の復元も含まれています.
"""

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt


def load_audio(filepath: str) -> tuple:
    """
    音声ファイルを読み込む.

    Parameters:
        filepath (str): 音声ファイルのパス

    Returns:
        tuple: 音声データとサンプリング周波数
    """
    data, sr = sf.read(filepath)  # 音声データを読み込む
    return data, sr


def my_stft(signal: np.ndarray, frame_size: int) -> np.ndarray:
    """
    短時間フーリエ変換 (STFT).

    Parameters:
        signal (np.ndarray): 音声データ
        frame_size (int): 窓関数の長さ

    Returns:
        np.ndarray: スペクトログラム
    """
    spectrogram = []
    window = np.hamming(frame_size)  # 窓関数としてハミング窓を作成
    overlap = 0.5  # オーバラップ率
    hop_size = int(frame_size * (1 - overlap))  # ホップサイズを計算

    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i : i + frame_size] * window  # 窓関数を適用
        spectrum = np.fft.rfft(frame)  # FFTを計算
        spectrogram.append(spectrum)  # スペクトルを保存

    return np.array(spectrogram).T  # 転置して [周波数 × 時間] にする


def my_istft(spectrogram: np.ndarray, frame_size: int) -> np.ndarray:
    """
    逆短時間フーリエ変換 (iSTFT).

    Parameters:
        spectrogram (np.ndarray): スペクトログラム
        frame_size (int): 窓関数の長さ

    Returns:
        np.ndarray: 復元された音声データ
    """
    window = np.hamming(frame_size)  # ハミング窓を作成
    overlap = 0.5  # オーバラップ率
    hop_size = int(frame_size * (1 - overlap))  # ホップサイズを計算
    signal = np.zeros(
        int(spectrogram.shape[1] * hop_size + frame_size)
    )  # 出力信号を初期化

    num_frames = spectrogram.shape[1]
    signal_len = frame_size + (num_frames - 1) * hop_size
    signal = np.zeros(signal_len)  # 出力信号を初期化
    window_sum = np.zeros(signal_len)  # 窓の重なり補正用

    for i in range(num_frames):
        spectrum = spectrogram[:, i]
        frame = np.fft.irfft(spectrum, n=frame_size)  # 逆FFT（元の長さで）
        start = i * hop_size
        signal[start : start + frame_size] += frame * window  # 加算合成
        window_sum[start : start + frame_size] += window  # 重みを記録

    # 重なりによる振幅変化を補正
    window_sum[window_sum == 0] = 1e-8  # 0除算を防ぐ
    return signal / window_sum


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int,
    hop_size: int,
    ax,
    duration: float,
    title: str = "Spectrogram",
) -> None:
    """スペクトログラムを描画する（時間軸を duration に揃える）."""
    time_axis = np.linspace(
        0, duration, spectrogram.shape[1]
    )  # ← duration に合わせて線形補間
    freq_axis = np.linspace(0, sr / 2, spectrogram.shape[0])

    im = ax.imshow(
        20 * np.log10(np.abs(spectrogram) + 1e-8),
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )
    ax.set_xlim([0, duration])
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    plt.colorbar(im, ax=ax, format="%+2.0f dB", label="Amplitude [dB]")


def main():
    """
    メイン関数.

    音声ファイルを読み込み、STFT と iSTFT を実行し、結果を描画する.
    """
    filepath = "audio.wav"
    frame_size = 1024

    # 音声読み込み
    signal, sr = load_audio(filepath)

    # STFT と iSTFT 実行
    spectrogram = my_stft(signal, frame_size)
    hop_size = int(frame_size * 0.5)
    restored_signal = my_istft(spectrogram, frame_size)

    # 共通の時間軸設定（オリジナルの長さに合わせる）
    duration = len(signal) / sr
    time = np.linspace(0, duration, num=len(signal))
    restored_time = np.linspace(0, duration, num=len(restored_signal))  # 同じ長さで調整

    # 描画
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # 1. 元の波形（青）
    axs[0].plot(time, signal, color="blue")
    axs[0].set_xlim([0, duration])
    axs[0].set_title("Original Audio Waveform")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")

    # 2. スペクトログラム（時間軸合わせ）
    plot_spectrogram(spectrogram, sr, hop_size, axs[1], duration, title="Spectrogram")

    # 3. 復元波形（青で統一）
    axs[2].plot(restored_time, restored_signal, color="blue")
    axs[2].set_xlim([0, duration])
    axs[2].set_title("Restored Audio Waveform")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # 復元音声の保存
    sf.write("restored_audio.wav", restored_signal, sr)


if __name__ == "__main__":
    main()
