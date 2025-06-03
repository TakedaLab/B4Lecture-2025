"""
このモジュールは、音声信号に対して短時間フーリエ変換 (STFT) を適用する機能を提供します.

また、スペクトログラムを生成し、その結果をプロットする機能も含まれています.
"""

import numpy as np


def stft(signal: np.ndarray, win_len: int) -> np.ndarray:
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


def spectrogram_dB(data: np.ndarray, win_len: int) -> np.ndarray:
    """
    スペクトログラムを計算.

    Parameters:
        data (np.ndarray): 音声データ
        win_len (int): 窓関数の長さ

    Returns:
        np.ndarray: スペクトログラム (dBスケール)
    """
    fft_results = stft(data, win_len)  # STFTを計算
    spectrogram = fft_results[:, : win_len // 2 + 1]  # ナイキスト周波数までの部分を取得
    spec_dB = 20 * np.log10(np.abs(spectrogram))  # dBスケールに変換
    return spec_dB
