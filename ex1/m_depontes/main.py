"""音声データのSTFTとISTFTを実装するプログラム.

音声データを読み込み，STFTを計算し，スペクトログラムを作成し，それを再構築する処理を記述する．
"""

import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pydub


# 音声ファイルの読み込み
def read_audio(path: str) -> np.ndarray:
    """音声ファイルを読み込む関数.

    指定されたパスから音声ファイルを読み込み，モノラルに変換し，numpy配列として返す．
    入力：
      path(str): 音声ファイルのパス
    出力：
      audio_data(np.ndarray).shape = (音声データ): 音声データのnumpy配列
    """
    if not os.path.exists(path):
        return np.array([])

    audio = pydub.AudioSegment.from_file(path)  # wavファイルを読み込み
    audio = audio.set_channels(1)  # モノラルに変換
    audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max  # 振幅の最大値が1になるようにスケーリング

    return audio_data


# stft実装クラス
class Stft:
    """短時間フーリエ変換（STFT）を実装するクラス.

    メンバ関数：
      cut_audio: 音声データをフレームに切り出す関数
      window_function: 窓関数を適用する関数
      fft: フレームに対してFFTを計算する関数
    属性：
      audio(np.ndarray): 音声データのnumpy配列
      frame_length(int): フレームの長さ
      frame_shift(int): フレームのシフト量
      sample_rate(int): サンプリングレート
    """

    def __init__(
        self,
        audio: np.ndarray,
        frame_length: int,
        frame_shift: int,
        sample_rate: int,
    ):
        """コンストラクタ."""
        self.audio = audio
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sample_rate = sample_rate

        # 与えたaudioに対して切り出し・窓関数の付加・fftの実行をする
        self.frames = self.cut_audio()
        self.windowed_frames = self.window_function()
        self.spectrogram = self.fft()
        self.times = (
            np.arange(len(self.spectrogram)) * self.frame_shift
        ) / self.sample_rate
        self.freqs = np.fft.fftfreq(self.frame_length, d=1 / self.sample_rate)[
            :, : self.frame_length // 2
        ]

    def cut_audio(self) -> List[np.ndarray]:
        """音声データの切り出し関数.

        STFTにおける，時間領域での音声データの切り出し（フレームの作成）を実装する関数．

        入力：
          audio(np.ndarray): 音声データのnumpy配列
          frame_length(int): フレームの長さ
          frame_shift(int): フレームのシフト量

        出力：
          frames(List[np.ndarray]): 切り出した音声データのnumpy配列の集合
        """
        frames = []
        for n in range(
            0,
            len(self.audio) - self.frame_length,
            self.frame_shift,
        ):
            frame = self.audio[n : n + self.frame_length]
            frames.append(frame)

        return frames

    # 音声データに窓関数を適用
    def window_function(self) -> List[np.ndarray]:
        """窓関数.

        STFTにおける，フレームそれぞれに窓関数を適応する実装を行う関数．

        入力：
          frames(List[np.ndarray]): 音声データのnumpy配列の集合

        出力：
          __(List[np.ndarray]): 窓関数を適用した音声データのnumpy配列の集合
        """
        return [np.hanning(len(audio)) * audio for audio in self.frames]

    # FFTを計算
    def fft(self) -> np.ndarray:
        """FFT関数.

        窓関数を適応したフレームに対してFFTを計算する実装を行う関数．

        入力：
          windowed_frames(List[np.ndarray]): 音声データのnumpy配列の集合
        出力：
          __(List[np.ndarray]): FFTの結果のnumpy配列の集合
        """
        if len(self.windowed_frames) == 0:
            return []

        return np.array([np.fft.fft(frame) for frame in self.windowed_frames])


# ISTFT実装クラス
class Istft:
    """短時間フーリエ逆変換（ISTFT）を実装するクラス.

    メンバ関数：
      ifft: iFFTを計算する関数
      reconstruct_wave: iFFTを適応した信号を利用して，元の音声データを再構築する関数
    属性：
      spectrogram(np.ndarray): スペクトログラムのnumpy配列
      frame_length(int): フレームの長さ
      frame_shift(int): フレームのシフト量
      sample_rate(int): サンプリングレート
    """

    def __init__(
        self,
        spectrogram: np.ndarray,
        frame_length: int,
        frame_shift: int,
        sample_rate: int,
    ):
        """コンストラクタ."""
        self.spectrogram = spectrogram
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sample_rate = sample_rate

        self.frames = self.ifft()
        self.output_len = (len(self.frames) - 1) * self.frame_shift
        self.output_len += self.frame_length
        self.reconstructed_wave = self.reconstruct_wave()

    # iFFTを計算
    def ifft(self) -> List[np.ndarray]:
        """iFFT関数.

        FFTを適応した信号に対して，iFFTを計算する実装を行う関数．

        入力：
          spectrogram(List[np.ndarray): スペクトログラムのnumpy配列の集合
        出力：
          __(List[np.ndarray]): iFFTの結果のnumpy配列
        """
        if len(self.spectrogram) == 0:
            return []

        return np.array([np.fft.ifft(spec).real for spec in self.spectrogram])

    def reconstruct_wave(self) -> List[np.ndarray]:
        """波形の再構築関数.

        iFFTを適応した信号を利用して，元の音声データを再構築する実装を行う関数．

        入力：
          frames(List[np.ndarray]): iFFTによって再構築した音声データのnumpy配列の集合
          output_len(int): 出力される信号の長さ
          frame_length(int): フレームの長さ
          frame_shift(int): フレームのシフト量

        出力：
          reconstructed(np.array): 再構築した音声データのnumpy配列
        """
        num_frames = len(self.spectrogram)  # フレーム数
        reconstructed = np.zeros(self.output_len)  # 再構築する音声データの枠
        window_sum = np.zeros(self.output_len)  # 補正する窓関数の合計値
        window = np.hanning(self.frame_length)  # 窓関数（ハミング関数）の定義

        # 各フレームに対して、窓関数を適用し、再構築する
        for i in range(num_frames):
            start = i * self.frame_shift  # フレームの開始位置
            reconstructed[start : start + self.frame_length] += (
                self.frames[i] * window
            )  # 窓関数によって重みづけ
            window_sum[start : start + self.frame_length] += window**2

        nonzero = window_sum > 1e-10
        reconstructed[nonzero] /= window_sum[nonzero]
        # 窓関数の合計値が0である(=窓関数によって歪曲されてない)箇所以外を取得
        # 窓関数の合計値で割ることで、歪曲された部分を補正する

        return reconstructed


###


def plot_waveform(
    audio: np.ndarray,
    title: str,
    sample_rate: int = 44100,
) -> None:
    """波形をプロットする関数.

    波形のnumpy配列を受け取り，時間軸を計算してプロットする関数．
    入力：
      audio(np.ndarray): 音声データのnumpy配列
      title(str): プロットのタイトル
      sample_rate(int): サンプリングレート
    出力：なし（波形を表示）
    """
    time = np.arange(0, len(audio)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()


# スペクトログラムを作成
def make_spectrogram(spectrogram: np.ndarray, times, freqs) -> None:
    """スペクトログラムを作成する関数.

    入力されたスペクトログラムの値をもとに描画を行う関数．
    入力：
      spectrogram(np.ndarray): スペクトログラムのnumpy配列
      times(List[float]): 時間軸要素の配列
      freqs(List[float]): 周波数軸要素の配列
    出力：なし（スペクトログラムを表示）
    """
    spectrogram = 20 * np.log10(np.abs(spectrogram))  # dBスケールに変換
    plt.imshow(
        spectrogram.T,
        aspect="auto",
        origin="lower",
        extent=[0, times[-1], freqs[0], freqs[-1]],
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.show()


# メイン関数
def main():
    """メイン関数.

    音声ファイルを読み込み，STFTを計算し，スペクトログラムを作成し，それを再構築する処理を記述する．
    入力：
      sys.argv(List)： コマンドライン引数
        sys.argv[1](str): 音声ファイルのパス
        sys.argv[2](int): サンプリングレート
        sys.argv[3](int): フレームの長さ
        sys.argv[4](int): フレームのシフト量
    出力：
      なし（音声データを表示）
    """
    args = sys.argv
    audio = read_audio(args[1])
    plot_waveform(audio, "original wave")

    sample_rate = args[2]
    frame_length = args[3]
    frame_shift = args[4]

    spectrogram = []
    stft_instance = Stft(audio, frame_length, frame_shift, sample_rate)

    spectrogram = stft_instance.spectrogram
    times = stft_instance.times
    freqs = stft_instance.freqs

    make_spectrogram(spectrogram[:, : frame_length // 2], times, freqs)

    istft_instance = Istft(spectrogram, frame_length, frame_shift, sample_rate)
    reconstructed_wave = istft_instance.reconstructed_wave

    plot_waveform(reconstructed_wave, "reconstructed wave")


if __name__ == "__main__":
    main()
