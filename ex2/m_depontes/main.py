"""デジタルフィルタ実装プログラム.

デジタルフィルタの設計を行い，フィルタリングを行うプログラムを記述する．
最初にFIRフィルタのHPF，LPF，BPF，BSPを構成し，そののちにIIRフィルタの作成を行う．
"""

import os

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


class FIR:
    """FIRフィルタの設計とフィルタリングを行うクラス.

    メンバ関数：
        convolsion_calculate: 畳み込み計算を行う関数
        digital_filter: デジタルフィルタを設計する関数
        filtering: フィルタリングを行う関数

    属性：
        M(int): フィルタの次数
        fc(int): カットオフ周波数．複数の場合は下限を意味する．
        fs(int): サンプリング周波数
        fc2(int): 上限カットオフ周波数
    """

    def __init__(self, M, fc, fs, fc2=None):
        """初期化関数.

        入力：
            M: フィルタの次数
            fc: カットオフ周波数
            fs: サンプリング周波数
        """
        self.M = M
        self.fc = fc
        self.fc2 = fc2
        self.fs = fs

    def convolution_calculate(self, x, h):
        """畳み込み計算を行う関数.

        入力：
            x: 入力信号
            h: フィルタ係数

        出力：
            y: 出力信号
        """
        y = [0 for _ in range(len(x) - len(h) + 1)]
        for i in range(len(x) - len(h) + 1):
            y[i] = 0
            for j in range(len(h)):
                y[i] += x[i + j] * h[j]
        return np.array(y)

    def low_path_filter(self, fc=None):
        """LPFを設計する関数.

        BPFでの再利用をするため，fcは外部からも指定可能とする．

        入力：
            M(int): フィルタの次数
            fc(int): 下側のカットオフ周波数
            fs(int): サンプリング周波数

        出力：
            filter(np.ndarray): フィルタ係数, shape = (filterValue)
        """
        if fc is None:
            fc = self.fc

        filter = np.zeros(2 * self.M + 1)  # フィルタ係数の初期化

        omega = 2 * fc / self.fs
        # 補足：np.sinc(x) = sin(np.pi * x) / (np.pi * x) のため，omegaのpiは省略される

        filter = omega * np.sinc(omega * np.arange(-self.M, self.M + 1))  # LPF

        return filter

    def high_pass_filter(self):
        """HPFを設計する関数.

        HPFは非因果的であるため，因果的にするためのシフト処理を行う．

        入力：
            M(int): フィルタの次数
            fc(int): 下側のカットオフ周波数
            fs(int): サンプリング周波数

        出力：
            filter(np.ndarray): フィルタ係数, shape = (filterValue)
        """
        filter = np.zeros(2 * self.M + 1)  # フィルタ係数の初期化
        omega = 2 * self.fc / self.fs  # デジタル周波数
        n = np.arange(0, 2 * self.M + 1)  # 因果的フィルタ変換
        filter = np.sinc(n - self.M) - omega * np.sinc(omega * (n - self.M))  # HPF

        return filter

    def band_pass_filter(self):
        """BPFを設計する関数.

        定義より，2つのカットオフ周波数における低域通過フィルタの組み合わせとして表現できる．

        入力：
            M(int): フィルタの次数
            fc(int): 下側のカットオフ周波数
            fc2(int): 上側のカットオフ周波数
            fs(int): サンプリング周波数

        出力：
            filter(np.ndarray): フィルタ係数, shape = (filterValue)
        """
        if self.fc > self.fc2:
            self.fc, self.fc2 = self.fc2, self.fc

        filter = np.zeros(2 * self.M + 1)
        filter = self.low_path_filter(self.fc2) - self.low_path_filter(self.fc)

        return filter

    def band_stop_filter(self):
        """BSFを設計する関数.

        定義より，全通過フィルタとバンドパスフィルタの組み合わせとして表現できる．

        入力：
            M(int): フィルタの次数
            fc(int): 下側のカットオフ周波数
            fc2(int): 上側のカットオフ周波数
            fs(int): サンプリング周波数
        出力：
            filter(np.ndarray): フィルタ係数, shape = (filterValue)
        """
        if self.fc > self.fc2:
            self.fc, self.fc2 = self.fc2, self.fc

        filter = np.zeros(2 * self.M + 1)
        n = np.arange(-self.M, self.M + 1)
        filter = np.sinc(n) - self.band_pass_filter()

        return filter

    def filtering(
        self,
        audio: np.ndarray,
        filter: np.ndarray,
    ) -> np.ndarray:
        """フィルタリングを行う関数.

        入力：
            audio(np.ndarray): 入力信号, shape = (data)
            filter(np.ndarray): フィルタ係数, shape = (filterValue)

        出力：
            y(np.ndarray): 出力信号, shape = (data)
        """
        window = np.hamming(len(filter))
        y = self.convolution_calculate(audio, filter * window)
        return y


def plot_filter_response(h: np.ndarray, fs: int, title: str) -> None:
    """フィルタの振幅特性と位相特性をプロットする関数.

    入力：
        h(np.ndarray): フィルタ係数, shape = (filterValue)
        fs(int): サンプリング周波数
        title(str): グラフのタイトル
    """
    N = 512  # 長めのFFTサイズでゼロパディング
    H = np.fft.rfft(h, n=N)
    w = np.fft.rfftfreq(N, d=1 / fs)

    # 振幅応答
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(H) + 1e-10))  # log(0)対策, 表記はデシベル
    plt.title(f"Amplitude Response of {title}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid()

    # 位相応答（アンラップして滑らかに）
    plt.figure()
    plt.plot(w, np.unwrap(np.angle(H)))
    plt.title(f"Phase Response of {title}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [radian]")
    plt.grid()

    plt.show()


def make_spectrogram(audio: np.ndarray, Fs: int, title: str) -> None:
    """スペクトログラムを表示する関数.

    入力：
        audio(np.ndarray): 音声データ, shape = (data)
        Fs(int): サンプリング周波数
        title(str): スペクトログラムのタイトル

    出力：
        None(スペクトログラムを出力)
    """
    import scipy.signal as signal

    f, t, Sxx = signal.spectrogram(audio, fs=Fs, nperseg=512)

    plt.figure()
    plt.pcolormesh(t, f, Sxx, vmax=1e-6)
    plt.xlabel("time [sec]")
    plt.ylabel("frequency [Hz]")
    plt.title(title)
    plt.show()


def parse_arguments():
    """コマンドライン引数を解析する関数.

    入力：
        なし

    出力：
        args: 入力のオブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FIRフィルタを使用した音声処理プログラム"
    )
    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        choices=["low", "high", "bandpass", "bandstop"],
        help="使用するフィルタの種類を指定 (low, high, bandpass, bandstop)",
    )
    parser.add_argument(
        "--fc",
        type=float,
        nargs="*",
        required=True,
        help="カットオフ周波数 (low/highの場合は1つ, bandpass/bandstopの場合は2つ指定)",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=64,
        help="フィルタの次数 (デフォルト: 64)",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=44100,
        help="サンプリング周波数 (デフォルト: 44100 Hz)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="入力音声ファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="filtered_audio.wav",
        help="出力音声ファイルのパス",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # FIRフィルタのインスタンスを生成
    if len(args.fc) == 1:
        fir = FIR(M=args.M, fc=args.fc[0], fs=args.fs)
    else:
        fir = FIR(M=args.M, fc=args.fc[0], fc2=args.fc[1], fs=args.fs)

    # 入力信号の読み込み
    audio = read_audio(args.input)
    if audio.size == 0:
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {args.input}")

    # フィルタの種類に応じて処理を分岐
    if args.filter == "low":
        filter_coeff = fir.low_path_filter()
        title = "Low-Pass Filter"
    elif args.filter == "high":
        filter_coeff = fir.high_pass_filter()
        title = "High-Pass Filter"
    elif args.filter == "bandpass":
        if len(args.fc) != 2:
            raise ValueError(
                "bandpassフィルタには2つのカットオフ周波数を指定してください"
            )
        filter_coeff = fir.band_pass_filter()
        title = "Band-Pass Filter"
    elif args.filter == "bandstop":
        if len(args.fc) != 2:
            raise ValueError(
                "bandstopフィルタには2つのカットオフ周波数を指定してください"
            )
        filter_coeff = fir.band_stop_filter()
        title = "Band-Stop Filter"

    # フィルタリングを実行
    window = np.hamming(len(filter_coeff))
    filtered_audio = fir.filtering(audio, filter_coeff)

    # フィルタの応答をプロット
    plot_filter_response(filter_coeff, args.fs, title)

    # スペクトログラムのプロット
    make_spectrogram(audio, args.fs, "Original Audio Spectrogram")
    make_spectrogram(filtered_audio, args.fs, "Filtered Audio Spectrogram")
