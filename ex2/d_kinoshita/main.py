"""
Audio signal processing module using FIR filters.

This script implements FIR filters such as LPF, HPF, BPF, and BEF,
applies them to an audio file, and outputs both spectrograms and filter characteristics.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile


def read_audio(path: str) -> tuple[int, np.ndarray]:
    """Read an audio file and convert it to mono and normalize it."""
    if not os.path.exists(path):
        return 0, np.array([])

    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # Convert to mono
    data = data.astype(np.float32) / np.iinfo(data.dtype).max  # Normalize
    return fs, data


def save_audio(path: str, data: np.ndarray, fs: int):
    """Save normalized audio data as a 16-bit integer WAV file."""
    scaled = np.int16(np.clip(data, -1.0, 1.0) * 32767)
    wavfile.write(path, fs, scaled)


class FIR:
    """Class for generating FIR filters and performing convolution."""

    def __init__(self, M, fc, fs, fc2=None):
        """Initialize FIR filter parameters."""
        self.M = M
        self.fc = fc
        self.fc2 = fc2
        self.fs = fs

    def convolution_calculate(self, x, h):
        """Manually perform convolution."""
        y = np.zeros(len(x) - len(h) + 1)
        for n in range(len(y)):
            y[n] = np.sum(x[n : n + len(h)] * h)
        return y

    def low_pass_filter(self, fc):
        """Generate the impulse response of a low-pass filter."""
        omega = 2 * fc / self.fs
        n = np.arange(-self.M, self.M + 1)
        h = omega * np.sinc(omega * n)
        return h

    def high_pass_filter(self):
        """Generate the impulse response of a high-pass filter."""
        omega = 2 * self.fc / self.fs
        n = np.arange(-self.M, self.M + 1)
        h = np.sinc(n) - omega * np.sinc(omega * n)
        return h

    def band_pass_filter(self):
        """Generate the impulse response of a band-pass filter."""
        low = self.low_pass_filter(self.fc2)
        high = self.low_pass_filter(self.fc)
        h = low - high
        return h

    def band_stop_filter(self):
        """Generate the impulse response of a band-stop filter."""
        h = np.sinc(np.arange(-self.M, self.M + 1)) - self.band_pass_filter()
        return h

    def filtering(self, audio: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """Apply a Hamming window and perform filtering."""
        window = np.hamming(len(filt))
        return self.convolution_calculate(audio, filt * window)


def plot_filter_response(h: np.ndarray, fs: int, title: str, folder: str):
    """Plot and save amplitude and phase response of a filter."""
    N = 512
    H = np.fft.rfft(h, n=N)
    w = np.fft.rfftfreq(N, d=1 / fs)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(w, 20 * np.log10(np.abs(H) + 1e-10))
    axs[0].set_title(f"Amplitude Response of {title}")
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel("Amplitude [dB]")
    axs[0].set_xlim([0, 8000])
    axs[0].grid()

    axs[1].plot(w, np.unwrap(np.angle(H)))
    axs[1].set_title(f"Phase Response of {title}")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Phase [radian]")
    axs[1].set_xlim([0, 8000])
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{title}_response.png"))
    plt.show()


def make_spectrogram(audio: np.ndarray, Fs: int, title: str, folder: str):
    """Create and save a spectrogram from audio data."""
    f, t, Sxx = signal.spectrogram(audio, fs=Fs, nperseg=512)
    plt.figure()
    plt.pcolormesh(t, f, Sxx, shading="gouraud", vmax=1e-6)
    plt.xlabel("time [sec]")
    plt.ylabel("frequency [Hz]")
    plt.title(title)
    plt.colorbar(label="Power")
    plt.ylim([0, 8000])
    plt.savefig(os.path.join(folder, f"{title}_spectrogram.png"))
    plt.show()


if __name__ == "__main__":
    input_path = "audio.wav"
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    M = 128
    fc = 3000
    fc2 = 6000

    fs, audio = read_audio(input_path)
    if audio.size == 0:
        raise FileNotFoundError(f"File not found: {input_path}")

    fir = FIR(M=M, fc=fc, fc2=fc2, fs=fs)

    make_spectrogram(audio, fs, "Original Audio Spectrogram", output_folder)

    filters = [
        (fir.low_pass_filter(fc), "Low-Pass Filter", "filtered_low.wav"),
        (fir.high_pass_filter(), "High-Pass Filter", "filtered_high.wav"),
        (fir.band_pass_filter(), "Band-Pass Filter", "filtered_bandpass.wav"),
        (fir.band_stop_filter(), "Band-Stop Filter", "filtered_bandstop.wav"),
    ]

    for filter_coeff, title, output_name in filters:
        filtered = fir.filtering(audio, filter_coeff)

        plot_filter_response(filter_coeff, fs, title, output_folder)
        make_spectrogram(
            filtered, fs, f"Filtered Audio Spectrogram ({title})", output_folder
        )

        save_audio(output_name, filtered, fs)
        print(f"Saved filtered audio: {output_name}")
