"""
STFT and ISTFT processing demo.

This module loads an audio file, computes its STFT,
reconstructs it via ISTFT, saves the reconstructed file,
and visualizes the original waveform, spectrogram, and reconstructed waveform.
"""

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def load_audio(file_path):
    """Load an audio file."""
    signal, sample_rate = sf.read(file_path)
    return signal, sample_rate


def stft(x, frame_size, hop_size, window):
    """Compute the Short-Time Fourier Transform (STFT)."""
    frames = []
    for i in range(0, len(x) - frame_size, hop_size):
        frame = x[i : i + frame_size] * window
        spectrum = np.fft.rfft(frame)
        frames.append(spectrum)
    return np.array(frames).T


def istft(spectrogram, frame_size, hop_size, window):
    """Perform inverse STFT to reconstruct the time-domain signal."""
    n_frames = spectrogram.shape[1]
    output_len = (n_frames - 1) * hop_size + frame_size
    output = np.zeros(output_len)
    window_sum = np.zeros(output_len)

    for i in range(n_frames):
        frame = np.fft.irfft(spectrogram[:, i])
        start = i * hop_size
        output[start : start + frame_size] += frame * window
        window_sum[start : start + frame_size] += window**2

    nonzero = window_sum > 1e-6
    output[nonzero] /= window_sum[nonzero]
    return output


def save_audio(file_path, signal, sample_rate):
    """Save audio to a file."""
    sf.write(file_path, signal, sample_rate)


def plot_waveforms_and_spectrogram(
    signal, reconstructed_signal, spectrogram, sample_rate, frame_size, hop_size
):
    """Plot original waveform, spectrogram, and reconstructed waveform."""
    time_original = np.arange(len(signal)) / sample_rate
    time_reconstructed = np.arange(len(reconstructed_signal)) / sample_rate
    frame_times = np.arange(spectrogram.shape[1]) * hop_size / sample_rate
    freqs = np.fft.rfftfreq(frame_size, 1 / sample_rate)

    plt.figure(figsize=(14, 8))

    # Original waveform
    plt.subplot(3, 1, 1)
    plt.plot(time_original, signal)
    plt.title("Original Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Spectrogram
    plt.subplot(3, 1, 2)
    magnitude_db = 20 * np.log10(np.abs(spectrogram) + 1e-6)
    plt.imshow(
        magnitude_db,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[frame_times[0], frame_times[-1], freqs[0], freqs[-1]],
    )
    plt.title("Spectrogram (dB)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    # Reconstructed waveform
    plt.subplot(3, 1, 3)
    plt.plot(time_reconstructed, reconstructed_signal)
    plt.title("Reconstructed Waveform (from ISTFT)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def main():
    """Perform STFT, ISTFT, save reconstructed audio, and visualize waveforms."""
    # --- パラメータ設定 ---
    file_path = "audio.wav"
    frame_size = 1024
    hop_size = 512
    window = np.hanning(frame_size)

    # --- 音声読み込み ---
    signal, sample_rate = load_audio(file_path)

    # --- STFT・ISTFT 実行 ---
    spectrogram = stft(signal, frame_size, hop_size, window)
    reconstructed_signal = istft(spectrogram, frame_size, hop_size, window)

    # --- 書き出し（任意） ---
    save_audio("reconstructed.wav", reconstructed_signal, sample_rate)

    # --- 描画 ---
    plot_waveforms_and_spectrogram(
        signal, reconstructed_signal, spectrogram, sample_rate, frame_size, hop_size
    )


if __name__ == "__main__":
    main()
