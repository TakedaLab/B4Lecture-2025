"""Apply FIR filtering to an audio file, visualize STFT spectrograms."""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from stft import inv_stft, stft
from window import get_filter


def magphase(
    complex_array: np.ndarray[complex],
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Split complex to magnitude and phase.

    Parameters
    ----------
    complex_array : np.ndarray[complex]
        Array containing complex numbers.

    Returns
    -------
    magnitude : np.ndarray[float]
        Array containing magnitudes.
    phase : np.ndarray[float]
        Array containing phases (in radians).
    """
    magnitude = np.abs(complex_array)
    phase = np.angle(complex_array)

    return magnitude, phase


def mag_to_db(magnitude: np.ndarray[float], eps: float = 1e-7) -> np.ndarray[float]:
    """Convert magnitude to dB.

    Parameters
    ----------
    magnitude : np.ndarray[float]
        Array containing magnitudes.

    eps : float, default 1e-7
        Small constant added to avoid taking the logarithm of zero.

    Returns
    -------
    db : np.ndarray[float]
        Array converted to dB scale.
    """
    magnitude = np.clip(magnitude, eps, None)
    db = 20 * np.log10(magnitude)

    return db


def conv(data: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Perform a 1D convolution of the input data with the given filter.

    Parameters
    ----------
    data : np.ndarray
        Input 1D array representing the signal or data to be convolved.
    filter : np.ndarray
        1D array representing the filter apply to the data.

    Returns
    -------
    result : np.ndarray
        The result of convolving `data` with `filter`.
    """
    result = np.zeros(len(data) + len(filter) - 1)
    for i, d in enumerate(data):
        result[i : i + len(filter)] += d * filter

    return result


class NameSpace:
    """Configuration namespace for audio filtering and STFT-based processing.

    Attributes
    ----------
    input_path : Path
        Path to the input audio file (.wav).
    output_image_path : Path
        Path to save the resulting spectrogram image (.png).
    output_audio_path : Path
        Path to save the filtered audio output (.wav).
    filter_length : int
        Length of the FIR filter.
    filter_name : {"lpf", "hpf", "bpf", "bsf"}
        Type of filter to apply:
        - "lpf": low-pass filter
        - "hpf": high-pass filter
        - "bpf": band-pass filter
        - "bsf": band-stop filter
    low_freq : int
        Lower cutoff frequency for filtering (Hz).
    high_freq : int
        Upper cutoff frequency for filtering (Hz).
    stft_length : int
        Length of each segment for STFT (window size).
    overlap_rate : float
        Overlap ratio between consecutive STFT windows (0 ~ 1).
    window : {"hamming", "hann", "rect"}
        Window function to apply to each STFT segment.
    """

    input_path: Path
    output_image_path: Path
    output_audio_path: Path
    filter_length: int
    filter_name: Literal["lpf", "hpf", "bpf", "bsf"]
    low_freq: int
    high_freq: int
    stft_length: int
    overlap_rate: float
    window: Literal["hamming", "hann", "rect"]


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including input/output paths, filter parameters,
        cutoff frequency, and window type.
    """
    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=Path, help="Path to input audio file. (.wav)"
    )
    parser.add_argument(
        "output_image_path",
        type=Path,
        help="Path to output result image file. (.png)",
    )
    parser.add_argument(
        "output_audio_path",
        type=Path,
        help="Path to output result audio file. (.wav)",
    )
    parser.add_argument(
        "--filter_length",
        type=int,
        default=101,
        help="Size of filter. Default 101",
    )
    parser.add_argument(
        "--filter_name",
        type=str,
        default="lpf",
        choices=["lpf", "hpf", "bpf", "bsf"],
        help="Using filter (lpf, hpf, bpf, bsf). Default lpf",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hamming",
        choices=["hamming", "hann", "rect"],
        help="Using window function for filter. (hamming, hann, rect). Default hamming",
    )
    parser.add_argument(
        "--low_freq",
        type=int,
        default=1000,
        help="Lower frequency. Default 1000",
    )
    parser.add_argument(
        "--high_freq",
        type=int,
        default=4000,
        help="Higher frequency. Default 4000",
    )
    parser.add_argument(
        "--stft_length",
        type=int,
        default=1024,
        help="Size of filter for STFT. Default 1024",
    )
    parser.add_argument(
        "--overlap_rate",
        type=float,
        default=0.5,
        help="Rate of overlap for STFT (0 ~ 1). Default 0.5",
    )
    parser.add_argument(
        "--stft_window",
        type=str,
        default="hamming",
        choices=["hamming", "hann", "rect"],
        help="Using window function for STFT. (hamming, hann, rect). Default hamming",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    audio_file_path = args.input_path
    result_image_path = args.output_image_path
    result_audio_path = args.output_audio_path
    filter_length = args.filter_length
    filter_name = args.filter_name
    window = args.window
    low_freq = args.low_freq
    high_freq = args.high_freq
    stft_length = args.stft_length
    overlap_rate = args.overlap_rate
    stft_window = args.window

    # load audio
    data, sample_rate = sf.read(args.input_path)
    nyquist_frequency = sample_rate // 2
    audio_frame = len(data)
    audio_time = audio_frame / sample_rate

    # get filter
    filter = get_filter(
        filter_name,
        filter_length,
        sample_rate,
        window,
        low_freq,
        high_freq,
    )

    # process FFT to filter
    filter_fft_result = np.fft.fft(filter)
    filter_magnitude, filter_phase = magphase(filter_fft_result)
    filter_magnitude = mag_to_db(filter_magnitude)

    # remove after nyquist frequency
    filter_magnitude = filter_magnitude[: filter_magnitude.shape[0] // 2]
    filter_phase = filter_phase[: filter_phase.shape[0] // 2]

    # unwrap phase
    filter_phase = np.unwrap(filter_phase)

    # process FIR convolution
    filtered_data = conv(data, filter)

    # process STFT to data
    origin_stft_result = stft(data, stft_length, overlap_rate, stft_window)
    origin_magnitude, _ = magphase(origin_stft_result)
    origin_spectrogram = mag_to_db(origin_magnitude)
    filtered_stft_result = stft(filtered_data, stft_length, overlap_rate, stft_window)
    filtered_magnitude, _ = magphase(filtered_stft_result)
    filtered_spectrogram = mag_to_db(filtered_magnitude)

    # remove after nyquist frequency
    origin_spectrogram = origin_spectrogram[: origin_spectrogram.shape[0] // 2]
    filtered_spectrogram = filtered_spectrogram[: filtered_spectrogram.shape[0] // 2]

    # process ISTFT
    istft_result = inv_stft(
        filtered_stft_result, audio_frame, stft_length, overlap_rate, stft_window
    )

    # save filtered audi0
    sf.write(result_audio_path, np.real(istft_result), sample_rate)

    # prepare for graph
    x = np.linspace(0, audio_time, audio_frame)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.5)

    # draw magnitude of filter
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(np.linspace(0, nyquist_frequency, len(filter_magnitude)), filter_magnitude)
    ax1.set_xlim(0, nyquist_frequency)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude [dB]")

    # draw phase of filter
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(np.linspace(0, nyquist_frequency, len(filter_phase)), filter_phase)
    ax2.set_xlim(0, nyquist_frequency)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Phase [rad]")

    # draw original spectrogram
    ax3 = fig.add_subplot(gs[2, 0])
    original_image = ax3.imshow(
        origin_spectrogram,
        cmap="jet",
        aspect="auto",
        vmin=-60,
        vmax=30,
        extent=[0, audio_time, 0, nyquist_frequency // 1000],
        origin="lower",
    )
    ax3.set_xlim(0, audio_time)
    ax3.set_ylim(0, nyquist_frequency // 1000)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Frequency [kHz]")
    ax3.set_title("Original audio spectrogram")
    cbar = fig.colorbar(original_image, ax=ax3)

    # draw filtered spectrogram
    ax4 = fig.add_subplot(gs[2, 1])
    filtered_image = ax4.imshow(
        filtered_spectrogram,
        cmap="jet",
        aspect="auto",
        vmin=-60,
        vmax=30,
        extent=[0, audio_time, 0, nyquist_frequency // 1000],
        origin="lower",
    )
    ax4.set_xlim(0, audio_time)
    ax4.set_ylim(0, nyquist_frequency // 1000)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Frequency [kHz]")
    ax4.set_title("Original audio spectrogram")
    cbar = fig.colorbar(filtered_image, ax=ax4)

    # show graph
    plt.savefig(result_image_path)
    plt.show()
