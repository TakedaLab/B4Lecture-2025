"""Visualize the STFT spectrogram of an audio file and reconstruct the waveform."""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from stft import inv_stft, stft


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


class NameSpace:
    """Configuration namespace for audio processing parameters.

    Parameters
    ----------
    input_path : Path
        Path to the input audio file. (.wav)
    output_path : Path
        Path where the processed output will be saved. (.png)
    filter_length : int
        Length of the filter to be applied, in samples.
    overlap_rate : float
        Overlap rate between consecutive windows. (0 ~ 1)
    window : {"hamming", "hann", "rect"}
        Type of window function to apply during STFT.
    """

    input_path: Path
    output_path: Path
    filter_length: int
    overlap_rate: float
    window: Literal["hamming", "hann", "rect"]


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including input/output paths, filter parameters, and window type.
    """
    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=Path, help="Path to input audio file. (.wav)"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output result file. (.png)",
    )
    parser.add_argument(
        "--filter_length",
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
        "--window",
        type=str,
        default="hamming",
        choices=["hamming", "hann", "rect"],
        help="Using window function (hamming, hann, rect). Default hamming",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    audio_file_path = args.input_path
    result_path = args.output_path
    filter_length = args.filter_length
    overlap_rate = args.overlap_rate
    window = args.window

    # load audio
    data, sample_rate = sf.read(audio_file_path)
    nyquist_frequency = sample_rate // 2
    audio_frame = len(data)
    audio_time = audio_frame / sample_rate

    # process STFT
    stft_result = stft(data, filter_length, overlap_rate, window)
    magnitude, _ = magphase(stft_result)
    spectrogram = mag_to_db(magnitude)

    # remove after nyquist frequency
    spectrogram = spectrogram[: spectrogram.shape[0] // 2]

    # process ISTFT
    istft_result = inv_stft(
        stft_result, audio_frame, filter_length, overlap_rate, window
    )

    # prepare for graph
    x = np.linspace(0, audio_time, audio_frame)
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[50, 1], wspace=0.05, hspace=0.7)

    # draw original audio signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, data)
    ax1.set_xlim(0, audio_time)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Original audio signal")

    # draw spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    image = ax2.imshow(
        spectrogram,
        cmap="jet",
        aspect="auto",
        extent=[0, audio_time, 0, nyquist_frequency // 1000],
        origin="lower",
    )
    ax2.set_xlim(0, audio_time)
    ax2.set_ylim(0, nyquist_frequency // 1000)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [kHz]")
    ax2.set_title("Spectrogram")

    # draw color bar
    cax = fig.add_subplot(gs[1, 1])
    fig.colorbar(image, cax=cax, format="%.0f dB")

    # draw re-synthesized audio signal
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(x, istft_result)
    ax3.set_xlim(0, audio_time)
    ax3.set_ylim(-1, 1)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Re-synthesized audio signal")

    # show graph
    plt.savefig(result_path)
    plt.show()
