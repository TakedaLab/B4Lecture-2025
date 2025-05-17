"""A solution to ex2 on digital filters."""

import argparse
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile as wavfile


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve two 1-D arrays and return the result.

    Params:
        x (ndarray): First 1-D input array
        h (ndarray): Second 1-D input array
    Returns:
        y (ndarray): Convolution result
    """
    # Designate x as the longer array
    if len(x) < len(h):
        x, h = h, x

    # Input lengths
    x_len = len(x)
    h_len = len(h)

    # Convolution slice maximum length
    slice_len_max = min(x_len, h_len)

    # Initialize output y with zeros
    y = np.zeros(x_len + h_len - 1)

    # Flip one input
    h = np.flip(h)

    for i, _ in enumerate(y):
        slice_x = slice(max(0, i - slice_len_max + 1), min(x_len, i + 1))
        slice_h = slice(max(-h_len, -i - 1), x_len - 1 - i if i >= x_len else None)

        y[i] = np.dot(x[slice_x], h[slice_h])

    return y


def lpf(size: int, cutoff: float, samplerate: int) -> np.ndarray:
    """Return the coefficients of a low-pass filter.

    Params:
        size (int): The size of the filter
        cutoff (float): The desired cutoff frequency in Hz
        samplerate (int): The sampling rate of the relevant signal
    Returns:
        h (ndarray): Filter coefficient array
    """
    # Cutoff frequency as fraction of sample rate for sinc equation
    cutoff = cutoff / samplerate

    # Calculate sinc filter
    h = sinc_lp(size, cutoff)

    # Windowing
    h *= np.hamming(size)

    return h


def hpf(size: int, cutoff: float, samplerate: int) -> np.ndarray:
    """Return the coefficients of a high-pass filter.

    Params:
        size (int): The size of the filter
        cutoff (float): The desired cutoff frequency in Hz
        samplerate (int): The sampling rate of the relevant signal
    Returns:
        h (ndarray): Filter coefficient array
    """
    # Cutoff frequency as fraction of the sample rate for sinc equation
    cutoff = cutoff / samplerate

    # Calculate high-pass filter through spectral inversion of low-pass filter.
    h = -sinc_lp(size, cutoff)
    h[size // 2] += 1

    # Windowing
    h *= np.hamming(size)

    return h


def bpf(size: int, cutoff1: float, cutoff2: float, samplerate: int) -> np.ndarray:
    """Return the coefficients of a band-pass filter.

    Params:
        size (int): The size of the filter
        cutoff1 (float): The desired low cutoff frequency in Hz
        cutoff2 (float): The desired high cutoff frequency in Hz
        samplerate (int): The sampling rate of the relevant signal
    Returns:
        h (ndarray): Filter coefficient array
    """
    # Size of fundamental HP/LP filters.
    basic_filter_size = (size + 1) // 2

    h = convolve(
        hpf(basic_filter_size, cutoff1, samplerate),
        lpf(basic_filter_size, cutoff2, samplerate),
    )

    return h


def bef(size: int, cutoff1: float, cutoff2: float, samplerate: int) -> np.ndarray:
    """Return the coefficients of a band elimination filter.

    Params:
        size (int): The size of the filter
        cutoff1 (float): The desired low cutoff frequency in Hz
        cutoff2 (float): The desired high cutoff frequency in Hz
        samplerate (int): The sampling rate of the relevant signal
    Returns:
        out (ndarray): Filter coefficient array
    """
    return lpf(size, cutoff1, samplerate) + hpf(size, cutoff2, samplerate)


def sinc_lp(size: int, cutoff: float) -> np.ndarray:
    """Return the raw coefficients for a low-pass filter based on the sinc function.

    Params:
        size (int): The size of the filter
        cutoff (float): The desired cutoff frequency as a fraction of the sample rate.
    Returns:
        out (ndarray): Raw filter coefficient array
    """
    return 2 * cutoff * np.sinc(2 * cutoff * np.arange(-(size // 2), size // 2 + 1))


def main(args) -> None:
    """Execute main routine.

    Params:
        args (Namespace): Command line arguments.
    Returns: None
    """
    # Open .wav file with builtin python interface
    with wave.open(args.file, "rb") as file:
        # Number of audio frames
        nframes = file.getnframes()
        # Sample rate
        samplerate = file.getframerate()
        # Read all frames and close file
        buffer = file.readframes(nframes)

    # Copy bytes to numpy array
    a = np.frombuffer(buffer, dtype=np.int16)

    # Calculate filter coefficients
    match (args.filtertype):
        case "lpf":
            h = lpf(args.size, args.cutoffs[0], samplerate)
        case "hpf":
            h = hpf(args.size, args.cutoffs[0], samplerate)
        case "bpf":
            h = bpf(args.size, args.cutoffs[0], args.cutoffs[1], samplerate)
        case "bef":
            h = bef(args.size, args.cutoffs[0], args.cutoffs[1], samplerate)
        case _:
            raise ValueError("Invalid filter type.")

    # Apply filtering
    y = convolve(a, h)

    # Write output signal to file
    wavfile.write(f"{args.file[:-4]}_filtered.wav", samplerate, y.astype(np.int16))

    # -----Plotting-----

    # Filter information
    plt.figure(1, (8, 6))

    # Frequency response
    FFT_SIZE = 1024
    H = np.fft.rfft(h, FFT_SIZE)
    H_dB = 20 * np.log10(np.abs(H) + 1e-10)
    sample_freqs = np.fft.rfftfreq(FFT_SIZE, 1 / samplerate)

    plt.subplot(211)
    plt.plot(sample_freqs, H_dB)
    plt.ylim(bottom=-90)
    plt.title("Filter frequency response")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True, linestyle="--")

    # Phase response
    plt.subplot(212)
    plt.plot(sample_freqs, np.unwrap(np.angle(H)))
    plt.title("Filter phase response")
    plt.xlabel("Frequency")
    plt.ylabel("Phase (rad)")
    plt.grid(True, linestyle="--")

    plt.tight_layout(h_pad=1.2)  # Padding to prevent text overlap

    # Spectrograms
    plt.figure(2, (8, 4))

    # Source
    plt.subplot(121)
    plt.title("Source")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    f, t, Sxx = scipy.signal.spectrogram(a, samplerate)
    plt.pcolormesh(t, f, Sxx, norm="symlog")

    # Filtered
    plt.subplot(122)
    plt.title("Filtered")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    f, t, Sxx = scipy.signal.spectrogram(y, samplerate)
    plt.pcolormesh(t, f, Sxx, norm="symlog")

    plt.tight_layout(h_pad=1.2)  # Padding to prevent text overlap

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--filtertype",
        default="lpf",
        choices=["lpf", "hpf", "bpf", "bef"],
        help="type of filtering to be applied",
    )
    parser.add_argument(
        "-c",
        "--cutoffs",
        nargs="*",
        type=int,
        default=[1000, 3000],
        help="the cutoff frequency/ies in Hz",
    )
    parser.add_argument("-s", "--size", type=int, default=101, help="the filter size")
    parsed_args = parser.parse_args()

    filepath = Path(parsed_args.file)

    if not filepath.is_file():
        raise FileNotFoundError(f"{filepath} does not exist or is not a file.")

    main(parsed_args)
