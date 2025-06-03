"""Provides STFT and inverse STFT functions for 1D audio signal processing."""

import numpy as np


def back_padding(data: np.ndarray, length: int) -> np.ndarray:
    """Pad a 1D array with zeros at the end.

    Parameters
    ----------
    data : np.ndarray
        1D array to be padded.

    length : int
        Desired total length of the output array.

    Returns
    -------
    padded_data : np.ndarray
        1D array of length `length`.
        If `data` is shorter than `length`, zeros are appended.
    """
    return np.append(data, np.zeros(length - len(data)))


def split_data(
    data: np.ndarray, length: int = 1024, overlap_rate: float = 0.5
) -> list[np.ndarray]:
    """Split a 1D array into overlapping segments.

    Parameters
    ----------
    data : np.ndarray
        1D array

    length : int, default 1024
        Segment length (number of samples per segment).

    overlap_rate : float, default 0
        Overlap rate between consecutive segments.
        Should be in the range [0, 1).

    Returns
    -------
    splitted_data : list[np.ndarray]
        List of 1D arrays, each of length `length`.
        If the last segment is shorter than `length`, zero-padded at the end.
    """
    if not (0 <= overlap_rate < 1):
        raise ValueError("'overlap_rate' should be in the range [0, 1)")

    step = int((1 - overlap_rate) * length)

    splitted_data = []
    for i in range(0, len(data), step):
        splitted_data.append(back_padding(data[i : i + length], length))

    return splitted_data


def get_window_func(window: str, length: int) -> np.ndarray:
    """Get window function.

    Parameters
    ----------
    window : str
        Using window function name. ("hamming", "hann", "rect")

    filter_length: int
        Size of window function.

    Returns
    -------
    window_function : np.ndarray
        Corresponding window function.
    """
    if window == "hamming":
        return np.hamming(length)
    elif window == "hann":
        return np.hanning(length)
    elif window == "rect":
        return np.ones(length)


def stft(
    audio_data: np.ndarray,
    filter_length: int,
    overlap_rate: float,
    window: str,
) -> np.ndarray:
    """Calculate Short-Time Fourier Transform (STFT) of a 1D audio signal.

    Parameters
    ----------
    audio_data : np.ndarray
        1D array of audio signal.

    filter_length: int
        Size of filter.

    overlap_rate: float
        Overlap rate for splitting data. (0 ~ 1)

    window : str
        Using window function name. ("hamming", "hann", "rect")

    Returns
    -------
    spectrogram : np.ndarray
        2D array of complex STFT coefficients.
    """
    # get window function
    window_func = get_window_func(window, filter_length)

    stft_result = []
    for splitted_audio_data in split_data(audio_data, filter_length, overlap_rate):
        # apply window function
        splitted_audio_data *= window_func

        # process FFT
        fft_result = np.fft.fft(splitted_audio_data)

        # add to result
        stft_result.append(fft_result)

    return np.stack(stft_result, axis=1)


def inv_stft(
    stft_result: np.ndarray[complex],
    original_audio_frame: int,
    filter_length: int,
    overlap_rate: float,
    window: str,
    eps: float = 1e-7,
) -> np.ndarray:
    """Reconstruct a 1D audio signal from STFT result.

    Parameters
    ----------
    stft_result : np.ndarray[complex]
        2D array of complex STFT coefficients.

    original_audio_frame: int
        Total number of frames in the original audio signal before STFT.

    filter_length: int
        Size of filter.

    overlap_rate: float
        Overlap rate for splitting data. (0 ~ 1)

    window : str
        Using window function name. ("hamming", "hann", "rect")

    eps : float, default 1e-7
        Small constant added to avoid taking the logarithm of zero.

    Returns
    -------
    audio_data : np.ndarray
        1D array of audio signal.
    """
    # get window function
    window_func = get_window_func(window, filter_length)

    step = int((1 - overlap_rate) * filter_length)

    audio_data = np.zeros(original_audio_frame + filter_length)
    norm = np.zeros(original_audio_frame + filter_length)
    for i, fft_result in enumerate(stft_result.T):
        # process IFFT
        ifft_result = np.fft.ifft(fft_result)

        # use only real value
        ifft_result = np.real(ifft_result)

        # prepare to remove window function
        ifft_result *= window_func
        norm[step * i : step * i + filter_length] += window_func**2

        # add to result
        audio_data[step * i : step * i + filter_length] += ifft_result

    # remove window function
    norm = np.clip(norm, eps, None)
    audio_data /= norm

    return np.array(audio_data[:original_audio_frame])
