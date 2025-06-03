"""Provides FIR filters (low-pass, high-pass, band-pass, and band-stop filters)."""

import numpy as np
from stft import get_window_func


def sinc(x: np.ndarray) -> np.ndarray:
    """Compute the unnormalized sinc function.

    This function computes sinc(0) = 1, otherwise sinc(x) = sin(x) / x.

    Parameters
    ----------
    x : np.ndarray
        Input array of values at which to compute the sinc function.

    Returns
    -------
    y : np.ndarray
        Output array of the same shape as `x`, containing the sinc values.
    """
    return np.sinc(x / np.pi)


def get_filter(  # noqa: PLR0913
    filter_name: str,
    length: int,
    sample_rate: int,
    window: str,
    low_freq: int | None = None,
    high_freq: int | None = None,
) -> np.ndarray:
    """Generate a FIR filter of the specified type.

    Parameters
    ----------
    filter_name : str
        Type of filter to generate. Must be one of:
        - 'lpf' (low-pass filter)
        - 'hpf' (high-pass filter)
        - 'bpf' (band-pass filter)
        - 'bsf' (band-stop filter)
    length : int
        Length of the filter.
    sample_rate : int
        Sampling rate of the signal the filter will be applied to, in Hz.
    window : str
        Name of the window function to use in filter design (e.g., 'hamming', 'hann').
    low_freq : int or None, optional
        Lower cutoff frequency in Hz (used for 'hpf', 'bpf', 'bsf').
    high_freq : int or None, optional
        Upper cutoff frequency in Hz (used for 'lpf', 'bpf', 'bsf').

    Returns
    -------
    filter : np.ndarray
        The FIR filter as a 1D array.
    """
    if filter_name == "lpf":
        return low_pass_filter(length, high_freq, sample_rate, window)
    elif filter_name == "hpf":
        return high_pass_filter(length, low_freq, sample_rate, window)
    elif filter_name == "bpf":
        return band_pass_filter(length, low_freq, high_freq, sample_rate, window)
    elif filter_name == "bsf":
        return band_stop_filter(length, low_freq, high_freq, sample_rate, window)
    else:
        raise ValueError(f"Unknown filter '{filter_name}'.")


def low_pass_filter(
    length: int, high_freq: int, sample_rate: int, window: str
) -> np.ndarray:
    """Generate low-pass FIR filter.

    Parameters
    ----------
    length : int
        Length of the filter.
    high_freq : int
        Cutoff frequency in Hz. Frequencies above this will be attenuated.
    sample_rate : int
        Sampling rate of the signal to which the filter will be applied, in Hz.
    window : str
        Name of the window function to apply to the ideal filter.

    Returns
    -------
    filter : np.ndarray
        The designed low-pass FIR filter as a 1D array.
    """
    window_func = get_window_func(window, length)

    high_ang_freq = high_freq / (sample_rate // 2) * np.pi
    n = np.arange(length) - (length // 2)

    ideal_lpf = high_ang_freq * sinc(high_ang_freq * n) / np.pi

    return ideal_lpf * window_func


def high_pass_filter(
    length: int, low_freq: int, sample_rate: int, window: str
) -> np.ndarray:
    """Generate high-pass FIR filter.

    Parameters
    ----------
    length : int
        Length of the filter.
    low_freq : int
        Cutoff frequency in Hz. Frequencies below this will be attenuated.
    sample_rate : int
        Sampling rate of the signal to which the filter will be applied, in Hz.
    window : str
        Name of the window function to apply to the ideal filter.

    Returns
    -------
    filter : np.ndarray
        The designed high-pass FIR filter as a 1D array.
    """
    window_func = get_window_func(window, length)

    low_ang_freq = low_freq / (sample_rate // 2) * np.pi
    n = np.arange(length) - (length // 2)

    ideal_hpf = sinc(n * np.pi) - low_ang_freq * sinc(low_ang_freq * n) / np.pi

    return ideal_hpf * window_func


def band_pass_filter(
    length: int, low_freq: int, high_freq: int, sample_rate: int, window: str
) -> np.ndarray:
    """Generate band-pass FIR filter.

    Parameters
    ----------
    length : int
        Length of the filter.
    low_freq : int
        Lower cutoff frequency in Hz. Frequencies below this will be attenuated.
    high_freq : int
        Upper cutoff frequency in Hz. Frequencies above this will be attenuated.
    sample_rate : int
        Sampling rate of the signal to which the filter will be applied, in Hz.
    window : str
        Name of the window function to apply to the ideal filter.

    Returns
    -------
    filter : np.ndarray
        The designed band-pass FIR filter as a 1D array.
    """
    window_func = get_window_func(window, length)

    low_ang_freq = low_freq / (sample_rate // 2) * np.pi
    high_ang_freq = high_freq / (sample_rate // 2) * np.pi
    n = np.arange(length) - (length // 2)

    ideal_bpf = (
        high_ang_freq * sinc(high_ang_freq * n) - low_ang_freq * sinc(low_ang_freq * n)
    ) / np.pi

    return ideal_bpf * window_func


def band_stop_filter(
    length: int, low_freq: int, high_freq: int, sample_rate: int, window: str
) -> np.ndarray:
    """Generate band-stop FIR filter.

    Parameters
    ----------
    length : int
        Length of the filter.
    low_freq : int
        Lower cutoff frequency in Hz. Frequencies below this will pass through.
    high_freq : int
        Upper cutoff frequency in Hz. Frequencies above this will pass through.
    sample_rate : int
        Sampling rate of the signal to which the filter will be applied, in Hz.
    window : str
        Name of the window function to apply to the ideal filter.

    Returns
    -------
    filter : np.ndarray
        The designed band-stop FIR filter as a 1D array.
    """
    window_func = get_window_func(window, length)

    low_ang_freq = low_freq / (sample_rate // 2) * np.pi
    high_ang_freq = high_freq / (sample_rate // 2) * np.pi
    n = np.arange(length) - (length // 2)

    ideal_bsf = (
        sinc(n * np.pi)
        - (
            high_ang_freq * sinc(high_ang_freq * n)
            - low_ang_freq * sinc(low_ang_freq * n)
        )
        / np.pi
    )

    return ideal_bsf * window_func
