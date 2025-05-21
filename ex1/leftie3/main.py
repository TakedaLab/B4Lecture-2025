"""A solution to ex1 implementing STFT and ISTFT on an example audio file."""

import wave

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def main() -> None:
    """Process an audio file and display the results.

    Reads a .wav file and performs STFT and ISTFT on it, and displays the original waveform,
    the STFT spectrogram as well as the resynthesized waveform on the screen using matplotlib.

    Params: None
    Returns: None
    """
    # Path to .wav file
    WAV_PATH = "./ex1/leftie3/kuma_lq.wav"

    # Open .wav file with builtin python interface
    with wave.open(WAV_PATH, "rb") as file:
        # Get number of audio frames and sample rate
        nframes = file.getnframes()
        sample_rate = file.getframerate()
        # Read all frames and close file
        buffer = file.readframes(nframes)

    # Copy bytes to numpy array
    x = np.frombuffer(buffer, dtype=np.int16)

    # Create time vector to display time in seconds
    t = np.linspace(0, len(x) / sample_rate, num=len(x))

    # Size of each frame for which we compute FFT for the spectrogram
    FFT_SIZE = 256

    # Hanning window used on each frame of the signal
    window = np.hanning(FFT_SIZE)

    # Spectrogram data
    spec = []
    spec_raw = []

    for i in range(0, len(x) - FFT_SIZE, FFT_SIZE):
        # Raw frame FFT
        frame = x[i : i + FFT_SIZE]
        frame_freq_raw = np.fft.fft(frame)
        # Save raw fft output to use in inverse fft later
        spec_raw.append(frame_freq_raw)

        # Windowing needed for spectrogram
        frame_windowed = frame * window
        # Only save the first half of each FFT result
        frame_freq = np.abs(np.fft.fft(frame_windowed))[: FFT_SIZE // 2]
        spec.append(frame_freq)

    # Convert to numpy array and rotate so time is in x-axis
    spec = np.rot90(np.asarray(spec))

    # Inverse FFT/Resynthesis
    re_x = []

    for i in range(0, len(spec_raw)):
        temp = np.fft.ifft(spec_raw[i])
        re_x.append(temp)

    re_x = np.int16(np.asarray(re_x).flatten())

    # Create another time vector for the resynthesized signal
    re_t = np.linspace(0, len(re_x) / sample_rate, num=len(re_x))

    # Write resynthesized signal to file
    wavfile.write(f"{WAV_PATH[:-4]}_resynthesized.wav", sample_rate, re_x)

    # -----Plotting-----
    plt.figure(1, (8, 8))

    # Original signal
    plt.subplot(311)
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.ylim(-1, 1)
    plt.plot(t, x / np.iinfo(np.int16).max)  # Normalize magnitude

    # Spectrogram
    plt.subplot(312)
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    # X-axis ticks
    xlocs = np.float32(np.linspace(0, spec.shape[1] - 1, 6))
    plt.xticks(
        xlocs,
        [
            "%.02f" % (i * spec.shape[1] * FFT_SIZE / (spec.shape[1] * sample_rate))
            for i in xlocs
        ],
    )

    # Y-axis ticks
    ylocs = np.int32(np.round(np.linspace(0, spec.shape[0] - 1, 6)))
    plt.yticks(
        ylocs,
        [
            "%.0f" % ((((FFT_SIZE // 2) - i - 1) * sample_rate) / FFT_SIZE)
            for i in ylocs
        ],
    )

    plt.imshow(spec, vmin=np.min(spec), vmax=np.max(spec), aspect="auto")

    # Resynthesized signal
    plt.subplot(313)
    plt.title("Resynthesized Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.ylim(-1, 1)
    plt.plot(re_t, re_x / np.iinfo(np.int16).max)  # Normalize magnitude

    plt.tight_layout(h_pad=1.2)  # Padding to prevent text overlap
    plt.show()


if __name__ == "__main__":
    main()
