#!/usr/bin/env python3

import math
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram


def is_raw_accelerometer_csv(csv_path: Path) -> bool:
    with open(csv_path, 'r') as f:
        line = f.readline()
        return bool(line) and line.startswith('#time,accel_x,accel_y,accel_z')


def load_accelerometer_data(csv_path: Path) -> np.ndarray:
    if is_raw_accelerometer_csv(csv_path):
        return np.loadtxt(csv_path, comments='#', delimiter=',')

    raise ValueError(
        f'File {csv_path} does not contain raw accelerometer data and therefore '
        'is not supported by this script. Please use the official Klipper '
        'calibrate_shaper.py script to process it instead.')


# This is Klipper's spectrogram generation function adapted to use Scipy
def compute_spectrogram(
        data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = data.shape[0]
    Fs = N / (data[-1, 0] - data[0, 0])
    # Round up to a power of 2 for faster FFT
    M = 1 << int(.5 * Fs - 1).bit_length()
    window = np.kaiser(M, 6.)

    def _specgram(x):
        return spectrogram(x,
                           fs=Fs,
                           window=window,
                           nperseg=M,
                           noverlap=M // 2,
                           detrend='constant',
                           scaling='density',
                           mode='psd')

    d = {'x': data[:, 1], 'y': data[:, 2], 'z': data[:, 3]}
    f, t, pdata = _specgram(d['x'])
    for axis in 'yz':
        pdata += _specgram(d[axis])[2]
    return pdata, t, f


# Compute natural resonant frequency and damping ratio by using the half power bandwidth method with interpolated frequencies
def compute_mechanical_parameters(
        psd: np.ndarray, freqs: np.ndarray) -> tuple[float, float, np.intp]:
    max_power_index: np.intp = np.argmax(psd)
    fr: float = freqs[max_power_index]
    max_power: float = psd[max_power_index]

    half_power = max_power / math.sqrt(2)
    idx_below = np.where(psd[:max_power_index] <= half_power)[0][-1]
    idx_above = np.where(
        psd[max_power_index:] <= half_power)[0][0] + max_power_index
    freq_below_half_power = freqs[idx_below] + (half_power - psd[idx_below]) * (
        freqs[idx_below + 1] - freqs[idx_below]) / (psd[idx_below + 1] -
                                                    psd[idx_below])
    freq_above_half_power = freqs[idx_above -
                                  1] + (half_power - psd[idx_above - 1]) * (
                                      freqs[idx_above] - freqs[idx_above - 1]
                                  ) / (psd[idx_above] - psd[idx_above - 1])

    bandwidth = freq_above_half_power - freq_below_half_power
    bw1 = math.pow(bandwidth / fr, 2)
    bw2 = math.pow(bandwidth / fr, 4)

    zeta = math.sqrt(0.5 - math.sqrt(1 / (4 + 4 * bw1 - bw2)))

    return fr, zeta, max_power_index


# This find all the peaks in a curve by looking at when the derivative term goes from positive to negative
# Then only the peaks found above a threshold are kept to avoid capturing peaks in the low amplitude noise of a signal
def detect_peaks(data: np.ndarray,
                 indices: np.ndarray,
                 detection_threshold: float,
                 relative_height_threshold: float | None = None,
                 window_size=5,
                 vicinity=3) -> tuple[int, np.ndarray, np.ndarray]:
    # Smooth the curve using a moving average to avoid catching peaks everywhere in noisy signals
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, kernel, mode='valid')
    mean_pad = [np.mean(data[:window_size])] * (window_size // 2)
    smoothed_data = np.concatenate((mean_pad, smoothed_data))

    # Find peaks on the smoothed curve
    smoothed_peaks = np.where((smoothed_data[:-2] < smoothed_data[1:-1])
                              &
                              (smoothed_data[1:-1] > smoothed_data[2:]))[0] + 1
    smoothed_peaks = smoothed_peaks[smoothed_data[smoothed_peaks] >
                                    detection_threshold]

    # Additional validation for peaks based on relative height
    valid_peaks = smoothed_peaks
    if relative_height_threshold is not None:
        valid_peaks = []
        for peak in smoothed_peaks:
            peak_height = smoothed_data[peak] - np.min(
                smoothed_data[max(0, peak -
                                  vicinity):min(len(smoothed_data), peak +
                                                vicinity + 1)])
            if peak_height > relative_height_threshold * smoothed_data[peak]:
                valid_peaks.append(peak)

    # Refine peak positions on the original curve
    refined_peaks = []
    for peak in valid_peaks:
        local_max = peak + np.argmax(
            data[max(0, peak - vicinity):min(len(data), peak + vicinity +
                                             1)]) - vicinity
        refined_peaks.append(local_max)

    num_peaks = len(refined_peaks)

    return num_peaks, np.array(refined_peaks), indices[refined_peaks]
