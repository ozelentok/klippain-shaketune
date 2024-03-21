#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from .common import load_accelerometer_data


def accel_signal_filter(data: np.ndarray,
                        cutoff=2,
                        fs=100,
                        order=5) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    filtered_data -= np.mean(filtered_data)
    return filtered_data


def find_first_spike(data: np.ndarray) -> tuple[str, np.intp]:
    min_index, max_index = np.argmin(data), np.argmax(data)
    return ('-', min_index) if min_index < max_index else ('', max_index)


def map_axes(csv_path: Path) -> str:
    raw_data = load_accelerometer_data(csv_path)

    filtered_data = [accel_signal_filter(raw_data[:, i + 1]) for i in range(3)]
    spikes = [find_first_spike(filtered_data[i]) for i in range(3)]
    spikes_sorted = sorted([(spikes[0], 'x'), (spikes[1], 'y'),
                            (spikes[2], 'z')],
                           key=lambda x: x[0][1])

    axes_map = ','.join([f'{s[0][0]}{s[1]}' for s in spikes_sorted])
    results = f'Detected axes_map: {axes_map}'

    return results
