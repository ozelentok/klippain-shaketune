#!/usr/bin/env python3

import operator
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.font_manager
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')

from ..shaper_calibrate import ShaperCalibrate
from .common import (compute_mechanical_parameters, detect_peaks,
                     load_accelerometer_data)

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_RELATIVE_HEIGHT_THRESHOLD = 0.04
VALLEY_DETECTION_THRESHOLD = 0.1  # Lower is more sensitive

KLIPPAIN_COLORS = {
    "purple": "#70088C",
    "orange": "#FF8D32",
    "dark_purple": "#150140",
    "dark_orange": "#F24130",
    "red_pink": "#F2055C"
}

######################################################################
# Computation
######################################################################


# Call to the official Klipper input shaper object to do the PSD computation
def calc_freq_response(data: np.ndarray):
    helper = ShaperCalibrate(printer=None)
    return helper.process_accelerometer_data(data)


def compute_vibration_spectrogram(datas, group, max_freq):
    psd_list = []
    first_freqs = None
    signal_axes = ['x', 'y', 'z', 'all']

    for i in range(0, len(datas), group):
        # Round up to the nearest power of 2 for faster FFT
        N = datas[i].shape[0]
        T = datas[i][-1, 0] - datas[i][0, 0]
        M = 1 << int((N / T) * 0.5 - 1).bit_length()
        if N <= M:
            # If there is not enough lines in the array to be able to round up to the
            # nearest power of 2, we need to pad some zeros at the end of the array to
            # avoid entering a blocking state from Klipper shaper_calibrate.py
            datas[i] = np.pad(datas[i], [(0, (M - N) + 1), (0, 0)],
                              mode='constant',
                              constant_values=0)

        freqrsp = calc_freq_response(datas[i])
        for n in range(group - 1):
            data = datas[i + n + 1]

            # Round up to the nearest power of 2 for faster FFT
            N = data.shape[0]
            T = data[-1, 0] - data[0, 0]
            M = 1 << int((N / T) * 0.5 - 1).bit_length()
            if N <= M:
                # If there is not enough lines in the array to be able to round up to the
                # nearest power of 2, we need to pad some zeros at the end of the array to
                # avoid entering a blocking state from Klipper shaper_calibrate.py
                data = np.pad(data, [(0, (M - N) + 1), (0, 0)],
                              mode='constant',
                              constant_values=0)

            freqrsp.add_data(calc_freq_response(data))

        if not psd_list:
            # First group, just put it in the result list
            first_freqs = freqrsp.freq_bins
            psd = freqrsp.psd_sum[first_freqs <= max_freq]
            px = freqrsp.psd_x[first_freqs <= max_freq]
            py = freqrsp.psd_y[first_freqs <= max_freq]
            pz = freqrsp.psd_z[first_freqs <= max_freq]
            psd_list.append([psd, px, py, pz])
        else:
            # Not the first group, we need to interpolate every new signals
            # to the first one to equalize the frequency_bins between them
            signal_normalized = dict()
            freqs = freqrsp.freq_bins
            for axe in signal_axes:
                signal = freqrsp.get_psd(axe)
                signal_normalized[axe] = np.interp(first_freqs, freqs, signal)

            # Remove data above max_freq on all axes and add to the result list
            psd = signal_normalized['all'][first_freqs <= max_freq]
            px = signal_normalized['x'][first_freqs <= max_freq]
            py = signal_normalized['y'][first_freqs <= max_freq]
            pz = signal_normalized['z'][first_freqs <= max_freq]
            psd_list.append([psd, px, py, pz])

    return np.array(first_freqs[first_freqs <= max_freq]), np.array(psd_list)


def compute_speed_profile(speeds, freqs, psd_list):
    # Preallocate arrays as psd_list is known and consistent
    pwrtot_sum = np.zeros(len(psd_list))
    pwrtot_x = np.zeros(len(psd_list))
    pwrtot_y = np.zeros(len(psd_list))
    pwrtot_z = np.zeros(len(psd_list))

    for i, psd in enumerate(psd_list):
        pwrtot_sum[i] = np.trapz(psd[0], freqs)
        pwrtot_x[i] = np.trapz(psd[1], freqs)
        pwrtot_y[i] = np.trapz(psd[2], freqs)
        pwrtot_z[i] = np.trapz(psd[3], freqs)

    # Resample the signals to get a better detection of the valleys of low energy
    # and avoid getting limited by the speed increment defined by the user
    resampled_speeds, resampled_power_sum = resample_signal(speeds, pwrtot_sum)
    _, resampled_pwrtot_x = resample_signal(speeds, pwrtot_x)
    _, resampled_pwrtot_y = resample_signal(speeds, pwrtot_y)
    _, resampled_pwrtot_z = resample_signal(speeds, pwrtot_z)

    return resampled_speeds, [
        resampled_power_sum, resampled_pwrtot_x, resampled_pwrtot_y,
        resampled_pwrtot_z
    ]


def compute_motor_profile(power_spectral_densities):
    # Sum the PSD across all speeds for each frequency of the spectrogram. Basically this
    # is equivalent to sum up all the spectrogram column by column to plot the total on the right
    motor_total_vibration = np.sum([psd[0] for psd in power_spectral_densities],
                                   axis=0)

    # Then a very little smoothing of the signal is applied to avoid too much noise and sharp peaks on it and simplify
    # the resonance frequency and damping ratio estimation later on. Also, too much smoothing is bad and would alter the results
    smoothed_motor_total_vibration = np.convolve(motor_total_vibration,
                                                 np.ones(10) / 10,
                                                 mode='same')

    return smoothed_motor_total_vibration


# The goal is to find zone outside of peaks (flat low energy zones) to advise them as good speeds range to use in the slicer
def identify_low_energy_zones(power_total):
    valleys = []

    # Calculate the mean and standard deviation of the entire power_total
    mean_energy = np.mean(power_total)
    std_energy = np.std(power_total)

    # Define a threshold value as mean minus a certain number of standard deviations
    threshold_value = mean_energy - VALLEY_DETECTION_THRESHOLD * std_energy

    # Find valleys in power_total based on the threshold
    in_valley = False
    start_idx = 0
    for i, value in enumerate(power_total):
        if not in_valley and value < threshold_value:
            in_valley = True
            start_idx = i
        elif in_valley and value >= threshold_value:
            in_valley = False
            valleys.append((start_idx, i))

    # If the last point is still in a valley, close the valley
    if in_valley:
        valleys.append((start_idx, len(power_total) - 1))

    max_signal = np.max(power_total)

    # Calculate mean energy for each valley as a percentage of the maximum of the signal
    valley_means_percentage = []
    for start, end in valleys:
        if not np.isnan(np.mean(power_total[start:end])):
            valley_means_percentage.append(
                (start, end,
                 (np.mean(power_total[start:end]) / max_signal) * 100))

    # Sort valleys based on mean percentage values
    sorted_valleys = sorted(valley_means_percentage, key=lambda x: x[2])

    return sorted_valleys


# Resample the signal to achieve denser data points in order to get more precise valley placing and
# avoid having to use the original sampling of the signal (that is equal to the speed increment used for the test)
def resample_signal(speeds, power_total, new_spacing=0.1):
    new_speeds = np.arange(speeds[0], speeds[-1] + new_spacing, new_spacing)
    new_power_total = np.interp(new_speeds, speeds, power_total)
    return np.array(new_speeds), np.array(new_power_total)


######################################################################
# Graphing
######################################################################


def plot_speed_profile(ax, speeds, power_total, num_peaks, peaks,
                       low_energy_zones):
    # For this function, we have two array for the speeds. Indeed, since the power total sum was resampled to better detect
    # the valleys of low energy later on, we also need the resampled speed array to plot it. For the rest
    ax.set_title("Machine speed profile",
                 fontsize=14,
                 color=KLIPPAIN_COLORS['dark_orange'],
                 weight='bold')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('Energy')

    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)

    max_y = power_total[0].max() + power_total[0].max() * 0.05
    ax.set_xlim([speeds.min(), speeds.max()])
    ax.set_ylim([0, max_y])
    ax2.set_ylim([0, max_y])

    ax.plot(speeds, power_total[0], label="X+Y+Z", color='purple', zorder=5)
    ax.plot(speeds, power_total[1], label="X", color='red')
    ax.plot(speeds, power_total[2], label="Y", color='green')
    ax.plot(speeds, power_total[3], label="Z", color='blue')

    if peaks.size:
        ax.plot(speeds[peaks],
                power_total[0][peaks],
                "x",
                color='black',
                markersize=8)
        for idx, peak in enumerate(peaks):
            fontcolor = 'red'
            fontweight = 'bold'
            ax.annotate(f"{idx+1}", (speeds[peak], power_total[0][peak]),
                        textcoords="offset points",
                        xytext=(8, 5),
                        ha='left',
                        fontsize=13,
                        color=fontcolor,
                        weight=fontweight)
        ax2.plot([], [], ' ', label=f'Number of peaks: {num_peaks}')
    else:
        ax2.plot([], [], ' ', label=f'No peaks detected')

    for idx, (start, end, energy) in enumerate(low_energy_zones):
        ax.axvline(speeds[start],
                   color='red',
                   linestyle='dotted',
                   linewidth=1.5)
        ax.axvline(speeds[end], color='red', linestyle='dotted', linewidth=1.5)
        ax2.fill_between(
            speeds[start:end],
            0,
            power_total[0][start:end],
            color='green',
            alpha=0.2,
            label=
            f'Zone {idx+1}: {speeds[start]:.1f} to {speeds[end]:.1f} mm/s (mean energy: {energy:.2f}%)'
        )

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper left', prop=fontP)
    ax2.legend(loc='upper right', prop=fontP)

    return


def plot_vibration_spectrogram(ax, speeds, freqs, power_spectral_densities,
                               peaks, fr, max_freq):
    # Prepare the spectrum data
    spectrum = np.empty([len(freqs), len(speeds)])
    for i in range(len(speeds)):
        for j in range(len(freqs)):
            spectrum[j, i] = power_spectral_densities[i][0][j]

    ax.set_title("Vibrations spectrogram",
                 fontsize=14,
                 color=KLIPPAIN_COLORS['dark_orange'],
                 weight='bold')
    # ax.pcolormesh(speeds, freqs, spectrum, norm=matplotlib.colors.LogNorm(),
    #         cmap='inferno', shading='gouraud')

    ax.imshow(spectrum,
              norm=matplotlib.colors.LogNorm(),
              cmap='inferno',
              aspect='auto',
              extent=[speeds[0], speeds[-1], freqs[0], freqs[-1]],
              origin='lower',
              interpolation='antialiased')

    # Add peaks lines in the spectrogram to get hint from peaks found in the first graph
    if peaks is not None:
        for idx, peak in enumerate(peaks):
            ax.axvline(peak, color='cyan', linestyle='dotted', linewidth=0.75)
            ax.annotate(f"Peak {idx+1}", (peak, freqs[-1] * 0.9),
                        textcoords="data",
                        color='cyan',
                        rotation=90,
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='right')

    # Add motor resonance line
    if fr is not None and fr > 25:
        ax.axhline(fr, color='cyan', linestyle='dotted', linewidth=1)
        ax.annotate(f"Motor resonance", (speeds[-1] * 0.95, fr + 2),
                    textcoords="data",
                    color='cyan',
                    fontsize=10,
                    verticalalignment='bottom',
                    horizontalalignment='right')

    ax.set_ylim([0., max_freq])
    ax.set_ylabel('Frequency (hz)')
    ax.set_xlabel('Speed (mm/s)')

    return


def plot_motor_profile(ax, freqs, motor_vibration_power, motor_fr, motor_zeta,
                       motor_max_power_index):
    ax.set_title("Motors frequency profile",
                 fontsize=14,
                 color=KLIPPAIN_COLORS['dark_orange'],
                 weight='bold')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency (hz)')

    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)

    ax.set_ylim([freqs.min(), freqs.max()])
    ax.set_xlim(
        [0, motor_vibration_power.max() + motor_vibration_power.max() * 0.1])

    # Plot the profile curve
    ax.plot(motor_vibration_power, freqs, color=KLIPPAIN_COLORS['orange'])

    # Tag the resonance peak
    ax.plot(motor_vibration_power[motor_max_power_index],
            freqs[motor_max_power_index],
            "x",
            color='black',
            markersize=8)
    fontcolor = KLIPPAIN_COLORS['purple']
    fontweight = 'bold'
    ax.annotate(f"R", (motor_vibration_power[motor_max_power_index],
                       freqs[motor_max_power_index]),
                textcoords="offset points",
                xytext=(8, 8),
                ha='right',
                fontsize=13,
                color=fontcolor,
                weight=fontweight)

    # Add the legend
    ax2.plot([], [],
             ' ',
             label="Motor resonant frequency (ω0): %.1fHz" % (motor_fr))
    ax2.plot([], [], ' ', label="Motor damping ratio (ζ): %.3f" % (motor_zeta))

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(which='major', color='grey')
    ax.grid(which='minor', color='lightgrey')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    ax2.legend(loc='upper right', prop=fontP)

    return


######################################################################
# Startup and main routines
######################################################################


def extract_speed(csv_path: Path) -> float:
    speed_pattern = re.compile('SP(.+)N')
    match = speed_pattern.search(csv_path.name)
    if not match:
        raise ValueError(
            f'File {csv_path.name} does not contain speed in its name'
            'and therefore is not supported by this script.')
    return float(match.group(1).replace('_', '.'))


def sort_and_slice(speeds_data: list[tuple[float, np.ndarray]], remove):
    # Sort to get the speeds and their datas aligned and in ascending order
    raw_speeds, raw_datas = zip(
        *sorted(speeds_data, key=operator.itemgetter(0)))

    # Optionally remove the beginning and end of each data file to get only
    # the constant speed part of the segments and remove the start/stop phase
    sliced_datas = []
    for data in raw_datas:
        sliced = round((len(data) * remove / 100) / 2)
        sliced_datas.append(data[sliced:len(data) - sliced])

    return raw_speeds, sliced_datas


def graph_vibrations_test_results(csv_paths: list[Path],
                                  axis_name: str | None = None,
                                  accel: float | None = None,
                                  max_freq=1000.0,
                                  remove=0):
    speeds_data = [(extract_speed(p), load_accelerometer_data(p))
                   for p in csv_paths]
    speeds, datas = sort_and_slice(speeds_data, remove)
    del speeds_data

    # As we assume that we have the same number of files for each speed increment, we can group
    # the PSD results by this number (to combine all the segments of the pattern at a constant speed)
    group_by = speeds.count(speeds[0])

    # Remove speeds duplicates and graph the processed datas
    speeds = list(OrderedDict((x, True) for x in speeds).keys())

    # Compute speed profile, vibration spectrogram and motor resonance profile
    freqs, psd = compute_vibration_spectrogram(datas, group_by, max_freq)
    upsampled_speeds, speeds_powers = compute_speed_profile(speeds, freqs, psd)
    motor_vibration_power = compute_motor_profile(psd)

    # Peak detection and low energy valleys (good speeds) identification between the peaks
    num_peaks, vibration_peaks, peaks_speeds = detect_peaks(
        speeds_powers[0], upsampled_speeds,
        PEAKS_DETECTION_THRESHOLD * speeds_powers[0].max(),
        PEAKS_RELATIVE_HEIGHT_THRESHOLD, 10, 10)
    low_energy_zones = identify_low_energy_zones(speeds_powers[0])

    # Print the vibration peaks info in the console
    formated_peaks_speeds = ', '.join(f'{ps:.1f}' for ps in peaks_speeds)
    print(
        f'Vibrations peaks detected: {num_peaks} @ {formated_peaks_speeds} mm/s '
        '(avoid setting a speed near these values in your slicer print profile)'
    )

    # Motor resonance estimation
    motor_fr, motor_zeta, motor_max_power_index = compute_mechanical_parameters(
        motor_vibration_power, freqs)
    if motor_fr > 25:
        print(f'Motors have a main resonant frequency at {motor_fr:.1f}Hz with '
              f'an estimated damping ratio of {motor_zeta:.3f}')
    else:
        print(
            f'The detected resonance frequency of the motors is too low ({motor_fr:.1f}Hz). '
            ' (This is probably due to the test running with acceleration too high)'
        )
        print(
            'Try lowering the ACCEL value before restarting the macro to ensure that only '
            'constant speeds are recorded and that the dynamic behavior in the corners is not impacting the measurements.'
        )

    # Create graph layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                 2,
                                                 gridspec_kw={
                                                     'height_ratios': [4, 3],
                                                     'width_ratios': [5, 3],
                                                     'bottom': 0.050,
                                                     'top': 0.890,
                                                     'left': 0.057,
                                                     'right': 0.985,
                                                     'hspace': 0.166,
                                                     'wspace': 0.138
                                                 })
    ax2.remove()  # top right graph is not used and left blank for now...
    fig.set_size_inches(14, 11.6)

    # Add title
    title_line1 = "VIBRATIONS MEASUREMENT TOOL"
    fig.text(0.075,
             0.965,
             title_line1,
             ha='left',
             va='bottom',
             fontsize=20,
             color=KLIPPAIN_COLORS['purple'],
             weight='bold')
    try:
        filename_parts = csv_paths[0].name.split('_')
        dt = datetime.strptime(f"{filename_parts[1]} {filename_parts[2]}",
                               "%Y%m%d %H%M%S")
        title_line2 = dt.strftime('%x %X')
        if axis_name is not None:
            title_line2 += ' -- ' + str(axis_name).upper() + ' axis'
        if accel is not None:
            title_line2 += ' at ' + str(accel) + ' mm/s²'
    except:
        print(
            f'Warning: CSV filename look to be different than expected ({csv_paths[0].name})'
        )
        title_line2 = csv_paths[0].name
    fig.text(0.075,
             0.957,
             title_line2,
             ha='left',
             va='top',
             fontsize=16,
             color=KLIPPAIN_COLORS['dark_purple'])

    # Plot the graphs
    plot_speed_profile(ax1, upsampled_speeds, speeds_powers, num_peaks,
                       vibration_peaks, low_energy_zones)
    plot_motor_profile(ax4, freqs, motor_vibration_power, motor_fr, motor_zeta,
                       motor_max_power_index)
    plot_vibration_spectrogram(ax3, speeds, freqs, psd, peaks_speeds, motor_fr,
                               max_freq)

    return fig
