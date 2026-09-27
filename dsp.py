import numpy as np
from scipy import signal


def _filter(signal_data, sampling_freq, cutoff, filter_type, filter_order):
    """Apply a short-window-safe, zero-phase Butterworth filter."""
    nyquist = sampling_freq / 2.0
    if len(signal_data) < 12 or cutoff <= 0 or cutoff >= nyquist:
        return signal_data.copy()

    sos = np.asarray(signal.butter(filter_order, cutoff, btype=filter_type, fs=sampling_freq, output='sos'))
    pad_length = min(len(signal_data) - 1, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, signal_data, padlen=pad_length)


def _statistics(values):
    """Return robust statistics in a stable, documented order."""
    centered = values - np.mean(values)
    return [
        np.mean(values),
        np.std(values),
        np.sqrt(np.mean(values * values)),
        np.min(values),
        np.max(values),
        np.median(values),
        np.percentile(values, 10),
        np.percentile(values, 90),
        np.percentile(values, 75) - np.percentile(values, 25),
        np.mean(np.abs(centered)),
        np.sqrt(np.mean(centered ** 4)) / (np.std(values) ** 2 + 1e-12),
    ]


def _power_spectrum(values, sampling_freq, spectral_window, fft_lengths_used=None):
    """Return frequencies and normalized power from a Hann-windowed spectrum."""
    windows = {
        'hanning': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'rectangular': lambda length: np.ones(length),
    }
    try:
        window = windows[spectral_window](len(values))
    except KeyError as error:
        raise ValueError('spectral_window must be hanning, hamming, blackman, or rectangular') from error
    windowed = (values - np.mean(values)) * window
    fft_length = 1 << (len(windowed) - 1).bit_length()
    if fft_lengths_used is not None:
        fft_lengths_used.append(fft_length)
    spectrum = np.abs(np.fft.rfft(windowed, n=fft_length)) ** 2
    frequencies = np.fft.rfftfreq(fft_length, 1.0 / sampling_freq)
    if len(spectrum) > 1:
        spectrum = spectrum[1:]
        frequencies = frequencies[1:]
    return frequencies, spectrum / (np.sum(spectrum) + 1e-12)


def _spectral_features(values, sampling_freq, spectral_window, fft_lengths_used=None):
    """Summarize a Hann-windowed one-sided power spectrum."""
    frequencies, probabilities = _power_spectrum(
        values, sampling_freq, spectral_window, fft_lengths_used
    )

    band_edges = (0.0, 0.5, 2.0, 5.0, 10.0, 20.0)
    band_power = []
    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        band_power.append(np.sum(probabilities[(frequencies >= lower) & (frequencies < upper)]))

    if len(probabilities) <= 1:
        spectral_entropy = 0.0
    else:
        spectral_entropy = -np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(len(probabilities))
    return [
        *band_power,
        frequencies[np.argmax(probabilities)] if len(probabilities) else 0.0,
        np.sum(frequencies * probabilities),
        spectral_entropy,
        np.max(probabilities) if len(probabilities) else 0.0,
    ]


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes,
                     gravity_cutoff=0.7, filter_order=2, spectral_window='hanning'):
    """Extract orientation-, motion-, and frequency-aware accelerometer features."""
    raw_data = np.asarray(raw_data, dtype=float).reshape(-1)
    if len(axes) < 1 or raw_data.size == 0 or raw_data.size % len(axes) != 0:
        raise ValueError('raw_data must contain complete samples for every axis')
    if sampling_freq <= 0:
        raise ValueError('sampling_freq must be greater than zero')
    if gravity_cutoff <= 0 or gravity_cutoff >= sampling_freq / 2.0:
        raise ValueError('gravity_cutoff must be between zero and the Nyquist frequency')
    if filter_order < 1 or filter_order > 4:
        raise ValueError('filter_order must be between 1 and 4')

    samples = raw_data.reshape(-1, len(axes)) * float(scale_axes)
    axis_names = [str(axis) for axis in axes]
    gravity = np.column_stack([
        _filter(samples[:, index], sampling_freq, gravity_cutoff, 'lowpass', filter_order)
        for index in range(samples.shape[1])
    ])
    motion = samples - gravity
    magnitude = np.linalg.norm(samples, axis=1)
    motion_magnitude = np.linalg.norm(motion, axis=1)

    features = []
    labels = []
    fft_lengths_used = []

    def add_group(name, values, group_labels):
        group_values = np.asarray(values, dtype=float)
        for label, value in zip(group_labels, _statistics(group_values)):
            labels.append(name + '_' + label)
            features.append(value)

    statistic_labels = ('mean', 'std', 'rms', 'min', 'max', 'median', 'p10', 'p90', 'iqr', 'mad', 'kurtosis')
    for index, name in enumerate(axis_names):
        add_group(name + '_gravity', gravity[:, index], statistic_labels)
        add_group(name + '_motion', motion[:, index], statistic_labels)

    add_group('magnitude', magnitude, statistic_labels)
    add_group('motion_magnitude', motion_magnitude, statistic_labels)

    spectral_labels = ('band_0_0_5', 'band_0_5_2', 'band_2_5', 'band_5_10', 'band_10_20',
                       'dominant_hz', 'centroid_hz', 'entropy', 'peak_ratio')
    for name, values in [('motion_magnitude', motion_magnitude), ('magnitude', magnitude)]:
        for label, value in zip(
            spectral_labels,
            _spectral_features(values, sampling_freq, spectral_window, fft_lengths_used),
        ):
            labels.append(name + '_' + label)
            features.append(value)

    if samples.shape[1] > 1:
        correlation = np.eye(samples.shape[1])
        motion_std = np.std(motion, axis=0)
        for first in range(samples.shape[1]):
            for second in range(first + 1, samples.shape[1]):
                if motion_std[first] > 1e-12 and motion_std[second] > 1e-12:
                    covariance = np.mean(
                        (motion[:, first] - np.mean(motion[:, first]))
                        * (motion[:, second] - np.mean(motion[:, second]))
                    )
                    correlation[first, second] = covariance / (motion_std[first] * motion_std[second])
                labels.append(axis_names[first] + '_' + axis_names[second] + '_motion_corr')
                features.append(correlation[first, second])

    graphs = []
    if draw_graphs:
        time_ms = (np.arange(samples.shape[0]) / sampling_freq * 1000.0).tolist()
        gravity_signals = {}
        motion_signals = {}
        for index, name in enumerate(axis_names):
            gravity_signals[name] = gravity[:, index].tolist()
            motion_signals[name] = motion[:, index].tolist()

        for name, values in [
            ('Estimated gravity', gravity_signals),
            ('Dynamic motion', motion_signals),
            ('Signal magnitude', {
                'total': magnitude.tolist(),
                'motion': motion_magnitude.tolist(),
            }),
        ]:
            graphs.append({
                'name': name,
                'X': values,
                'y': time_ms,
                'suggestedYMin': float(np.min(samples)),
                'suggestedYMax': float(np.max(samples)),
            })

        spectrum_frequencies, spectrum_power = _power_spectrum(
            motion_magnitude, sampling_freq, spectral_window, fft_lengths_used
        )
        graphs.append({
            'name': 'Motion spectrum',
            'X': {'normalized power': spectrum_power.tolist()},
            'y': spectrum_frequencies.tolist(),
            'suggestedYMin': 0,
            'suggestedYMax': 1.0,
        })

    return {
        'features': [float(value) for value in features],
        'labels': labels,
        'graphs': graphs,
        'fft_used': sorted(set(fft_lengths_used)),
        'output_config': {
            'type': 'flat',
            'shape': {'width': len(features)},
        },
    }
