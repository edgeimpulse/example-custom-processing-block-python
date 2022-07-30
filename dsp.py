import pandas as pd
import heartpy as hp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, sosfilt
from scipy.signal import find_peaks
from scipy import stats
import io, base64

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, filter_low, filter_high):
    # features is a 1D array, reshape so we have a matrix
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(float(raw_data[ix][ax]))

        sos = butter(6, [float(filter_low), float(filter_high)], 'band', output='sos', fs=sampling_freq)
        slice = sosfilt(sos, X)

        peaks, _ = find_peaks(slice, prominence=200, distance=0.3*sampling_freq)
        hr = len(peaks) * (60*sampling_freq / (len(X)))
        features.append(hr)

        if draw_graphs:
            plt.cla()
            plt.plot(peaks, slice[peaks], "xr");
            plt.plot(slice)

            buf = io.BytesIO()

            plt.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0)

            buf.seek(0)
            image = (base64.b64encode(buf.getvalue()).decode('ascii'))

            buf.close()

            graphs.append({
                'name': 'PPG Peaks (after filter)',
                'image': image,
                'imageMimeType': 'image/svg+xml',
                'type': 'image'
            })

    return {
        'features': features,
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        'fft_used': [],
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(features)
            }
        }
    }
