import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib import cm
import io, base64, os

matplotlib.use('Svg')

eng = matlab.engine.start_matlab()

dir_path = os.path.dirname(os.path.realpath(__file__))
temp_file = os.path.join(dir_path, 'temp.png')

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, window, noverlap, fft):
    # features is a 1D array, reshape so we have a matrix
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    width = 0
    height = 0

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(float(raw_data[ix][ax]))

        # X now contains only the current axis
        fx = np.array(X)

        # normalize to -1..1
        fx = fx / 32767

        mfe = np.array(eng.custom_spectrogram(matlab.double(fx.tolist()), temp_file if draw_graphs else "",
            window, noverlap, fft))

        flattened = mfe.flatten()

        features = np.concatenate((features, flattened))

        width = np.shape(mfe)[0]
        height = np.shape(mfe)[1]

        if draw_graphs:
            with open(temp_file, 'rb') as f:
                data = f.read()
                image = (base64.b64encode(data).decode('ascii'))

                graphs.append({
                    'name': 'Spectrogram',
                    'image': image,
                    'imageMimeType': 'image/png',
                    'type': 'image'
                })

    return {
        'features': features,
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        'fft_used': [],
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'spectrogram',
            'shape': {
                'width': width,
                'height': height
            }
        }
    }
