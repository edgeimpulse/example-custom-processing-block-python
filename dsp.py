import json
import numpy as np
import sys

def generate_features(draw_graphs, raw_data, axes, sampling_freq, scale_axes):
    if (scale_axes == 1):
        return { 'features': raw_data, 'graphs': [] }

    return { 'features': raw_data * scale_axes, 'graphs': [] }
