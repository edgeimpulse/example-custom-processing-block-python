import jax.numpy as jnp
from jax.experimental import jax2tf
import tensorflow as tf

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes,
                      average, minimum, maximum, rms, stdev):
    get_features_fn = get_dsp_impl(implementation_version, len(axes), sampling_freq, scale_axes,
        average, minimum, maximum, rms, stdev)

    features = get_features_fn(jnp.array(raw_data))
    graphs = []

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
                'width': features.size
            }
        }
    }

# this returns a function to generate features
def get_dsp_impl(implementation_version, axes_length, sampling_freq, scale_axes,
                 average, minimum, maximum, rms, stdev):
    # calc out feature count outside of the function
    out_feature_count = jnp.count_nonzero(jnp.array([ average, minimum, maximum, rms, stdev ]))

    def get_features(raw_data):
        # data is interleaved (so a,b,c,a,b,c) so reshape and transpose - this yields one
        # row per axis (e.g. [ [ a, a ], [ b, b ], [ c, c ] ])
        raw_data = raw_data.reshape((-1, axes_length)).transpose()

        out_features = jnp.zeros(out_feature_count)
        out_feature_ix = 0

        features = []

        # multiply by scale_axes
        raw_data = raw_data * scale_axes

        if (average):
            features.append(jnp.average(raw_data, axis=-1))

        if (minimum):
            features.append(jnp.min(raw_data, axis=-1))

        if (maximum):
            features.append(jnp.max(raw_data, axis=-1))

        if (rms):
            features.append(jnp.sqrt(jnp.mean(jnp.square(raw_data), axis=-1)))

        if (stdev):
            features.append(jnp.std(raw_data, axis=-1))

        # and transpose back and flatten back
        return jnp.array(features).transpose().flatten()

    return get_features

# Get the corresponding TFLite model that maps to generate_features w/ the same parameters
def get_tflite_implementation(implementation_version, input_shape, axes, sampling_freq, scale_axes,
                              average, minimum, maximum, rms, stdev):
    print('get_tflite_implementation', input_shape)

    get_features_fn = get_dsp_impl(implementation_version, len(axes), sampling_freq, scale_axes,
        average, minimum, maximum, rms, stdev)

    tf_predict = tf.function(
        jax2tf.convert(get_features_fn, enable_xla=False),
        input_signature=[
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')
        ],
        autograph=False)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        # tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_float_model = converter.convert()
    return tflite_float_model
