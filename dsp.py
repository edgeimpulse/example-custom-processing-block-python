import jax.numpy as jnp
from jax.experimental import jax2tf
import jax.numpy as jnp
import tensorflow as tf

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes):
    get_features_fn = get_dsp_impl(implementation_version, len(axes), sampling_freq, scale_axes)

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
def get_dsp_impl(implementation_version, axes_length, sampling_freq, scale_axes):
    def get_features(raw_data):
        # data is interleaved (so a,b,c,a,b,c) so transpose and reshape - this yields one
        # row per axis (e.g. [ [ a, a ], [ b, b ], [ c, c ] ])
        raw_data = raw_data.transpose().reshape(-1, axes_length)

        # multiply by scale_axes
        raw_data = raw_data * scale_axes

        # and transpose back and flatten back
        return raw_data.flatten()

    return get_features

# Get the corresponding TFLite model that maps to generate_features w/ the same parameters
def get_tflite_implementation(implementation_version, input_shape, axes, sampling_freq, scale_axes):
    get_features_fn = get_dsp_impl(implementation_version, len(axes), sampling_freq, scale_axes)

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
