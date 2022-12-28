import math
import jax.numpy as jnp
import jax

def lowpass_filter(raw_data, sampling_freq, filter_cutoff, filter_order):

    n_steps = int(filter_order / 2)
    a = math.tan(math.pi * filter_cutoff / sampling_freq)
    a2 = math.pow(a, 2)

    R = jnp.sin(math.pi * ((2.0 * jnp.arange(0, 3)) + 1.0) / (2.0 * 6))
    sampling_freq = a2 + (2.0 * a * R) + 1.0
    A = a2 / sampling_freq
    d1 = 2.0 * (1 - a2) / sampling_freq
    d2 = -(a2 - (2.0 * a * R) + 1.0) / sampling_freq

    def per_feature_inner(data_ix, val_to_unpack_1):
        out_state, w0, w1, w2 = val_to_unpack_1

        # print('per_feature_inner', data_ix)

        # inner loop (ran for every feature in the signal)
        def filter_inner_fn(i, val_to_unpack_2):
            dest, w0, w1, w2 = val_to_unpack_2

            # print('w0 setting to', d1[i] * w1[i] + d2[i] * w2[i] + dest[0])
            w0 = w0.at[i].set(d1[i] * w1[i] + d2[i] * w2[i] + dest)
            # print('w0 is now', w0[i])
            dest = A[i] * (w0[i] + (2.0 * w1[i]) + w2[i])
            w2 = w2.at[i].set(w1[i])
            w1 = w1.at[i].set(w0[i])
            # print('i', i, 'dest', dest)
            return [ dest, w0, w1, w2 ]

        # data = jnp.r
        # epeat(raw_data[data_ix], n_steps)
        res = jax.lax.fori_loop(0, n_steps,
            filter_inner_fn,
            [ raw_data[data_ix], w0, w1, w2 ])

        out_state = out_state.at[data_ix].set(res[0])
        w0 = res[1]
        w1 = res[2]
        w2 = res[3]

        return [ out_state, w0, w1, w2 ]


    res = jax.lax.fori_loop(0, jnp.size(raw_data), per_feature_inner,
        # initial state
        [
            # out vector
            jnp.zeros(jnp.size(raw_data)),
            # w0
            jnp.zeros(n_steps),
            # w1
            jnp.zeros(n_steps),
            # w2
            jnp.zeros(n_steps)
        ])

    return res[0]