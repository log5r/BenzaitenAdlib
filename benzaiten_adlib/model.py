"""Keras 3 VAE with the parameter layout used by the original TFP model."""
import tensorflow as tf

ENCODED_DIM = 32
LSTM_DIM = 1024


@tf.keras.utils.register_keras_serializable(package="benzaiten_adlib")
class GaussianSampling(tf.keras.layers.Layer):
    """Sample a full-covariance Gaussian and regularize against N(0, I).

    The dense parameters and triangular ordering match TFP's
    MultivariateNormalTriL / FillScaleTriL, including its diagonal shift.
    This layer has no weights, so legacy HDF5 weights load by topology.
    """

    def __init__(self, latent_dim=ENCODED_DIM, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def distribution_parameters(self, inputs):
        n = self.latent_dim
        loc = inputs[..., :n]
        packed = inputs[..., n:]
        # TFP fill_triangular(lower=True), not row-major triangle packing.
        square = tf.concat([packed[..., n:], tf.reverse(packed, axis=[-1])], axis=-1)
        square = tf.reshape(square, tf.concat([tf.shape(inputs)[:-1], [n, n]], axis=0))
        scale = tf.linalg.band_part(square, -1, 0)
        diagonal = tf.nn.softplus(tf.linalg.diag_part(scale)) + tf.cast(1e-5, inputs.dtype)
        return loc, tf.linalg.set_diag(scale, diagonal)

    def call(self, inputs):
        loc, scale = self.distribution_parameters(inputs)
        noise = tf.random.normal(tf.shape(loc), dtype=loc.dtype)
        sample = loc + tf.linalg.matvec(scale, noise)
        # Monte Carlo log q(z) - log p(z), using the same draw as the decoder.
        # Keras 2 divided activity_regularizer's batch sum by the batch size.
        kl = 0.5 * tf.reduce_sum(tf.square(sample) - tf.square(noise), axis=-1)
        kl -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(scale)), axis=-1)
        self.add_loss(self.kl_weight * tf.reduce_mean(kl))
        return sample

    def get_config(self):
        return {**super().get_config(), "latent_dim": self.latent_dim,
                "kl_weight": self.kl_weight}


def make_model(seq_length, input_dim, output_dim, *, latent_dim=ENCODED_DIM,
               lstm_dim=LSTM_DIM, compile_model=True):
    """Build the original LSTM/full-covariance VAE using Keras 3 layers."""
    encoder = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_length, input_dim)),
        tf.keras.layers.LSTM(lstm_dim, activation="tanh"),
        tf.keras.layers.Dense(latent_dim + latent_dim * (latent_dim + 1) // 2),
        GaussianSampling(latent_dim),
    ], name="encoder")
    decoder = tf.keras.Sequential([
        tf.keras.Input(shape=(latent_dim,)),
        tf.keras.layers.RepeatVector(seq_length),
        tf.keras.layers.LSTM(lstm_dim, activation="tanh", return_sequences=True),
        tf.keras.layers.Dense(output_dim, activation="softmax"),
    ], name="decoder")
    # Keep encoder layers at the top level, as in the legacy HDF5 topology.
    model = tf.keras.Model(encoder.inputs[0], decoder(encoder.outputs[0]), name="benzaiten_vae")
    if compile_model:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model
