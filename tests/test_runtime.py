"""Numerical and file-format regressions for the supported ML environment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import music21
import numpy as np
import tensorflow as tf

from benzaiten_adlib import core
from benzaiten_adlib.model import GaussianSampling, make_model
from benzaiten_adlib.model_io import load_trained_model


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

    def test_full_covariance_layout_and_kl(self):
        # TFP's lower triangular packing of [1..6] is [[4,0,0],[6,5,0],[3,2,1]].
        params = tf.constant([[0.1, 0.2, 0.3, 1., 2., 3., 4., 5., 6.]])
        layer = GaussianSampling(3)
        loc, scale = layer.distribution_parameters(params)
        expected = np.array([[4., 0., 0.], [6., 5., 0.], [3., 2., 1.]])
        np.fill_diagonal(expected, np.logaddexp(0, [4., 5., 1.]) + 1e-5)
        np.testing.assert_allclose(scale.numpy()[0], expected, rtol=1e-6)
        np.testing.assert_allclose(loc.numpy()[0], [0.1, 0.2, 0.3])
        noise = np.array([[0.5, -0.25, 0.75]], dtype=np.float32)
        with patch('tensorflow.random.normal', return_value=tf.constant(noise)):
            sample = layer(params).numpy()[0]
        expected_sample = np.array([0.1, 0.2, 0.3]) + expected @ noise[0]
        np.testing.assert_allclose(sample, expected_sample, rtol=1e-6)
        # Independent Gaussian log-density calculation checks the KL estimator.
        covariance = expected @ expected.T
        delta = sample - loc.numpy()[0]
        log_q = -0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
                        + delta @ np.linalg.solve(covariance, delta))
        log_p = -0.5 * (3 * np.log(2 * np.pi) + sample @ sample)
        self.assertAlmostEqual(float(layer.losses[0]), 0.001 * (log_q - log_p), places=6)

    def test_training_weights_and_serialization(self):
        model = make_model(4, 5, 3, latent_dim=2, lstm_dim=8)
        x = np.ones((2, 4, 5), dtype=np.float32)
        y = np.eye(3, dtype=np.float32)[np.zeros((2, 4), dtype=int)]
        before = [v.numpy().copy() for v in model.trainable_variables]
        metrics = model.train_on_batch(x, y, return_dict=True)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not np.array_equal(a, b.numpy())
                            for a, b in zip(before, model.trainable_variables)))
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, prediction))
            loss += tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
        np.testing.assert_allclose(prediction.numpy().sum(axis=-1), 1, rtol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'test.weights.h5'
            model.save_weights(weights)
            restored = make_model(4, 5, 3, latent_dim=2, lstm_dim=8)
            restored.optimizer.build(restored.trainable_variables)
            restored.load_weights(weights)
            for old, new in zip(model.get_weights(), restored.get_weights()):
                np.testing.assert_array_equal(old, new)
            full = Path(directory) / 'test.keras'
            model.save(full)
            restored = tf.keras.models.load_model(full)
            self.assertEqual(restored(x).shape, (2, 4, 3))
            self.assertTrue(np.isfinite(restored.train_on_batch(x, y)).all())

    def test_legacy_and_current_weight_selection(self):
        original = make_model(4, 5, 3, latent_dim=2, lstm_dim=8)
        def fresh(*shape, **kwargs):
            return make_model(*shape, latent_dim=2, lstm_dim=8, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'C_major.benzaitenconfig').write_text('4\n5\n3')
            # Legacy .h5 topology loading must not deserialize the saved layer.
            original.save(root / 'mymodel_C_major.h5')
            with patch('benzaiten_adlib.model.make_model', side_effect=fresh):
                restored = load_trained_model('C_major', root)
                for old, new in zip(original.get_weights(), restored.get_weights()):
                    np.testing.assert_array_equal(old, new)
                restored.save_weights(root / 'mymodel_C_major.weights.h5')
                (root / 'mymodel_C_major.h5').write_bytes(b'not used when current weights exist')
                current = load_trained_model('C_major', root)
                self.assertEqual(current.output_shape, (None, 4, 3))
            (root / 'C_major.benzaitenconfig').write_text('4\n0\n3')
            with self.assertRaises(ValueError):
                load_trained_model('C_major', root)

    def test_musicxml_to_training_arrays(self):
        score = music21.stream.Score()
        part = music21.stream.Part()
        measure = music21.stream.Measure(number=1)
        measure.insert(0, music21.harmony.ChordSymbol('C'))
        measure.insert(0, music21.note.Note('C4', quarterLength=1))
        measure.insert(1, music21.note.Note('E4', quarterLength=1))
        part.append(measure)
        score.append(part)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'score.musicxml'
            score.write('musicxml', fp=path)
            notes, chords = core.make_note_and_chord_seq_from_musicxml(music21.converter.parse(path))
        self.assertEqual(notes[0].pitch.midi, 60)
        self.assertEqual(notes[4].pitch.midi, 64)
        onehot = core.add_rest_nodes(core.note_seq_to_onehot(notes))
        chroma = core.chord_seq_to_chroma(chords[:16])
        self.assertEqual(onehot.shape[1], 49)
        self.assertEqual(chroma.shape, (16, 24))
        self.assertEqual(set(np.flatnonzero(chroma[0])), {0, 4, 7})


if __name__ == '__main__':
    unittest.main()
