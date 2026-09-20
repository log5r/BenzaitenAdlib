"""Training and generation orchestration without external models or audio tools."""
import contextlib
import datetime
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import mido
import music21
import numpy as np

from benzaiten_adlib import core, features, generate, learn, model_io, paths


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        random.seed(42)
        self.addCleanup(random.setstate, state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, suffix in [('MODEL_DIR', 'models'), ('MIDI_DIR', 'midi'),
                             ('SOLO_DIR', 'solo'), ('WAV_DIR', 'wav'),
                             ('SAMPLE_DIR', 'sample'), ('SOUNDFONT_DIR', 'soundfonts')]:
            patcher = patch.object(paths, name, self.root / suffix)
            patcher.start()
            self.addCleanup(patcher.stop)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_training_saves_dimensions_and_current_weights(self):
        x = np.ones((2, 64, 73))
        y = np.ones((2, 64, 49))
        model = Mock()
        with patch.object(core, 'make_model', return_value=model) as build:
            learn.learn_and_generate_model(x, y, 'C_major', epochs=3)
        build.assert_called_once_with(64, 73, 49)
        model.fit.assert_called_once_with(x, y, epochs=3)
        model.save_weights.assert_called_once_with(paths.MODEL_DIR / 'mymodel_C_major.weights.h5')
        self.assertEqual((paths.MODEL_DIR / 'C_major.benzaitenconfig').read_text(), '64\n73\n49')

    def test_invalid_epochs_fail_before_writing(self):
        for epochs in (0, -1):
            with self.subTest(epochs=epochs), self.assertRaisesRegex(ValueError, 'positive'):
                learn.learn_and_generate_model(None, None, 'C_major', epochs=epochs)
        self.assertFalse(paths.MODEL_DIR.exists())

    def test_training_cli_selects_models_and_epochs(self):
        x, y = np.ones((1, 4, 5)), np.ones((1, 4, 3))
        with patch('sys.argv', ['learn', '--models', 'C_major', 'A_minor', '--epochs', '2']), \
                patch.object(core, 'read_mus_xml_files', return_value=(x, y)) as read, \
                patch.object(learn, 'learn_and_generate_model') as train:
            learn.main()
        self.assertEqual([call.args[2:] for call in read.call_args_list], [('C', 'major'), ('A', 'minor')])
        self.assertEqual([call.args[2] for call in train.call_args_list], ['C_major', 'A_minor'])
        self.assertTrue(all(call.kwargs == {'epochs': 2} for call in train.call_args_list))

    def test_training_cli_rejects_invalid_epochs_and_empty_data(self):
        for args, expected in [(['--epochs', '0'], 'positive'), ([], 'No training sequences')]:
            stderr = io.StringIO()
            with patch('sys.argv', ['learn', *args]), \
                    patch.object(core, 'read_mus_xml_files', return_value=(np.array([]), np.array([]))), \
                    patch.object(learn, 'learn_and_generate_model') as train, \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                learn.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(expected, stderr.getvalue())
            train.assert_not_called()

    def test_model_loader_rejects_invalid_or_missing_files(self):
        paths.MODEL_DIR.mkdir()
        config = paths.MODEL_DIR / 'C_major.benzaitenconfig'
        with patch('benzaiten_adlib.model.make_model') as build:
            with self.assertRaises(FileNotFoundError):
                model_io.load_trained_model('C_major')
            for contents in ('4 5', '4 5 3 2', '4 -5 3', 'four 5 3'):
                config.write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    model_io.load_trained_model('C_major')
            config.write_text('4 5 3')
            with self.assertRaisesRegex(FileNotFoundError, 'Model weights not found'):
                model_io.load_trained_model('C_major')
            build.assert_not_called()

    def test_musicxml_file_loading_transposes_to_requested_tonic(self):
        music_dir = self.root / 'music'
        (music_dir / 'C_major').mkdir(parents=True)
        score = music21.stream.Score()
        part = music21.stream.Part()
        measure = music21.stream.Measure(number=1)
        measure.insert(0, music21.harmony.ChordSymbol('D'))
        measure.insert(0, music21.note.Note('D4', quarterLength=1))
        part.append(measure)
        score.append(part)
        score.write('musicxml', fp=music_dir / 'C_major' / 'example.xml')
        with patch.object(core, 'MUS_DIR', str(music_dir) + '/'), \
                patch.object(music21.stream.Score, 'analyze', return_value=music21.key.Key('D')):
            x, y = core.read_mus_xml_files([], [], 'C', 'major')
            empty_x, empty_y = core.read_mus_xml_files([], [], 'A', 'minor')
        self.assertEqual(x.shape, (1, 64, 73))
        self.assertEqual(y.shape, (1, 64, 49))
        self.assertEqual(core.calc_notenums_from_pianoroll(y[0])[0], 60)
        self.assertEqual(set(np.flatnonzero(x[0, 0, 49:])), {0, 4, 7})
        self.assertEqual(empty_x.size, 0)
        self.assertEqual(empty_y.size, 0)

    def test_generation_writes_playable_backing_and_solo_for_each_correction(self):
        paths.SAMPLE_DIR.mkdir()
        (paths.SAMPLE_DIR / 'sample_chord.csv').write_text('0,0,C,major,C\n')
        midi = mido.MidiFile()
        midi.tracks.extend([mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=600000)]), mido.MidiTrack()])
        midi.save(paths.SAMPLE_DIR / 'sample_backing.mid')
        def predict(batch):
            self.assertEqual(batch.shape, (1, 64, 73))
            np.testing.assert_array_equal(batch[0, :, :49], 0)
            self.assertEqual(set(np.flatnonzero(batch[0, 0, 49:])), {0, 4, 7})
            prediction = np.zeros((1, 64, 49))
            prediction[:, :, 24] = 1
            return prediction
        for options in (None, [features.CORRECTION_TYPE2],
                        [features.CORRECTION_TYPE3, features.BUMP_UP_LOW_TONE]):
            with self.subTest(options=options), \
                    patch.object(model_io, 'load_trained_model', return_value=Mock(predict=Mock(side_effect=predict))) as load, \
                    patch.object(core, 'generate_wav_file') as render:
                generate.generate_adlib_files('C_major', options)
                self.assertEqual(load.return_value.predict.call_count, 2)
                load.assert_called_once_with('C_major')
                suffix, output = render.call_args.args
                self.assertEqual(suffix, 'C_major_' + '_'.join(options or []))
                rendered = mido.MidiFile(output)
                solo_path = paths.SOLO_DIR / (Path(output).stem + '_solo.mid')
                solo = mido.MidiFile(solo_path)
                self.assertEqual(len(rendered.tracks), 2)
                self.assertEqual(len(solo.tracks), 1)
                self.assertEqual(rendered.tracks[0][0].tempo, 600000)
                self.assertEqual(list(rendered.tracks[1]), list(solo.tracks[0]))
                self.assertTrue(any(m.type == 'note_on' for m in solo.tracks[0]))
                self.assertTrue(all(m.time >= 0 for m in solo.tracks[0]))

    def test_default_generation_variants(self):
        with patch.object(generate, 'generate_adlib_files') as run:
            generate.main()
        expected = [[features.CORRECTION_TYPE1],
                    [features.CORRECTION_TYPE1, features.V2_SHUFFLE, features.TRIPLET_SEMIQUAVER],
                    [features.CORRECTION_TYPE3], [features.CORRECTION_TYPE3, features.V2_SHUFFLE]]
        self.assertEqual([call.args[0] for call in run.call_args_list], ['C_major'] * 4 + ['A_minor'] * 4)
        self.assertEqual([call.kwargs['features'] for call in run.call_args_list], expected * 2)

    def test_timer_preserves_arguments_and_return_value(self):
        function = Mock(return_value=42, __name__='example')
        with patch.object(generate.time, 'process_time', side_effect=[1, 3]):
            self.assertEqual(generate.print_proc_time(function)(1, enabled=True), 42)
        function.assert_called_once_with(1, enabled=True)

    def test_wav_renderer_receives_soundfont_and_output_paths(self):
        now = datetime.datetime(2026, 1, 2, 3, 4, 5)
        with patch.object(core.midi2audio, 'FluidSynth') as synth, \
                patch.object(core.datetime, 'datetime') as date:
            date.now.return_value = now
            core.generate_wav_file('C_major_type1', 'input.mid')
        synth.assert_called_once_with(sound_font=str(paths.SOUNDFONT_DIR / 'FluidR3_GM.sf2'))
        synth.return_value.midi_to_audio.assert_called_once_with(
            'input.mid', str(paths.WAV_DIR / '2026-01-02_03-04-05_C_major_type1_output.wav'))
        self.assertTrue(paths.WAV_DIR.is_dir())
