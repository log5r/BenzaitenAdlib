"""Musical data and MIDI regressions using small, synthetic inputs."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import mido
import music21
import numpy as np

from benzaiten_adlib import config as cfg, core, features, music_utils as mu, submission


def timed_notes(track):
    tick = 0
    result = []
    for message in track:
        tick += message.time
        if message.type in ('note_on', 'note_off'):
            result.append((message.type, message.note, tick))
    return result


class MusicProcessingTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        state = random.getstate()
        random.seed(42)
        self.addCleanup(random.setstate, state)

    def test_note_encoding_rest_and_octave_wrapping(self):
        notes = [music21.note.Note(n) if n is not None else None
                 for n in (36, 83, None, 84, 35)]
        encoded = core.add_rest_nodes(core.note_seq_to_onehot(notes))
        self.assertEqual(encoded.shape, (5, 49))
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(5))
        self.assertEqual(core.calc_notenums_from_pianoroll(encoded), [36, 83, -1, 36, 83])
        self.assertEqual(core.make_empty_pianoroll(8).shape, (8, 49))
        self.assertFalse(core.make_empty_pianoroll(8).any())

    def test_chord_csv_changes_inheritance_and_ending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chords.csv'
            path.write_text('0,0,C,major,C\n0,2,G,dominant-seventh,G\n8,0,A,minor,A\n')
            chords = core.read_chord_file(path)
            extended = core.read_chord_file(path, appending=1)
        self.assertEqual(len(chords), 32)
        self.assertEqual([c.figure for c in chords[:4]], ['C', 'C', 'G7', 'G7'])
        self.assertTrue(all(c.figure == 'G7' for c in chords[2:]))
        self.assertEqual([c.figure for c in extended[32:]], ['Am'] * 4)
        self.assertEqual(core.parse_chord_for_magenta(chords[:4]), '"C C G7 G7"')
        expanded = core.make_chord_seq([chords[0], 'G7'], 4)
        self.assertEqual([c.figure for c in expanded], ['C'] * 4 + ['G7'] * 4)
        chroma = core.chord_seq_to_chroma([None, chords[0]])
        self.assertFalse(chroma[0].any())
        self.assertEqual(set(np.flatnonzero(chroma[1])), {0, 4, 7})

    def test_training_windows_skip_only_silent_segments(self):
        notes = np.zeros((128, 49))
        notes[:, -1] = 1
        notes[64, -1] = 0
        notes[64, 24] = 1
        chords = np.arange(128 * 24).reshape(128, 24)
        xs, ys = [], []
        with patch.object(cfg, 'TOTAL_MEASURES', 8):
            core.divide_seq(notes, chords, xs, ys)
        self.assertEqual(len(xs), 1)
        np.testing.assert_array_equal(ys[0], notes[64:128])
        np.testing.assert_array_equal(xs[0][:, :49], notes[64:128])
        np.testing.assert_array_equal(xs[0][:, 49:], chords[64:128])

    def test_duration_merging_preserves_rests_and_total_timing(self):
        durations, notes = core.calc_durations([60, 60, -1, -1, 64, 64, 64, 67])
        self.assertEqual(notes, [60, 0, -1, -1, 64, 0, 0, 67])
        self.assertEqual(durations, [2, 1, 1, 1, 3, 1, 1, 1])
        self.assertEqual(core.calc_durations([]), ([], []))
        chords = [music21.harmony.ChordSymbol('C')] * 2
        track = core.make_midi_track(notes, chords, durations, 12, 480, [])
        self.assertEqual(timed_notes(track), [
            ('note_on', 72, 7680), ('note_off', 72, 7920),
            ('note_on', 76, 8160), ('note_off', 76, 8520),
            ('note_on', 79, 8520), ('note_off', 79, 8640)])

    def test_shuffle_and_triplets_preserve_beat_length(self):
        notes = [-1] * 16 + [60, 64, 67, 72]
        chords = [music21.harmony.ChordSymbol('C')] * 5
        straight = core.make_midi_track(notes, chords, [1] * 20, 0, 480, [])
        shuffled = core.make_midi_track(notes, chords, [1] * 20, 0, 480, [features.V2_SHUFFLE])
        triplets = core.make_midi_track(notes, chords, [1] * 20, 0, 480,
                                       [features.V2_SHUFFLE, features.TRIPLET_SEMIQUAVER])
        self.assertEqual([n[2] for n in timed_notes(shuffled) if n[0] == 'note_on'],
                         [9600, 9760, 9840, 10000])
        self.assertEqual(len(timed_notes(triplets)), 12)
        for track in (straight, shuffled, triplets):
            self.assertEqual(timed_notes(track)[-1][2], 10080)
            self.assertTrue(all(m.time >= 0 for m in track))

    def test_midi_export_roundtrip_and_backing_preservation(self):
        backing = mido.MidiFile()
        backing.tracks.extend([mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=600000)]),
                               mido.MidiTrack()])
        tempo_track = list(backing.tracks[0])
        result = core.make_midi([60, 0, -1, 64], [music21.harmony.ChordSymbol('C')],
                                [2, 1, 1, 1], 12, backing, [])
        self.assertEqual(result.tracks[0], tempo_track)
        solo = submission.make_midi_for_submission_using_midi(result)
        self.assertEqual(len(solo.tracks), 1)
        self.assertEqual(timed_notes(solo.tracks[0]), timed_notes(result.tracks[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'solo.mid'
            submission.make_midi_for_submission([60, 0, -1, 64], [2, 1, 1, 1], 12, path)
            self.assertEqual(timed_notes(core.read_midi_file(path).tracks[0]), timed_notes(solo.tracks[0]))
            source = Path(directory) / 'backing.mid'
            result.save(source)
            submission.make_midi_for_check([60], [1], 0, source, path)
            self.assertEqual(len(mido.MidiFile(path).tracks), 3)

    def test_program_change_only_targets_melody_channel(self):
        midi = mido.MidiFile()
        midi.tracks.append(mido.MidiTrack([
            mido.Message('program_change', channel=1, program=10),
            mido.Message('program_change', channel=cfg.MELODY_CH, program=20)]))
        self.assertIs(submission.replace_prog_chg(midi), midi)
        self.assertEqual([m.program for m in midi.tracks[0]], [10, cfg.MELODY_PROG_CHG])

    def test_bends_preserve_note_timestamps(self):
        for duration in (120, 240, 360):
            with self.subTest(duration=duration):
                midi = mido.MidiFile()
                track = mido.MidiTrack([mido.Message('program_change', program=73),
                    mido.Message('note_on', note=60, time=7680),
                    mido.Message('note_off', note=60, time=duration),
                    mido.Message('note_on', note=64),
                    mido.Message('note_off', note=64, time=120)])
                midi.tracks.extend([mido.MidiTrack(), track])
                expected = timed_notes(track)
                arranged = mu.arrange_using_midi(midi)
                self.assertEqual(timed_notes(arranged.tracks[1]), expected)
                bends = [m.pitch for m in arranged.tracks[1] if m.type == 'pitchwheel']
                self.assertEqual(len(bends), 0 if duration == 120 else 11)
                if bends:
                    self.assertEqual(bends[-1], 0)
                    self.assertEqual(bends, sorted(bends))

    def test_last_note_ignores_trailing_rests(self):
        for notes, expected in [([], 60), ([-1, -1], 60), ([72, 76, -1], 76), ([48], 48)]:
            with self.subTest(notes=notes):
                self.assertEqual(mu.get_last_note(notes), expected)

    def test_pitch_helpers(self):
        for chord, expected in [('C', 'C'), ('Am', 'Am'), ('Am7', 'Am'), ('Cmaj7', 'C')]:
            self.assertEqual(mu.remove_chord_suffix(chord), expected)
        self.assertEqual(mu.bump_up_low_note([-1, 59, 60, 66, 67, 84]), [-1, 59, 72, 78, 67, 84])
        self.assertEqual(list(mu.linear_increasing_bend_curve()), list(range(-4000, 0, 400)))
        with patch.object(mu.random, 'shuffle'):
            for note, offbeat, expected in [(-1, 0, -1), (60, 0, 60), (61, 0, 64),
                                            (61, 1, 61), (65, 1, 67), (71, 0, 71)]:
                self.assertEqual(mu.fixed_note_num(note, [0, 4, 7], [5], offbeat), expected)

    def test_dominant_classification(self):
        for symbol, expected in [('G7', True), ('G', True), ('D', True),
                                 ('C', False), ('F', False), ('Am', False)]:
            with self.subTest(symbol=symbol):
                self.assertEqual(mu.is_dominant(music21.harmony.ChordSymbol(symbol), None), expected)

    def test_corrections_produce_valid_notes_and_sustained_ending(self):
        chords = [music21.harmony.ChordSymbol('C')] * 36
        source = ([60, 61, -1, 83, 36, 64, 65, 72] * 16)
        for correction in (mu.corrected_note_num_list_type1, mu.corrected_note_num_list_type2,
                           mu.corrected_note_num_list_type3):
            for options in ([], [features.COPY_SQ2_TO_SQ3]):
                with self.subTest(correction=correction.__name__, options=options):
                    result = correction(source, chords, 'C_major', options)
                    self.assertGreater(len(result), len(source))
                    self.assertTrue(all(n == -1 or 0 <= n <= 127 for n in result))
                    self.assertEqual(len(set(result[-8:])), 1)
                    if correction == mu.corrected_note_num_list_type1:
                        self.assertTrue(all(58 <= n <= 84 for n in result))
                    else:
                        self.assertEqual(result[2], -1)
                        self.assertEqual(result[-16:], [72] * 8 + [76] * 8)
