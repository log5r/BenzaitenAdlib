import mido
import benzaiten_config as cfg


# MIDIトラックを生成（make_midiから呼び出される）

def make_midi_track(notenums, durations, transpose, ticks_per_beat):
    track = mido.MidiTrack()
    init_tick = cfg.INTRO_BLANK_MEASURES * cfg.N_BEATS * ticks_per_beat
    prev_tick = 0
    for i in range(len(notenums)):
        if notenums[i] > 0:
            curr_tick = int(i * ticks_per_beat / cfg.BEAT_RESO) + init_tick
            track.append(mido.Message('note_on', note=notenums[i] + transpose,
                                      velocity=100, time=curr_tick - prev_tick))
            prev_tick = curr_tick
            curr_tick = int((i + durations[i]) * ticks_per_beat / cfg.BEAT_RESO) + init_tick
            track.append(mido.Message('note_off', note=notenums[i] + transpose,
                                      velocity=100, time=curr_tick - prev_tick))
            prev_tick = curr_tick
    return track


# プログラムチェンジを指定したものに差し替え
def replace_prog_chg(midi):
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'program_change' and msg.channel == cfg.MELODY_CH:
                msg.program = cfg.MELODY_PROG_CHG
                return midi


# MIDIファイル（提出用、伴奏なし）を生成
def make_midi_for_submission(notenums, durations, transpose, dst_filename):
    midi = mido.MidiFile(type=1)
    midi.ticks_per_beat = cfg.TICKS_PER_BEAT
    midi.tracks.append(make_midi_track(notenums, durations, transpose, cfg.TICKS_PER_BEAT))
    midi.save(dst_filename)


def make_midi_for_submission_using_midi(source_midi):
    midi = mido.MidiFile(type=1)
    midi.ticks_per_beat = cfg.TICKS_PER_BEAT
    midi.tracks.append(source_midi.tracks[1])
    return midi


# MIDIファイル（チェック用、伴奏あり）を生成
def make_midi_for_check(notenums, durations, transpose, src_filename, dst_filename):
    midi = mido.MidiFile(src_filename)
    replace_prog_chg(midi)
    midi.tracks.append(make_midi_track(notenums, durations, transpose, midi.ticks_per_beat))
    midi.save(dst_filename)
