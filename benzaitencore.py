import random

import music21
import numpy as np
import matplotlib.pyplot as plt
import mido
import csv
import midi2audio
import glob
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import functools
import benzaiten_config as cfg
import common_features as Feature
import music_utils as mu
import math

# ディレクトリ定義
BASE_DIR = "./"
MUS_DIR = BASE_DIR + "omnibook/"

# VAEモデル関連
ENCODED_DIM = 32  # 潜在空間の次元数
LSTM_DIM = 1024  # LSTM層のノード数


# エンコーダを構築
def make_encoder(prior, seq_length, input_dim):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.LSTM(LSTM_DIM,
                                     input_shape=(seq_length, input_dim),
                                     use_bias=True, activation="tanh",
                                     return_sequences=False))
    encoder.add(tf.keras.layers.Dense(
        tfp.layers.MultivariateNormalTriL.params_size(ENCODED_DIM),
        activation=None))
    encoder.add(tfp.layers.MultivariateNormalTriL(
        ENCODED_DIM,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(
            prior, weight=0.001)))
    return encoder


# デコーダを構築
def make_decoder(seq_length, output_dim):
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.RepeatVector(seq_length, input_dim=ENCODED_DIM))
    decoder.add(tf.keras.layers.LSTM(LSTM_DIM, use_bias=True, activation="tanh", return_sequences=True))
    decoder.add(tf.keras.layers.Dense(output_dim, use_bias=True, activation="softmax"))
    return decoder


# VAEに用いる事前分布を定義
def make_prior():
    tfd = tfp.distributions
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(ENCODED_DIM), scale=1),
        reinterpreted_batch_ndims=1)
    return prior


# エンコーダとデコーダを構築し、それらを結合したモデルを構築する
# (入力:エンコーダの入力、
#  出力:エンコーダの出力をデコーダに入力して得られる出力)
def make_model(seq_length, input_dim, output_dim):
    encoder = make_encoder(make_prior(), seq_length, input_dim)
    decoder = make_decoder(seq_length, output_dim)
    vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss="categorical_crossentropy", metrics="categorical_accuracy")
    return vae


# MusicXMLデータからNote列とChordSymbol列を生成
# 時間分解能は BEAT_RESO にて指定
def make_note_and_chord_seq_from_musicxml(score):
    note_seq_arr_size = cfg.TOTAL_MEASURES * cfg.N_BEATS * cfg.BEAT_RESO
    note_seq = [None] * note_seq_arr_size
    chord_seq = [None] * note_seq_arr_size
    for element in score.parts[0].elements:
        if isinstance(element, music21.stream.Measure):
            measure_offset = element.offset
            for note in element.notes:
                if isinstance(note, music21.note.Note):
                    onset = measure_offset + note._activeSiteStoredOffset
                    offset = onset + note._duration.quarterLength
                    for i in range(int(onset * cfg.BEAT_RESO), int(offset * cfg.BEAT_RESO + 1)):
                        note_seq[i] = note
                if isinstance(note, music21.harmony.ChordSymbol):
                    chord_offset = measure_offset + note.offset
                    for i in range(int(chord_offset * cfg.BEAT_RESO),
                                   int((measure_offset + cfg.N_BEATS) * cfg.BEAT_RESO + 1)):
                        chord_seq[i] = note
    return note_seq, chord_seq


# Note列をone-hot vector列(休符はすべて0)に変換
def note_seq_to_onehot(note_seq):
    M = cfg.NOTENUM_THRU - cfg.NOTENUM_FROM
    N = len(note_seq)
    matrix = np.zeros((N, M))
    for i in range(N):
        if note_seq[i] is not None:
            matrix_idx_r = (note_seq[i].pitch.midi - cfg.NOTENUM_FROM) % M
            matrix[i, matrix_idx_r] = 1
    return matrix


# 音符列を表すone-hot vector列に休符要素を追加
def add_rest_nodes(onehot_seq):
    rest = 1 - np.sum(onehot_seq, axis=1)
    rest = np.expand_dims(rest, 1)
    return np.concatenate([onehot_seq, rest], axis=1)


# 指定された仕様のcsvファイルを読み込んで
# ChordSymbol列を返す
def read_chord_file(file, appending=0):
    chord_seq = [None] * ((cfg.MELODY_LENGTH + appending) * cfg.N_BEATS)
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            m = int(row[0])  # 小節番号(0始まり)
            if m < cfg.MELODY_LENGTH + appending:
                b = int(row[1])  # 拍番号(0始まり、今回は0または2)
                smbl = music21.harmony.ChordSymbol(root=row[2], kind=row[3], bass=row[4])
                assign_idx = m * 4 + b
                chord_seq[assign_idx] = smbl
    for i in range(len(chord_seq)):
        if chord_seq[i] is not None:
            chord = chord_seq[i]
        else:
            chord_seq[i] = chord
    return chord_seq


# コード進行からChordSymbol列を生成
# divisionは1小節に何個コードを入れるか
def make_chord_seq(chord_prog, division):
    T = int(cfg.N_BEATS * cfg.BEAT_RESO / division)
    seq = [None] * (T * len(chord_prog))
    for i in range(len(chord_prog)):
        for t in range(T):
            if isinstance(chord_prog[i], music21.harmony.ChordSymbol):
                seq[i * T + t] = chord_prog[i]
            else:
                seq[i * T + t] = music21.harmony.ChordSymbol(chord_prog[i])
    return seq


# ChordSymbol列をmany-hot (chroma) vector列に変換
def chord_seq_to_chroma(chord_seq):
    N = len(chord_seq)
    matrix = np.zeros((N, 12))
    for i in range(N):
        if chord_seq[i] is not None:
            for note in chord_seq[i]._notes:
                matrix[i, note.pitch.midi % 12] = 1
    return matrix


# 空(全要素がゼロ)のピアノロールを生成
def make_empty_pianoroll(length):
    return np.zeros((length, cfg.NOTENUM_THRU - cfg.NOTENUM_FROM + 1))


# ピアノロール(one-hot vector列)をノートナンバー列に変換
def calc_notenums_from_pianoroll(pianoroll):
    notenums = []
    for i in range(pianoroll.shape[0]):
        n = np.argmax(pianoroll[i, :])
        nn = -1 if n == pianoroll.shape[1] - 1 else n + cfg.NOTENUM_FROM
        notenums.append(nn)
    return notenums


# 連続するノートナンバーを統合して (notenums, durations) に変換
def calc_durations(target_note_num_list):
    note_num_list = target_note_num_list
    note_nums_length = len(note_num_list)
    duration = [1] * note_nums_length
    for i in range(note_nums_length):
        k = 1
        while i + k < note_nums_length:
            merge_condition = [
                note_num_list[i] == note_num_list[i + k],
                note_num_list[i + k] == 0
            ]
            if note_num_list[i] > 0 and any(merge_condition):
                note_num_list[i + k] = 0
                duration[i] += 1
            else:
                break
            k += 1
    return duration, note_num_list


# MIDIファイルを読み込み
def read_midi_file(src_filename):
    return mido.MidiFile(src_filename)


# MIDIファイルを生成
def make_midi(note_num_list, durations, transpose, backing_midi, features):
    midi = backing_midi
    midi.tracks[1] = make_midi_track(note_num_list, durations, transpose, cfg.TICKS_PER_BEAT, features)
    return midi


def make_midi_track(note_nums, durations, transpose, ticks_per_beat, features):

    use_shuffle_mode = Feature.V2_SHUFFLE in features
    use_triplet_semiquaver = Feature.TRIPLET_SEMIQUAVER in features

    track = mido.MidiTrack()
    track.append(mido.Message('program_change', program=cfg.MELODY_PROG_CHG, time=0))
    init_tick = cfg.INTRO_BLANK_MEASURES * cfg.N_BEATS * ticks_per_beat
    prev_tick = 0
    prev_note = 0
    semiquiv = ticks_per_beat / cfg.BEAT_RESO
    triplet_sqv = ticks_per_beat / (3 * int(cfg.BEAT_RESO / 2))
    tick_table = [0, semiquiv, semiquiv * 2, semiquiv * 3]
    if use_shuffle_mode:
        tick_table = [0, triplet_sqv * 2, triplet_sqv * 3, triplet_sqv * 5]

    def add_note_on(note, tick):
        track.append(mido.Message('note_on', note=note, velocity=127, time=tick))
        print('note-on: %s => %s : %s : %s | %s' % (note, tick, i, i + durations[i], i % cfg.BEAT_RESO))

    def add_note_off(note, tick):
        track.append(mido.Message('note_off', note=note, velocity=127, time=tick))
        print('note-off: %s => %s : %s : %s | %s' % (note, tick, i, i + durations[i], i % cfg.BEAT_RESO))

    for i in range(len(note_nums)):
        this_note = note_nums[i]
        if this_note > 0:
            curr_tick = int(math.floor(i / cfg.BEAT_RESO) * ticks_per_beat + tick_table[i % cfg.BEAT_RESO] + init_tick)
            add_note_on(this_note + transpose, curr_tick - prev_tick)
            prev_tick = curr_tick
            curr_tick = int(math.floor((i + durations[i]) / cfg.BEAT_RESO) * ticks_per_beat
                            + tick_table[(i + durations[i]) % cfg.BEAT_RESO]
                            + init_tick)
            sub_tick_unit = int((curr_tick - prev_tick) / 2)
            if (use_triplet_semiquaver
                    and durations[i] == 1
                    and sub_tick_unit > 79
                    and any(mu.use_triplet_semiquaver_condition(i))):
                add_note_off(this_note + transpose, sub_tick_unit)
                mid_cmp_note = int((this_note + note_nums[i + 1]) / 2)
                add_note_on(mid_cmp_note + transpose, 0)
                add_note_off(mid_cmp_note + transpose, sub_tick_unit)
            else:
                add_note_off(this_note + transpose, curr_tick - prev_tick)
            prev_tick = curr_tick
        prev_note = this_note
    return track


# ピアノロールを描画
def plot_pianoroll(pianoroll):
    plt.matshow(np.transpose(pianoroll))
    plt.show()


# WAVを生成
def generate_wav_file(model_idf, dst_filename):
    sf_path = "soundfonts/FluidR3_GM.sf2"
    fs = midi2audio.FluidSynth(sound_font=sf_path)
    timestamp = format(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')
    generated_filename = "%s_%s_output.wav" % (timestamp, model_idf)
    fs.midi_to_audio(dst_filename, generated_filename)


# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列に対して、
# UNIT_MEASURES小節分だけ切り出したものを返す
def extract_seq(i, onehot_seq, chroma_seq):
    n_beats_reso = cfg.N_BEATS * cfg.BEAT_RESO
    offset = (i + cfg.UNIT_MEASURES) * cfg.N_BEATS * cfg.BEAT_RESO
    o = onehot_seq[i * n_beats_reso: offset, :]
    c = chroma_seq[i * n_beats_reso: offset, :]
    return o, c


# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から、
# モデルの入力、出力用のデータい整えて返す
def calc_xy(o, c):
    x = np.concatenate([o, c], axis=1)
    y = o
    return x, y


# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から
# モデルの入力、出力用のデータを作成して、配列に逐次格納する
def divide_seq(onehot_seq, chroma_seq, x_all, y_all):
    for i in range(0, cfg.TOTAL_MEASURES, cfg.UNIT_MEASURES):
        o, c, = extract_seq(i, onehot_seq, chroma_seq)
        if np.any(o[:, 0:-1] != 0):
            x, y = calc_xy(o, c)
            x_all.append(x)
            y_all.append(y)


# ファイルの読み込み
def read_mus_xml_files(x, y, key_root, key_mode):
    for f in glob.glob(MUS_DIR + "/*.xml"):
        print(f)
        score = music21.converter.parse(f)
        key = score.analyze("key")
        if key.mode == key_mode:
            inter = music21.interval.Interval(key.tonic, music21.pitch.Pitch(key_root))
            score = score.transpose(inter)
            note_seq, chord_seq = make_note_and_chord_seq_from_musicxml(score)
            main_onehot_seq = add_rest_nodes(note_seq_to_onehot(note_seq))
            main_chroma_seq = chord_seq_to_chroma(chord_seq)
            divide_seq(main_onehot_seq, main_chroma_seq, x, y)
    x_all = np.array(x)
    y_all = np.array(y)
    return x_all, y_all
