import music21
import numpy as np
import matplotlib.pyplot as plt
import mido
import csv
import IPython.display as ipd
import midi2audio
import glob
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import functools

TOTAL_MEASURES = 240  # 学習用MusicXMLを読み込む際の小節数の上限
UNIT_MEASURES = 4  # 1回の生成で扱う旋律の長さ
BEAT_RESO = 4  # 1拍を何個に分割するか(4の場合は16分音符単位)
N_BEATS = 4  # 1小節の拍数(今回は4/4なので常に4)
NOTENUM_FROM = 36  # 扱う音域の下限(この値を含む)
NOTENUM_THRU = 84  # 扱う音域の上限(この値を含まない)
INTRO_BLANK_MEASURES = 4  # ブランクおよび伴奏の小節数の合計
MELODY_LENGTH = 8  # 生成するメロディの長さ(小節数)

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
    note_seq = [None] * (TOTAL_MEASURES * N_BEATS * BEAT_RESO)
    chord_seq = [None] * (TOTAL_MEASURES * N_BEATS * BEAT_RESO)
    for element in score.parts[0].elements:
        if isinstance(element, music21.stream.Measure):
            measure_offset = element.offset
            for note in element.notes:
                if isinstance(note, music21.note.Note):
                    onset = measure_offset + note._activeSiteStoredOffset
                    offset = onset + note._duration.quarterLength
                    for i in range(int(onset * BEAT_RESO), int(offset * BEAT_RESO + 1)):
                        note_seq[i] = note
                if isinstance(note, music21.harmony.ChordSymbol):
                    chord_offset = measure_offset + note.offset
                    for i in range(int(chord_offset * BEAT_RESO),
                                   int((measure_offset + N_BEATS) * BEAT_RESO + 1)):
                        chord_seq[i] = note
    return note_seq, chord_seq


# Note列をone-hot vector列(休符はすべて0)に変換
def note_seq_to_onehot(note_seq):
    M = NOTENUM_THRU - NOTENUM_FROM
    N = len(note_seq)
    matrix = np.zeros((N, M))
    for i in range(N):
        if note_seq[i] is not None:
            matrix_idx_r = (note_seq[i].pitch.midi - NOTENUM_FROM) % M
            matrix[i, matrix_idx_r] = 1
    return matrix


# 音符列を表すone-hot vector列に休符要素を追加
def add_rest_nodes(onehot_seq):
    rest = 1 - np.sum(onehot_seq, axis=1)
    rest = np.expand_dims(rest, 1)
    return np.concatenate([onehot_seq, rest], axis=1)


# 指定された仕様のcsvファイルを読み込んで
# ChordSymbol列を返す
def read_chord_file(file):
    chord_seq = [None] * (MELODY_LENGTH * N_BEATS)
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            m = int(row[0])  # 小節番号(0始まり)
            if m < MELODY_LENGTH:
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
    T = int(N_BEATS * BEAT_RESO / division)
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
    return np.zeros((length, NOTENUM_THRU - NOTENUM_FROM + 1))


# ピアノロール(one-hot vector列)をノートナンバー列に変換
def calc_notenums_from_pianoroll(pianoroll):
    notenums = []
    for i in range(pianoroll.shape[0]):
        n = np.argmax(pianoroll[i, :])
        nn = -1 if n == pianoroll.shape[1] - 1 else n + NOTENUM_FROM
        notenums.append(nn)
    return notenums


# 連続するノートナンバーを統合して (notenums, durations) に変換
def calc_durations(notenums):
    N = len(notenums)
    duration = [1] * N
    for i in range(N):
        k = 1
        while i + k < N:
            if notenums[i] > 0 and notenums[i] == notenums[i + k]:
                notenums[i + k] = 0
                duration[i] += 1
            else:
                break
        k += 1
    return notenums, duration


# MIDIファイルを生成
def make_midi(notenums, durations, transpose, src_filename, dst_filename):
    midi = mido.MidiFile(src_filename)
    MIDI_DIVISION = midi.ticks_per_beat
    init_tick = INTRO_BLANK_MEASURES * N_BEATS * MIDI_DIVISION
    midi.tracks[1].pop()
    prev_tick = functools.reduce(lambda x, y: x + y, map(lambda u: u.time, midi.tracks[1]))
    for i, e in enumerate(notenums):
        if e > 0:
            curr_note = e + transpose

            note_on_tick = int(i * MIDI_DIVISION / BEAT_RESO) + init_tick
            note_on_time = note_on_tick - prev_tick
            note_on_msg = mido.Message('note_on', note=curr_note, velocity=100, time=note_on_time)
            midi.tracks[1].append(note_on_msg)

            note_off_tick = int((i + durations[i]) * MIDI_DIVISION / BEAT_RESO) + init_tick
            note_off_time = note_off_tick - note_on_tick
            note_off_msg = mido.Message('note_off', note=curr_note, velocity=100, time=note_off_time)
            midi.tracks[1].append(note_off_msg)

            prev_tick = note_off_tick

    midi.save(dst_filename)


# ピアノロールを描画
def plot_pianoroll(pianoroll):
    plt.matshow(np.transpose(pianoroll))
    plt.show()


# MIDIとWAVを生成
def generate_midi_and_wav(notes, transpose, src_filename, dst_filename, model_idf):
    notenums, durations = calc_durations(notes)
    make_midi(notenums, durations, transpose, src_filename, dst_filename)
    sf_path = "soundfonts/FluidR3_GM.sf2"
    fs = midi2audio.FluidSynth(sound_font=sf_path)
    timestamp = format(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')
    generated_filename = "%s_%s_output.wav" % (timestamp, model_idf)
    fs.midi_to_audio(dst_filename, generated_filename)
    ipd.display(ipd.Audio(generated_filename))


# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列に対して、
# UNIT_MEASURES小節分だけ切り出したものを返す
def extract_seq(i, onehot_seq, chroma_seq):
    o = onehot_seq[i * N_BEATS * BEAT_RESO: (i + UNIT_MEASURES) * N_BEATS * BEAT_RESO, :]
    c = chroma_seq[i * N_BEATS * BEAT_RESO: (i + UNIT_MEASURES) * N_BEATS * BEAT_RESO, :]
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
    for i in range(0, TOTAL_MEASURES, UNIT_MEASURES):
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
