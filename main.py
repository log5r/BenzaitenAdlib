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

TOTAL_MEASURES = 240  # 学習用MusicXMLを読み込む際の小節数の上限
UNIT_MEASURES = 4  # 1回の生成で扱う旋律の長さ
BEAT_RESO = 4  # 1拍を何個に分割するか(4の場合は16分音符単位)
N_BEATS = 4  # 1小節の拍数(今回は4/4なので常に4)
NOTENUM_FROM = 36  # 扱う音域の下限(この値を含む)
NOTENUM_THRU = 84  # 扱う音域の上限(この値を含まない)
INTRO_BLANK_MEASURES = 4  # ブランクおよび伴奏の小節数の合計
MELODY_LENGTH = 8  # 生成するメロディの長さ(小節数)
KEY_ROOT = "C"  # 生成するメロディの調のルート("C" or "A")
KEY_MODE = "major"  # 生成するメロディの調のモード("major" or "minor")


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
        if note_seq[i] != None:
            matrix[i, note_seq[i].pitch.midi - NOTENUM_FROM] = 1
    return matrix


# 音符列を表すone-hot vector列に休符要素を追加
def add_rest_nodes(onehot_seq):
    rest = 1 - np.sum(onehot_seq, axis=1)
    rest = np.expand_dims(rest, 1)
    return np.concatenate([onehot_seq, rest], axis=1)


# 指定された仕様のcsvファイルを読み込んで # ChordSymbol列を返す
def read_chord_file(file):
    chord_seq = [None] * (MELODY_LENGTH * N_BEATS)
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            m = int(row[0])  # 小節番号(0始まり) if m < MELODY_LENGTH:
            b = int(row[
                        1])  # 拍番号(0始まり、今回は0または2) chord_seq[m*4+b] = music21.harmony.ChordSymbol(root=row[2], kind=row[3], bass=row[4])
            for i in range(len(chord_seq)):
                if chord_seq[i] != None:
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
        if chord_seq[i] != None:
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
    track = mido.MidiTrack()
    midi.tracks.append(track)
    init_tick = INTRO_BLANK_MEASURES * N_BEATS * MIDI_DIVISION
    prev_tick = 0
    for i in range(len(notenums)):
        if notenums[i] > 0:
            curr_tick = int(i * MIDI_DIVISION / BEAT_RESO) + init_tick
            track.append(mido.Message('note_on', note=notenums[i] + transpose,
                                      velocity=100, time=curr_tick - prev_tick))
            prev_tick = curr_tick
            curr_tick = int((i + durations[i]) * MIDI_DIVISION / BEAT_RESO) + init_tick
            track.append(mido.Message('note_off', note=notenums[i] + transpose,
                                      velocity=100, time=curr_tick - prev_tick))
            prev_tick = curr_tick
    midi.save(dst_filename)


# ピアノロールを描画し、MIDIファイルを再生
def show_and_play_midi(pianoroll, transpose, src_filename, dst_filename):
    plt.matshow(np.transpose(pianoroll))
    plt.show()
    notenums = calc_notenums_from_pianoroll(pianoroll)
    notenums, durations = calc_durations(notenums)
    make_midi(notenums, durations, transpose, src_filename, dst_filename)
    fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    fs.midi_to_audio(dst_filename, "output.wav")
    ipd.display(ipd.Audio("output.wav"))


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


# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から # モデルの入力、出力用のデータを作成して、配列に逐次格納する
def divide_seq(onehot_seq, chroma_seq, x_all, y_all):
    for i in range(0, TOTAL_MEASURES, UNIT_MEASURES):
        o, c, = extract_seq(i, onehot_seq, chroma_seq)
        if np.any(o[:, 0:-1] != 0):
            x, y = calc_xy(o, c)
            x_all.append(x)
            y_all.append(y)


# 学習用MusicXMLを読み込む
basedir = "/Users/judau/dev/music/benzaiten2023/adlib/"
dir = basedir + "omnibook/"
main_x_all = []
main_y_all = []
for f in glob.glob(dir + "/*.xml"):
    print(f)
    score = music21.converter.parse(f)
    key = score.analyze("key")
    if key.mode == KEY_MODE:
        inter = music21.interval.Interval(key.tonic, music21.pitch.Pitch(KEY_ROOT))
        score = score.transpose(inter)
        note_seq, chord_seq = make_note_and_chord_seq_from_musicxml(score)
        main_onehot_seq = add_rest_nodes(note_seq_to_onehot(note_seq))
        main_chroma_seq = chord_seq_to_chroma(chord_seq)
        divide_seq(main_onehot_seq, main_chroma_seq, main_x_all, main_y_all)
x_all = np.array(main_x_all)
y_all = np.array(main_y_all)

# VAEのモデルを構築するための関数を定義する


encoded_dim = 32  # 潜在空間の次元数
seq_length = x_all.shape[1]  # 時間軸上の要素数
input_dim = x_all.shape[2]  # 入力データにおける各時刻のベクトルの次元数
output_dim = y_all.shape[2]  # 出力データにおける各時刻のベクトルの次元数
lstm_dim = 1024  # LSTM層のノード数


# VAEに用いる事前分布を定義
def make_prior():
    tfd = tfp.distributions
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(encoded_dim), scale=1),
        reinterpreted_batch_ndims=1)
    return prior


# エンコーダを構築
def make_encoder(prior):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.LSTM(lstm_dim,
                                     input_shape=(seq_length, input_dim),
                                     use_bias=True, activation="tanh",
                                     return_sequences=False))
    encoder.add(tf.keras.layers.Dense(
        tfp.layers.MultivariateNormalTriL.params_size(encoded_dim),
        activation=None))
    encoder.add(tfp.layers.MultivariateNormalTriL(
        encoded_dim,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(
            prior, weight=0.001)))
    return encoder


# デコーダを構築
def make_decoder():
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.RepeatVector(seq_length, input_dim=encoded_dim))
    decoder.add(tf.keras.layers.LSTM(lstm_dim, use_bias=True, activation="tanh", return_sequences=True))
    decoder.add(tf.keras.layers.Dense(output_dim, use_bias=True, activation="softmax"))
    return decoder


# エンコーダとデコーダを構築し、それらを結合したモデルを構築する
# (入力:エンコーダの入力、
#  出力:エンコーダの出力をデコーダに入力して得られる出力)
def make_model():
    encoder = make_encoder(make_prior())
    decoder = make_decoder()
    vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss="categorical_crossentropy", metrics="categorical_accuracy")
    return vae


main_vae = make_model()
main_vae.fit(x_all, y_all, epochs=50)
main_vae.save(basedir + "/mymodel.h5")

backing_file = "sample_backing.mid"
chord_file = "sample_chord.csv"
output_file = "output.mid"
main_vae = make_model()
main_vae.load_weights(basedir + "/mymodel.h5")

chord_prog = read_chord_file(basedir + chord_file)
chroma_vec = chord_seq_to_chroma(make_chord_seq(chord_prog, N_BEATS))
pianoroll = make_empty_pianoroll(chroma_vec.shape[0])
for i in range(0, MELODY_LENGTH, UNIT_MEASURES):
    o, c = extract_seq(i, pianoroll, chroma_vec)
    x, y = calc_xy(o, c)
    y_new = main_vae.predict(np.array([x]))
    index_from = i * (N_BEATS * BEAT_RESO)
    pianoroll[index_from: index_from + y_new[0].shape[0], :] = y_new[0]

show_and_play_midi(pianoroll, 12, basedir + "/" + backing_file, basedir + output_file)
