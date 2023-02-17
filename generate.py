import benzaitencore as bc
import numpy as np
import datetime

# タイムスタンプ
timestamp = format(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')

# ファイル定義
backing_file = "sample/sample_backing.mid"
chord_file = "sample/sample_chord.csv"
output_file = "output/%s_output.mid" % timestamp


# config読み込み
config_file = open('config.benzaitenconfig', 'r')
configurations = config_file.readlines()
seq_length = int(configurations[0])
input_dim = int(configurations[1])
output_dim = int(configurations[2])
config_file.close()

# VAEモデルの読み込み
main_vae = bc.make_model(seq_length, input_dim, output_dim)
main_vae.load_weights(bc.BASE_DIR + "/mymodel.h5")

chord_prog = bc.read_chord_file(bc.BASE_DIR + chord_file)
chroma_vec = bc.chord_seq_to_chroma(bc.make_chord_seq(chord_prog, bc.N_BEATS))
pianoroll = bc.make_empty_pianoroll(chroma_vec.shape[0])
for i in range(0, bc.MELODY_LENGTH, bc.UNIT_MEASURES):
    o, c = bc.extract_seq(i, pianoroll, chroma_vec)
    x, y = bc.calc_xy(o, c)
    y_new = main_vae.predict(np.array([x]))
    index_from = i * (bc.N_BEATS * bc.BEAT_RESO)
    pianoroll[index_from: index_from + y_new[0].shape[0], :] = y_new[0]


notenumlist = bc.calc_notenums_from_pianoroll(pianoroll)
assert(len(notenumlist) == 128)

# 補正
for i, e in enumerate(notenumlist):
    safenotes = chord_prog[i // 4]
    target_class = e % 12
    if (e % 12) not in safenotes.pitchClasses:
        dist = 999
        for k in safenotes:
            expected_class = k.pitch.midi % 12
            buf = abs(target_class - expected_class)
            if buf < dist:
                dist = buf
        isMinusWork = abs(50 - (e - dist)) < abs(50 - (e + dist))
        fixed_note = e - dist if isMinusWork else e + dist
        notenumlist[i] = fixed_note

# ピアノロール表示
bc.plot_pianoroll(pianoroll)
bc.show_and_play_midi(notenumlist, 12, bc.BASE_DIR + "/" + backing_file, bc.BASE_DIR + output_file)
