import benzaitencore as bc
import music_utils as mu
import numpy as np
import datetime
import music21.harmony as harmony

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
fixednotenumlist = []

# 補正
for i, e in enumerate(notenumlist):
    area_chord = chord_prog[i // 4]
    # IF REMOVE_CHORD_SUFFIX
    # fixed_chord_str = str(mu.remove_chord_suffix(area_chord.figure))
    # goal_chord = harmony.ChordSymbol(fixed_chord_str)
    # ELSE
    goal_chord = area_chord
    # END IF
    fixed_note = e
    target_class = e % 12

    if (e % 12) not in goal_chord.pitchClasses:
        conditions = [
            # 1
            (i < 16),
            # 2
            (16 <= i <= 20),
            (20 <= i < 32 and i % 2 != 1),
            # 3
            (32 <= i < 48 and i % 8 != 1),
            # 4
            (48 <= i < 52),
            (52 <= i < 60 and i % 2 != 1),
            (60 <= i < 64),
            # 5
            (64 <= i < 80 and i % 4 != 1),
            # 6
            (80 <= i < 82),
            (82 <= i < 96 and i % 2 != 1),
            # 7
            (96 <= i < 112 and i % 16 != 4),
            # 8
            (112 <= i)
        ]
        if any(conditions):
            clist = []
            for k in goal_chord:
                expected_class = k.pitch.midi % 12
                buf = expected_class - target_class
                clist.append([abs(buf), buf])
            clist.sort(key=lambda z: z[0])
            fixed_note = e + clist[0][1]
            if i > 0 and fixednotenumlist[-1] == fixed_note:
                fixed_note = clist[1][1]
    fixednotenumlist.append(fixed_note)

# MIDIとWAVファイルを生成
bc.generate_midi_and_wav(fixednotenumlist, 12, bc.BASE_DIR + "/" + backing_file, bc.BASE_DIR + output_file)
