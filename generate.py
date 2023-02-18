import benzaitencore as bc
import music_utils as mu
import numpy as np
import datetime
import music21.harmony as harmony
import random


def note_correction_conditions(i):
    return [
        # 1
        (i < 16),
        # 2
        (16 <= i < 20),
        (20 <= i < 28 and i % 3 == 0),
        (28 <= i < 32),
        # 3
        (32 <= i < 48 and i % 4 == 0),
        # 4
        (48 <= i < 52),
        (52 <= i < 60 and i % 2 == 0),
        (60 <= i < 64),
        # 5
        (64 <= i < 80 and i % 3 == 0),
        # 6
        (80 <= i < 82),
        (82 <= i < 92 and i % 2 == 0),
        (92 <= i < 96),
        # 7
        (96 <= i < 112 and i % 16 != 4),
        # 8
        (112 <= i)
    ]


def generate_files(model_idf, remove_suffix_prob, strict_mode=False):
    # タイムスタンプ
    timestamp = format(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')

    # ファイル定義
    backing_file = "sample/sample_backing.mid"
    chord_file = "sample/sample_chord.csv"
    output_file = "output/%s_%s_output.mid" % (timestamp, model_idf)

    # config読み込み
    model_idf = model_idf.replace("_ST", "")
    model_idf = model_idf.replace("_O", "")
    config_file = open("%s.benzaitenconfig" % model_idf, 'r')
    configurations = config_file.readlines()
    seq_length = int(configurations[0])
    input_dim = int(configurations[1])
    output_dim = int(configurations[2])
    config_file.close()

    # VAEモデルの読み込み
    main_vae = bc.make_model(seq_length, input_dim, output_dim)
    main_vae.load_weights(bc.BASE_DIR + "/mymodel_%s.h5" % model_idf)

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
    same_note_chain = 0
    through_count = 0
    bottom_count = 0
    for i, e in enumerate(notenumlist):
        area_chord = chord_prog[i // 4]

        if random.random() < remove_suffix_prob:
            fixed_chord_str = str(mu.remove_chord_suffix(area_chord.figure))
            goal_chord = harmony.ChordSymbol(fixed_chord_str)
        else:
            goal_chord = area_chord

        fixed_note = e
        target_class = e % 12

        if e < 24:
            next_n_idx = bottom_count % len(goal_chord.pitchClasses)
            fixed_note = 48 + goal_chord.pitchClasses[next_n_idx]
            bottom_count += 1
        elif (e % 12) not in goal_chord.pitchClasses:
            if any(note_correction_conditions(i)):
                clist = []
                for k in goal_chord:
                    expected_class = k.pitch.midi % 12
                    buf = expected_class - target_class
                    clist.append([abs(buf), buf])
                clist.sort(key=lambda z: z[0])
                fixed_note = e + clist[0][1]
                if i > 0 and fixednotenumlist[-1] == fixed_note:
                    fixed_note = e + clist[1][1]
            elif (e % 12) in [1, 3, 6, 8, 10]:
                through_count += 1
                if through_count % 7 != 0:
                    fixed_note = e + random.choice([1, -1])
        # fixed_note += random.choice([0, 0, 12])
        if len(fixednotenumlist) > 0 and fixednotenumlist[-1] == fixed_note:
            same_note_chain += 1
            if same_note_chain > random.choice([1, 2, 3]):
                fixed_note = list(filter(lambda x: x != fixed_note, goal_chord.pitchClasses))[0]
                same_note_chain = 0
        fixednotenumlist.append(fixed_note)

    for i, e in enumerate(notenumlist):
        print("%s: %s" % (i, e))

    # MIDIとWAVファイルを生成
    bf_path = bc.BASE_DIR + "/" + backing_file
    out_path = bc.BASE_DIR + output_file
    suffix = "%s_r%s_%s" % (model_idf, str(remove_suffix_prob), "ST" if strict_mode else "O")
    bc.generate_midi_and_wav(fixednotenumlist, 12, bf_path, out_path, suffix)


# generate_files("C_major_O", 0)
# generate_files("C_major_O", 1)
# generate_files("C_major_O", 0.5)
#
# generate_files("A_minor_O", 0)
# generate_files("A_minor_O", 1)
# generate_files("A_minor_O", 0.5)

generate_files("C_major_ST", 0, True)
# generate_files("C_major_ST", 1, True)
# generate_files("C_major_ST", 0.5, True)

# generate_files("A_minor_ST", 0, True)
# generate_files("A_minor_ST", 1, True)
# generate_files("A_minor_ST", 0.5, True)
