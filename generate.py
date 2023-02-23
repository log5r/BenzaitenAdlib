import benzaitencore as bc
import music_utils as mu
import numpy as np
import datetime


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

    note_num_list = bc.calc_notenums_from_pianoroll(pianoroll)

    # 伴奏ファイルの生成
    backing_mus_path = bc.BASE_DIR + "/" + backing_file
    target_midi = bc.read_midi_file(backing_mus_path)

    # 補正
    corrected_notes = mu.corrected_note_num_list(note_num_list, chord_prog, remove_suffix_prob, strict_mode)

    # 同一ノートを結合
    durations, dur_fixed_notes = bc.calc_durations(corrected_notes)

    # サフィックス定義
    suffix = "%s_r%s_%s" % (model_idf, str(remove_suffix_prob), "ST" if strict_mode else "O")

    # MIDIファイル生成
    res_midi = bc.make_midi(dur_fixed_notes, durations, 12, target_midi)

    # MIDIファイル補正

    # TBD

    # MIDIファイルのセーブ
    midi_out_path = bc.BASE_DIR + output_file
    res_midi.save(midi_out_path)

    # MWAVファイルを生成
    bc.generate_wav_file(suffix, midi_out_path)


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
