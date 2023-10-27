import benzaitencore as bc
import music_utils as mu
import numpy as np
import datetime
import time
import common_model_type as ModelType
import common_features as Features
import benzaiten_submit_util as bsu
import benzaiten_config as cfg


def print_proc_time(f):
    def print_proc_time_func(*args, **kwargs):
        start_time = time.process_time()
        return_val = f(*args, **kwargs)
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        print("FUNCTION: %s (%s sec)" % (f.__name__, elapsed_time))
        return return_val

    return print_proc_time_func


def generate_adlib_files(model_type, features=None):
    # 引数処理
    if features is None:
        features = []

    # タイムスタンプ
    timestamp = format(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')

    # ファイル定義
    backing_file = "sample/sample_backing.mid"
    chord_file = "sample/sample_chord.csv"

    # config読み込み
    config_file = open("%s.benzaitenconfig" % model_type, 'r')
    configurations = config_file.readlines()
    seq_length = int(configurations[0])
    input_dim = int(configurations[1])
    output_dim = int(configurations[2])
    config_file.close()

    # VAEモデルの読み込み
    main_vae = bc.make_model(seq_length, input_dim, output_dim)
    main_vae.load_weights(bc.BASE_DIR + "/mymodel_%s.h5" % model_type)

    chord_prog = bc.read_chord_file(bc.BASE_DIR + chord_file)
    chord_prog_append = bc.read_chord_file(bc.BASE_DIR + chord_file, 1)
    chroma_vec = bc.chord_seq_to_chroma(bc.make_chord_seq(chord_prog, cfg.N_BEATS))
    pianoroll = bc.make_empty_pianoroll(chroma_vec.shape[0])
    for i in range(0, cfg.MELODY_LENGTH, cfg.UNIT_MEASURES):
        o, c = bc.extract_seq(i, pianoroll, chroma_vec)
        x, y = bc.calc_xy(o, c)
        y_new = main_vae.predict(np.array([x]))
        index_from = i * (cfg.N_BEATS * cfg.BEAT_RESO)
        pianoroll[index_from: index_from + y_new[0].shape[0], :] = y_new[0]

    note_num_list = bc.calc_notenums_from_pianoroll(pianoroll)

    # 伴奏ファイルの生成
    backing_mus_path = bc.BASE_DIR + "/" + backing_file
    target_midi = bc.read_midi_file(backing_mus_path)

    # 補正
    if Features.CORRECTION_TYPE2 in features:
        corrected_notes = mu.corrected_note_num_list_type2(note_num_list, chord_prog_append, model_type, features)
    elif Features.CORRECTION_TYPE3 in features:
        corrected_notes = mu.corrected_note_num_list_type3(note_num_list, chord_prog_append, model_type, features)
    else:
        corrected_notes = mu.corrected_note_num_list_type1(note_num_list, chord_prog_append, model_type, features)

    # 同一ノートを結合
    durations, dur_fixed_notes = bc.calc_durations(corrected_notes)

    # サフィックス定義
    suffix = "%s_%s" % (model_type, "_".join(features))

    # MIDIファイル生成
    res_midi = bc.make_midi(dur_fixed_notes, chord_prog_append, durations, 12, target_midi, features)

    # MIDIファイル補正
    arranged_midi = mu.arrange_using_midi(res_midi)
    # TBD
    for i, tr in enumerate(arranged_midi.tracks[1]):
        print(i, tr)

    fixed_midi = bsu.replace_prog_chg(arranged_midi)

    # MIDIファイルのセーブ
    output_file = "output/%s_output_%s.mid" % (timestamp, suffix)
    midi_out_path = bc.BASE_DIR + output_file
    fixed_midi.save(midi_out_path)

    # 【弁財天第2幕用】提出用MIDIファイル生成
    sbm_midi = bsu.make_midi_for_submission_using_midi(fixed_midi)
    sbm_output_file = "contest_submit/%s_output_%s_solo.mid" % (timestamp, suffix)
    sbm_midi_out_path = bc.BASE_DIR + sbm_output_file
    sbm_midi.save(sbm_midi_out_path)

    # MWAVファイルを生成
    bc.generate_wav_file(suffix, midi_out_path)


@print_proc_time
def generate_file_set():
    generate_adlib_files(ModelType.C_MAJOR, features=[Features.CORRECTION_TYPE1])
    generate_adlib_files(ModelType.C_MAJOR, features=[Features.CORRECTION_TYPE1, Features.V2_SHUFFLE, Features.TRIPLET_SEMIQUAVER])
    # generate_adlib_files(ModelType.C_MAJOR, features=[Features.CORRECTION_TYPE2])
    # generate_adlib_files(ModelType.C_MAJOR, features=[Features.CORRECTION_TYPE2, Features.V2_SHUFFLE])
    generate_adlib_files(ModelType.C_MAJOR, features=[Features.CORRECTION_TYPE3])
    generate_adlib_files(ModelType.C_MAJOR, features=[Features.CORRECTION_TYPE3, Features.V2_SHUFFLE])

    generate_adlib_files(ModelType.A_MINOR, features=[Features.CORRECTION_TYPE1])
    generate_adlib_files(ModelType.A_MINOR, features=[Features.CORRECTION_TYPE1, Features.V2_SHUFFLE, Features.TRIPLET_SEMIQUAVER])
    # generate_adlib_files(ModelType.A_MINOR, features=[Features.CORRECTION_TYPE2])
    # generate_adlib_files(ModelType.A_MINOR, features=[Features.CORRECTION_TYPE2, Features.V2_SHUFFLE])
    generate_adlib_files(ModelType.A_MINOR, features=[Features.CORRECTION_TYPE3])
    generate_adlib_files(ModelType.A_MINOR, features=[Features.CORRECTION_TYPE3, Features.V2_SHUFFLE])
    # generate_adlib_files(ModelType.A_MINR, features=[Features.STRICT_MODE])


generate_file_set()
