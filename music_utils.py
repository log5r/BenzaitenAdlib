import random
import music21.harmony as harmony
import music21.midi
import mido
import common_features as Features
import numpy as np

# コードのサフィックスを除外
def remove_chord_suffix(chord_string):
    lst = list(chord_string)
    if not chord_has_suffix(chord_string):
        return chord_string
    if lst[1] == "m" and lst[2] != "a":
        return lst[0] + lst[1]
    else:
        return lst[0]


# コードがサフィックスを持っていればTrue、なければFalse
def chord_has_suffix(chord_string):
    lst = list(chord_string)
    if len(lst) <= 2:
        if len(lst) <= 1:
            return False
        if len(lst) == 2 and lst[1] == "m":
            return False
    return True


# Note補正を行う条件の定義
# 設定ファイルに切り出すべきとは思うけどシンプルに面倒...
def note_canonicalization_conditions(i):
    return [
        # 1
        (i < 16),
        # 2
        (16 <= i < 20),
        (20 <= i < 28 and i % 3 == 0),
        (28 <= i < 32),
        # 3
        (32 <= i < 48 and i % 4 == 2 or i % 16 == 0),
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


def note_chain_breaking_condition(i):
    return [
        # 1
        (9 <= i < 16),
        # 2
        (16 <= i < 24),
        (28 <= i < 32),
        # 3
        # -- NONE --
        # 4
        # -- NONE --
        # 5
        (68 <= i < 80),
        # 6
        (80 <= i < 96),
        # 7
        (96 <= i < 100),
        (108 <= i < 112),
        # 8
        (112 <= i < 116),
        (120 <= i < 128)
    ]


# ノート補正
def corrected_note_num_list(note_num_list, chord_prog, model_type, features):
    res_note_list = []
    same_note_chain = 0
    through_count = 0
    bottom_count = 0

    ignore_chord_suffix = Features.IGNORE_CHORD_SUFFIX in features
    use_swing_mode_via2t3 = Features.COPY_SQ2_TO_SQ3 in features

    key, mode = model_type.split("_")
    base_key = music21.key.Key(key, mode)

    prev_note = 0

    for i, e in enumerate(note_num_list):
        area_chord = chord_prog[i // 4]

        if ignore_chord_suffix:
            goal_chord = harmony.ChordSymbol(str(remove_chord_suffix(area_chord.figure)))
        else:
            goal_chord = area_chord

        fixed_note = e
        target_class = fixed_note % 12

        good_notes = goal_chord.pitchClasses

        # --- 攻めの補正 ---

        if use_swing_mode_via2t3:
            # TODO: this is test impl
            if i % 4 == 2:
                fixed_note = res_note_list[-1]

        if i == 0:
            if random.random() < 0.6:
                fixed_note = 60 + good_notes[-1]
            else:
                fixed_note = 60 + random.choice(good_notes)
        elif 1 <= i < 3:
            fixed_note = res_note_list[-1]
        elif 64 <= i < 76:
            pnb = ((6 + res_note_list[-1]) // 12) * 12
            fixed_note = [pnb + good_notes[-2], pnb + good_notes[-1], pnb + good_notes[-1] + 2][i % 3]
        else:
            # --- 守りの補正 ---
            if fixed_note < 24:
                # 低すぎる音が出たら適宜取り繕う
                next_n_idx = bottom_count % len(good_notes)
                if len(res_note_list) > 0:
                    if random.random() < 0.5:
                        fixed_note = res_note_list[-1]
                    else:
                        fixed_note = ((6 + res_note_list[-1]) // 12) * 12 + good_notes[next_n_idx]
                        bottom_count += 1
            elif (fixed_note % 12) not in good_notes:
                # コード構成音にない音について、特定条件のもとで補正する
                if any(note_canonicalization_conditions(i)):
                    clist = []
                    for k in goal_chord:
                        expected_class = k.pitch.midi % 12
                        buf = expected_class - target_class
                        clist.append([abs(buf), buf])
                    clist.sort(key=lambda z: z[0])
                    fixed_note = fixed_note + clist[0][1]
                else:
                    # Maj 7th の降下は禁止
                    if i != 0 and (prev_note - e) == 11:
                        e = prev_note + 1
                    # コードを取得
                    area_chord = chord_prog[i // 4]  # 1拍ごとにとりだし（4分音符単位のため）
                    # 有効コード音を取得
                    valid_notes = list(map(lambda x: x.pitch.midi % 12, area_chord.notes))
                    valid_notes = np.unique(valid_notes)
                    # アボイドノートを取得
                    root_note = area_chord.root().midi % 12
                    avoid_notes = get_avoid_notes(area_chord, base_key, root_note)
                    # 音補正
                    fixed_note = fixed_note_num(e, valid_notes, avoid_notes)
                    print("note: %d %d" % (fixed_note, (fixed_note % 12 if (fixed_note != -1) else -1)) + "|" + str(area_chord.notes) + "|" + str(valid_notes))

            # 同一ノートが連続したときの処理
            if i != 0 and len(res_note_list) > 0 and res_note_list[-1] == fixed_note:
                same_note_chain += 1
                if any(note_chain_breaking_condition(i)):
                    err_note_class = [res_note_list[-1] % 12]
                    if same_note_chain % 4 != 1 and len(res_note_list) > 1:
                        err_note_class += [res_note_list[-2] % 12]
                        if len(res_note_list) > 2 and len(good_notes) > 3:
                            err_note_class += [res_note_list[-3] % 12]
                    delta = random.choice(list(filter(lambda nt: nt not in err_note_class, good_notes)))
                    fixed_note = (fixed_note // 12) * 12 + delta

        res_note_list.append(fixed_note)
        # 前の音を保存
        prev_note = e

    # 最後の小節
    end_note_scale = (res_note_list[-1] // 12) * 12
    for cls in chord_prog[-1].pitchClasses:
        res_note_list.append(end_note_scale + cls)
    res_note_list.append(end_note_scale + 12 + chord_prog[-1].pitchClasses[0])
    res_note_list += [res_note_list[-1]] * 8

    # 高さをまとめて検査
    for i in range(len(res_note_list)):
        # 高すぎる音は結構耳につくので引いておく
        while res_note_list[i] > 84:
            res_note_list[i] -= 12
        # 低すぎるのもおかしい...
        while res_note_list[i] < 58:
            res_note_list[i] += 12

    return res_note_list


def convex_increasing_bend_curve():
    curve = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return list(map(lambda x: x - 4097, curve))


def concave_increasing_bend_curve():
    curve = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return reversed(list(map(lambda x: 0 - x, curve)))


def linear_increasing_bend_curve():
    curve = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    return reversed(list(map(lambda x: 0 - x, curve)))


def arrange_using_midi(target_midi: music21.midi):
    res_midi = target_midi
    res_main_tml = []

    note_on_t = 0
    note_off_t = 0
    note_value = 0

    bend_chk_counter = 9999
    for msg in target_midi.tracks[1]:
        if msg.type == 'note_on':
            note_on_t = msg.time
            note_value = msg.note
        elif msg.type == 'note_off':
            note_off_t = msg.time
            bend_chk_counter += note_off_t
            if note_off_t >= 240 and bend_chk_counter > 1439:
                note_on_msg = mido.Message('note_on', note=note_value, velocity=127, time=note_on_t)
                res_main_tml.append(note_on_msg)
                # -- bend --
                bend_curve = concave_increasing_bend_curve()
                if note_off_t >= 359:
                    bend_curve = convex_increasing_bend_curve()
                for c in bend_curve:
                    bend_msg_on = mido.Message('pitchwheel', channel=0, pitch=c, time=12)
                    res_main_tml.append(bend_msg_on)
                bend_reset_msg = mido.Message('pitchwheel', channel=0, pitch=0, time=0)
                res_main_tml.append(bend_reset_msg)

                note_off_msg = mido.Message('note_off', note=note_value, velocity=127, time=note_off_t - 120)
                res_main_tml.append(note_off_msg)

                bend_chk_counter = 0
            else:
                note_on_msg = mido.Message('note_on', note=note_value, velocity=127, time=note_on_t)
                res_main_tml.append(note_on_msg)
                note_off_msg = mido.Message('note_off', note=note_value, velocity=127, time=note_off_t)
                res_main_tml.append(note_off_msg)

        else:
            res_main_tml.append(msg)

    res_midi.tracks[1] = res_main_tml
    return res_midi


# 音補正処理
def fixed_note_num(note_num, valid_notes, avoid_notes):
    if note_num == -1:
        return note_num
    elif note_num % 12 in avoid_notes:
        # アボイドノート時は最寄りの音に補正
        for i, e in enumerate(valid_notes):
            if (e - note_num % 12) > 0:
                fixed_note = note_num + (e - note_num % 12)
                print("fixed note: %d -> %d %d" % (note_num, fixed_note, fixed_note % 12))
                return fixed_note
    # それ以外は有効な音階とみなす
    return note_num


# ドミナント判定（すごく強力なコード。通常のドミナントと代理のセカンダリードミナントを含む）
def is_dominant(area_chord, base_key):
    # ドミナント7thの場合はTrue
    if area_chord.isDominantSeventh():
        return True
    # 7thが含まれなくてもメジャーの I or IV は M7系なので除外
    if area_chord.isMajorTriad() and (area_chord.root().midi % 12 == 0 or area_chord.root().midi % 12 == 5):
        return False
    # それ以外のII,III,VI,VIIはマイナー系のはずだが、メジャー系の場合は、ドミナントとして扱う(セカンダリードミナント）
    # Vのメジャーも7thが省略されているだけで特に指定がなければ、ドミナントとして扱う
    if area_chord.isMajorTriad():
        return True
    return False


# アボイドノートを取得する
def get_avoid_notes(area_chord, base_key, root_note):
    # ドミナントセブンスもしくはIonianの場合は4thを避ける
    if is_dominant(area_chord, base_key) or root_note == 0:
        avoid_notes = [root_note + 5]
    # Dorianの場合は6thを避ける
    elif root_note == 2:
        avoid_notes = [root_note + 9]
    # Phrygianの場合はb2th, b6thを避ける
    elif root_note == 4:
        avoid_notes = [root_note + 1, root_note + 8]
    # Aeolianの場合はb6thを避ける
    elif root_note == 9:
        avoid_notes = [root_note + 8]
    # Locrianの場合はb2thを避ける
    elif root_note == 11:
        avoid_notes = [root_note + 1]
    else:
        avoid_notes = []
    return avoid_notes

