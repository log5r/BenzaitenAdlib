import random
import music21.harmony as harmony
import music21.midi


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
        (112 <= i)
    ]


# ノート補正
def corrected_note_num_list(notenumlist, chord_prog, remove_suffix_prob, strict_mode):
    fixed_note_num_list = []
    same_note_chain = 0
    through_count = 0
    bottom_count = 0
    for i, e in enumerate(notenumlist):
        area_chord = chord_prog[i // 4]

        if random.random() < remove_suffix_prob:
            goal_chord = harmony.ChordSymbol(str(remove_chord_suffix(area_chord.figure)))
        else:
            goal_chord = area_chord

        fixed_note = e
        target_class = fixed_note % 12

        if i == 0:
            if random.random() < 0.6:
                fixed_note = 60 + goal_chord.pitchClasses[-1]
            else:
                fixed_note = 60 + random.choice(goal_chord.pitchClasses)
        elif fixed_note < 24:
            # 低すぎる音が出たら適宜取り繕う
            next_n_idx = bottom_count % len(goal_chord.pitchClasses)
            if len(fixed_note_num_list) > 0:
                if random.random() < 0.5:
                    fixed_note = fixed_note_num_list[-1]
                else:
                    fixed_note = (fixed_note_num_list[-1] // 12) * 12 + goal_chord.pitchClasses[next_n_idx]
                    bottom_count += 1
        elif (fixed_note % 12) not in goal_chord.pitchClasses:
            # コード構成音にない音について、特定条件のもとで補正する
            if any(note_canonicalization_conditions(i)):
                clist = []
                for k in goal_chord:
                    expected_class = k.pitch.midi % 12
                    buf = expected_class - target_class
                    clist.append([abs(buf), buf])
                clist.sort(key=lambda z: z[0])
                fixed_note = fixed_note + clist[0][1]

        # C or A で黒鍵は登場しないはずなので、補正する。ただし、サフィックス付きコードはたまに許す
        if i != 0 and (fixed_note % 12) in [1, 3, 6, 8, 10]:
            if strict_mode or (through_count % 5 != 1 and not chord_has_suffix(goal_chord.figure)):
                through_count += 1
                fixed_note += random.choice([1, -1])

        # 同一ノートが連続したときの処理
        if i != 0 and len(fixed_note_num_list) > 0 and fixed_note_num_list[-1] == fixed_note:
            same_note_chain += 1
            if any(note_chain_breaking_condition(i)):
                err_note_class = [fixed_note_num_list[-1] % 12]
                if same_note_chain % 4 != 1 and len(fixed_note_num_list) > 1:
                    err_note_class += [fixed_note_num_list[-2] % 12]
                    if len(fixed_note_num_list) > 2 and len(goal_chord.pitchClasses) > 3:
                        err_note_class += [fixed_note_num_list[-3] % 12]
                delta = random.choice(list(filter(lambda nt: nt not in err_note_class, goal_chord.pitchClasses)))
                fixed_note = (fixed_note // 12) * 12 + delta

        fixed_note_num_list.append(fixed_note)

    # 最後の小節
    end_note_scale = (fixed_note_num_list[-1] // 12) * 12
    for cls in chord_prog[-1].pitchClasses:
        fixed_note_num_list.append(end_note_scale + cls)
    fixed_note_num_list.append(end_note_scale + 12 + chord_prog[-1].pitchClasses[0])
    fixed_note_num_list += [fixed_note_num_list[-1]] * 8

    return fixed_note_num_list
