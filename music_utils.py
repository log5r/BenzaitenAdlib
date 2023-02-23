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
            if any(note_correction_conditions(i)):
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

        # 同一ノートが連続したときの処理。２回まではゆるすが、３回以降は変える
        if i != 0 and len(fixed_note_num_list) > 0 and fixed_note_num_list[-1] == fixed_note:
            same_note_chain += 1
            if same_note_chain > random.choice([1, 2, 3]):
                delta = list(filter(lambda nt: nt != fixed_note, goal_chord.pitchClasses))[0]
                fixed_note = (fixed_note // 12) * 12 + delta
                same_note_chain = 0

        # サフィックス除去後のコード構成音と半音をなすような音が出てきたら、プラマイ1する
        safe_chord_list = harmony.ChordSymbol(str(remove_chord_suffix(area_chord.figure)))
        if i != 0 and any(map(lambda en: abs(fixed_note - en) == 1, safe_chord_list.pitchClasses)):
            fixed_note += random.choice([1, -1])

        fixed_note_num_list.append(fixed_note)

    # 最後の音（2部音符＋8部音符だけ持続）
    lnmp = list(
        map(lambda ln: [abs(ln - fixed_note_num_list[-1]), ln - fixed_note_num_list[-1]], [48, 52, 60, 64, 72, 76]))
    lnmp.sort(key=lambda u: u[0])
    fixed_note_num_list += [fixed_note_num_list[-1] + lnmp[0][1]] * 10

    return fixed_note_num_list


def correct_note_using_accomplishment(note_num_list, back_midi: music21.midi):
    fixed_note_num_list = []
