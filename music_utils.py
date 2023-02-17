# コードのサフィックスを除外
def remove_chord_suffix(chord_string):
    lst = list(chord_string)
    if len(lst) <= 2:
        # サフィックスがないなら処理しない
        if len(lst) <= 1:
            return chord_string
        if len(lst) == 2 and lst[1] == "m":
            return chord_string
    if lst[1] == "m" and lst[2] != "a":
        return lst[0] + lst[1]
    else:
        return lst[0]