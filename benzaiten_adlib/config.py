TOTAL_MEASURES = 240        # 学習用MusicXMLを読み込む際の小節数の上限
UNIT_MEASURES = 4           # 1回の生成で扱う旋律の長さ
BEAT_RESO = 4               # 1拍を何個に分割するか（4の場合は16分音符単位）
N_BEATS = 4                 # 1小節の拍数（今回は4/4なので常に4）
NOTENUM_FROM = 36           # 扱う音域の下限（この値を含む）
NOTENUM_THRU = 84           # 扱う音域の上限（この値を含まない）
INTRO_BLANK_MEASURES = 4    # ブランクおよび伴奏の小節数の合計
MELODY_LENGTH = 8           # 生成するメロディの長さ（小節数）

TICKS_PER_BEAT = 480        # 四分音符を何ticksに分割するか
MELODY_PROG_CHG = 73        # メロディの音色（プログラムチェンジ）
MELODY_CH = 0               # メロディのチャンネル
