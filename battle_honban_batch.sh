#!/usr/bin/env sh

# これは試合本番で受け取ったファイルが想定と違ったので名前を変更したりするためにあわててかきあげたshell

mv battle/battle*_backing.mid sample/sample_backing.mid
mv battle/battle*_chord.csv sample/sample_chord.csv
rm battle/battle*_key.txt
