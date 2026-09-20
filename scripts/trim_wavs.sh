#!/bin/bash

# 生成先の .wav ファイルを処理
cd "$(dirname "$0")/../output/wav" || exit 1
for file in *.wav; do
  # 入力ファイルが実際に存在するかどうかを確認
  if [ -f "$file" ]; then
    # 出力ファイル名を決定（元の名前に '_trimmed' を追加）
    output="${file%.wav}_trimmed.wav"

    # ffmpegを使用して、ファイルの最初の4秒をカット
    ffmpeg -n -i "$file" -ss 4 -c copy "$output" && rm "${file}"
  else
    echo "No .wav files found in the current directory."
    break
  fi
done
