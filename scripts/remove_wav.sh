#!/usr/bin/env sh

cd "$(dirname "$0")/.." || exit 1
rm -f ./output/wav/*.wav ./output/midi/*.mid ./output/solo/*.mid
