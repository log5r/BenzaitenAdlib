#!/usr/bin/env sh

cd "$(dirname "$0")" || exit 1
mkdir -p sample omnibook/C_major omnibook/A_minor soundfonts models/current output/midi output/solo output/wav

echo "please add soundfont at soundfonts directory."
echo "please add xml for learning at omnibook directory."
echo "please add sample file set at sample directory."
