#!/usr/bin/env sh

thisrepo="BenzaitenAdlibCode"
mkdir $thisrepo
cp ./LICENSE ./$thisrepo/LICENSE
cp ./*py ./$thisrepo/
cp ./*sh ./$thisrepo/
mkdir ./$thisrepo/omnibook
mkdir ./$thisrepo/sample
mkdir -p ./$thisrepo/output/midi ./$thisrepo/output/solo ./$thisrepo/output/wav ./$thisrepo/models/current
mkdir ./$thisrepo/soundfonts

zip -r ./${thisrepo}.zip ./${thisrepo}/
rm -rf ./${thisrepo}/
