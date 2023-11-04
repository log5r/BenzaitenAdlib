#!/usr/bin/env sh

thisrepo="BenzaitenAdlibCode"
mkdir $thisrepo
cp ./LICENSE ./$thisrepo/LICENSE
cp ./*py ./$thisrepo/
cp ./*sh ./$thisrepo/
mkdir ./$thisrepo/omnibook
mkdir ./$thisrepo/sample
mkdir ./$thisrepo/output
mkdir ./$thisrepo/soundfonts

zip -r ./${thisrepo}.zip ./${thisrepo}/
rm -rf ./${thisrepo}/
