#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
project_root=$(pwd)
thisrepo="BenzaitenAdlibCode"
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT HUP INT TERM
destination="$staging/$thisrepo"
mkdir -p "$destination/benzaiten_adlib" "$destination/experiments" "$destination/scripts/legacy" "$destination/docs" "$destination/tests"
cp LICENSE README.md README.ja.md requirements.txt pyproject.toml "$destination/"
cp benzaiten_adlib/*.py "$destination/benzaiten_adlib/"
cp tests/*.py "$destination/tests/"
cp experiments/*.py "$destination/experiments/"
cp scripts/*.sh "$destination/scripts/"
cp scripts/legacy/*.sh "$destination/scripts/legacy/"
cp docs/*.md docs/*.json "$destination/docs/"
sh "$destination/scripts/setup_required_folders.sh"

# Build a fresh archive so deleted source files cannot survive from an older ZIP.
cd "$staging"
zip -qr "$thisrepo.zip" "$thisrepo"
mv "$thisrepo.zip" "$project_root/$thisrepo.zip"
