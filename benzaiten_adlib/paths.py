"""Locations shared by training, generation, and conversion scripts."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BENZAITEN_ROOT", Path(__file__).resolve().parent.parent)).expanduser().resolve()
SAMPLE_DIR = ROOT / "sample"
MUSIC_DIR = ROOT / "omnibook"
SOUNDFONT_DIR = ROOT / "soundfonts"
MODEL_DIR = ROOT / "models" / "current"
MIDI_DIR = ROOT / "output" / "midi"
SOLO_DIR = ROOT / "output" / "solo"
WAV_DIR = ROOT / "output" / "wav"


def ensure_output_dirs():
    for directory in (MIDI_DIR, SOLO_DIR, WAV_DIR):
        directory.mkdir(parents=True, exist_ok=True)
