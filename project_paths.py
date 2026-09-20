"""Locations shared by training, generation, and conversion scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "current"
MIDI_DIR = ROOT / "output" / "midi"
SOLO_DIR = ROOT / "output" / "solo"
WAV_DIR = ROOT / "output" / "wav"


def ensure_output_dirs():
    for directory in (MIDI_DIR, SOLO_DIR, WAV_DIR):
        directory.mkdir(parents=True, exist_ok=True)
