# BenzaitenAdlib

English | [日本語](README.ja.md)

BenzaitenAdlib is a Python program that generates improvised melodies from a chord progression and exports them as MIDI and WAV files with accompaniment. Developed as experimental code for the Benzaiten music generation contest, it combines a variational autoencoder (VAE) built with TensorFlow / Keras 3 with rules for adjusting pitch and rhythm. It was originally provided as a sample for M1 Macs.

Training uses MusicXML scores containing melody notes and chord symbols. Generation uses a backing MIDI file and a chord progression CSV. By default, the program generates eight measures in four-measure segments, adds ending notes during post-processing, and places the melody after a four-measure introduction.

```text
MusicXML → benzaiten_adlib/learn.py → Trained model and shape configuration
                                      ↓
Backing MIDI + chord CSV → benzaiten_adlib/generate.py → MIDI with backing / solo MIDI / WAV
```

## Before you start

Currently, `benzaiten_adlib/learn.py` trains only the C major model, while `benzaiten_adlib/generate.py` loads both the C major and A minor models. If you are training from scratch, either enable A minor training as described below or limit generation to C major.

Trained models, input samples, and the SoundFont are not tracked in Git. You can reuse them if they are already available locally, but cloning the repository alone does not provide everything needed for generation. Run all commands below from the project root.

## 1. Set up the environment

Use **Python 3.13** (Python 3.12 is also allowed by the dependency set). Python 3.14 is not supported: [TensorFlow 2.21 provides official builds through Python 3.13](https://www.tensorflow.org/install/source). On macOS, the current TensorFlow wheel requires Apple Silicon and macOS 12 or later. Intel Macs are outside this updated environment's scope.

Create a separate environment to preserve any previous installation:

```sh
python3.13 -m venv .venv-py313
source .venv-py313/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

`requirements.txt` pins TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3, music21 10.5.0, Matplotlib 3.11.2, mido 1.3.3, and midi2audio 0.1.1. TensorFlow Probability and legacy `tf-keras` are no longer required. See [the migration and validation notes](docs/PYTHON_UPGRADE.md) for the tested environment and model compatibility.

WAV rendering requires [FluidSynth itself in addition to the `midi2audio` Python package](https://github.com/bzamecnik/midi2audio). With Homebrew, install it as follows:

```sh
brew install fluidsynth
```

Create the input and output directories. `benzaiten_adlib/generate.py` also creates its output directories automatically.

```sh
sh scripts/setup_required_folders.sh
```

## 2. Prepare the backing track, chords, and SoundFont

Place the following three files at these exact paths, which are hardcoded in the source.

| Path | Contents |
| --- | --- |
| `sample/sample_backing.mid` | Backing MIDI into which the generated melody is inserted |
| `sample/sample_chord.csv` | Chord progression used for generation and pitch correction |
| `soundfonts/FluidR3_GM.sf2` | SoundFont used for WAV rendering |

The original README points to the [Benzaiten sample folder](https://drive.google.com/drive/folders/1jZSMX14B-i98x06QowaNL_9VGXeJZJbd) for backing tracks and chord progressions. For example, rename `sample1_backing.mid` and `sample1_chord.csv` to the names above. It also lists the [FluidR3_GM download page](https://member.keymusician.com/Member/FluidR3_GM/index.html) as a source for the SoundFont.

### Backing MIDI requirements

Generation **replaces the second track (`tracks[1]`) of the backing MIDI with the melody track**. Use a MIDI file with at least two tracks, with its second track reserved for replacement. The defaults assume 4/4 time, 480 ticks per quarter note, and a melody starting after four measures. Prepare custom backing files to match this layout.

### Chord CSV format

Use a CSV without a header, with one row for each chord change. Columns appear in this order:

```text
measure,beat,root,chord_kind,bass
```

For example, these rows place Fmaj7 on the first beat and E7 on the third beat of the first measure:

```csv
0,0,F,major-seventh,F
0,2,E,dominant-seventh,E
```

Measure and beat numbers start at zero. Measure numbers are relative to the start of the melody and exclude the four-measure introduction. Use chord kinds accepted by music21's `ChordSymbol`, such as `major-seventh`, `minor-seventh`, and `dominant-seventh`. Beats without an explicit chord inherit the preceding chord.

Always specify a chord at `0,0`. With the default settings, measures 0–7 are used for generation, and post-processing also reads measure 8 for the ending. To change the final chord, add a row such as `8,0,A,minor-seventh,A`.

## 3. Prepare the trained models

Each model needs a `.weights.h5` file (or a legacy `.h5` file) and a `.benzaitenconfig` file containing its shape settings. The paths below show the legacy files, which remain supported without conversion. If you already have a matching pair, you can skip training and proceed to generation.

| Model | Required files |
| --- | --- |
| C major | `models/current/mymodel_C_major.h5`, `models/current/C_major.benzaitenconfig` |
| A minor | `models/current/mymodel_A_minor.h5`, `models/current/A_minor.benzaitenconfig` |

### Training from MusicXML

Obtain training scores, for example from the [Omnibook MusicXML collection](https://homepages.loria.fr/evincent/omnibook/), and separate major-key and minor-key pieces into these directories:

```text
omnibook/
├── C_major/
│   └── major_key_piece.xml
└── A_minor/
    └── minor_key_piece.xml
```

Only `*.xml` files directly inside each model directory are loaded. Files placed directly in `omnibook/` are not read. The code analyzes each piece's key and transposes it to the specified tonic, but does not sort pieces into major and minor groups. Classify them before training. The first part of each score is expected to contain a single-note melody and chord symbols.

Train C major with the default 50 epochs:

```sh
python -m benzaiten_adlib.learn
```

To train both models, or change the number of epochs:

```sh
python -m benzaiten_adlib.learn --models C_major A_minor --epochs 50
```

Training writes `mymodel_<model>.weights.h5` and `<model>.benzaitenconfig` to `models/current/`. Existing files with those names are overwritten; legacy `mymodel_<model>.h5` files are preserved. The configuration stores sequence length, input dimension, and output dimension. Generation prefers `.weights.h5` when both formats exist. To try only C major, comment out the four active calls using `ModelType.A_MINOR` in `generate_file_set()` in `benzaiten_adlib/generate.py`.

## 4. Generate improvisations

Once the models and input files are ready, run:

```sh
python -m benzaiten_adlib.generate
```

By default, the program generates the following four variants for each of the C major and A minor models. Both models use the same backing track and chord progression.

| Filename suffix | Pitch and rhythm processing |
| --- | --- |
| `type1` | Pitch correction based on chords and transitions between notes |
| `type1_V2SH_16Tri` | Type 1 with shuffle timing and added sixteenth-note triplet notes |
| `type3` | Pitch correction based on a pentatonic scale, with range, leap, and avoid-note adjustments |
| `type3_V2SH` | Type 3 with shuffle timing |

Each variant produces three files. A complete run of all eight default variants produces 24 files.

| Output path | Contents |
| --- | --- |
| `output/midi/<timestamp>_output_<model>_<suffix>.mid` | MIDI with accompaniment |
| `output/solo/<timestamp>_output_<model>_<suffix>_solo.mid` | Melody-only MIDI for submission |
| `output/wav/<timestamp>_<model>_<suffix>_output.wav` | Audio rendered from the MIDI with accompaniment, saved in `output/wav/` |

The solo MIDI retains the initial four-measure delay. The backing track's tempo track is not copied, so standalone playback may use a different tempo from the MIDI with accompaniment. Generation and correction use randomness, so identical inputs do not necessarily produce identical melodies.

## Configuration

Edit `generate_file_set()` in `benzaiten_adlib/generate.py` to choose which variants to generate, and `benzaiten_adlib/config.py` to change the basic music settings. Music settings and generation variants are edited in code; training accepts `--models` and `--epochs`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `TOTAL_MEASURES` | `240` | Number of measures allocated for training data |
| `UNIT_MEASURES` | `4` | Measures per training or generation segment |
| `BEAT_RESO` | `4` | Subdivisions per beat: sixteenth-note resolution |
| `N_BEATS` | `4` | Beats per measure |
| `NOTENUM_FROM` / `NOTENUM_THRU` | `36` / `84` | Model's MIDI note range, with an exclusive upper bound |
| `INTRO_BLANK_MEASURES` | `4` | Measures before the melody starts |
| `MELODY_LENGTH` | `8` | Generated measures before ending adjustments |
| `TICKS_PER_BEAT` | `480` | MIDI ticks per quarter note |
| `MELODY_PROG_CHG` | `73` | Melody program number, starting at zero |

Some processing still contains hardcoded assumptions of four beats per measure and four subdivisions per beat. Changing the meter or resolution requires code changes as well as configuration changes. If you change sequence length or note range, also check compatibility with the trained model's shape. The melody is transposed up 12 semitones when written to MIDI, so the output range is not identical to the model's note range.

## Main source files

| File | Role |
| --- | --- |
| `benzaiten_adlib/learn.py` | Loading MusicXML, training, and saving models |
| `benzaiten_adlib/generate.py` | Loading models, generating melodies, and exporting files |
| `benzaiten_adlib/core.py` | Music data conversion and MIDI/WAV generation |
| `benzaiten_adlib/model.py` / `benzaiten_adlib/model_io.py` | Keras 3 VAE and loading current or legacy model weights |
| `benzaiten_adlib/music_utils.py` | Pitch correction, ending notes, and performance effects such as pitch bends |
| `benzaiten_adlib/submission.py` | Creating solo MIDI files for submission and replacing program changes |
| `benzaiten_adlib/config.py` | Measure, note range, and MIDI settings |
| `benzaiten_adlib/model_types.py` / `benzaiten_adlib/features.py` | Model and correction-feature identifiers |

`experiments/converter.py` optionally copies legacy weights into the new format: run `python -m experiments.converter C_major` (or `A_minor`). It preserves the source `.h5` and refuses to overwrite an existing `.weights.h5`. Conversion is not required for generation.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing `models/current/A_minor.benzaitenconfig` or `models/current/mymodel_A_minor.h5` | Enable A minor training or limit generation to C major. |
| Array shape errors during training | Check that training files exist at the expected paths, such as `omnibook/C_major/*.xml`. |
| Missing directory when saving MIDI | Run `sh scripts/setup_required_folders.sh` and check output directory permissions. |
| MIDI files are created but WAV files are not | Check that the `fluidsynth` command and `soundfonts/FluidR3_GM.sf2` are available. |
| Model weights cannot be loaded | Match the settings and library versions used for training and generation. Generation reconstructs the model and loads weights from `.weights.h5` or a legacy `.h5` file. |

## Original material and license

The original README credits this [Benzaiten document](https://docs.google.com/document/d/1CizJ6b9i2yZ9OIDPrBWUROyJahlZrlqe-naxh4brACQ/edit) as the basis for the implementation.

The repository's code is released under the MIT License; see [LICENSE](LICENSE). Check the respective providers' terms for training scores, backing samples, and SoundFonts.

## Local file organization

See [the directory guide](docs/DIRECTORY_GUIDE.md) for contest archives, experiments, and the move manifest. Original input ZIPs are preserved in `data/input-originals/2023-02/` (samples 1–3) and `data/input-originals/2023-10/` (samples 4–5). These date assignments are inferred from local file evidence. Keep these archives; extract a selected sample into a separate temporary directory before copying the MIDI/CSV into `sample/`. Verify originals from the project root with `shasum -a 256 -c data/input-originals/SHA256SUMS`. Archived files remain local and are excluded from Git.

## Python project layout

```text
benzaiten_adlib/    Application package (learn, generate, core, config, paths, utilities)
scripts/           Setup, output cleanup, WAV trimming, and code ZIP tools
scripts/legacy/    Historical contest input preparation
experiments/       Optional legacy model weight conversion
tests/            Regression tests
pyproject.toml     Package metadata and console commands
requirements.txt   Pinned runtime dependencies
```

Run `python -m benzaiten_adlib.learn` or `python -m benzaiten_adlib.generate` from the project root. Direct execution of individual package files is not supported. Importing the modules does not start training or generation. Run the optional conversion with `python -m experiments.converter C_major`.

For commands available outside the project directory, install the checkout with `python -m pip install -e .` in the Python 3.13 environment described above, then use `benzaiten-learn` or `benzaiten-generate`. Dependencies are read from `requirements.txt`. Data paths default to this checkout's root; set `BENZAITEN_ROOT` to an absolute data directory to override them. A non-editable installation requires this variable to point to the prepared data directory. Models, samples, scores, and SoundFonts are not included in the Python package.

Run all tests in the configured environment with `python -m unittest discover -s tests -v`. To run only the dependency-free structure tests, use `python -m unittest discover -s tests -p test_project_layout.py`. Create a source ZIP with `sh scripts/make_zip_of_code.sh`; this includes the package, scripts, experiments, tests, and documentation. Existing model and data directories retain their locations.
