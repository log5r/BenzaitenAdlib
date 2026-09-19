# BenzaitenAdlib

[English](README.md) | 日本語

BenzaitenAdlibは、コード進行に合わせたアドリブの旋律を生成し、伴奏付きのMIDIとWAVに出力するPythonプログラムです。音楽生成コンテスト「弁財天」向けの実験用コードで、TensorFlow / TensorFlow ProbabilityによるVAE（変分オートエンコーダ）と、音高・リズムを補正するルールを組み合わせています。もともとはM1 Mac向けのサンプルとして作成されました。

学習には旋律とコード記号を含むMusicXML、生成には伴奏MIDIとコード進行CSVを使います。既定では4小節単位で計8小節の旋律を生成し、補正処理で終止の音を加え、冒頭4小節の後から演奏させます。

```text
MusicXML → learn.py → 学習済みモデルと形状設定
                              ↓
伴奏MIDI + コード進行CSV → generate.py → 伴奏付きMIDI / ソロMIDI / WAV
```

## 実行前に確認すること

現在のコードでは、`learn.py`はC majorモデルだけを学習します。一方、`generate.py`はC majorとA minorの両モデルを読み込みます。初めて学習する場合は、後述の手順でA minorの学習も有効にするか、生成対象をC majorだけに変更してください。

学習済みモデル、入力サンプル、SoundFontはGit管理されていません。手元にファイルがあれば再利用できますが、クローンしただけでは生成に必要な一式はそろいません。以下のコマンドは、すべてプロジェクトのルートディレクトリで実行します。

## 1. 実行環境を準備する

以下はmacOSでPython 3.10を使うセットアップ例です。依存バージョンは手元の`requirements.txt`に合わせています。[music21 9.1.0はPython 3.10以上を要求](https://pypi.org/pypi/music21/9.1.0/json)し、[NumPy 1.22.4はPython 3.10向けのmacOS用wheelを提供](https://pypi.org/project/numpy/1.22.4/)しています。このREADMEの改訂時に、新規環境での学習・生成の通し実行は検証していません。

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  music21==9.1.0 \
  midi2audio==0.1.1 \
  mido==1.3.0 \
  matplotlib==3.7.2 \
  tensorflow==2.13.0 \
  tensorflow-probability==0.21.0 \
  numpy==1.22.4
```

WAVへの変換には、Pythonパッケージの`midi2audio`に加えて、[FluidSynth本体が必要](https://github.com/bzamecnik/midi2audio)です。Homebrewを使う場合は次のようにインストールします。

```sh
brew install fluidsynth
```

入力・出力用のディレクトリも作成します。`generate.py`は出力先を自動作成しません。

```sh
mkdir -p sample omnibook/C_major omnibook/A_minor soundfonts output contest_submit
```

## 2. 伴奏・コード進行・音源を用意する

次の3ファイルを配置します。ファイル名とパスはコード内で固定されています。

| 配置先 | 内容 |
| --- | --- |
| `sample/sample_backing.mid` | 生成した旋律を組み込む伴奏MIDI |
| `sample/sample_chord.csv` | 旋律生成と補正に使うコード進行 |
| `soundfonts/FluidR3_GM.sf2` | WAV変換用のSoundFont |

伴奏とコード進行の入手先は、従来のREADMEで案内している[弁財天のサンプル配布フォルダ](https://drive.google.com/drive/folders/1jZSMX14B-i98x06QowaNL_9VGXeJZJbd)です。たとえば`sample1_backing.mid`と`sample1_chord.csv`を使う場合は、それぞれ上の名前に変更します。SoundFontの入手先としては[FluidR3_GMの配布ページ](https://member.keymusician.com/Member/FluidR3_GM/index.html)が案内されています。

### 伴奏MIDIの条件

生成処理は、伴奏MIDIの**2番目のトラック（`tracks[1]`）を旋律トラックに置き換えます**。少なくとも2トラックを持ち、2番目が差し替え用になっているMIDIを用意してください。既定の設定は4/4拍子、四分音符あたり480 ticksで、旋律の開始位置は冒頭から4小節後です。独自の伴奏を使う場合は、この構成に合わせます。

### コード進行CSVの形式

ヘッダーなしで、コードが変わる位置を1行ずつ記述します。各列は次の順序です。

```text
小節番号,拍番号,ルート音,コード種類,ベース音
```

たとえば、次の2行は最初の小節の1拍目をFmaj7、3拍目をE7にします。

```csv
0,0,F,major-seventh,F
0,2,E,dominant-seventh,E
```

小節番号と拍番号は0始まりです。小節番号は旋律部分の先頭を基準とし、冒頭4小節は含めません。コード種類には`major-seventh`、`minor-seventh`、`dominant-seventh`など、music21の`ChordSymbol`が受け付ける名前を使います。コードを指定していない拍には、直前のコードが引き継がれます。

先頭の`0,0`には必ずコードを指定してください。既定では小節番号0〜7を生成に使い、補正処理では終止用に小節番号8も読み込みます。最後のコードを変えたい場合は、たとえば`8,0,A,minor-seventh,A`を追加します。

## 3. 学習済みモデルを用意する

各モデルには、重みを読み込むための`.h5`ファイルと、モデルの形状を記録した`.benzaitenconfig`ファイルが必要です。対応する2ファイルがすでにある場合は、学習を省略して生成に進めます。

| モデル | プロジェクト直下に必要なファイル |
| --- | --- |
| C major | `mymodel_C_major.h5`、`C_major.benzaitenconfig` |
| A minor | `mymodel_A_minor.h5`、`A_minor.benzaitenconfig` |

### MusicXMLから学習する場合

[OmnibookのMusicXML配布ページ](https://homepages.loria.fr/evincent/omnibook/)などから学習用の楽譜を用意し、長調・短調に分けて配置します。

```text
omnibook/
├── C_major/
│   └── 長調の楽曲.xml
└── A_minor/
    └── 短調の楽曲.xml
```

読み込み対象は各ディレクトリ直下の`*.xml`だけです。`omnibook/`直下に置いたファイルは読み込まれません。コードは楽曲の調を解析して指定の主音へ移調しますが、長調・短調の振り分けは行わないため、事前に分類してください。楽譜の最初のパートに単音の旋律とコード記号が入っていることを前提としています。

両モデルを作る場合は、`learn.py`末尾の次の4行のコメントを外します。

```python
x_all_a_minor = []
y_all_a_minor = []
x_all_am, y_all_am = bc.read_mus_xml_files(x_all_a_minor, y_all_a_minor, "A", "minor")
learn_and_generate_model(x_all_am, y_all_am, "A_minor")
```

その後、学習を実行します。

```sh
python learn.py
```

各モデルを50エポック学習し、プロジェクト直下に`.h5`と`.benzaitenconfig`を書き出します。同名ファイルがある場合は上書きします。形状設定には、系列長・入力次元・出力次元の3値を記録します。

C majorだけを試す場合は、`learn.py`をそのまま実行し、`generate.py`の`generate_file_set()`内にある`ModelType.A_MINOR`を指定した4つの有効な呼び出しをコメントアウトしてください。

## 4. アドリブを生成する

モデルと入力ファイルを用意したら、次を実行します。

```sh
python generate.py
```

既定ではC majorとA minorのそれぞれについて、次の4パターンを生成します。両モデルとも同じ伴奏とコード進行を使います。

| ファイル名に付く識別子 | 補正・リズム処理 |
| --- | --- |
| `type1` | コードや音のつながりを考慮した音高補正 |
| `type1_V2SH_16Tri` | type1にシャッフルと16分三連音符の補完を追加 |
| `type3` | ヨナ抜き音階を基準に、音域・跳躍・アボイドノートを補正 |
| `type3_V2SH` | type3にシャッフルを追加 |

各パターンから、次の3ファイルを出力します。既定の8パターンを最後まで実行すると計24ファイルになります。

| 出力先 | 内容 |
| --- | --- |
| `output/<日時>_output_<モデル>_<識別子>.mid` | 伴奏付きMIDI |
| `contest_submit/<日時>_output_<モデル>_<識別子>_solo.mid` | 提出用の旋律トラックだけのMIDI |
| `<日時>_<モデル>_<識別子>_output.wav` | 伴奏付きMIDIを音声化したWAV（プロジェクト直下） |

ソロMIDIにも冒頭4小節分の待ち時間が残ります。元の伴奏のテンポ用トラックはコピーされないため、単体再生時には伴奏付きMIDIとテンポが異なる場合があります。生成と補正には乱数を使うため、同じ入力でも毎回同じ旋律になるとは限りません。

## 設定を変更する

生成する組み合わせは`generate.py`の`generate_file_set()`で、音楽上の基本設定は`benzaiten_config.py`で変更します。コマンドライン引数による指定には対応していません。

| 設定 | 既定値 | 用途 |
| --- | --- | --- |
| `TOTAL_MEASURES` | `240` | 学習用に確保する小節数 |
| `UNIT_MEASURES` | `4` | 1回の学習・生成で扱う小節数 |
| `BEAT_RESO` | `4` | 1拍の分割数（16分音符単位） |
| `N_BEATS` | `4` | 1小節の拍数 |
| `NOTENUM_FROM` / `NOTENUM_THRU` | `36` / `84` | モデルが扱うMIDIノート番号の範囲（上限を含まない） |
| `INTRO_BLANK_MEASURES` | `4` | 旋律開始までの小節数 |
| `MELODY_LENGTH` | `8` | 終止の補正前の生成小節数 |
| `TICKS_PER_BEAT` | `480` | MIDIの四分音符あたりのticks |
| `MELODY_PROG_CHG` | `73` | 旋律のプログラム番号（0始まり） |

現在の処理には4拍子・1拍4分割を前提とする固定値が残っています。拍子や分解能の変更には、設定値だけでなく実装の修正も必要です。系列長や音域を変える場合は、学習済みモデルとの形状の整合も確認してください。また、生成時には旋律を12半音上げてMIDI化するため、出力音域はモデルの音域設定と同一ではありません。

## 主なソースファイル

| ファイル | 役割 |
| --- | --- |
| `learn.py` | MusicXMLの読み込みとモデルの学習・保存 |
| `generate.py` | モデルの読み込み、旋律生成、各形式への出力 |
| `benzaitencore.py` | LSTMを使うVAE、音楽データの変換、MIDI・WAV生成 |
| `music_utils.py` | 音高補正、終止の追加、ピッチベンドなどの演奏処理 |
| `benzaiten_submit_util.py` | 提出用ソロMIDIの作成と音色の差し替え |
| `benzaiten_config.py` | 小節数・音域・MIDI関連の設定 |
| `common_model_type.py` / `common_features.py` | モデル名と補正機能の識別子 |

`converter.py`はモデル読み込みを試す補助コードで、通常の学習・生成手順では使いません。

## エラーが出たとき

| 症状 | 確認する点 |
| --- | --- |
| `A_minor.benzaitenconfig`や`mymodel_A_minor.h5`が見つからない | A minorの学習を有効にするか、生成対象をC majorだけに変更します。 |
| 学習時に配列の形状に関するエラーが出る | `omnibook/C_major/*.xml`など、対象の場所に学習用ファイルがあるか確認します。 |
| MIDI保存時にディレクトリが見つからない | `output/`と`contest_submit/`を作成し、プロジェクト直下から実行します。 |
| MIDIはできるがWAVができない | `fluidsynth`コマンドと`soundfonts/FluidR3_GM.sf2`の有無を確認します。 |
| モデルの重みを読み込めない | 学習時と生成時の設定・ライブラリのバージョンを合わせます。生成処理はモデルを再構築して`.h5`から重みを読み込む方式です。 |

## 元資料とライセンス

実装の元資料は、従来のREADMEで参照していた[弁財天の資料](https://docs.google.com/document/d/1CizJ6b9i2yZ9OIDPrBWUROyJahlZrlqe-naxh4brACQ/edit)です。

本リポジトリのコードはMITライセンスです。詳細は[LICENSE](LICENSE)を参照してください。学習用楽譜、伴奏サンプル、SoundFontの利用条件は、それぞれの配布元で確認してください。
