# BenzaitenAdlib

[English](README.md) | 日本語

BenzaitenAdlibは、コード進行に合わせたアドリブの旋律を生成し、伴奏付きのMIDIとWAVに出力するPythonプログラムです。音楽生成コンテスト「弁財天」向けの実験用コードで、TensorFlow / Keras 3によるVAE（変分オートエンコーダ）と、音高・リズムを補正するルールを組み合わせています。もともとはM1 Mac向けのサンプルとして作成されました。

学習には旋律とコード記号を含むMusicXML、生成には伴奏MIDIとコード進行CSVを使います。既定では4小節単位で計8小節の旋律を生成し、補正処理で終止の音を加え、冒頭4小節の後から演奏させます。

```text
MusicXML → benzaiten_adlib/learn.py → 学習済みモデルと形状設定
                              ↓
伴奏MIDI + コード進行CSV → benzaiten_adlib/generate.py → 伴奏付きMIDI / ソロMIDI / WAV
```

## 実行前に確認すること

現在のコードでは、`benzaiten_adlib/learn.py`はC majorモデルだけを学習します。一方、`benzaiten_adlib/generate.py`はC majorとA minorの両モデルを読み込みます。初めて学習する場合は、後述の手順でA minorの学習も有効にするか、生成対象をC majorだけに変更してください。

学習済みモデル、入力サンプル、SoundFontはGit管理されていません。手元にファイルがあれば再利用できますが、クローンしただけでは生成に必要な一式はそろいません。以下のコマンドは、すべてプロジェクトのルートディレクトリで実行します。

## 1. 実行環境を準備する

**Python 3.13**を使用します。依存関係の指定上はPython 3.12にも対応します。Python 3.14には対応していません。[TensorFlow 2.21の公式ビルドはPython 3.13まで](https://www.tensorflow.org/install/source)です。macOSではTensorFlowがApple SiliconとmacOS 12以降を要求します。Intel Macは今回の更新対象に含めていません。

既存の環境を残して、新しい仮想環境を作成します。

```sh
python3.13 -m venv .venv-py313
source .venv-py313/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

`requirements.txt`ではTensorFlow 2.21.0、Keras 3.15.1、NumPy 2.5.3、music21 10.5.0、Matplotlib 3.11.2、mido 1.3.3、midi2audio 0.1.1を固定しています。TensorFlow Probabilityと旧`tf-keras`は不要です。検証環境とモデル互換性は[移行・検証記録](docs/PYTHON_UPGRADE.md)を参照してください。

WAVへの変換には、Pythonパッケージの`midi2audio`に加えて、[FluidSynth本体が必要](https://github.com/bzamecnik/midi2audio)です。Homebrewを使う場合は次のようにインストールします。

```sh
brew install fluidsynth
```

入力・出力用のディレクトリも作成します。`benzaiten_adlib/generate.py`も出力先を自動作成します。

```sh
sh scripts/setup_required_folders.sh
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

各モデルには、重みを読み込むための`.weights.h5`（または旧形式の`.h5`）ファイルと、モデルの形状を記録した`.benzaitenconfig`ファイルが必要です。対応する2ファイルがすでにある場合は、学習を省略して生成に進めます。

下表は変換なしで読み込める旧形式のファイルです。新しく学習した場合は`.h5`の代わりに`.weights.h5`を使用します。

| モデル | 必要なファイル |
| --- | --- |
| C major | `models/current/mymodel_C_major.h5`、`models/current/C_major.benzaitenconfig` |
| A minor | `models/current/mymodel_A_minor.h5`、`models/current/A_minor.benzaitenconfig` |

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

既定ではC majorを50エポック学習します。

```sh
python -m benzaiten_adlib.learn
```

両モデルの学習やエポック数の変更は、引数で指定できます。

```sh
python -m benzaiten_adlib.learn --models C_major A_minor --epochs 50
```

学習結果は`models/current/`の`mymodel_<モデル名>.weights.h5`と`<モデル名>.benzaitenconfig`に保存します。同名のファイルは上書きしますが、旧形式の`mymodel_<モデル名>.h5`は残します。形状設定には系列長・入力次元・出力次元の3値を記録します。両形式が存在する場合、生成時には`.weights.h5`を優先します。C majorだけを試す場合は、`benzaiten_adlib/generate.py`の`generate_file_set()`内にある`ModelType.A_MINOR`を指定した4つの有効な呼び出しをコメントアウトしてください。

## 4. アドリブを生成する

モデルと入力ファイルを用意したら、次を実行します。

```sh
python -m benzaiten_adlib.generate
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
| `output/midi/<日時>_output_<モデル>_<識別子>.mid` | 伴奏付きMIDI |
| `output/solo/<日時>_output_<モデル>_<識別子>_solo.mid` | 提出用の旋律トラックだけのMIDI |
| `output/wav/<日時>_<モデル>_<識別子>_output.wav` | 伴奏付きMIDIを音声化したWAV（`output/wav/`） |

ソロMIDIにも冒頭4小節分の待ち時間が残ります。元の伴奏のテンポ用トラックはコピーされないため、単体再生時には伴奏付きMIDIとテンポが異なる場合があります。生成と補正には乱数を使うため、同じ入力でも毎回同じ旋律になるとは限りません。

## 設定を変更する

生成する組み合わせは`benzaiten_adlib/generate.py`の`generate_file_set()`で、音楽上の基本設定は`benzaiten_adlib/config.py`で変更します。音楽上の設定と生成パターンはコードで変更します。学習では`--models`と`--epochs`を使用できます。

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
| `benzaiten_adlib/learn.py` | MusicXMLの読み込みとモデルの学習・保存 |
| `benzaiten_adlib/generate.py` | モデルの読み込み、旋律生成、各形式への出力 |
| `benzaiten_adlib/core.py` | 音楽データの変換とMIDI・WAV生成 |
| `benzaiten_adlib/model.py` / `benzaiten_adlib/model_io.py` | Keras 3のVAEと新旧形式のモデル重みの読み込み |
| `benzaiten_adlib/music_utils.py` | 音高補正、終止の追加、ピッチベンドなどの演奏処理 |
| `benzaiten_adlib/submission.py` | 提出用ソロMIDIの作成と音色の差し替え |
| `benzaiten_adlib/config.py` | 小節数・音域・MIDI関連の設定 |
| `benzaiten_adlib/model_types.py` / `benzaiten_adlib/features.py` | モデル名と補正機能の識別子 |

`experiments/converter.py`は旧形式の重みを新形式へコピーする補助コードです。`python -m experiments.converter C_major`（または`A_minor`）で実行します。元の`.h5`は残し、同名の`.weights.h5`がある場合は上書きせず終了します。生成するだけなら変換は不要です。

## エラーが出たとき

| 症状 | 確認する点 |
| --- | --- |
| `models/current/A_minor.benzaitenconfig`や`models/current/mymodel_A_minor.h5`が見つからない | A minorの学習を有効にするか、生成対象をC majorだけに変更します。 |
| 学習時に配列の形状に関するエラーが出る | `omnibook/C_major/*.xml`など、対象の場所に学習用ファイルがあるか確認します。 |
| MIDI保存時にディレクトリが見つからない | `sh scripts/setup_required_folders.sh`を実行し、出力先の書き込み権限を確認します。 |
| MIDIはできるがWAVができない | `fluidsynth`コマンドと`soundfonts/FluidR3_GM.sf2`の有無を確認します。 |
| モデルの重みを読み込めない | 学習時と生成時の設定・ライブラリのバージョンを合わせます。生成処理はモデルを再構築して`.weights.h5`または旧形式の`.h5`から重みを読み込む方式です。 |

## 元資料とライセンス

実装の元資料は、従来のREADMEで参照していた[弁財天の資料](https://docs.google.com/document/d/1CizJ6b9i2yZ9OIDPrBWUROyJahlZrlqe-naxh4brACQ/edit)です。

本リポジトリのコードはMITライセンスです。詳細は[LICENSE](LICENSE)を参照してください。学習用楽譜、伴奏サンプル、SoundFontの利用条件は、それぞれの配布元で確認してください。

## ローカルファイルの整理

大会別の保存場所と移動履歴は[ディレクトリ案内](docs/DIRECTORY_GUIDE.md)に記載しています。入力ZIP原本は`data/input-originals/`に保管しています。再入手が困難なため、原本を残したまま別の作業フォルダへ展開し、使うMIDIとCSVを`sample/`へコピーしてください。

## Pythonプロジェクトの構成

```text
benzaiten_adlib/    学習・生成と共通処理のPythonパッケージ
scripts/           フォルダ作成・出力整理・WAV加工・コードZIP作成
scripts/legacy/    過去の大会用の入力調整スクリプト
experiments/       旧形式モデルの重み変換
tests/            回帰テスト
pyproject.toml     パッケージ定義と起動コマンド
requirements.txt   実行時の依存ライブラリと固定バージョン
```

プロジェクト直下から`python -m benzaiten_adlib.learn`または`python -m benzaiten_adlib.generate`で起動します。パッケージ内のファイルを直接実行する形式には対応していません。モジュールをimportしただけでは学習・生成は始まりません。任意のモデル変換は`python -m experiments.converter C_major`で実行します。

別のディレクトリから起動する場合は、前述のPython 3.13環境で`python -m pip install -e .`を実行すると、`benzaiten-learn`と`benzaiten-generate`を使用できます。依存ライブラリは`requirements.txt`から読み込みます。データの参照先は既定でこのチェックアウトのルートです。変更する場合は環境変数`BENZAITEN_ROOT`にデータ用ディレクトリの絶対パスを指定します。通常のインストール（`-e`なし）では、この環境変数の指定が必要です。モデル、サンプル、学習用楽譜、SoundFontはPythonパッケージに含めません。

環境構築後、全テストを`python -m unittest discover -s tests -v`で実行できます。外部ライブラリ不要の構成テストだけを実行する場合は`python -m unittest discover -s tests -p test_project_layout.py`を使用します。`sh scripts/make_zip_of_code.sh`はパッケージ、補助スクリプト、実験、テスト、ドキュメントを含むコードZIPを作成します。既存のモデルやデータの配置は維持しています。
