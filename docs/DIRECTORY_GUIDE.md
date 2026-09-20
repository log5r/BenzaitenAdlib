# ディレクトリ案内

2026年9月20日に、入力原本・学習済みモデル・過去の生成物を分類しました。ファイル名と内容は維持し、移動した150ファイルすべてで移動前後のSHA-256が一致することを確認しました。

| 保存先 | 内容 |
| --- | --- |
| `benzaiten_adlib/` | 学習・生成と共通処理のPythonパッケージ |
| `scripts/` | フォルダ作成、出力整理、コードZIP作成の補助シェル |
| `scripts/legacy/` | 過去の大会で使用した入力調整スクリプト |
| `experiments/` | 旧形式モデルの重みを新形式へコピーする補助コード |
| `tests/` | import時の副作用とデータ参照先の回帰テスト |
| `pyproject.toml` | パッケージ定義とインストール後の起動コマンド |
| `data/input-originals/2023-02/` | `sample1.zip`～`sample3.zip`の入力原本 |
| `data/input-originals/2023-10/` | `sample4.zip`・`sample5.zip`の入力原本 |
| `data/training-originals/` | 学習データの原本`omnibook.zip` |
| `sample/` | 現在使用する伴奏MIDIとコードCSV |
| `omnibook/` | 学習用MusicXMLと既存の補助ファイル |
| `soundfonts/` | WAV生成用SoundFont |
| `models/current/` | 現在のモデル・対応する形状設定・変換済みモデル・チェックポイント一式 |
| `models/legacy/` | 旧`modelbk/`・`models_1000/`・`model/`と旧`config.benzaitenconfig` |
| `archive/contests/2023-02/` | 第1回開催月の出力ZIPと`alpha_mus/`のWAV |
| `archive/contests/2023-10/` | 第2回の予選候補・提出フォルダ、開催月の生成結果、コードZIP |
| `archive/experiments/2023-05/` | 旧`良さそう/`の選別済みMIDI |
| `archive/experiments/2025-05/` | 2025年5月の生成結果 |
| `output/midi/` | 今後生成する伴奏付きMIDI |
| `output/solo/` | 今後生成する旋律のみのMIDI |
| `output/wav/` | 今後生成するWAV |
| `battle/` | 既存の本番入力調整スクリプト用作業場所 |
| `docs/` | この案内と移動記録 |

大会の開催月はユーザー提供情報（第1回：2023年2月、第2回：2023年10月）に基づきます。[大会サイト](https://benzaiten.studio.site/)も参照しましたが、各ファイルの用途の判定にはローカルのファイル名、ZIP内部の日時、既存フォルダ名を使っています。入力ZIPの大会別分類はこの情報からの推定です。大会開催月の生成物も、実際に提出・使用したかは未確認です。元の`第二幕予選候補/提出/`という区分はそのまま残しています。

`models/legacy/`のモデルは大会ごとの対応関係を確認できないため、元のフォルダ名で保持しています。`models_1000`という名前だけでは学習条件を確定できません。仮想環境、IDE設定、キャッシュは既存の場所に残しています。

## 入力ZIP原本の扱い

`sample*.zip`は再入手が困難な原本です。5ファイルともZIP内部のCRC検査に合格し、移動前後のSHA-256も一致しました。原本は展開・再圧縮・削除せずに保管しています。使用時には別の作業フォルダへ展開してから、必要なMIDIとCSVを`sample/`へコピーしてください。現在の入力を変更するときは、その作業用ファイルも先に退避してください。

プロジェクト直下から次のコマンドで原本を検証できます。

```sh
shasum -a 256 -c data/input-originals/SHA256SUMS
```

この整理では独立したバックアップは作成していません。ZIP原本は引き続きGitの対象外なので、Gitへのコミットだけでは保護されません。別ディスクなどへ原本と`SHA256SUMS`を一緒にコピーすると、保管先でも内容を照合できます。

## 実行と移動履歴

コマンドは引き続きプロジェクト直下から実行します。`benzaiten_adlib/learn.py`・`benzaiten_adlib/generate.py`・`experiments/converter.py`は`models/current/`を参照し、生成物は`output/`の各サブフォルダへ保存します。学習は従来どおり同名の現行モデルと設定を上書きします。

`scripts/remove_wav.sh`は現在の`output/midi/`・`output/solo/`・`output/wav/`だけを消去します。`scripts/trim_wavs.sh`も`output/wav/`を対象にしています。過去の生成物は`archive/`に保管されています。`scripts/legacy/battle_honban_batch.sh`は従来どおり`sample/`の作業用入力を置き換えるため、保管用ZIPの展開には使わないでください。

移動元・移動先・バイト数・SHA-256は[移動記録](organization-manifest-2026-09-20.json)に記録しました。元に戻す必要がある場合は、この対応表に従い、元の場所に同名ファイルがないことを確認して移動できます。コードの参照先も併せて戻す必要があります。

アーカイブ、モデル、生成物、原本ZIPはローカル保存です。`.gitignore`は誤って大きなファイルをコミットすることを防ぎますが、バックアップにはなりません。

## Pythonコードの整理

Python実装は`benzaiten_adlib/`、補助シェルは`scripts/`、実験は`experiments/`へ分離しました。学習と生成は`python -m benzaiten_adlib.learn`と`python -m benzaiten_adlib.generate`で実行します。設定は`benzaiten_adlib/config.py`、データの参照先は`benzaiten_adlib/paths.py`に集約しています。インストール手順と`BENZAITEN_ROOT`の指定方法は[README](../README.ja.md#pythonプロジェクトの構成)を参照してください。上記の移動記録はデータ整理時の履歴で、今回のソースコード移動はGitの差分で確認できます。

Python 3.13とKeras 3への更新内容、既存モデルとの互換性、検証範囲は[移行・検証記録](PYTHON_UPGRADE.md)にまとめています。新しい学習結果は`.weights.h5`で保存し、旧`.h5`も生成時に読み込めます。
