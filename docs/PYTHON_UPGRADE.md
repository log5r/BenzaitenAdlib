# Python 3.13・Keras 3への移行記録

2026年9月20日に、Apple Silicon上のPython 3.13.13で依存関係と実装を更新・検証しました。既存の仮想環境を残し、`.venv-py313/`に新環境を作成しています。利用時はプロジェクト直下で`source .venv-py313/bin/activate`を実行してください。

## 対応バージョン

| 項目 | 更新前 | 更新後 |
| --- | --- | --- |
| Python | READMEでは3.10を使用 | 3.13推奨、依存関係の指定範囲は3.12以上3.14未満 |
| TensorFlow | 2.13.0 | 2.21.0 |
| Keras | TensorFlowに付属する旧API | 3.15.1 |
| TensorFlow Probability | 0.21.0 | 依存を削除 |
| NumPy | 1.22.4 | 2.5.3 |
| music21 | 9.1.0 | 10.5.0 |
| Matplotlib | 3.7.2 | 3.11.2 |
| mido | 1.3.0 | 1.3.3 |
| midi2audio | 0.1.1 | 0.1.1（継続） |

Pythonの最新安定系列は3.14ですが、[TensorFlow 2.21の公式ビルドは3.13まで](https://www.tensorflow.org/install/source)です。そのため、`pyproject.toml`でもPython 3.14を対象外としています。Python 3.12は依存ライブラリの対応範囲に含まれますが、今回の実行検証は3.13で行いました。

macOSではApple Siliconを対象とします。TensorFlow 2.21のwheelはmacOS 12以降向けです。NumPy 2.5.3にはmacOS 11以降向けと14以降向けのwheelがあり、pipが環境に合わせて選択します。Intel Mac、CUDA、Metalでの実行は検証していません。検証時はCPUとFluidSynth 2.3.5を使用しました。

## VAEとモデルファイルの互換性

[TensorFlow Probability 0.25はKeras 3と非互換](https://github.com/tensorflow/probability/releases/tag/v0.25.0)のため、確率分布層を`benzaiten_adlib/model.py`の`GaussianSampling`へ置き換えました。潜在次元32、LSTMの1024ユニット、完全共分散の正規分布、三角行列へのパラメーターの配置、対角要素のsoftplusと微小値、標準正規分布に対するKL項の重み0.001を維持しています。KL項は従来と同様にサンプルによる推定を使い、バッチ平均として加算します。

旧ファイルの重みを対応する層へ読み込めるよう、エンコーダーの層をモデル直下に置き、デコーダーをSequentialとしてまとめる構造も維持しました。古いTFP層をデシリアライズせず、構造を再構築して重みだけを読み込みます。乱数や演算実装が変わるため、以前と同一の旋律や学習過程を再現することは保証しません。

生成時には、対応する`.benzaitenconfig`と次の順序で重みを読み込みます。

1. `mymodel_<モデル名>.weights.h5`があれば使用する。
2. なければ、旧形式の`mymodel_<モデル名>.h5`を使用する。

新しい学習結果は`.weights.h5`へ保存します。旧`.h5`は上書きしませんが、同名の`.weights.h5`と`.benzaitenconfig`は上書きします。新しい重みと古い形状設定を混ぜないでください。今回追加した読み込み処理は推論用で、旧オプティマイザーの状態からの学習再開には対応していません。

変換は任意です。プロジェクト直下から次を実行すると旧重みを新形式へコピーできます。

```sh
python -m experiments.converter C_major
python -m experiments.converter A_minor
```

変換先がすでにある場合は終了し、上書きしません。今回の変換検証では一時ディレクトリを使用し、既存のモデルと設定を保持しました。

music21については、音符のオフセット・長さ・コード構成音の取得を公開APIへ変更しました。

## 学習とテスト

既定ではC majorを50エポック学習します。両モデルを指定する場合や短い動作確認では、次の引数を使用できます。

```sh
python -m benzaiten_adlib.learn --models C_major A_minor --epochs 50
python -m benzaiten_adlib.learn --models C_major --epochs 1
python -m unittest discover -s tests -v
python -m pip check
```

`--epochs 1`も実際の学習であり、上記の保存先へ書き込みます。手元のモデルを保護して検証する場合は、`BENZAITEN_ROOT`で別のデータディレクトリを指定し、その配下の`omnibook/C_major/`に検証用XMLを配置してください。

今回、次の項目を確認しました。

- 回帰テスト7件：importとパス、補助シェルとZIP、共分散行列とKL項、勾配と学習、重みとKeras形式の保存・再読み込み、新旧形式の選択、MusicXMLの読み込み。
- `pip check`で依存関係の不整合がないこと。
- 手元のC major・A minorの旧モデルを読み込み、既定の8パターンで伴奏付きMIDI 8個、ソロMIDI 8個、WAV 8個を生成したこと。MIDIのノートイベントとWAVのフレーム数も確認。
- 既存モデルと設定のSHA-256が生成の前後で一致したこと。
- リポジトリの`X_SCALE_CODE.xml`から4小節を取り出し、別ディレクトリで1エポック学習し、保存した`.weights.h5`を再読み込みして推論できたこと。
- 両モデルを新形式へ変換し、すべての重み配列が変換前と一致したこと。変換先の上書きを拒否し、新形式からも両モデルでMIDI・WAVを生成できたこと。

50エポックの本学習、音楽的品質の比較、旧環境と同じ乱数列による出力比較は行っていません。旧環境の`.venv/`・`venv/`と、保存済みのモデル・入力・生成物は残しています。
