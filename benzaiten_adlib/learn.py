import argparse

from . import core as bc
from . import paths


def learn_and_generate_model(x, y, model_idf, *, epochs=50):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    # VAEのモデルを構築するための関数を定義する
    seq_length = x.shape[1]  # 時間軸上の要素数
    input_dim = x.shape[2]  # 入力データにおける各時刻のベクトルの次元数
    output_dim = y.shape[2]  # 出力データにおける各時刻のベクトルの次元数

    # generateフェーズ用に数値を保存
    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config_file = open(paths.MODEL_DIR / ("%s.benzaitenconfig" % model_idf), 'w')
    config_file.write("%s\n%s\n%s" % (seq_length, input_dim, output_dim))
    config_file.close()

    # VAEモデル作成
    main_vae = bc.make_model(seq_length, input_dim, output_dim)
    main_vae.fit(x, y, epochs=epochs)
    main_vae.save_weights(paths.MODEL_DIR / ("mymodel_%s.weights.h5" % model_idf))


def main():
    parser = argparse.ArgumentParser(description="Train BenzaitenAdlib models from MusicXML.")
    parser.add_argument("--models", nargs="+", choices=["C_major", "A_minor"], default=["C_major"])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for model_id in args.models:
        tonic, mode = model_id.split("_")
        x, y = bc.read_mus_xml_files([], [], tonic, mode)
        if x.size == 0:
            parser.error(f"No training sequences found for {model_id} in {paths.MUSIC_DIR}")
        learn_and_generate_model(x, y, model_id, epochs=args.epochs)


if __name__ == "__main__":
    main()
