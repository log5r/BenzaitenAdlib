import benzaitencore as bc

# 学習用MusicXMLを読み込む
main_x_all = []
main_y_all = []
x_all, y_all = bc.read_mus_xml_files(main_x_all, main_y_all) 

# VAEのモデルを構築するための関数を定義する
seq_length = x_all.shape[1]  # 時間軸上の要素数
input_dim = x_all.shape[2]  # 入力データにおける各時刻のベクトルの次元数
output_dim = y_all.shape[2]  # 出力データにおける各時刻のベクトルの次元数

# generateフェーズ用に数値を保存
config_file = open('config.benzaitenconfig', 'w')
config_file.write("%s\n%s\n%s" % (seq_length, input_dim, output_dim))
config_file.close()

# VAEモデル作成
main_vae = bc.make_model(seq_length, input_dim, output_dim)
main_vae.fit(x_all, y_all, epochs=50)
main_vae.save(bc.BASE_DIR + "/mymodel.h5")
