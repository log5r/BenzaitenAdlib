import benzaitencore as bc


def learn_and_generate_model(x, y, model_idf):
    # VAEのモデルを構築するための関数を定義する
    seq_length = x.shape[1]  # 時間軸上の要素数
    input_dim = x.shape[2]  # 入力データにおける各時刻のベクトルの次元数
    output_dim = y.shape[2]  # 出力データにおける各時刻のベクトルの次元数

    # generateフェーズ用に数値を保存
    config_file = open("%s.benzaitenconfig" % model_idf, 'w')
    config_file.write("%s\n%s\n%s" % (seq_length, input_dim, output_dim))
    config_file.close()

    # VAEモデル作成
    main_vae = bc.make_model(seq_length, input_dim, output_dim)
    main_vae.fit(x, y, epochs=500)
    main_vae.save(bc.BASE_DIR + "/mymodel_%s.h5" % model_idf)


# 学習用MusicXMLを読み込んでモデルを生成
x_all_c_major = []
y_all_c_major = []
x_all_c, y_all_c = bc.read_mus_xml_files(x_all_c_major, y_all_c_major, "C", "major")
learn_and_generate_model(x_all_c, y_all_c, "C_major")

x_all_a_minor = []
y_all_a_minor = []
x_all_am, y_all_am = bc.read_mus_xml_files(x_all_a_minor, y_all_a_minor, "A", "minor")
learn_and_generate_model(x_all_am, y_all_am, "A_minor")
