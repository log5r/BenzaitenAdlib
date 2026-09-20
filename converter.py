import os
import tensorflow as tf
import tensorflow_probability as tfp
import project_paths as paths

# --- DistributionLambda のシグネチャに trainable を追加し、from_config をオーバーライド ---
class DistributionLambdaWrapper(tfp.layers.DistributionLambda):
    def __init__(
        self,
        make_distribution_fn,
        convert_to_tensor_fn=None,
        trainable=True,
        **kwargs
    ):
        # trainable は base Layer.__init__ で扱われないので自前で保持
        super().__init__(make_distribution_fn, convert_to_tensor_fn, **kwargs)
        self._trainable = trainable  # 内部状態に持たせる

    @classmethod
    def from_config(cls, config):
        # config に含まれる trainable をそのまま取り出して渡す
        trainable = config.pop("trainable", True)
        return cls(**config, trainable=trainable)

# モデルファイルのパス
h5_model_path = str(paths.MODEL_DIR / "mymodel_C_major.h5")

# custom_objects にラッパーを登録
custom_objects = {
    "MultivariateNormalTriL": tfp.distributions.MultivariateNormalTriL,
    "DistributionLambda": DistributionLambdaWrapper,
}

# モデルロード（load_model が trainable キーを渡しても OK）
keras_model = tf.keras.models.load_model(
    h5_model_path,
    custom_objects=custom_objects
)

# ここで全体を凍結
keras_model.trainable = False

# 以降、keras_model を用いた処理…
