"""Load legacy HDF5 weights or the current Keras weights format."""
from pathlib import Path

from . import paths


def load_trained_model(model_id, model_dir=None):
    from .model import make_model

    directory = Path(model_dir) if model_dir is not None else paths.MODEL_DIR
    config_path = directory / f"{model_id}.benzaitenconfig"
    shape = [int(value) for value in config_path.read_text().split()]
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Expected three positive model dimensions in {config_path}")
    current = directory / f"mymodel_{model_id}.weights.h5"
    legacy = directory / f"mymodel_{model_id}.h5"
    weights = current if current.is_file() else legacy
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {current} or {legacy}")
    model = make_model(*shape, compile_model=False)
    # Reconstruct the architecture; never deserialize the old TFP Lambda layer.
    model.load_weights(weights)
    return model
