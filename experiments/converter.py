"""Copy legacy model weights into Keras 3's .weights.h5 format."""
import argparse

from benzaiten_adlib import paths
from benzaiten_adlib.model_io import load_trained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['C_major', 'A_minor'])
    args = parser.parse_args()
    target = paths.MODEL_DIR / f'mymodel_{args.model}.weights.h5'
    if target.exists():
        parser.error(f'Output already exists: {target}')
    model = load_trained_model(args.model)
    model.save_weights(target)
    print(target)


if __name__ == '__main__':
    main()
