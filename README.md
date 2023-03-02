# BenzaitenAdlib
Benzaiten Adlib TensorFlow version Sample for M1 Mac

based on https://docs.google.com/document/d/1CizJ6b9i2yZ9OIDPrBWUROyJahlZrlqe-naxh4brACQ/edit

## How to setup

1. Set up Python and TensorFlow for the M1Mac environment.
   * e.g. https://zenn.dev/log5/scraps/af1d7f748ed789 
1. Install each module.   
   * Since the following import statements appear in the program, please install the respective modules so that these can be resolved:
   ```python
    import music21
    import numpy as np
    import matplotlib.pyplot as plt
    import mido
    import csv
    import IPython.display as ipd
    import midi2audio
    import glob
    import tensorflow as tf
    import tensorflow_probability as tfp
   ```

1. Download the training musicXML from https://homepages.loria.fr/evincent/omnibook/ and place it in the `omnibook` directory.

1. Download the following sample file from the [here](https://drive.google.com/drive/folders/1jZSMX14B-i98x06QowaNL_9VGXeJZJbd) and place it in the `sample` directory. 
1. Rename sample files as follows:
   - sample1_backing.mid -> sample_backing.mid
   - sample1_chord.csv -> sample_chord.csv

1. Obtain a soundfont and place it in the `soundfont` directory.
   - For example, get `FluidR3_GM.sf2` from https://member.keymusician.com/Member/FluidR3_GM/index.html and place it in the soundfont directory.

1. Run `learn.py` to generate the model.

1. Run `generate.py` to output a midi file in the `output` directory with the melody on the accompaniment.
   - At the same time, a wav file is output to the root directory.



