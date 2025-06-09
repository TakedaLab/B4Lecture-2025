"""確認用."""

import sys

from tensorflow.keras.models import load_model

model = load_model(sys.argv[1])
model.summary()
