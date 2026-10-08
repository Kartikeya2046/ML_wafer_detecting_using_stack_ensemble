#!/bin/sh
# Create the project venv on Linux (Python 3.12, NVIDIA driver with CUDA 12 support). Used by run_gpu.sh / run_cpu.sh.
# TF 2.17 + tf_keras: TF_USE_LEGACY_KERAS=1 (set in run_*.sh) makes tf.keras = Keras 2, the API of the TF 2.10 Stage 1 runs.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install "tensorflow[and-cuda]==2.17.1" "tf_keras==2.17.0" "numpy<2" scikit-learn==1.7.2 xgboost==2.1.4 \
  scipy pandas==2.3.3 matplotlib scikit-image jupyter nbformat
# tf_keras 2.17 bug under Python 3.12: after set_random_seed, unseeded layers call randint(1, 1e9) with a float -> TypeError.
F=$(.venv/bin/python -c "import tf_keras, os; print(os.path.join(os.path.dirname(tf_keras.__file__), 'src', 'backend.py'))")
sed -i 's/generator.randint(1, 1e9)/generator.randint(1, int(1e9))/' "$F"
sh run_gpu.sh -c "import tensorflow as tf; print(tf.__version__, tf.keras.__name__, tf.config.list_physical_devices('GPU'))"
