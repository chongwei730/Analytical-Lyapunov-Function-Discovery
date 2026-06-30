#!/bin/bash
# Build the 'alfd' conda env on Anvil and compile the DSO Cython extension.
# Idempotent: safe to re-run after a partial failure.
set -eo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
cd "$ROOT"
echo "[build] loading conda module"
module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "/alfd$"; then
  echo "[build] alfd env already exists"
else
  echo "[build] creating alfd (python 3.9)"
  conda create -y --name alfd python=3.9
fi
conda activate alfd

echo "[build] pip tooling (pin setuptools<58 for use_2to3 packages, add wheel)"
pip install --upgrade pip
pip install "setuptools<58.0.0" wheel

echo "[build] torch (cu118 wheel)"
python -c "import torch" 2>/dev/null && echo "  torch present" || \
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo "[build] deap 1.3.0 WITHOUT build isolation (uses pinned setuptools<58)"
python -c "import deap" 2>/dev/null && echo "  deap present" || \
  pip install --no-build-isolation deap==1.3.0

echo "[build] requirements (deap already satisfied)"
pip install -r requirements.txt

echo "[build] install DSO via legacy setup.py develop (avoids pip build-isolation + PEP660)"
cd ./libs/sd3/dso
# Cython/numpy already in env -> fetch_build_eggs is a no-op; develop avoids PEP660.
python setup.py develop
python setup.py build_ext --inplace
cd "$ROOT"

echo "[build] sanity import"
python -c "import torch, dso, numpy, sympy, deap, numba; print('OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[build] DONE"
