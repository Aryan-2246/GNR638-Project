#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.bash  –  Map Tile Stitcher + Visual QA
# Run this ONCE before inference.
# Internet IS available when this runs.
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on first error

# ── 1. Clone the project repository ──────────────────────────────────────────
REPO_URL="https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git"
REPO_DIR="map_stitcher_project"

if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ── 2. Create conda environment ───────────────────────────────────────────────
conda create -y -n gnr_project_env python=3.11

# Activate (works inside bash scripts when using 'source')
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnr_project_env

# ── 3. Install Python dependencies ────────────────────────────────────────────
pip install --upgrade pip

pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install \
    opencv-python-headless \
    Pillow \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    pandas \
    easyocr \
    timm \
    transformers \
    accelerate

# ── 4. Pre-download EasyOCR model weights (needs internet) ────────────────────
python - <<'PYEOF'
import easyocr
# This triggers the model download; GPU not needed here
reader = easyocr.Reader(['en'], gpu=False, verbose=True)
print("EasyOCR models downloaded successfully.")
PYEOF

echo ""
echo "✅  setup.bash complete."
echo "    Activate with:  conda activate gnr_project_env"
echo "    Run inference:  python inference.py --test_dir <path>"
