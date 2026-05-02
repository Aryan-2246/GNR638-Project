# 🗺️ Map Tile Stitcher + Visual QA System

An end-to-end pipeline that downloads OpenStreetMap tiles for Mumbai neighbourhoods, trains a CNN encoder to learn tile positions, stitches scrambled tiles back into a full map, and answers spatial multiple-choice questions about the reconstructed map using OCR and fuzzy text search.

---

## 📁 File Overview

| File | Internet needed | Purpose |
|---|---|---|
| `setup.sh` | ✅ Yes | Installs all dependencies + downloads all 20 Mumbai OSM maps |
| `1_install_dependencies.sh` | ✅ Yes | Installs dependencies only (no map download) |
| `2_download_maps.py` | ✅ Yes | Downloads maps only (no install) |
| `inference.py` | ❌ No | Full pipeline — slicing, training, stitching, OCR, QA |
| `map_stitcher_qa.ipynb` | ❌ No | Same pipeline as an interactive Jupyter notebook |

Run `setup.sh` once with internet, then `inference.py` offline. The notebook is an interactive alternative to `inference.py`.

---

## 🚀 Quickstart

### Step 1 — Run setup (internet required)

```bash
bash setup.sh
```

This does everything in one go:
- Installs Tesseract OCR (via `apt-get` or `brew`)
- Installs all Python packages
- Downloads 20 Mumbai neighbourhood maps from OpenStreetMap into `./full_maps/`

If you want to run these steps separately, use `1_install_dependencies.sh` and `2_download_maps.py` individually.

### Step 2 — Run inference (no internet needed)

```bash
python inference.py
```

Or open the notebook interactively:

```bash
jupyter notebook map_stitcher_qa.ipynb
```

---

## 🗂️ Pipeline Stages

### 1. Data Preparation
Slices each full 2560×2560 map into a 10×10 grid of 256×256 tiles. Saves tiles to `./tiles/` and writes ground-truth positions to `tiles_meta.json`.

### 2. Model — TileEncoder + PositionRegressor
- **TileEncoder**: EfficientNet-B0 backbone (via `timm`) projecting to a 256-d L2-normalised embedding
- **PositionRegressor**: Small MLP that predicts normalised (row, col) from the embedding
- Trained jointly with a position regression loss (SmoothL1) and a contrastive loss that pulls spatially adjacent tiles together in embedding space

### 3. Training
Runs for 50 epochs with AdamW + cosine LR schedule. Saves checkpoint to `tile_encoder.pth` and plots training loss to `training_loss.png`.

### 4. Map Stitching
Given ~100 shuffled tile images, the stitcher embeds all tiles, uses the Hungarian algorithm to uniquely assign each tile to a grid cell, and renders the reconstructed map. A secondary feature-based stitcher uses edge pixel similarity + mutual matching + a refinement pass as an alternative approach.

### 5. OCR + Visual QA
Runs EasyOCR on the reconstructed map to detect all text regions. Builds a searchable DataFrame of text, position, and confidence. For each question in `test.csv`, applies fuzzy string matching per answer option, applies spatial constraints (north/south/east/west keywords), uses proximity to an anchor location if one can be identified, and picks the highest-scoring option.

---

## 🗂️ Directory Structure

```
project/
├── setup.sh                      ← run this first (internet)
├── 1_install_dependencies.sh     ← install only (internet)
├── 2_download_maps.py            ← download only (internet)
├── inference.py                  ← full offline pipeline
├── map_stitcher_qa.ipynb         ← interactive notebook
│
├── full_maps/                    ← created by setup.sh / 2_download_maps.py
│   ├── map_19.076_72.8777.png    # Andheri
│   ├── map_19.0596_72.8295.png   # Bandra
│   └── ...                       # 18 more neighbourhoods
│
├── tiles/                        ← created by inference.py (stage 1)
│   ├── map_19.076_72.8777_r00_c00.png
│   └── ...
│
├── tiles_meta.json               ← ground-truth (row, col) per tile
├── tile_encoder.pth              ← trained model checkpoint
├── training_loss.png             ← loss curve
│
├── patches/                      ← your scrambled input tiles (you provide)
│   ├── patch_001.png
│   └── ...
│
├── reconstructed_map_fixed.png   ← stitched output map
├── test.csv                      ← your QA questions (you provide)
└── answers.csv                   ← predicted answers (output)
```

---

## 📋 Input Format

**`test.csv`** — questions file you must provide:

| id | question | option_1 | option_2 | option_3 | option_4 |
|---|---|---|---|---|---|
| q1 | What road runs north of Dadar station? | Tilak Road | LBS Marg | Senapati Bapat Marg | Gokhale Road |

**`./patches/`** — folder of shuffled tile PNGs to reconstruct (15×15 grid = 225 tiles expected by the feature stitcher).

---

## ⚙️ Requirements

- Python 3.9+
- CUDA GPU recommended for training and OCR (CPU works but is slow)
- ~3 GB disk space for 20 full-resolution maps
- Tesseract system binary (installed by `setup.sh`)

### Python packages installed by `setup.sh`

```
torch  torchvision  timm  opencv-python-headless  pillow
matplotlib  scikit-learn  scipy  numpy  tqdm
transformers  accelerate  requests  easyocr  pandas  pytesseract
```

---

## 🗺️ Mumbai Neighbourhoods Covered

Andheri · Bandra · Colaba · Goregaon · Thane border · Dadar · Kurla · Powai · Worli · Vikhroli · Jogeshwari · Borivali · Kalyan · Mahim · Chembur · Prabhadevi · Chandivali · Mulund · Matunga · Malad

All maps are zoom level 16, 2560×2560 px, fetched from OpenStreetMap.

> Please respect the [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/). Requests are sent with `User-Agent: MapStitcherTraining/1.0`.
