# 🗺️ Map Tile Stitcher + Visual QA System

A pipeline that downloads OpenStreetMap tiles, trains a CNN encoder to learn tile positions, stitches scrambled tiles back into a full map, and answers spatial questions about the map using OCR + fuzzy text search.

---

## 📁 File Overview

| File | Internet needed | Purpose |
|---|---|---|
| `1_install_dependencies.sh` | ✅ Yes | Installs all Python packages + Tesseract OCR |
| `2_download_maps.py` | ✅ Yes | Downloads 20 Mumbai neighbourhood maps from OSM |
| `map_stitcher_qa.ipynb` | ❌ No | Main notebook — training, stitching, QA |

---

## 🚀 Setup & Usage

### Step 1 — Install dependencies (internet required)

```bash
bash 1_install_dependencies.sh
```

This installs:
- **PyTorch + torchvision** — model training
- **timm** — pretrained CNN backbones
- **opencv-python-headless** — image processing
- **Pillow** — tile stitching
- **easyocr + pytesseract** — OCR for map text
- **pandas, scikit-learn, scipy, matplotlib, numpy, tqdm, transformers, accelerate, requests**
- **Tesseract** system binary (via apt-get or brew)

---

### Step 2 — Download map data (internet required)

```bash
python 2_download_maps.py
```

Downloads 20 Mumbai neighbourhood maps (Andheri, Bandra, Colaba, Goregaon, …) from OpenStreetMap at zoom level 16. Each map is a **2560 × 2560 px** PNG assembled from 100 individual 256×256 tiles.

Output files land in `./full_maps/`:
```
full_maps/
  map_19.076_72.8777.png   # Andheri
  map_19.0596_72.8295.png  # Bandra
  ...
```

> OSM tile usage: this script sends requests with `User-Agent: MapStitcherTraining/1.0` and is intended for research/training use. Please respect the [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/).

---

### Step 3 — Run the notebook (no internet needed)

Open `map_stitcher_qa.ipynb` in Jupyter and run cells top to bottom.

```bash
jupyter notebook map_stitcher_qa.ipynb
```

#### What the notebook does

| Section | Description |
|---|---|
| **1. Install** | (skip — already done) |
| **2. Imports** | Loads all libraries, sets CUDA/CPU device |
| **3. Data Prep** | Slices full maps into 256×256 tiles for training |
| **4. Model** | Defines `TileEncoder` CNN (timm backbone + position head) |
| **5. Training** | Trains on triplet + position-regression loss |
| **6. Load Model** | Loads a saved checkpoint |
| **7. Stitcher** | Given ~100 unsorted tiles, predicts each tile's (row, col) |
| **8. Run Stitching** | Reconstructs the full map from scrambled tiles |
| **9–11. QA** | Runs OCR on the reconstructed map, builds a searchable text index, answers spatial multiple-choice questions |

---

## 🗂️ Expected Directory Structure (after setup)

```
project/
├── 1_install_dependencies.sh
├── 2_download_maps.py
├── map_stitcher_qa.ipynb
├── full_maps/               ← created by 2_download_maps.py
│   ├── map_19.076_72.8777.png
│   └── ...
├── tiles/                   ← created by notebook (Section 3)
├── checkpoints/             ← created by notebook (Section 5)
├── reconstructed_map.png  ← output of stitcher (Section 8)
├── test.csv                 ← your QA questions (provide this)
└── answers.csv              ← QA predictions (output of Section 11)
```

---

## ⚙️ Requirements

- Python 3.9+
- CUDA GPU recommended for training (CPU works but is slow)
- ~3 GB disk space for 20 full-resolution maps
- `test.csv` with columns: `id, question, option_1, option_2, option_3, option_4`
