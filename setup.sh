#!/usr/bin/env bash
# =============================================================================
# setup.sh
# Run this ONCE with internet access before opening the notebook.
# Does everything: installs dependencies + downloads all 20 Mumbai OSM maps.
# =============================================================================

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1 — System dependencies (Tesseract OCR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v apt-get &>/dev/null; then
    apt-get update -q && apt-get install -y -q tesseract-ocr libgl1
    echo "✅ System packages installed via apt-get"
elif command -v brew &>/dev/null; then
    brew install tesseract
    echo "✅ System packages installed via brew"
else
    echo "⚠️  Could not detect apt-get or brew."
    echo "   Please install Tesseract manually: https://github.com/tesseract-ocr/tesseract"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2 — Python packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install \
    torch \
    torchvision \
    timm \
    opencv-python-headless \
    pillow \
    matplotlib \
    scikit-learn \
    scipy \
    numpy \
    tqdm \
    transformers \
    accelerate \
    requests \
    easyocr \
    pandas \
    pytesseract \
    -q

echo "✅ All Python packages installed."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 3 — Download Mumbai OSM maps (20 locations, zoom 16)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 - <<'PYEOF'
import requests, os, math
from PIL import Image
from io import BytesIO

def download_osm_map(lat, lon, zoom=16, width=2560, height=2560, out_path=None, verify=False):
    """Download a large OSM map image centred at (lat, lon)."""

    def deg2tile(lat, lon, zoom):
        lat_r = math.radians(lat)
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(lat_r) + 1/math.cos(lat_r)) / math.pi) / 2 * n)
        return x, y

    tx, ty = deg2tile(lat, lon, zoom)
    cols, rows = width // 256, height // 256
    canvas = Image.new('RGB', (cols*256, rows*256))

    for dr in range(rows):
        for dc in range(cols):
            url = f"https://tile.openstreetmap.org/{zoom}/{tx+dc}/{ty+dr}.png"
            headers = {'User-Agent': 'MapStitcherTraining/1.0'}
            r = requests.get(url, headers=headers, timeout=10, verify=verify)
            if r.status_code == 200:
                tile = Image.open(BytesIO(r.content))
                canvas.paste(tile, (dc*256, dr*256))
            else:
                print(f"  ⚠️  Failed to fetch tile: {url}")

    os.makedirs('./full_maps', exist_ok=True)
    path = out_path or f'./full_maps/map_{lat}_{lon}.png'
    canvas.save(path)
    print(f'  ✅ Saved: {path}')

locations = [
    (19.0760, 72.8777),   # Andheri
    (19.0596, 72.8295),   # Bandra
    (18.9388, 72.8354),   # Colaba
    (19.1136, 72.8697),   # Goregaon
    (19.1759, 72.9479),   # Thane border
    (19.0330, 72.8550),   # Dadar
    (19.0454, 72.8927),   # Kurla
    (19.1197, 72.9051),   # Powai
    (18.9647, 72.8258),   # Worli
    (19.0728, 72.9010),   # Vikhroli
    (19.0883, 72.8372),   # Jogeshwari
    (19.1543, 72.8567),   # Borivali
    (19.2183, 72.9781),   # Kalyan
    (19.0221, 72.8569),   # Mahim
    (19.0176, 72.9104),   # Chembur
    (18.9956, 72.8371),   # Prabhadevi
    (19.0825, 72.8826),   # Chandivali
    (19.1393, 72.9002),   # Mulund
    (19.0048, 72.8318),   # Matunga
    (19.1025, 72.8526),   # Malad
]

print(f"Downloading {len(locations)} maps ...\n")
for i, (lat, lon) in enumerate(locations, 1):
    print(f"[{i:2d}/{len(locations)}] lat={lat}, lon={lon}")
    download_osm_map(lat, lon, zoom=16, verify=False)
PYEOF

echo ""
echo "🎉 Setup complete! All maps saved to ./full_maps/"
echo "   Next step → open map_stitcher_qa.ipynb (no internet needed)"
