#!/usr/bin/env python3
# =============================================================================
# inference.py
# Map Tile Stitcher + Visual QA System
#
# Converts map_stitcher_qa.ipynb to a single runnable Python script.
# Run after setup.sh has been executed (no internet needed).
#
# Usage:
#   python inference.py
#
# Outputs:
#   ./tiles/                      — sliced training tiles
#   ./tiles_meta.json             — tile ground-truth positions
#   ./tile_encoder.pth            — trained model checkpoint
#   ./reconstructed_map.png — stitched map
#   ./submission.csv                 — QA predictions
# =============================================================================


# ── Section 2: Imports ────────────────────────────────────────────────────────

import os, glob, random, math, json, re
from pathlib import Path
from difflib import SequenceMatcher

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

import easyocr
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


# ── Section 3: Config & Tile Slicing ─────────────────────────────────────────

MAPS_DIR   = './full_maps'   # folder with full training map images
TILES_DIR  = './tiles'       # where sliced tiles are saved
TILE_SIZE  = 256             # px per tile (square)
GRID_COLS  = 10              # tiles per row when slicing
GRID_ROWS  = 10              # tiles per column (10×10 = 100 tiles)
OVERLAP    = 0               # overlap px between tiles

os.makedirs(TILES_DIR, exist_ok=True)


def slice_map_to_tiles(img_path: str, out_dir: str,
                       tile_h=TILE_SIZE, tile_w=TILE_SIZE,
                       rows=GRID_ROWS, cols=GRID_COLS,
                       overlap=OVERLAP) -> dict:
    """Slice a full map image into a rows×cols grid of overlapping tiles.
    Returns a metadata dict {tile_filename: (row, col)} for ground-truth."""
    img = Image.open(img_path).convert('RGB')
    target_w = cols * tile_w
    target_h = rows * tile_h
    img = img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.array(img)

    stem = Path(img_path).stem
    meta = {}
    for r in range(rows):
        for c in range(cols):
            y0 = max(r * tile_h - overlap, 0)
            x0 = max(c * tile_w - overlap, 0)
            y1 = min(y0 + tile_h + 2 * overlap, target_h)
            x1 = min(x0 + tile_w + 2 * overlap, target_w)
            tile = arr[y0:y1, x0:x1]
            fname = f'{stem}_r{r:02d}_c{c:02d}.png'
            Image.fromarray(tile).save(os.path.join(out_dir, fname))
            meta[fname] = (r, c)
    print(f'  Sliced {rows * cols} tiles → {out_dir}/')
    return meta


all_meta = {}
map_files = (glob.glob(os.path.join(MAPS_DIR, '*.png')) +
             glob.glob(os.path.join(MAPS_DIR, '*.jpg')) +
             glob.glob(os.path.join(MAPS_DIR, '*.tif')))

if map_files:
    for mf in map_files:
        print(f'Slicing: {mf}')
        m = slice_map_to_tiles(mf, TILES_DIR)
        all_meta[Path(mf).stem] = m
    with open('tiles_meta.json', 'w') as f:
        json.dump(all_meta, f)
    print(f'\nTotal maps processed: {len(map_files)}')
else:
    print(f'⚠️  No map images found in {MAPS_DIR}.')
    print('   Run setup.sh first to download maps.')


# ── Section 4: Model Definitions ─────────────────────────────────────────────

class TileDataset(Dataset):
    """Yields (anchor, positive, row, col) triplets for training."""

    def __init__(self, tile_dir, meta_json='tiles_meta.json',
                 img_size=224, augment=True):
        self.tile_dir = tile_dir
        with open(meta_json) as f:
            self.all_meta = json.load(f)

        self.samples = []
        for stem, meta in self.all_meta.items():
            for fname, (r, c) in meta.items():
                fp = os.path.join(tile_dir, fname)
                if os.path.exists(fp):
                    self.samples.append((fp, r, c))

        aug = ([T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.1, 0.05),
                T.RandomGrayscale(0.05)] if augment else [])
        self.tf = T.Compose(aug + [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, r, c = self.samples[idx]
        img = Image.open(fp).convert('RGB')
        return self.tf(img), torch.tensor([r, c], dtype=torch.float32)


class TileEncoder(nn.Module):
    """EfficientNet-B0 backbone → 256-d L2-normalised embedding."""

    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        feats = self.backbone(x)
        emb = self.proj(feats)
        return F.normalize(emb, dim=-1)


class PositionRegressor(nn.Module):
    """Predicts normalised (row, col) position from embedding."""

    def __init__(self, embed_dim=256, grid_rows=10, grid_cols=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def forward(self, emb):
        return self.net(emb)


print('Model classes defined.')


# ── Section 5: Training ───────────────────────────────────────────────────────

EMBED_DIM  = 256
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 3e-4
CHECKPOINT = 'tile_encoder.pth'


def train_encoder():
    dataset = TileDataset(TILES_DIR, augment=True)
    if len(dataset) == 0:
        print('No tiles found. Run the slicing step first.')
        return None, None

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0, pin_memory=True)

    encoder   = TileEncoder(embed_dim=EMBED_DIM).to(DEVICE)
    regressor = PositionRegressor(EMBED_DIM, GRID_ROWS, GRID_COLS).to(DEVICE)

    params = list(encoder.parameters()) + list(regressor.parameters())
    opt   = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    pos_loss_fn = nn.SmoothL1Loss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); regressor.train()
        epoch_loss = 0
        for imgs, pos in tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False):
            imgs, pos = imgs.to(DEVICE), pos.to(DEVICE)

            emb      = encoder(imgs)
            pred_pos = regressor(emb)

            gt_norm = pos / torch.tensor(
                [GRID_ROWS - 1, GRID_COLS - 1],
                dtype=torch.float32, device=DEVICE)

            loss_pos = pos_loss_fn(torch.sigmoid(pred_pos), gt_norm)

            dist_grid = torch.cdist(pos, pos, p=2)
            dist_emb  = torch.cdist(emb, emb, p=2)
            margin    = 1.0
            adj    = (dist_grid < 2).float()
            far    = (dist_grid > 4).float()
            loss_c = (adj * dist_emb ** 2 +
                      far * F.relu(margin - dist_emb) ** 2).mean()

            loss = loss_pos + 0.5 * loss_c

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg = epoch_loss / len(loader)
        history.append(avg)
        print(f'Epoch {epoch:3d} | loss {avg:.4f}')

    torch.save({'encoder': encoder.state_dict(),
                'regressor': regressor.state_dict()}, CHECKPOINT)
    print(f'\nModel saved → {CHECKPOINT}')

    plt.figure(figsize=(8, 4))
    plt.plot(history, marker='o')
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True); plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

    return encoder, regressor


encoder, regressor = train_encoder()


# ── Section 6: Load Model ─────────────────────────────────────────────────────

def load_model(checkpoint=CHECKPOINT):
    enc = TileEncoder(embed_dim=EMBED_DIM).to(DEVICE)
    reg = PositionRegressor(EMBED_DIM, GRID_ROWS, GRID_COLS).to(DEVICE)
    ckpt = torch.load(checkpoint, map_location=DEVICE)
    enc.load_state_dict(ckpt['encoder'])
    reg.load_state_dict(ckpt['regressor'])
    enc.eval(); reg.eval()
    print(f'Loaded checkpoint: {checkpoint}')
    return enc, reg


# ── Section 7: Stitcher Functions ─────────────────────────────────────────────

INF_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@torch.no_grad()
def embed_tiles(tile_paths, enc, batch_size=32):
    """Return embeddings + raw (row,col) predictions for a list of tile paths."""
    embeddings, pred_positions = [], []
    for i in range(0, len(tile_paths), batch_size):
        batch_paths = tile_paths[i:i + batch_size]
        imgs = torch.stack([
            INF_TF(Image.open(p).convert('RGB')) for p in batch_paths
        ]).to(DEVICE)
        emb = enc(imgs)
        pos = torch.sigmoid(regressor(emb))
        embeddings.append(emb.cpu())
        pred_positions.append(pos.cpu())
    return torch.cat(embeddings), torch.cat(pred_positions)


def assign_grid_positions(pred_pos, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
    """Use Hungarian algorithm to uniquely assign each tile to a grid cell."""
    pred_rc  = pred_pos.numpy() * np.array([grid_rows - 1, grid_cols - 1])
    all_cells = np.array([[r, c]
                           for r in range(grid_rows)
                           for c in range(grid_cols)], dtype=float)
    cost = np.linalg.norm(
        pred_rc[:, None, :] - all_cells[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    assignments = {}
    for ti, ci in zip(row_ind, col_ind):
        assignments[ti] = (int(all_cells[ci][0]), int(all_cells[ci][1]))
    return assignments


def stitch_tiles(tile_paths, enc, reg,
                 grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
                 tile_size=TILE_SIZE):
    """Main stitching function. Returns the reconstructed PIL map."""
    print(f'Embedding {len(tile_paths)} tiles...')
    emb, pred_pos = embed_tiles(tile_paths, enc)

    print('Assigning grid positions (Hungarian matching)...')
    assignments = assign_grid_positions(pred_pos, grid_rows, grid_cols)

    canvas = np.zeros(
        (grid_rows * tile_size, grid_cols * tile_size, 3), dtype=np.uint8)

    for ti, path in enumerate(tile_paths):
        if ti not in assignments:
            continue
        r, c = assignments[ti]
        tile_img = np.array(
            Image.open(path).convert('RGB').resize(
                (tile_size, tile_size), Image.LANCZOS))
        y0, x0 = r * tile_size, c * tile_size
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = tile_img

    result = Image.fromarray(canvas)
    print(f'Map reconstructed: {result.size}')
    return result, assignments


print('Stitcher functions defined.')


# ── Section 8: Feature-Based Stitching ───────────────────────────────────────

def feature_stitch(tile_paths, grid_rows=15, grid_cols=15, tile_size=256):
    n     = len(tile_paths)
    tiles = [cv2.imread(p) for p in tile_paths]
    gray  = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in tiles]
    STRIP = 32

    def edge_sim(i, j, edge):
        if edge == 'right':
            s1 = gray[i][:, -STRIP:].astype(np.float32)
            s2 = gray[j][:, :STRIP].astype(np.float32)
        else:
            s1 = gray[i][-STRIP:, :].astype(np.float32)
            s2 = gray[j][:STRIP, :].astype(np.float32)
        s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-5)
        s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-5)
        return -np.mean((s1 - s2) ** 2)

    print("Computing edge similarities...")
    right_sim  = np.zeros((n, n))
    bottom_sim = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            right_sim[i, j]  = edge_sim(i, j, 'right')
            bottom_sim[i, j] = edge_sim(i, j, 'bottom')

    print("Applying mutual matching filter...")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.argmax(right_sim[i]) != j or np.argmax(right_sim[:, j]) != i:
                right_sim[i, j] *= 0.5
            if np.argmax(bottom_sim[i]) != j or np.argmax(bottom_sim[:, j]) != i:
                bottom_sim[i, j] *= 0.5

    def best_candidate(scores):
        scores = sorted(scores, reverse=True)
        if len(scores) > 1:
            if scores[0][0] - scores[1][0] < 0.05:
                return None
        return scores[0][1]

    print("Assembling grid...")
    grid = [[None] * grid_cols for _ in range(grid_rows)]
    used = set()

    best_score, best_i, best_j = -np.inf, 0, 1
    for i in range(n):
        for j in range(n):
            if i != j and right_sim[i, j] > best_score:
                best_score = right_sim[i, j]
                best_i, best_j = i, j

    grid[0][0] = best_i
    grid[0][1] = best_j
    used.add(best_i); used.add(best_j)
    print(f"Starting pair: {best_i} | {best_j} (score {best_score:.4f})")

    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid[r][c] is None:
                continue
            cur = grid[r][c]

            if c + 1 < grid_cols and grid[r][c + 1] is None:
                scores = [(right_sim[cur, j], j)
                          for j in range(n) if j not in used]
                best = best_candidate(scores)
                if best is not None:
                    grid[r][c + 1] = best; used.add(best)

            if r + 1 < grid_rows and grid[r + 1][c] is None:
                scores = []
                for j in range(n):
                    if j in used:
                        continue
                    score = bottom_sim[cur, j]
                    if c > 0 and grid[r + 1][c - 1] is not None:
                        score += 0.7 * right_sim[grid[r + 1][c - 1], j]
                    scores.append((score, j))
                best = best_candidate(scores)
                if best is not None:
                    grid[r + 1][c] = best; used.add(best)

    remaining = [i for i in range(n) if i not in used]
    print(f"Unplaced tiles: {len(remaining)}")
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid[r][c] is None and remaining:
                grid[r][c] = remaining.pop(0)

    print("Refinement pass...")
    for _ in range(2):
        for r in range(grid_rows):
            for c in range(grid_cols):
                best_score_ref, best_tile = -np.inf, grid[r][c]
                for j in range(n):
                    score = 0
                    if c > 0:             score += right_sim[grid[r][c - 1], j]
                    if c < grid_cols - 1: score += right_sim[j, grid[r][c + 1]]
                    if r > 0:             score += bottom_sim[grid[r - 1][c], j]
                    if r < grid_rows - 1: score += bottom_sim[j, grid[r + 1][c]]
                    if score > best_score_ref:
                        best_score_ref = score; best_tile = j
                grid[r][c] = best_tile

    canvas = np.zeros(
        (grid_rows * tile_size, grid_cols * tile_size, 3), dtype=np.uint8)
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = grid[r][c]
            if idx is not None:
                tile = cv2.resize(tiles[idx], (tile_size, tile_size))
                canvas[r * tile_size:(r + 1) * tile_size,
                       c * tile_size:(c + 1) * tile_size] = cv2.cvtColor(
                    tile, cv2.COLOR_BGR2RGB)

    return Image.fromarray(canvas), grid


# Run feature stitching
tile_paths = sorted(
    glob.glob('./patches/*.png') +
    glob.glob('./patch_*.png') +
    glob.glob('patch_*.png')
)

print(f"Found {len(tile_paths)} tiles")

if tile_paths:
    stitched, grid = feature_stitch(tile_paths, grid_rows=15, grid_cols=15)
    stitched.save('./reconstructed_map.png')
    print("Saved: ./reconstructed_map.png")
else:
    print("⚠️  No patch tiles found. Skipping feature stitching.")


# ── Sections 11–14: OCR + QA ──────────────────────────────────────────────────

img_path = './reconstructed_map.png'

if not os.path.exists(img_path):
    raise FileNotFoundError(
        f"Could not find the map image at: {os.path.abspath(img_path)}\n"
        "Run the stitching step first.")

map_img = cv2.imread(img_path)
if map_img is None:
    raise ValueError(
        "The file exists but OpenCV cannot read it. It may be corrupted.")

h, w = map_img.shape[:2]

# Step 1: OCR
print("\nRunning OCR on map (this takes 1-2 mins)...")
reader    = easyocr.Reader(['en'], gpu=(DEVICE == 'cuda'))
ocr_data  = [(bbox, text.strip(), conf)
             for bbox, text, conf in reader.readtext(map_img)
             if conf > 0.3]
print(f"Detected {len(ocr_data)} text regions")
for bbox, text, conf in ocr_data[:20]:
    print(f"  '{text}' (conf={conf:.2f})")


# Step 2: Build searchable text DataFrame
def get_bbox_centre(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return np.mean(xs), np.mean(ys)


text_locations = []
for bbox, text, conf in ocr_data:
    cx, cy = get_bbox_centre(bbox)
    text_locations.append({
        'text':       text,
        'text_lower': text.lower(),
        'cx': cx, 'cy': cy,
        'conf':  conf,
        'rel_x': cx / w,
        'rel_y': cy / h,
    })

df_ocr = pd.DataFrame(text_locations)
print(df_ocr[['text', 'rel_x', 'rel_y', 'conf']].head(20))


# Step 3: QA functions
def find_text_on_map(query, df_ocr, threshold=0.6):
    """Find all OCR detections that fuzzy-match the query."""
    query_l = query.lower()
    matches = []
    for _, row in df_ocr.iterrows():
        ratio = SequenceMatcher(None, query_l, row['text_lower']).ratio()
        if query_l in row['text_lower'] or row['text_lower'] in query_l:
            ratio = max(ratio, 0.8)
        if ratio >= threshold:
            matches.append({**row.to_dict(), 'match_score': ratio})
    matches.sort(key=lambda x: -x['match_score'])
    return matches


def answer_question(question, options, df_ocr, map_shape):
    """
    Strategy:
    1. Search for each option's text on the map
    2. Apply spatial constraints from question keywords
       (north/south/east/west/near/visible)
    3. Pick option with strongest evidence
    """
    h, w       = map_shape[:2]
    question_l = question.lower()

    spatial = None
    if 'north' in question_l: spatial = 'north'
    if 'south' in question_l: spatial = 'south'
    if 'east'  in question_l: spatial = 'east'
    if 'west'  in question_l: spatial = 'west'

    anchor_pos = None
    q_words    = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', question)
    for phrase in q_words:
        hits = find_text_on_map(phrase, df_ocr, threshold=0.65)
        if hits:
            anchor_pos = (hits[0]['cx'], hits[0]['cy'])
            print(f"  Anchor '{phrase}' found at "
                  f"({anchor_pos[0]:.0f}, {anchor_pos[1]:.0f})")
            break

    scores = {}
    for opt_num, opt_text in options.items():
        hits = find_text_on_map(opt_text, df_ocr, threshold=0.55)
        if not hits:
            scores[opt_num] = 0
            continue

        best  = hits[0]
        score = best['match_score'] * best['conf']

        if spatial == 'north' and best['rel_y'] > 0.4: score *= 0.3
        if spatial == 'south' and best['rel_y'] < 0.6: score *= 0.3
        if spatial == 'east'  and best['rel_x'] < 0.6: score *= 0.3
        if spatial == 'west'  and best['rel_x'] > 0.4: score *= 0.3

        if anchor_pos is not None:
            dist = np.sqrt((best['cx'] - anchor_pos[0]) ** 2 +
                           (best['cy'] - anchor_pos[1]) ** 2)
            proximity_score = 1 - (dist / max(w, h))
            score *= (1 + proximity_score)

        scores[opt_num] = score
        print(f"  Option {opt_num} '{opt_text}': score={score:.3f} "
              f"pos=({best['rel_x']:.2f}, {best['rel_y']:.2f})")

    best_option = max(scores, key=scores.get)
    return best_option, scores


# Step 4: Run on questions CSV
questions_df = pd.read_csv('./test.csv')
map_img      = cv2.imread('./reconstructed_map.png')

answers = []
for _, row in questions_df.iterrows():
    q_id     = row['id']
    question = row['question']
    options  = {
        1: str(row['option_1']),
        2: str(row['option_2']),
        3: str(row['option_3']),
        4: str(row['option_4']),
    }

    print(f"\n{q_id}: {question}")
    answer, scores = answer_question(question, options, df_ocr, map_img.shape)
    print(f"  → Predicted: Option {answer} ({options[answer]})")

    answers.append({
        'id':           q_id,
        'question_ref': q_id,
        'option':       answer,
    })

out_df = pd.DataFrame(answers)
out_df.to_csv('./submission.csv', index=False)
print("\n\nFinal Answers:")
print(out_df)
