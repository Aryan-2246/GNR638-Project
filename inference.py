"""
inference.py  –  Map Tile Stitcher + Visual QA
Usage:  python inference.py --test_dir <absolute_path_to_test_dir>
Output: submission.csv  (in current working directory)
"""

import os, sys, glob, re, argparse
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, required=True,
                    help='Absolute path to test directory')
args = parser.parse_args()

TEST_DIR = Path(args.test_dir)
print(f"[INFO] test_dir = {TEST_DIR}")

# ── Locate test.csv ────────────────────────────────────────────────────────────
test_csv_path = TEST_DIR / 'test.csv'
if not test_csv_path.exists():
    # try one level up
    test_csv_path = TEST_DIR.parent / 'test.csv'
if not test_csv_path.exists():
    raise FileNotFoundError(f"test.csv not found in {TEST_DIR}")

questions_df = pd.read_csv(test_csv_path)
print(f"[INFO] Loaded {len(questions_df)} questions from {test_csv_path}")

# ── Locate tile images ─────────────────────────────────────────────────────────
# Images can be directly inside test_dir OR inside a sub-folder (e.g. images/)
def find_image_dir(base: Path):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    for folder in [base] + list(base.iterdir() if base.is_dir() else []):
        if not Path(folder).is_dir():
            continue
        imgs = []
        for e in exts:
            imgs += glob.glob(str(Path(folder) / e))
        if imgs:
            return Path(folder), imgs
    return base, []

img_dir, tile_paths = find_image_dir(TEST_DIR)
print(f"[INFO] Found {len(tile_paths)} tile images in {img_dir}")

# ── Determine grid size ────────────────────────────────────────────────────────
# We expect N² tiles (10×10 = 100, 15×15 = 225, etc.)
# Fall back gracefully if count is odd
n_tiles = len(tile_paths)
if n_tiles == 0:
    raise RuntimeError("No image tiles found in test_dir.")

import math
grid_side = int(round(math.sqrt(n_tiles)))
if grid_side * grid_side != n_tiles:
    # pick nearest perfect square that covers all tiles
    grid_side = math.ceil(math.sqrt(n_tiles))

GRID_ROWS = grid_side
GRID_COLS = grid_side
TILE_SIZE  = 256
print(f"[INFO] Grid size: {GRID_ROWS}×{GRID_COLS}  (tiles={n_tiles})")

# ── Edge-similarity stitching (pure CV, no deep model needed) ─────────────────
def stitch_tiles_cv(tile_paths, grid_rows, grid_cols, tile_size=256):
    n = len(tile_paths)
    tiles  = [cv2.imread(p) for p in tile_paths]
    # resize all tiles to the same size
    tiles  = [cv2.resize(t, (tile_size, tile_size)) if t is not None else
              np.zeros((tile_size, tile_size, 3), np.uint8) for t in tiles]
    gray   = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in tiles]

    STRIP = 32

    def edge_sim(i, j, edge):
        if edge == 'right':
            s1 = gray[i][:, -STRIP:].astype(np.float32)
            s2 = gray[j][:,  :STRIP].astype(np.float32)
        else:
            s1 = gray[i][-STRIP:, :].astype(np.float32)
            s2 = gray[j][ :STRIP, :].astype(np.float32)
        s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-5)
        s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-5)
        return -np.mean((s1 - s2) ** 2)

    print("[STITCH] Computing edge similarities …")
    right_sim  = np.zeros((n, n))
    bottom_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            right_sim[i, j]  = edge_sim(i, j, 'right')
            bottom_sim[i, j] = edge_sim(i, j, 'bottom')

    # Mutual-match filter
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.argmax(right_sim[i])  != j or np.argmax(right_sim[:, j])  != i:
                right_sim[i, j]  *= 0.5
            if np.argmax(bottom_sim[i]) != j or np.argmax(bottom_sim[:, j]) != i:
                bottom_sim[i, j] *= 0.5

    def best_candidate(scores):
        scores = sorted(scores, reverse=True)
        if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.05:
            return None
        return scores[0][1] if scores else None

    print("[STITCH] Assembling grid …")
    grid = [[None]*grid_cols for _ in range(grid_rows)]
    used = set()

    # Find best starting pair
    best_score, best_i, best_j = -np.inf, 0, 1
    for i in range(n):
        for j in range(n):
            if i != j and right_sim[i, j] > best_score:
                best_score, best_i, best_j = right_sim[i, j], i, j

    grid[0][0] = best_i; grid[0][1] = best_j
    used.add(best_i); used.add(best_j)

    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid[r][c] is None:
                continue
            cur = grid[r][c]
            if c + 1 < grid_cols and grid[r][c+1] is None:
                scores = [(right_sim[cur, j], j) for j in range(n) if j not in used]
                best = best_candidate(scores)
                if best is not None:
                    grid[r][c+1] = best; used.add(best)
            if r + 1 < grid_rows and grid[r+1][c] is None:
                scores = []
                for j in range(n):
                    if j in used:
                        continue
                    score = bottom_sim[cur, j]
                    if c > 0 and grid[r+1][c-1] is not None:
                        score += 0.7 * right_sim[grid[r+1][c-1], j]
                    scores.append((score, j))
                best = best_candidate(scores)
                if best is not None:
                    grid[r+1][c] = best; used.add(best)

    # Fill gaps
    remaining = [i for i in range(n) if i not in used]
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid[r][c] is None and remaining:
                grid[r][c] = remaining.pop(0)

    # Refinement pass (2 iterations)
    for _ in range(2):
        for r in range(grid_rows):
            for c in range(grid_cols):
                best_s, best_t = -np.inf, grid[r][c]
                for j in range(n):
                    score = 0
                    if c > 0:           score += right_sim[grid[r][c-1], j]
                    if c < grid_cols-1: score += right_sim[j, grid[r][c+1]]
                    if r > 0:           score += bottom_sim[grid[r-1][c], j]
                    if r < grid_rows-1: score += bottom_sim[j, grid[r+1][c]]
                    if score > best_s:  best_s, best_t = score, j
                grid[r][c] = best_t

    # Render
    canvas = np.zeros((grid_rows*tile_size, grid_cols*tile_size, 3), np.uint8)
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = grid[r][c]
            if idx is not None:
                tile = cv2.resize(tiles[idx], (tile_size, tile_size))
                canvas[r*tile_size:(r+1)*tile_size,
                       c*tile_size:(c+1)*tile_size] = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    return Image.fromarray(canvas), grid


tile_paths_sorted = sorted(tile_paths)
stitched_img, grid = stitch_tiles_cv(tile_paths_sorted, GRID_ROWS, GRID_COLS, TILE_SIZE)

recon_path = './reconstructed_map.png'
stitched_img.save(recon_path)
print(f"[INFO] Stitched map saved → {recon_path}")

# ── OCR ────────────────────────────────────────────────────────────────────────
print("[OCR] Running EasyOCR on stitched map …")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    map_img = cv2.imread(recon_path)
    results_ocr = reader.readtext(map_img)
    ocr_data = [(bbox, text.strip(), conf)
                for bbox, text, conf in results_ocr if conf > 0.25]
    print(f"[OCR] Detected {len(ocr_data)} text regions")
    ocr_available = True
except Exception as e:
    print(f"[OCR] EasyOCR failed ({e}), will use fallback heuristics")
    ocr_data = []
    ocr_available = False

# Build OCR dataframe
def get_bbox_centre(bbox):
    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    return float(np.mean(xs)), float(np.mean(ys))

if ocr_available and ocr_data:
    map_cv = cv2.imread(recon_path)
    h, w = map_cv.shape[:2]
    rows = []
    for bbox, text, conf in ocr_data:
        cx, cy = get_bbox_centre(bbox)
        rows.append({'text': text, 'text_lower': text.lower(),
                     'cx': cx, 'cy': cy, 'conf': conf,
                     'rel_x': cx/w, 'rel_y': cy/h})
    df_ocr = pd.DataFrame(rows)
else:
    df_ocr = pd.DataFrame(columns=['text','text_lower','cx','cy','conf','rel_x','rel_y'])
    map_cv = cv2.imread(recon_path)
    h, w = (map_cv.shape[:2] if map_cv is not None else (1,1))

# ── QA engine ─────────────────────────────────────────────────────────────────
from difflib import SequenceMatcher

def find_text_on_map(query, df, threshold=0.55):
    q = query.lower().strip()
    matches = []
    for _, row in df.iterrows():
        t = row['text_lower']
        ratio = SequenceMatcher(None, q, t).ratio()
        if q in t or t in q:
            ratio = max(ratio, 0.80)
        if ratio >= threshold:
            matches.append({**row.to_dict(), 'match_score': ratio})
    matches.sort(key=lambda x: -x['match_score'])
    return matches


def answer_question(question, options, df_ocr, map_shape):
    h, w = map_shape[:2]
    q_l = question.lower()

    # Spatial constraint
    spatial = None
    for kw in ('north','south','east','west'):
        if kw in q_l:
            spatial = kw; break

    # Find anchor (capitalised noun phrase from question)
    anchor_pos = None
    phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', question)
    for phrase in phrases:
        hits = find_text_on_map(phrase, df_ocr, threshold=0.60)
        if hits:
            anchor_pos = (hits[0]['cx'], hits[0]['cy'])
            print(f"  Anchor '{phrase}' @ ({anchor_pos[0]:.0f},{anchor_pos[1]:.0f})")
            break

    scores = {}
    for opt_num, opt_text in options.items():
        hits = find_text_on_map(str(opt_text), df_ocr, threshold=0.50)
        if not hits:
            scores[opt_num] = 0.0
            continue
        best = hits[0]
        score = best['match_score'] * (best['conf'] + 0.1)

        # Spatial penalty
        if spatial == 'north'  and best['rel_y'] > 0.40: score *= 0.25
        if spatial == 'south'  and best['rel_y'] < 0.60: score *= 0.25
        if spatial == 'east'   and best['rel_x'] < 0.60: score *= 0.25
        if spatial == 'west'   and best['rel_x'] > 0.40: score *= 0.25

        # Proximity to anchor
        if anchor_pos is not None:
            dist = np.hypot(best['cx'] - anchor_pos[0], best['cy'] - anchor_pos[1])
            prox = 1.0 - dist / max(w, h)
            score *= (1.0 + max(prox, 0.0))

        scores[opt_num] = score
        print(f"  Opt {opt_num} '{opt_text}': score={score:.3f}  "
              f"pos=({best['rel_x']:.2f},{best['rel_y']:.2f})")

    # If all zero (no OCR hits), pick randomly but deterministically
    if max(scores.values()) == 0:
        print("  [WARN] No OCR evidence – picking option 1 as fallback")
        return 1, scores

    return max(scores, key=scores.get), scores


# ── Run QA over all questions ──────────────────────────────────────────────────
answers = []
for _, row in questions_df.iterrows():
    q_id     = row['id']
    question = row['question']
    options  = {
        1: str(row.get('option_1', '')),
        2: str(row.get('option_2', '')),
        3: str(row.get('option_3', '')),
        4: str(row.get('option_4', '')),
    }
    print(f"\nQ {q_id}: {question}")
    answer, scores = answer_question(question, options, df_ocr,
                                     map_cv.shape if map_cv is not None else (1,1))
    print(f"  → Option {answer} : {options[answer]}")
    answers.append({'id': q_id, 'answer': answer})

# ── Write submission.csv ───────────────────────────────────────────────────────
out_df = pd.DataFrame(answers)
out_df.to_csv('./submission.csv', index=False)
print(f"\n[DONE] submission.csv written ({len(out_df)} rows)")
print(out_df.head(10).to_string())
