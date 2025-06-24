import os
import shutil
import random

# Patch shutil to use a larger buffer (e.g. 16 MB) for better performance on Windows
import shutil as _shutil
def _fast_copyfileobj(fsrc, fdst, length=16*1024*1024):
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
_shutil.copyfileobj = _fast_copyfileobj

# Paths
orig_train = r"D:\OFFICE WORK\KDD Summer 2025\emotion-detection\data\train"
orig_test  = r"D:\OFFICE WORK\KDD Summer 2025\emotion-detection\data\test"
out_root   = r"D:\OFFICE WORK\KDD Summer 2025\emotion-detection\split_data"

splits = {'train': 0.6, 'val': 0.2, 'test': 0.2}
random.seed(42)  # for reproducibility

# Get list of emotions from train folder
emotions = sorted([d for d in os.listdir(orig_train)
                   if os.path.isdir(os.path.join(orig_train, d))])

# Create output directories
for split in splits:
    for emo in emotions:
        os.makedirs(os.path.join(out_root, split, emo), exist_ok=True)

# Process each emotion
for emo in emotions:
    # gather images from both original train and test
    img_paths = []
    for src_dir in (orig_train, orig_test):
        folder = os.path.join(src_dir, emo)
        img_paths += [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(img_paths)
    n = len(img_paths)
    n_train = int(splits['train'] * n)
    n_val   = int(splits['val'] * n)
    boundaries = {
        'train': (0, n_train),
        'val':   (n_train, n_train + n_val),
        'test':  (n_train + n_val, n),
    }
    for split, (i0, i1) in boundaries.items():
        for src in img_paths[i0:i1]:
            dst = os.path.join(out_root, split, emo, os.path.basename(src))
            with open(src, 'rb') as fin, open(dst, 'wb') as fout:
                _shutil.copyfileobj(fin, fout)
    print(f"[{emo}] total={n}, train={n_train}, val={n_val}, test={n - n_train - n_val}")
