#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: time series of the hottest temporal-variance pixels across dumps."""
import glob
import os
import sys

import numpy as np
from skimage import io

BASE = "output/tmp_glassjitter"
DUMPS = ["none", "fsr1", "taa", "taa_fix", "taa_nojitter", "fsr2", "fsr2_fix"]
SKIP = 20


def load_luma(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    f = img.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def series(dump, x, y):
    out = []
    for f in sorted(glob.glob(os.path.join(BASE, dump, "frame_*.png")))[SKIP:]:
        out.append(load_luma(f)[y, x])
    return np.array(out)


# hottest pixels from the taa dump
files = sorted(glob.glob(os.path.join(BASE, "taa", "frame_*.png")))[SKIP:]
acc = None
acc2 = None
for f in files:
    y = load_luma(f)
    if acc is None:
        acc = np.zeros_like(y)
        acc2 = np.zeros_like(y)
    acc += y
    acc2 += y * y
std = np.sqrt(np.maximum(acc2 / len(files) - (acc / len(files)) ** 2, 0.0))

flat = np.argsort(std.ravel())[::-1][:8]
seen = set()
top = []
for idx in flat:
    y, x = divmod(int(idx), std.shape[1])
    if any(abs(x - sx) < 8 and abs(y - sy) < 8 for sx, sy in seen):
        continue
    seen.add((x, y))
    top.append((x, y))

for x, y in top[:4]:
    print(f"=== pixel ({x},{y})  std_taa={std[y,x]:.4f}")
    for d in DUMPS:
        if not os.path.isdir(os.path.join(BASE, d)):
            continue
        s = series(d, x, y)
        print(f"  {d:14s} std={s.std():.4f}  " +
              " ".join(f"{v:.2f}" for v in s[:16]))
