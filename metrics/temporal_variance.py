#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal stability metric for frame dumps (SR_FRAME_DUMP_DIR).

Computes per-pixel temporal standard deviation of luma across a frame range
and reports aggregate stats; optionally writes a false-colour heatmap PNG.

Usage:
    python temporal_variance.py <dump_dir> [--skip 20] [--heatmap out.png]
"""
import argparse
import glob
import os

import numpy as np
from skimage import io


def load_luma(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    f = img.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--skip", type=int, default=20, help="frames to skip (warmup)")
    ap.add_argument("--heatmap", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "frame_*.png")))[args.skip:]
    if len(files) < 2:
        raise SystemExit(f"need >=2 frames, found {len(files)} in {args.dump_dir}")

    acc = None
    acc2 = None
    n = 0
    for f in files:
        y = load_luma(f)
        if acc is None:
            acc = np.zeros_like(y)
            acc2 = np.zeros_like(y)
        acc += y
        acc2 += y * y
        n += 1
    mean = acc / n
    var = np.maximum(acc2 / n - mean * mean, 0.0)
    std = np.sqrt(var)

    qs = np.percentile(std, [50, 90, 95, 99, 99.9])
    print(f"{args.dump_dir}: frames={n}")
    print(f"  luma std  mean={std.mean():.5f}  p50={qs[0]:.5f}  p90={qs[1]:.5f}  "
          f"p95={qs[2]:.5f}  p99={qs[3]:.5f}  p99.9={qs[4]:.5f}  max={std.max():.5f}")
    for thr in (0.01, 0.02, 0.05):
        frac = float((std > thr).mean()) * 100.0
        print(f"  pixels std>{thr:.2f}: {frac:.3f}%")

    if args.heatmap:
        scale = np.clip(std / 0.10, 0.0, 1.0)  # 0.10 luma std -> full red
        heat = np.stack([scale, np.clip(mean, 0, 1) * 0.3, 1.0 - scale], axis=-1)
        io.imsave(args.heatmap, (heat * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
