#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线画质指标计算：对 captures/<run> 中每个算法与 GT 逐帧配对计算 PSNR/SSIM/LPIPS/FLIP。

用法：
    python compute_metrics.py --run captures/<run> [--metrics psnr,ssim,lpips,flip] [--max-frames N]

口径说明：
    - 输入 PNG 按 sRGB 读取（uint8 -> float [0,1]）。
    - PSNR / SSIM 直接在 sRGB 域计算，data_range=1.0。
    - LPIPS 使用 net=alex，输入归一化到 [-1,1]（lpips 官方约定）。
    - FLIP 使用 LDR-FLIP（flip_evaluator），输入为 sRGB float [0,1]，默认 ppd=67。
    - 每个 (算法, 指标) 输出 mean / median / p5 / num_frames。
"""

import argparse
import glob
import os
import sys

import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
from skimage.metrics import structural_similarity as _sk_ssim

DEFAULT_METRICS = ("psnr", "ssim", "lpips", "flip")
VALID_METRICS = set(DEFAULT_METRICS)


# --------------------------------------------------------------------------- #
# 图像读取
# --------------------------------------------------------------------------- #
def load_srgb(path):
    """读取 sRGB PNG，返回 HxWx3 的 float32 [0,1] 数组。"""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"无法识别的图像通道布局 {img.shape}: {path}")

    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr
    raise ValueError(f"不支持的图像 dtype {img.dtype}: {path}")


# --------------------------------------------------------------------------- #
# PSNR / SSIM
# --------------------------------------------------------------------------- #
def psnr(ref, img):
    """sRGB 域 PSNR（dB），data_range=1.0；两图完全相同返回 inf。"""
    return float(_sk_psnr(ref, img, data_range=1.0))


def ssim(ref, img):
    """sRGB 域 SSIM（逐通道平均），data_range=1.0。"""
    return float(_sk_ssim(ref, img, data_range=1.0, channel_axis=-1))


# --------------------------------------------------------------------------- #
# LPIPS（惰性加载，避免不需要时拖慢启动）
# --------------------------------------------------------------------------- #
_LPIPS_FN = None


def _limit_torch_threads(torch):
    raw = os.environ.get("METRICS_TORCH_THREADS")
    n = None
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = None
    if n is None:
        n = max(1, min(4, (os.cpu_count() or 1)))
    torch.set_num_threads(n)


def _get_lpips_fn():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import torch
        _limit_torch_threads(torch)
        import lpips  # noqa: E402
        _LPIPS_FN = lpips.LPIPS(net="alex")
        _LPIPS_FN.eval()
    return _LPIPS_FN


def _to_lpips_tensor(x):
    # x: HxWx3 float [0,1] -> 1x3xHxW float [-1,1]
    import torch
    arr = np.ascontiguousarray(x.transpose(2, 0, 1))
    t = torch.from_numpy(arr).unsqueeze(0).float()
    return t * 2.0 - 1.0


def lpips_score(ref, img):
    import torch
    fn = _get_lpips_fn()
    a = _to_lpips_tensor(ref)
    b = _to_lpips_tensor(img)
    with torch.no_grad():
        return float(fn(a, b).item())


# --------------------------------------------------------------------------- #
# FLIP（惰性加载）
# --------------------------------------------------------------------------- #
def flip_score(ref, img, ppd=67.0, out_map_path=None):
    """LDR-FLIP，返回 mean FLIP error；可选保存 error map。"""
    import flip_evaluator as flip
    err_map, mean_err, _ = flip.evaluate(
        ref, img, "LDR", inputsRGB=True, applyMagma=False,
        computeMeanError=True, parameters={"ppd": float(ppd)},
    )
    if out_map_path:
        save_flip_map(err_map, out_map_path)
    return float(mean_err)


def save_flip_map(err_map, out_path):
    """把 FLIP error map 保存为 PNG（灰度值映射到 uint8 RGB）。"""
    from PIL import Image
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.asarray(err_map)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = arr[..., :3]
    elif arr.ndim == 3:
        arr = arr[..., 0]

    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    elif arr.dtype == np.uint8:
        pass
    elif arr.dtype == np.uint16:
        arr = (arr / 257).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray(arr).save(out_path)


# --------------------------------------------------------------------------- #
# 帧配对与计算
# --------------------------------------------------------------------------- #
def _frame_files(d):
    return sorted(glob.glob(os.path.join(d, "frame_*.png")))


def discover_algorithms(run_dir):
    algos = []
    for name in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, name)
        if os.path.isdir(p) and name != "GT":
            algos.append(name)
    return algos


def compute_run_quality(run_dir, metrics=DEFAULT_METRICS, max_frames=None, flip_ppd=67.0):
    """对 run 目录逐算法逐帧计算指标，返回长表记录列表。

    每条记录：algo, metric, mean, median, p5, num_frames。
    """
    run_dir = os.path.abspath(run_dir)
    gt_dir = os.path.join(run_dir, "GT")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"未找到 GT 目录: {gt_dir}")

    gt_files = _frame_files(gt_dir)
    if not gt_files:
        raise FileNotFoundError(f"GT 目录下没有 frame_*.png: {gt_dir}")
    gt_names = [os.path.basename(f) for f in gt_files]

    metrics = [m.lower() for m in metrics]
    for m in metrics:
        if m not in VALID_METRICS:
            raise ValueError(f"未知指标: {m}（可选 {sorted(VALID_METRICS)}）")

    records = []
    for algo in discover_algorithms(run_dir):
        algo_dir = os.path.join(run_dir, algo)
        algo_files = {os.path.basename(f): f for f in _frame_files(algo_dir)}
        common = [n for n in gt_names if n in algo_files]
        if not common:
            print(f"[warn] 算法 {algo} 没有可与 GT 配对的帧，跳过", file=sys.stderr)
            continue
        if max_frames is not None:
            common = common[: int(max_frames)]

        per_metric = {m: [] for m in metrics}
        for name in common:
            ref = load_srgb(os.path.join(gt_dir, name))
            img = load_srgb(algo_files[name])
            if ref.shape[:2] != img.shape[:2]:
                raise RuntimeError(
                    f"尺寸不一致：GT {name} 为 {ref.shape[:2]}，而 {algo}/{name} 为 "
                    f"{img.shape[:2]}。请确认该算法输出与 GT 使用了相同渲染分辨率。"
                )
            if "psnr" in metrics:
                per_metric["psnr"].append(psnr(ref, img))
            if "ssim" in metrics:
                per_metric["ssim"].append(ssim(ref, img))
            if "lpips" in metrics:
                per_metric["lpips"].append(lpips_score(ref, img))
            if "flip" in metrics:
                map_path = os.path.join(algo_dir, "flip_maps", name)
                per_metric["flip"].append(flip_score(ref, img, ppd=flip_ppd, out_map_path=map_path))

        for m in metrics:
            vals = per_metric[m]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            with np.errstate(all="ignore"):
                mean = float(np.mean(arr))
                median = float(np.median(arr))
                p5 = float(np.percentile(arr, 5))
                if np.isnan(p5):  # 全 inf 时百分位插值会得到 NaN，退化为 min
                    p5 = float(np.min(arr))
            records.append({
                "algo": algo,
                "metric": m,
                "mean": mean,
                "median": median,
                "p5": p5,
                "num_frames": int(len(vals)),
            })
    return records


def write_quality_csv(records, out_path):
    import pandas as pd
    df = pd.DataFrame(records, columns=["algo", "metric", "mean", "median", "p5", "num_frames"])
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.8g")
    return out_path


def check_metrics_available(metrics):
    problems = []
    if "lpips" in metrics:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            import lpips  # noqa: F401
        except ImportError as e:
            problems.append(f"lpips: {e}")
    if "flip" in metrics:
        try:
            import flip_evaluator  # noqa: F401
        except ImportError as e:
            problems.append(f"flip: {e}")
    if problems:
        raise ImportError("；".join(problems))


def main(argv=None):
    ap = argparse.ArgumentParser(description="计算 captures/<run> 中各算法相对 GT 的画质指标")
    ap.add_argument("--run", required=True, help="run 目录，例如 captures/20260818_1500")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help="逗号分隔的指标列表，可选 psnr,ssim,lpips,flip")
    ap.add_argument("--max-frames", type=int, default=None, help="每个算法最多处理的帧数")
    ap.add_argument("--flip-ppd", type=float, default=67.0, help="FLIP 的 pixels per degree")
    ap.add_argument("--out", default=None, help="输出 CSV 路径（默认 output/quality_<run>.csv）")
    args = ap.parse_args(argv)

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("--metrics 不能为空")
    check_metrics_available(metrics)

    records = compute_run_quality(args.run, metrics=metrics,
                                  max_frames=args.max_frames, flip_ppd=args.flip_ppd)
    run_name = os.path.basename(os.path.abspath(args.run))
    out = args.out or os.path.join("output", f"quality_{run_name}.csv")
    out = write_quality_csv(records, out)
    print(f"完成：{len(records)} 条指标记录 -> {out}")


if __name__ == "__main__":
    main()
