#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成图像自测：验证 compute_metrics.py 与 report.py 的核心逻辑。

覆盖：
    - 相同图：PSNR=inf / SSIM≈1 / FLIP≈0 / LPIPS≈0
    - 退化图（模糊/位移/噪声）：指标单调变化
    - 尺寸不一致时报错
    - report.py 用合成 CSV 产出 report.md
"""

import csv
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image
from skimage.filters import gaussian

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import compute_metrics as cm  # noqa: E402
import report as rp  # noqa: E402

OUT_DIR = os.path.join(SCRIPT_DIR, "selftest_out")
SIZE = 256
FRAMES = 4


# --------------------------------------------------------------------------- #
# 合成图像
# --------------------------------------------------------------------------- #
def make_frame(idx, size=SIZE):
    """带渐变、移动圆、矩形条的合成图（含边缘结构，便于检测模糊/位移）。"""
    H = W = size
    y, x = np.mgrid[0:H, 0:W]
    base = x / (W - 1) * 0.6 + y / (H - 1) * 0.4
    img = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        img[..., c] = np.clip(base * (0.4 + 0.3 * c), 0, 1)

    cx = 50 + idx * 40
    cy = 80 + idx * 25
    r = 38
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    circle = (d < r).astype(np.float32)
    rect = ((x > 40) & (x < 120) & (y > 160) & (y < 200)).astype(np.float32)
    for c in range(3):
        img[..., c] = np.clip(
            img[..., c] + circle * (0.85 - 0.25 * c) + rect * (0.15 + 0.25 * c), 0, 1)
    return img


def save_png(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(path)


def build_run(run_dir):
    gt_dir = os.path.join(run_dir, "GT")
    rng = np.random.default_rng(1234)
    algos = {
        "identical": lambda a: a.copy(),
        "blur_mild": lambda a: gaussian(a, sigma=1.0, channel_axis=-1, mode="reflect"),
        "blur_strong": lambda a: gaussian(a, sigma=4.0, channel_axis=-1, mode="reflect"),
        "shift": lambda a: np.roll(a, (4, 7), axis=(0, 1)),
        "noise": lambda a: np.clip(a + rng.normal(0, 0.08, a.shape), 0, 1),
    }
    for i in range(FRAMES):
        ref = make_frame(i)
        save_png(os.path.join(gt_dir, f"frame_{i:04d}.png"), ref)
        for name, fn in algos.items():
            save_png(os.path.join(run_dir, name, f"frame_{i:04d}.png"), fn(ref))


# --------------------------------------------------------------------------- #
# 断言辅助
# --------------------------------------------------------------------------- #
def to_map(records):
    out = {}
    for r in records:
        out[(r["algo"], r["metric"])] = r
    return out


def check_core(records):
    m = to_map(records)
    assert m[("identical", "psnr")]["num_frames"] == FRAMES, "帧数应等于合成帧数"

    # 相同图
    assert math.isinf(m[("identical", "psnr")]["mean"]), "相同图 PSNR 应为 inf"
    assert m[("identical", "ssim")]["mean"] > 0.999, "相同图 SSIM 应≈1"
    assert m[("identical", "flip")]["mean"] < 0.01, "相同图 FLIP 应≈0"

    # 退化图比相同图差
    for a in ["blur_mild", "blur_strong", "shift", "noise"]:
        assert math.isfinite(m[(a, "psnr")]["mean"]), f"{a} PSNR 应为有限值"
        assert m[(a, "ssim")]["mean"] < m[("identical", "ssim")]["mean"], f"{a} SSIM 应低于 identical"
        assert m[(a, "flip")]["mean"] > m[("identical", "flip")]["mean"], f"{a} FLIP 应高于 identical"

    # 单调：strong blur 比 mild blur 更差
    assert m[("blur_strong", "ssim")]["mean"] < m[("blur_mild", "ssim")]["mean"], "SSIM 应随模糊增强而下降"
    assert m[("blur_strong", "psnr")]["mean"] < m[("blur_mild", "psnr")]["mean"], "PSNR 应随模糊增强而下降"
    assert m[("blur_strong", "flip")]["mean"] > m[("blur_mild", "flip")]["mean"], "FLIP 应随模糊增强而上升"


def check_lpips(records):
    m = to_map(records)
    assert m[("identical", "lpips")]["mean"] < 0.05, "相同图 LPIPS 应≈0"
    assert m[("blur_strong", "lpips")]["mean"] > m[("blur_mild", "lpips")]["mean"], "LPIPS 应随模糊增强而上升"
    for a in ["blur_mild", "blur_strong", "shift", "noise"]:
        assert m[(a, "lpips")]["mean"] > m[("identical", "lpips")]["mean"], f"{a} LPIPS 应高于 identical"


def test_size_mismatch():
    d = os.path.join(OUT_DIR, "mismatch_run")
    shutil.rmtree(d, ignore_errors=True)
    save_png(os.path.join(d, "GT", "frame_0000.png"), make_frame(0, size=256))
    save_png(os.path.join(d, "bad", "frame_0000.png"), make_frame(0, size=128))
    try:
        cm.compute_run_quality(d, metrics=["psnr"])
        raise AssertionError("尺寸不一致时应报错，但未报错")
    except RuntimeError as e:
        assert "尺寸不一致" in str(e), f"错误信息应包含尺寸提示: {e}"
    print("  [OK] 尺寸不一致时正确报错")


# --------------------------------------------------------------------------- #
# 报告自测
# --------------------------------------------------------------------------- #
BENCH_HEADER = ["algo", "resolution", "upscale_factor", "upscale_pass_ms_avg",
                "upscale_pass_ms_p50", "frame_ms_avg", "fps_avg", "fps_p50",
                "fps_1pct_low", "vram_algo_bytes", "vram_total_bytes",
                "gpu_name", "driver_version", "notes"]

BENCH_ROWS = [
    ["FSR2", "2160p", "2.0", "2.5", "2.4", "20.0", "50.0", "50.5", "38.0",
     "800000000", "8000000000", "RTX 4070", "551.86", ""],
    ["FSR2", "1080p", "2.0", "1.2", "1.1", "14.0", "71.4", "72.0", "55.0",
     "500000000", "6000000000", "RTX 4070", "551.86", ""],
    ["XeSS", "2160p", "2.0", "3.1", "3.0", "22.0", "45.4", "46.0", "34.0",
     "900000000", "9000000000", "RTX 4070", "551.86", ""],
    ["NSS", "1080p", "2.0", "0.8", "0.8", "13.0", "76.9", "77.0", "60.0",
     "300000000", "5500000000", "RTX 4070", "551.86", "移动导向/模拟推理，仅供参考"],
    ["SGSR", "2160p", "2.0", "1.5", "1.4", "18.0", "55.5", "56.0", "42.0",
     "400000000", "7000000000", "RTX 4070", "551.86", "移动导向/模拟推理，仅供参考"],
]


def write_bench_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BENCH_HEADER)
        w.writerows(BENCH_ROWS)


def test_report(quality_csv_path):
    bench_path = os.path.join(OUT_DIR, "bench_synth.csv")
    write_bench_csv(bench_path)
    bench_df = rp.load_bench(bench_path)
    quality_df = rp.load_quality(quality_csv_path)
    md = rp.generate_report(bench_df, quality_df)
    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    for needle in ["## 性能", "## 画质", "分辨率@倍率", "移动导向", "NSS", "无数据"]:
        assert needle in md, f"报告缺少内容: {needle}"
    print("  [OK] report.md 生成（含性能/画质表、要点、notes 透传、缺失脚注）")
    return report_path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    run_dir = os.path.join(OUT_DIR, "run")
    print("== 生成合成图像 ==")
    build_run(run_dir)

    print("== 计算核心指标（psnr/ssim/flip） ==")
    core = cm.compute_run_quality(run_dir, metrics=["psnr", "ssim", "flip"])
    check_core(core)
    print("  [OK] 相同图 PSNR=inf / SSIM≈1 / FLIP≈0")
    print("  [OK] 退化图指标单调")

    print("== 计算 LPIPS ==")
    lpips_records = []
    lpips_ok = True
    lpips_err = None
    try:
        lpips_records = cm.compute_run_quality(run_dir, metrics=["lpips"])
        check_lpips(lpips_records)
        print("  [OK] LPIPS 单调")
    except Exception as e:  # noqa: BLE001
        lpips_ok = False
        lpips_err = e
        print(f"  [WARN] LPIPS 跳过: {type(e).__name__}: {e}")

    records = core + lpips_records
    quality_csv = os.path.join(OUT_DIR, "quality_run.csv")
    cm.write_quality_csv(records, quality_csv)
    print(f"  [OK] 画质 CSV -> {quality_csv}")

    print("== 尺寸不一致测试 ==")
    test_size_mismatch()

    print("== 报告生成测试 ==")
    report_path = test_report(quality_csv)

    print("== 全部通过 ==")
    print(f"  run 目录: {run_dir}")
    print(f"  画质 CSV: {quality_csv}")
    print(f"  报告: {report_path}")
    if not lpips_ok:
        print(f"  注意：LPIPS 未通过（{lpips_err}），其余指标全部通过。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
