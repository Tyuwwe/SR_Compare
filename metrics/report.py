#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据 bench CSV 与画质 CSV 生成 Markdown 对比报告。

用法：
    python report.py --bench output/bench_xxx.csv --quality output/quality_xxx.csv --out report.md
"""

import argparse
import datetime
import os
import re

import numpy as np
import pandas as pd

METRIC_ORDER = ["psnr", "ssim", "lpips", "flip"]

BENCH_NUM_COLS = [
    "upscale_pass_ms_avg", "upscale_pass_ms_p50", "frame_ms_avg",
    "fps_avg", "fps_p50", "fps_1pct_low", "vram_algo_bytes", "vram_total_bytes",
]


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def load_bench(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in BENCH_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_quality(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ["mean", "median", "p5", "num_frames"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# 格式化辅助
# --------------------------------------------------------------------------- #
def human_bytes(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if x < 0:
        return ""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{x:.0f}{units[i]}"
    return f"{x:.1f}{units[i]}"


def fmt_num(x, nd=2):
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)) and np.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def _fmt_metric(m, v):
    if pd.isna(v):
        return ""
    v = float(v)
    if m == "psnr":
        return "inf" if np.isinf(v) else f"{v:.2f}"
    return f"{v:.4f}"


def _esc(s):
    """Escape pipe characters so a cell value cannot break the Markdown table."""
    return str(s).replace("|", "\\|")


def _res_num(res):
    s = str(res).strip().lower()
    m = re.match(r"(\d+)\s*k", s)
    if m:
        return int(m.group(1)) * 1000
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def _bench_key(row):
    res = str(row.get("resolution", "")).strip()
    factor = row.get("upscale_factor")
    if pd.isna(factor):
        factor_s = ""
    else:
        factor_s = str(factor).strip()
        if factor_s.lower() in ("", "nan", "none"):
            factor_s = ""
        elif factor_s.lower().endswith("x"):
            factor_s = factor_s[:-1]
    return f"{res}@{factor_s}x" if factor_s else res


# --------------------------------------------------------------------------- #
# 表格构建
# --------------------------------------------------------------------------- #
def build_perf_pivot(df):
    if df.empty:
        return None, []
    work = df.copy()
    work["_key"] = work.apply(_bench_key, axis=1)
    algos = sorted(work["algo"].dropna().unique().tolist())
    keys = sorted(work["_key"].unique().tolist(),
                  key=lambda k: _res_num(k.split("@")[0]), reverse=True)

    def cell(row):
        parts = []
        if pd.notna(row.get("upscale_pass_ms_avg")):
            parts.append(f"pass {row['upscale_pass_ms_avg']:.1f}ms")
        if pd.notna(row.get("fps_avg")):
            parts.append(f"{row['fps_avg']:.1f}fps")
        if pd.notna(row.get("vram_algo_bytes")):
            parts.append(human_bytes(row["vram_algo_bytes"]))
        return " · ".join(parts) if parts else ""

    matrix = {}
    for _, row in work.iterrows():
        matrix[(str(row["algo"]), str(row["_key"]))] = cell(row)

    lines = ["| 算法 \\ 分辨率@倍率 | " + " | ".join(keys) + " |",
             "|---|" + "---|" * len(keys)]
    missing = []
    for a in algos:
        cells = []
        for k in keys:
            v = matrix.get((a, k), "")
            cells.append(v)
            if not v:
                missing.append((a, k))
        lines.append("| " + a + " | " + " | ".join(cells) + " |")
    return "\n".join(lines), missing


def build_perf_detail(df):
    if df.empty:
        return None
    header = ["算法", "分辨率", "倍率", "pass均值(ms)", "pass中位(ms)", "帧耗时(ms)",
              "FPS均值", "FPS中位", "FPS 1%低", "算法显存", "总显存", "GPU", "驱动", "备注"]
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for _, row in df.iterrows():
        cells = [
            _esc(row.get("algo", "")),
            _esc(row.get("resolution", "")),
            _esc(row.get("upscale_factor", "")),
            fmt_num(row.get("upscale_pass_ms_avg"), 2),
            fmt_num(row.get("upscale_pass_ms_p50"), 2),
            fmt_num(row.get("frame_ms_avg"), 2),
            fmt_num(row.get("fps_avg"), 1),
            fmt_num(row.get("fps_p50"), 1),
            fmt_num(row.get("fps_1pct_low"), 1),
            human_bytes(row.get("vram_algo_bytes")),
            human_bytes(row.get("vram_total_bytes")),
            _esc(row.get("gpu_name", "")),
            _esc(row.get("driver_version", "")),
            _esc(row.get("notes", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_quality_pivot(df):
    if df.empty:
        return None
    piv = df.pivot_table(index="algo", columns="metric", values="mean", aggfunc="first")
    metrics = [m for m in METRIC_ORDER if m in piv.columns]
    algos = sorted(piv.index.astype(str).tolist())
    lines = ["| 算法 \\ 指标(mean) | " + " | ".join(m.upper() for m in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for a in algos:
        cells = []
        for m in metrics:
            v = piv.loc[a, m]
            cells.append("" if pd.isna(v) else _fmt_metric(m, v))
        lines.append("| " + a + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_quality_detail(df):
    if df.empty:
        return None
    lines = ["| 算法 | 指标 | mean | median | p5 | 帧数 |",
             "|---|---|---|---|---|---|"]
    for _, row in df.iterrows():
        m = str(row.get("metric", ""))
        num_frames = row.get("num_frames")
        num_frames_s = "" if pd.isna(num_frames) else str(int(num_frames))
        lines.append(
            f"| {row.get('algo', '')} | {m} | {_fmt_metric(m, row.get('mean'))} "
            f"| {_fmt_metric(m, row.get('median'))} | {_fmt_metric(m, row.get('p5'))} "
            f"| {num_frames_s} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 要点总结
# --------------------------------------------------------------------------- #
def summarize_perf(df):
    if df.empty or "resolution" not in df.columns:
        return ["（无性能数据，无法生成要点）"]
    lines = []
    resolutions = sorted(df["resolution"].dropna().unique().tolist(),
                         key=_res_num, reverse=True)
    if "upscale_pass_ms_avg" in df.columns:
        for res in resolutions:
            g = df[(df["resolution"] == res) & df["upscale_pass_ms_avg"].notna()]
            if g.empty:
                continue
            row = g.loc[g["upscale_pass_ms_avg"].idxmin()]
            factor = str(row.get("upscale_factor", "")).strip()
            where = f"{res}（{factor}x档）" if factor else str(res)
            lines.append(f"{row['algo']} 在 {where} 的 upscale pass 耗时最低（{row['upscale_pass_ms_avg']:.2f} ms）。")
            break
    if "fps_avg" in df.columns:
        g = df[df["fps_avg"].notna()]
        if not g.empty:
            row = g.loc[g["fps_avg"].idxmax()]
            lines.append(f"{row['algo']} 的平均帧率最高（{row['fps_avg']:.1f} fps @ {row['resolution']}）。")
    if "vram_algo_bytes" in df.columns:
        g = df[df["vram_algo_bytes"].notna()]
        if not g.empty:
            row = g.loc[g["vram_algo_bytes"].idxmin()]
            lines.append(f"{row['algo']} 的算法侧显存占用最低（{human_bytes(row['vram_algo_bytes'])}）。")
    if not lines:
        lines.append("（性能数据不足，无法生成要点）")
    return lines[:3]


def summarize_quality(df):
    if df.empty:
        return ["（无画质数据，无法生成要点）"]
    piv = df.pivot_table(index="algo", columns="metric", values="mean", aggfunc="first")
    lines = []
    for m, better in [("psnr", "high"), ("ssim", "high"), ("lpips", "low"), ("flip", "low")]:
        if m not in piv.columns:
            continue
        col = piv[m].dropna()
        if col.empty:
            continue
        if better == "high":
            best, val = col.idxmax(), col.max()
        else:
            best, val = col.idxmin(), col.min()
        lines.append(f"{m.upper()} 方面 {best} 最优（mean={_fmt_metric(m, val)}）。")
    if not lines:
        lines.append("（画质数据不足，无法生成要点）")
    return lines[:3]


def collect_notes(df):
    notes = []
    if "notes" not in df.columns:
        return notes
    seen = set()
    for _, row in df.iterrows():
        n = str(row.get("notes", "")).strip()
        if n and n.lower() not in ("", "nan", "none") and n not in seen:
            seen.add(n)
            notes.append(f"- {row.get('algo', '')}: {n}")
    return notes


# --------------------------------------------------------------------------- #
# 报告生成
# --------------------------------------------------------------------------- #
def generate_report(bench_df, quality_df):
    parts = []
    parts.append("# 超分算法对比报告")
    parts.append("")
    parts.append(f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append(f"- 性能记录数：{len(bench_df)}；画质记录数：{len(quality_df)}")
    parts.append("")

    # ---- 性能 ----
    parts.append("## 性能")
    parts.append("")
    if bench_df.empty:
        perf_pivot, missing = None, []
        parts.append("（无性能数据）")
    else:
        perf_pivot, missing = build_perf_pivot(bench_df)
        parts.append("下表按算法 × 分辨率@倍率 透视，单元格为 `pass耗时 / 平均FPS / 算法侧显存`：")
        parts.append("")
        parts.append(perf_pivot)
    parts.append("")
    parts.append("**要点**")
    parts.append("")
    for s in summarize_perf(bench_df):
        parts.append(f"- {s}")
    parts.append("")

    if not bench_df.empty:
        detail = build_perf_detail(bench_df)
        if detail:
            parts.append("### 性能明细")
            parts.append("")
            parts.append(detail)
            parts.append("")

    footnotes = []
    if missing:
        if len(missing) <= 12:
            detail = "；".join(f"{a}@{k}" for a, k in missing)
        else:
            detail = f"{len(missing)} 个 (算法, 分辨率@倍率) 组合"
        footnotes.append(f"- 透视表中空单元格表示对应组合无数据（缺失：{detail}）。")
    else:
        footnotes.append("- 透视表中空单元格表示对应组合无数据。")
    footnotes.extend(collect_notes(bench_df))
    if footnotes:
        parts.append("**脚注**")
        parts.append("")
        parts.extend(footnotes)
        parts.append("")

    # ---- 画质 ----
    parts.append("## 画质")
    parts.append("")
    qpivot = build_quality_pivot(quality_df)
    if qpivot:
        parts.append("下表为各算法各指标的均值（PSNR/SSIM 越大越好，LPIPS/FLIP 越小越好）：")
        parts.append("")
        parts.append(qpivot)
    else:
        parts.append("（无画质数据）")
    parts.append("")
    parts.append("**要点**")
    parts.append("")
    for s in summarize_quality(quality_df):
        parts.append(f"- {s}")
    parts.append("")

    qdetail = build_quality_detail(quality_df)
    if qdetail:
        parts.append("### 画质明细")
        parts.append("")
        parts.append(qdetail)
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("*口径：PSNR/SSIM 在 sRGB 域计算（data_range=1.0）；LPIPS 输入归一化到 [-1,1]（net=alex）；FLIP 为 LDR-FLIP（默认 ppd=67）。*")
    parts.append("")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 Markdown 对比报告")
    ap.add_argument("--bench", required=True, help="bench CSV 路径")
    ap.add_argument("--quality", default=None, help="画质 CSV 路径（可选；缺省或文件不存在时只出性能报告）")
    ap.add_argument("--out", default="report.md", help="输出 Markdown 路径")
    args = ap.parse_args(argv)

    bench_df = load_bench(args.bench)
    if args.quality and os.path.isfile(args.quality):
        quality_df = load_quality(args.quality)
    else:
        quality_df = pd.DataFrame(columns=["algo", "metric", "mean", "median", "p5", "num_frames"])
    md = generate_report(bench_df, quality_df)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已生成：{out}（{len(md)} 字符）")


if __name__ == "__main__":
    main()
