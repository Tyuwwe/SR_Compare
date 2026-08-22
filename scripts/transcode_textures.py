#!/usr/bin/env python3
"""Batch-transcode scene textures to BC7 KTX2 with pre-baked mip chains.

Uses AMD CompressonatorCLI (pinned, downloaded into third_party/compressonator
on first run — same download pattern as fetch_sdks.py).  Every .png/.jpg under
assets/ gets a sibling <name>.ktx2 (BC7, full mip chain, no supercompression);
the renderer's glTF loader prefers the .ktx2 automatically and falls back to
the original PNG/JPG when it is absent (renderer/scene/Scene.h, Phase 7b).

BC7 is used for all usages (base color and normal/MR/AO data alike) to keep a
single path; BC5 for normal maps is a possible quality follow-up.  sRGB vs
linear is a sampling-time property of the VkFormat the loader picks, not of
the blocks, so one file serves both variants.

Usage:
    python scripts/transcode_textures.py                 # all of assets/
    python scripts/transcode_textures.py --scene bistro  # one assets subdir
    python scripts/transcode_textures.py --force         # re-encode existing
    python scripts/transcode_textures.py --jobs 8        # parallel encodes
"""

import argparse
import concurrent.futures
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TOOL_DIR = ROOT / "third_party" / "compressonator"

# Pinned so re-running reproduces identical BC7 output.
CMP_VERSION = "4.5.52"
CMP_URL = (
    "https://github.com/GPUOpen-Tools/compressonator/releases/download/"
    f"V{CMP_VERSION}/compressonatorcli-{CMP_VERSION}-win64.zip"
)

UA = {"User-Agent": "sr_compare-transcode/1.0 (python-urllib)"}


def log(msg: str) -> None:
    print(msg, flush=True)


def find_cli() -> Path | None:
    for p in sorted(TOOL_DIR.rglob("compressonatorcli.exe")):
        return p
    return None


def ensure_cli() -> Path:
    cli = find_cli()
    if cli:
        return cli
    log(f"[tool] downloading CompressonatorCLI {CMP_VERSION}")
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TOOL_DIR / "compressonatorcli.zip"
    req = urllib.request.Request(CMP_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f, 1024 * 1024)
    log(f"[tool] extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TOOL_DIR)
    zip_path.unlink()
    cli = find_cli()
    if not cli:
        raise RuntimeError("compressonatorcli.exe not found after extraction")
    return cli


def transcode_one(cli: Path, src: Path, force: bool, tool_threads: int) -> tuple[Path, str]:
    dst = src.with_suffix(".ktx2")
    if dst.exists() and not force:
        return src, "skip"
    # -miplevels 20 clamps to the full chain (down to 1x1).  BC7 CPU encoder,
    # default quality.  Mip filtering runs in the source's native space,
    # matching the previous runtime-blit path's approximation.
    cmd = [
        str(cli),
        "-fd", "BC7",
        "-miplevels", "20",
        "-NumThreads", str(tool_threads),
        "-noprogress",
        "-silent",
        str(src),
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        return src, f"FAIL: {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}"
    return src, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="only assets/<scene> (e.g. bistro, sponza)")
    parser.add_argument("--force", action="store_true", help="re-encode existing .ktx2")
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel encodes (each spawns --tool-threads encoder threads)")
    parser.add_argument("--tool-threads", type=int, default=4,
                        help="BC7 encoder threads per job (jobs * tool-threads ~= core count)")
    args = parser.parse_args()

    base = ASSETS / args.scene if args.scene else ASSETS
    if not base.is_dir():
        log(f"error: {base} does not exist")
        return 2

    cli = ensure_cli()
    log(f"[tool] {cli}")

    sources = sorted(
        p for p in base.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    todo = [p for p in sources if args.force or not p.with_suffix(".ktx2").exists()]
    log(f"[scan] {len(sources)} textures under {base}, {len(todo)} to encode")
    if not todo:
        return 0

    ok = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(transcode_one, cli, p, args.force, args.tool_threads): p for p in todo
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            src, status = fut.result()
            if status == "ok":
                ok += 1
            elif status.startswith("FAIL"):
                fail += 1
                log(f"  FAIL {src}: {status}")
            if i % 25 == 0 or i == len(todo):
                log(f"  [{i}/{len(todo)}] ok={ok} fail={fail}")
    log(f"done: {ok} encoded, {fail} failed, {len(sources) - len(todo)} pre-existing")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
