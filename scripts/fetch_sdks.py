#!/usr/bin/env python3
"""Download and stage the third-party SDKs used by sr_compare.

Everything written by this script lives under <repo>/third_party/ and
<repo>/assets/ only.  Downloads are streamed with a progress bar, retried on
failure, and zip archives are deleted after extraction to save disk space.

Usage:
    python scripts/fetch_sdks.py                 # fetch everything
    python scripts/fetch_sdks.py --only fsr,xess # fetch a subset
    python scripts/fetch_sdks.py --force         # re-fetch even if present
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
ASSETS = ROOT / "assets"
TMP = THIRD_PARTY / ".tmp"

UA = {"User-Agent": "sr_compare-fetch-sdks/1.0 (python-urllib)"}

ALL_TASKS = ["fsr", "xess", "streamline", "sgsr", "nss", "sponza"]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def human(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def run(cmd, **kw) -> None:
    log(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def rmtree_force(path: Path) -> None:
    """Remove a tree even when it contains read-only git objects (Windows)."""
    path = Path(path)
    if not path.exists():
        return

    def _onerror(func, p, _exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_onerror)


def _progress(done: int, total: int, start: float, final: bool = False) -> None:
    if total > 0:
        pct = done / total * 100.0
        width = 24
        filled = int(pct // (100.0 / width))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {human(done)}/{human(total)}"
        )
    else:
        sys.stdout.write(f"\r  downloaded {human(done)}")
    if final:
        sys.stdout.write("\n")
    sys.stdout.flush()


def download(url: str, dest: Path, retries: int = 3) -> Path:
    """Stream *url* to *dest* with a progress bar and retry on failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                start = time.time()
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        _progress(done, total, start)
                _progress(done, total, start, final=True)
            part.replace(dest)
            log(f"  -> {dest} ({human(dest.stat().st_size)})")
            return dest
        except Exception as exc:  # noqa: BLE001 - retry any download error
            last_err = exc
            log(f"  download failed (attempt {attempt}/{retries}): {exc}")
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def gh_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={**UA, "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def find_asset(repo: str, tag: str | None = None, name_regex: str | None = None):
    """Return (tag_name, asset_dict) from a GitHub release.

    Uses ``/releases/tags/<tag>`` when *tag* is given, otherwise
    ``/releases/latest``.  When *name_regex* is provided the first asset whose
    ``name`` matches is returned (avoiding hard-coded, versioned filenames).
    """
    if tag:
        url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    else:
        url = f"https://api.github.com/repos/{repo}/releases/latest"
    rel = gh_json(url)
    tag_name = rel["tag_name"]
    assets = rel.get("assets", [])
    if name_regex:
        rx = re.compile(name_regex, re.IGNORECASE)
        for asset in assets:
            if rx.search(asset["name"]):
                return tag_name, asset
        raise RuntimeError(
            f"no asset matching {name_regex!r} for {repo} {tag_name}; "
            f"available: {[a['name'] for a in assets]}"
        )
    return tag_name, assets


def git_clone(
    url: str,
    dest: Path,
    shallow: bool = True,
    sparse_paths: list[str] | None = None,
) -> None:
    dest = Path(dest)
    if dest.exists() and (dest / ".git").exists():
        log(f"  exists (already cloned), skip: {dest}")
        return
    if dest.exists() and any(dest.iterdir()):
        log(f"  exists non-empty, skip: {dest}")
        return
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    if sparse_paths:
        cmd += ["--filter=blob:none", "--sparse"]
    cmd += [url, str(dest)]
    run(cmd)
    if sparse_paths:
        run(["git", "-C", str(dest), "sparse-checkout", "set", *sparse_paths])


def _norm_zpath(name: str) -> str:
    p = name.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _detect_zip_root(names: list[str]) -> str:
    # A single top-level directory exists only when there are no files
    # directly at the archive root and every directory entry shares the same
    # first path component.  This avoids mis-detecting a dominant directory
    # (e.g. ``doc/``) inside a flat archive as the root.
    first_components: set[str] = set()
    has_root_file = False
    for name in names:
        p = _norm_zpath(name)
        if "/" not in p:
            has_root_file = True
        else:
            first_components.add(p.split("/")[0])
    if has_root_file or len(first_components) != 1:
        return ""
    return first_components.pop()


def extract_zip_selected(
    zip_path: Path,
    dest: Path,
    include_res: list[str],
    strip_prefix: str = "",
    extra_root_files: list[str] | None = None,
) -> int:
    """Extract a subset of a zip archive into *dest*.

    The archive's single top-level directory (when present) is stripped first,
    then a member is kept when its relative path matches any regex in
    *include_res* (or, for files directly at the archive root, any regex in
    *extra_root_files*).  *strip_prefix* is additionally removed from every
    kept relative path.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    include_rx = [re.compile(r, re.IGNORECASE) for r in include_res]
    extra_rx = [re.compile(r, re.IGNORECASE) for r in (extra_root_files or [])]
    dest_resolved = str(dest.resolve())

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        root = _detect_zip_root(names)
        extracted = 0
        for name in names:
            p = _norm_zpath(name)
            rel = p
            if root:
                if p == root:
                    continue
                if p.startswith(root + "/"):
                    rel = p[len(root) + 1 :]
                else:
                    continue

            matched = any(rx.search(rel) for rx in include_rx)
            if not matched and extra_rx and "/" not in rel:
                matched = any(rx.search(rel) for rx in extra_rx)
            if not matched:
                continue

            out_rel = rel
            if strip_prefix and out_rel.startswith(strip_prefix):
                out_rel = out_rel[len(strip_prefix) :]
            out = (dest / out_rel).resolve()
            if not str(out).startswith(dest_resolved + ("\\", "/")[0]):
                raise RuntimeError(f"unsafe zip path in {zip_path}: {name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            extracted += 1
    return extracted


def verify_files(base: Path, patterns: list[str]) -> list[tuple[str, Path | None]]:
    """Return (pattern, first_match) for each glob pattern under *base*."""
    results = []
    for pat in patterns:
        matches = sorted(p for p in Path(base).rglob(pat) if p.is_file())
        results.append((pat, matches[0] if matches else None))
    return results


def report_verify(title: str, base: Path, patterns: list[str]) -> list[tuple[str, Path | None]]:
    results = verify_files(base, patterns)
    log(f"[{title}] key files:")
    for pat, found in results:
        if found:
            log(f"  OK   {pat:38s} -> {found.relative_to(base)} ({human(found.stat().st_size)})")
        else:
            log(f"  MISS {pat:38s} -> NOT FOUND")
    return results


def dir_total(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# individual tasks
# --------------------------------------------------------------------------- #
def task_fsr(force: bool = False) -> dict:
    dest = THIRD_PARTY / "fidelityfx"
    marker = dest / "include" / "FidelityFX" / "host" / "ffx_fsr2.h"
    if marker.exists() and not force:
        log("[fsr] already present, skip")
        return _done("fsr", dest, "v1.1.4", "skipped (present)")

    repo = "GPUOpen-LibrariesAndSDKs/FidelityFX-SDK"
    tag, asset = find_asset(repo, tag="v1.1.4", name_regex=r"FidelityFX-SDK-v1\.1\.4\.zip")
    zip_path = TMP / "fidelityfx.zip"
    log(f"[fsr] downloading {asset['name']} ({human(asset['size'])}) from {repo} {tag}")
    download(asset["browser_download_url"], zip_path)
    log("[fsr] extracting sdk/include + sdk/src ...")
    n = extract_zip_selected(
        zip_path,
        dest,
        include_res=[r"^sdk/(include|src)/", r"^sdk/LICENSE(\.txt)?$"],
        strip_prefix="sdk/",
        extra_root_files=[r"^LICENSE(\.txt)?$", r"^readme\.md$"],
    )
    zip_path.unlink(missing_ok=True)
    log(f"[fsr] extracted {n} files")
    report_verify(
        "fsr",
        dest,
        [
            "include/FidelityFX/host/ffx_fsr1.h",
            "include/FidelityFX/host/ffx_fsr2.h",
            "include/FidelityFX/host/ffx_fsr3.h",
            "include/FidelityFX/host/ffx_fsr3upscaler.h",
            "src/components/fsr2/*",
            "src/components/fsr3/*",
        ],
    )
    return _done("fsr", dest, tag, "ok", n_files=n)


def task_xess(force: bool = False) -> dict:
    dest = THIRD_PARTY / "xess"
    marker = dest / "bin"
    if marker.exists() and not force:
        log("[xess] already present, skip")
        return _done("xess", dest, "?", "skipped (present)")

    repo = "intel/xess"
    tag, asset = find_asset(repo, name_regex=r"XeSS_SDK_.*\.zip")
    zip_path = TMP / "xess.zip"
    log(f"[xess] downloading {asset['name']} ({human(asset['size'])}) from {repo} {tag}")
    download(asset["browser_download_url"], zip_path)
    log("[xess] extracting inc/include + lib + bin ...")
    n = extract_zip_selected(
        zip_path,
        dest,
        include_res=[r"^(inc|include)/", r"^lib/", r"^bin/"],
        extra_root_files=[r"^(LICENSE|EULA|README).*", r"\.txt$", r"\.md$"],
    )
    zip_path.unlink(missing_ok=True)
    log(f"[xess] extracted {n} files")
    report_verify(
        "xess",
        dest,
        [
            "inc/xess/xess.h",
            "inc/xess/xess_vk.h",
            "inc/xess/xess_debug.h",
            "bin/libxess.dll",
            "lib/libxess.lib",
        ],
    )
    return _done("xess", dest, tag, "ok", n_files=n)


def task_streamline(force: bool = False) -> dict:
    dest = THIRD_PARTY / "streamline"
    marker = dest / "bin"
    if marker.exists() and not force:
        log("[streamline] already present, skip")
        return _done("streamline", dest, "?", "skipped (present)")

    repo = "NVIDIA-RTX/Streamline"
    tag, asset = find_asset(repo, name_regex=r"streamline-sdk.*\.zip")
    zip_path = TMP / "streamline.zip"
    log(f"[streamline] downloading {asset['name']} ({human(asset['size'])}) from {repo} {tag}")
    download(asset["browser_download_url"], zip_path)
    log("[streamline] extracting include + lib + bin ...")
    n = extract_zip_selected(
        zip_path,
        dest,
        include_res=[r"^include/", r"^lib/", r"^bin/"],
        extra_root_files=[r"^license\.txt$", r"^README\.md$", r"3rd-party-licenses\.md$"],
    )
    zip_path.unlink(missing_ok=True)
    log(f"[streamline] extracted {n} files")
    report_verify(
        "streamline",
        dest,
        [
            "include/sl.h",
            "include/sl_dlss.h",
            "lib/**/sl.interposer.lib",
            "bin/**/sl.interposer.dll",
            "bin/**/sl.common.dll",
            "bin/**/sl.dlss.dll",
            "bin/**/nvngx_dlss.dll",
            "bin/**/development/nvngx_dlss.dll",
        ],
    )
    return _done("streamline", dest, tag, "ok", n_files=n)


def task_sgsr(force: bool = False) -> dict:
    dest = THIRD_PARTY / "snapdragon-gsr"
    if dest.exists() and any(dest.iterdir()) and not force:
        log("[sgsr] already present, skip")
        return _done("sgsr", dest, "git main", "skipped (present)")
    log("[sgsr] git clone (shallow) SnapdragonGameStudios/snapdragon-gsr")
    git_clone("https://github.com/SnapdragonGameStudios/snapdragon-gsr", dest)
    sha = _git_head(dest)
    report_verify("sgsr", dest, ["*.hlsl", "*.h", "*.fx", "*.cs", "README*", "LICENSE*"])
    return _done("sgsr", dest, f"git {sha[:8] if sha else 'main'}", "ok")


def task_nss(force: bool = False) -> dict:
    dest = THIRD_PARTY / "arm-nss"
    dest.mkdir(parents=True, exist_ok=True)

    # 1) Arm NSS SDK source
    sdk_dest = dest  # SDK repo lands directly in arm-nss/
    sdk_marker = sdk_dest / ".git"
    if sdk_marker.exists() or (sdk_dest.exists() and (sdk_dest / "README.md").exists()):
        log("[nss] SDK already present, skip")
    else:
        log("[nss] git clone (shallow) arm/neural-graphics-sdk-for-game-engines")
        git_clone("https://github.com/arm/neural-graphics-sdk-for-game-engines", sdk_dest)

    # 2) HuggingFace prebuilt VGF models (git-lfs)
    models_dest = dest / "models"
    if models_dest.exists() and any(models_dest.iterdir()) and not force:
        log("[nss] models already present, skip")
    else:
        log("[nss] git clone (shallow, lfs) huggingface.co/Arm/neural-super-sampling")
        git_clone("https://huggingface.co/Arm/neural-super-sampling", models_dest)

    # 3) Vulkan ML emulation layer
    emu_dest = dest / "emulation-layer"
    if emu_dest.exists() and any(emu_dest.iterdir()) and not force:
        log("[nss] emulation-layer already present, skip")
    else:
        log("[nss] git clone (shallow) arm/ai-ml-emulation-layer-for-vulkan")
        git_clone("https://github.com/arm/ai-ml-emulation-layer-for-vulkan", emu_dest)

    report_verify(
        "nss/models",
        models_dest,
        ["nss_v1_0_1_high_int8.vgf", "nss_v1_0_1_mid_low_int8.vgf", "*.vgf", "LICENSE*"],
    )
    report_verify("nss/sdk", sdk_dest, ["README*", "LICENSE*", "*.cpp", "*.h"])
    report_verify("nss/emulation-layer", emu_dest, ["README*", "LICENSE*", "*.cpp", "*.h"])

    return _done("nss", dest, "see subdirs", "ok")


def task_sponza(force: bool = False) -> dict:
    dest = ASSETS / "sponza"
    marker = dest / "Sponza.gltf"
    if marker.exists() and not force:
        log("[sponza] already present, skip")
        return _done("sponza", dest, "Khronos glTF-Sample-Assets main", "skipped (present)")

    clone_dir = TMP / "gltf-sample-assets"
    log("[sponza] sparse-cloning KhronosGroup/glTF-Sample-Assets (Models/Sponza)")
    if not (clone_dir / ".git").exists():
        rmtree_force(clone_dir)
        git_clone(
            "https://github.com/KhronosGroup/glTF-Sample-Assets",
            clone_dir,
            sparse_paths=["Models/Sponza"],
        )

    src = clone_dir / "Models" / "Sponza"
    dest.mkdir(parents=True, exist_ok=True)
    gltf_dir = src / "glTF"
    log(f"[sponza] copying {gltf_dir} -> {dest}")
    for p in gltf_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, dest / p.name)
        elif p.is_dir():
            shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
    for name in ("LICENSE.md", "README.md", "metadata.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    report_verify(
        "sponza",
        dest,
        ["Sponza.gltf", "Sponza.bin", "*.jpg", "*.png", "LICENSE.md"],
    )
    return _done("sponza", dest, "Khronos glTF-Sample-Assets main", "ok")


def _git_head(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _done(name: str, dest: Path, version: str, status: str, n_files: int | None = None) -> dict:
    return {
        "task": name,
        "dest": str(dest),
        "version": version,
        "status": status,
        "n_files": n_files,
        "size_bytes": dir_total(dest) if dest.exists() else 0,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="comma-separated subset of tasks: " + ",".join(ALL_TASKS),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch even when the target already exists",
    )
    args = parser.parse_args()

    tasks = ALL_TASKS
    if args.only:
        tasks = [t.strip() for t in args.only.split(",") if t.strip()]
        unknown = [t for t in tasks if t not in ALL_TASKS]
        if unknown:
            log(f"unknown task(s): {unknown}")
            return 2

    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    results = []
    for name in tasks:
        log(f"\n=== {name} ===")
        try:
            fn = {
                "fsr": task_fsr,
                "xess": task_xess,
                "streamline": task_streamline,
                "sgsr": task_sgsr,
                "nss": task_nss,
                "sponza": task_sponza,
            }[name]
            results.append(fn(force=args.force))
        except Exception as exc:  # noqa: BLE001 - keep going on other tasks
            log(f"[{name}] FAILED: {exc}")
            results.append({"task": name, "status": f"FAILED: {exc}", "version": "-"})

    # cleanup temporary archive/clone directory
    rmtree_force(TMP)

    log("\n\n===== SUMMARY =====")
    total = 0
    for r in results:
        size = r.get("size_bytes", 0)
        total += size
        log(
            f"  {r['task']:12s} {r['version']:28s} {r['status']:16s} "
            f"{human(size):>10s}  files={r.get('n_files','-')}"
        )
    log(f"  {'TOTAL':12s} {'':28s} {'':16s} {human(total):>10s}")

    failed = [r for r in results if str(r.get("status", "")).startswith("FAILED")]
    if failed:
        log(f"{len(failed)} task(s) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
