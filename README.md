# sr_compare — Super-Resolution Comparison Tool

A Vulkan renderer for side-by-side quality and performance comparison of
upscaling algorithms, running the same scene under the same camera path.

## Algorithms

| Algorithm | Type | Source / License |
|---|---|---|
| TAA (own baseline) | temporal | this repo |
| FSR 1 / 2 / 3.1 | spatial / temporal | AMD FidelityFX SDK 1.1.4 (MIT) |
| Intel XeSS | ML (DP4a) | intel/xess (closed DLL, Intel Simplified License) |
| NVIDIA DLSS 4.5 (Preset K / L / M) | ML (Transformer) | Streamline SDK (framework MIT; nvngx_dlss.dll closed, needs an NVIDIA Developer applicationId) |
| Arm NSS v1.0.1 (high/mid_low int8) | ML (mobile) | arm/neural-graphics-sdk-for-game-engines (MIT) + HF models (Arm AI Model Community License) |
| SGSR 1 / 2 | spatial / temporal (mobile) | SnapdragonGameStudios/snapdragon-gsr (BSD-3-Clause) |

Arm NSS runs on PC through the Vulkan ML Emulation Layer (software inference);
its performance numbers are not representative of real hardware.

## Renderer

- Deferred PBR: Cook-Torrance GGX, two point lights, full IBL (irradiance +
  prefiltered specular + BRDF LUT), normal/ORM/AO/emissive maps, alphaMode
  MASK cutouts, mipmaps + 16x anisotropy, frustum culling, merged scene buffers
- SSAO (16-sample normal-hemisphere + cross blur, ~0.5 ms @1080p)
- Forward transparency pass (after lighting, before upscaling): dielectric
  glass model (Fresnel-driven opacity), back-to-front sorted, writes camera
  motion and a translucent coverage mask
- The coverage mask is wired into every upscaler interface that officially
  supports it: TAA (history weight), FSR2/FSR3 (reactive + T&C mask), DLSS
  (bias-current-color / reactive / transparency hints), XeSS (responsive
  pixel mask). NSS and SGSR2 have no such input.
- Ground truth: native display resolution, optional 200% SSAA (4K→1080p)

## Modes

- **gui** — ImGui front end with Viewer / Compare / Bench tabs. Scene and
  algorithm changes load asynchronously on a worker thread with a progress
  overlay; the current frame keeps rendering. Rebuilds are tiered: switching
  only algorithms does not reload the scene.
- **viewer** — free-fly single-algorithm fullscreen preview.
- **compare** — split-screen columns: native GT on the left, one column per
  algorithm, live PSNR/SSIM against GT; aspect-preserving crop, wheel zoom
  (metrics recomputed for the visible region), optional GT 200% SSAA.
- **bench** — fixed camera path, fixed frame count; GPU time of the upscale
  pass, frame time/FPS (avg/median/1% low), VRAM (VK_EXT_memory_budget);
  CSV/JSON export.

![Viewer](docs/screenshot_viewer.png)

![Compare](docs/screenshot_compare.png)

```bat
sr_compare gui
sr_compare viewer --scene sponza --upscaler fsr2
sr_compare compare --scene boxes --upscalers taa,fsr2,xess,dlss-m --render-scale 0.5
sr_compare bench --scene sponza --frames 300 --warmup 60
sr_compare viewer --list-upscalers
```

## Scenes (`--scene`)

- `boxes` (alias `procedural`): built-in procedural box room, no assets needed
- `sponza`: Sponza atrium — Khronos glTF sample of Crytek Sponza
- `bistro_exterior` / `bistro_interior`: Amazon Lumberyard Bistro (NVIDIA ORCA)
- `<any glTF path>`: load a custom scene

Assets under `assets/` are not committed. After a fresh clone, fetch Sponza
and (optionally) convert Bistro as below.

### Sponza

Source: [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
`Models/Sponza/glTF/`. Model files: Cryengine Limited License Agreement;
docs/metadata: CC-BY-4.0.

Already a loadable glTF — no conversion. From the repo root:

```bat
python scripts/fetch_sdks.py --only sponza
```

That sparse-clones the Khronos repo and copies the glTF folder into
`assets/sponza/` (`Sponza.gltf` + `Sponza.bin` + textures + license files).

### Bistro

Source: [Amazon Lumberyard Bistro, NVIDIA ORCA](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)
(CC-BY 4.0). Download the FBX package from that page and unpack it. The
layout this repo expects is:

```
Bistro_v5_2/
  BistroExterior.fbx
  BistroInterior.fbx
  Textures/          :: DDS: *_BaseColor, *_Normal (DirectX Y-),
                     ::       *_Specular (actually ORM: R=AO, G=Roughness, B=Metalness),
                     ::       *_Emissive
```

Needs **Blender 5.1+** (native C++ FBX importer; the old Python addon crashes
on this file's lights) and its bundled NumPy. From the repo root (Git Bash
example; adjust the Blender and unpack paths):

```bash
blender=/c/Program\ Files/Blender\ Foundation/Blender\ 5.1/blender.exe
src=D:/Code/Bistro_v5_2

"$blender" --background --factory-startup --python scripts/convert_bistro_v2.py -- \
    --input "$src/BistroExterior.fbx" --textures "$src/Textures" \
    --output assets/bistro --name BistroExterior

"$blender" --background --factory-startup --python scripts/convert_bistro_v2.py -- \
    --input "$src/BistroInterior.fbx" --textures "$src/Textures" \
    --output assets/bistro --name BistroInterior

python scripts/fixup_bistro_alpha.py
```

`convert_bistro_v2.py` writes `GLTF_SEPARATE` (`.gltf` + `.bin` + PNG) with
full PBR: baseColor+alpha, G-flipped normals (DirectX → glTF), ORM as
metallicRoughness + occlusion, emissive, MASK cutouts. Then
`fixup_bistro_alpha.py` (plain Python, no Blender) corrects the remaining
alpha-mode mistakes: uniform-zero MASK → OPAQUE, real glass → BLEND with a
constant alpha, and false BLEND (opaque FBX flags) → OPAQUE. The result is
`assets/bistro/BistroExterior.gltf` and `BistroInterior.gltf`.

## Building

```bat
build_all.bat          :: one-click configure + Release build
```

Requirements: Windows 11, Visual Studio (C++ workload; discovered via
vswhere), CMake >= 3.24 (on PATH or the VS-bundled one), Vulkan SDK 1.4+
(`VULKAN_SDK` provides glslangValidator), Python 3.10+ (only for
`scripts/fetch_sdks.py` and `metrics/`).

DLSS needs a free NVIDIA Developer applicationId (env `SR_DLSS_APP_ID`).

### Moving the tree to another machine

The source tree is fully self-contained (third_party, assets, shaders; no
hardcoded paths in CMake) — copy the whole folder, install the requirements
above, run `build_all.bat`. `build*` directories are machine-local caches and
can be deleted, with one exception: `build-nss-emu/` holds the NSS emulation
layer's dependencies (git clones with applied patches). If it was deleted,
run `scripts\nss_emu_fetch_deps.bat` (git + network), then
`scripts\nss_emu_configure.bat` and `scripts\nss_emu_build.bat`
(also needs Ninja + Python 3). Without it the main build still works; the NSS
plugin simply disables itself.

## Packaging

`build/app/Release/` is self-contained after a build — copy it anywhere and
run (no source or build tools needed):

- `sr_compare.exe` + all runtime DLLs (XeSS / Streamline / DLSS / NSS, CRT)
- `run_gui.bat` (one-click GUI)
- `shaders/`, `assets/` (Bistro full-PBR ≈ 1.3 GB, trim as needed), `nss-emu/`

Assets and shaders resolve relative to the exe, so the package works from any
working directory. Target machine: a Vulkan 1.3 capable driver; DLSS requires
an RTX 20+ GPU (other algorithms are unaffected without one).

## Directory layout

- `renderer/` — Vulkan renderer core (context/frame loop/glTF scene/deferred
  GBuffer/lighting/transparency/SSAO/IBL/camera paths)
- `upscalers/` — unified upscaler plugin layer (`IUpscaler.h` is the contract,
  one subdirectory per algorithm)
- `compare/`, `bench/`, `gui/`, `app/` — front ends and entry point
- `metrics/` — offline metric verification + Markdown reports (Python)
- `scripts/` — SDK download, Bistro conversion, NSS emu layer builds
- `third_party/` — SDKs (fetched by script, not committed)

## License

This repository's original source is MIT (see `LICENSE`). Third-party SDKs
under `third_party/` and scene assets under `assets/` keep their own
licenses — see `third_party/README.md` and the scene notes above.

## Notes

- Frame generation (FSR3-FG / XeSS-FG / DLSS-FG) is out of scope.
- SGSR targets mobile Adreno; PC performance is indicative only.
- Transparent surfaces are rendered at render resolution and upscaled with the
  frame (the industry-standard structure); per-pixel refraction/transmission
  is intentionally not implemented.
