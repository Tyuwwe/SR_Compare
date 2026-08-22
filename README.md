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
| FSR3 frame interpolation | frame gen (GUI) | AMD FidelityFX SDK 1.1.4 (MIT); dispatch to texture, no FI swapchain |
| Arm NFRU | frame gen (GUI) | same Neural Graphics SDK + ML emu layer as NSS |

Arm NSS runs on PC through the Vulkan ML Emulation Layer (software inference);
its performance numbers are not representative of real hardware.

## Renderer

- Deferred PBR: Heitz height-correlated GGX + Hammon 2017 diffuse (no Lambert),
  punctual lights, full IBL (irradiance + prefiltered specular + BRDF LUT) with
  Fdez-Aguera multi-scatter compensation, normal/ORM/AO/emissive maps,
  alphaMode MASK cutouts, mipmaps + 16x anisotropy, merged scene buffers
- Sky: Hillaire 2020 atmosphere (transmittance/multiscatter LUTs) rendered
  into the IBL env cube, sun direction driven by the scene preset (CLI
  `--sun-elev` / `--sun-az`); `--env-map <hdr>` swaps in a static equirect
  HDR instead.  Bistro's golden hour gets its sun color from the atmosphere
- Clustered shading: 64 px tiles × 24 exponential depth slices, up to 1024
  point + spot lights in an SSBO (≤64 per cluster), per-cluster assignment on
  the GPU; the CSM sun bypasses the clusters
- Shadows: CSM sun (4 × 2048² cascades, bounding-sphere stabilized, IGN
  dithered transitions), spot shadow atlas (4096², 16 × 1024² tiles, per-frame
  importance selection), screen-space contact shadows for the sun
  (`--no-contact-shadows`)
- SSR: full-screen opaque Hi-Z cone march sampling the color mip by roughness,
  energy-conserving replacement of the IBL specular term, temporal EMA
  accumulation (off by default; `--ssr` enables, `--ssr-strength <0..1>`
  scales the weight, default 0.6).  Reflection fallback chain: SSR → baked local
  probes (box projection, two-probe blending; bake with `--bake-probes`) → env
- GTAO (XeGTAO-style depth-mip sampling + temporal accumulation; Jimenez 2016
  3 slices × 3 steps/side + 5×5 depth-aware denoise)
- Auto exposure: log-luminance histogram + smoothed EV, fixed-dt deterministic,
  feeds TAA/upscaler preExposure; CLI `--exposure <f>` switches to manual
- TAA: YCoCg history clipping, depth-based disocclusion, adaptive alpha, RCAS
  sharpening
- Volumetric fog: froxel grid (inject/light/temporal/march/composite) lit by
  the CSM + cluster lights, god rays; scene presets carry the media params
  (`--no-volfog`)
- Post chain: thresholded bloom pyramid (13-tap down / tent up; on by default,
  `--no-bloom`), HDR motion blur (McGuire 2012 tile-max gather) and depth of
  field (UE4 scatter-as-gather CoC, centre-texel autofocus) — both off by
  default in every mode, opt in with `--motion-blur` / `--dof` (viewer +
  compare) or the GUI checkboxes; same algorithm and parameters on the LR, GT
  and GT-SSAA paths, deterministic frame-index-driven noise
- Color grading: simplified-ACEScc log domain (temperature/tint/contrast/
  saturation, CLI `--temperature`/`--tint`/`--contrast`/`--saturation`, GUI
  sliders) + `.cube` 3D LUT (`--lut`), mirrored on the CPU for PNG
  screenshots; terminal lens chain (chromatic aberration / dirt / vignette /
  grain, `--no-lens-fx`); HDR swapchain output probing HDR10 PQ then scRGB
  with SDR fallback (viewer `--hdr`)
- Display transform: Stephen Hill fitted ACES + gamma 2.2 (shared by present,
  compare compose, GPU metrics, and CPU screenshots)
- Forward transparency pass (after lighting, before upscaling): dielectric
  shop-window model (coated Fresnel, clip-space SSR / McGuire DDA against the
  opaque HDR, EnvBRDF composite), back-to-front sorted, writes camera motion
  and a translucent coverage mask
- The coverage mask is wired into every upscaler interface that officially
  supports it: TAA (history weight), FSR2/FSR3 (reactive + T&C mask), DLSS
  (bias-current-color / reactive / transparency hints), XeSS (responsive
  pixel mask). NSS and SGSR2 have no such input.
- Animation: glTF node trees, animations and GPU skinning (double-buffered
  joint palettes), poses sampled at frame index × fixed dt (never wall clock);
  per-object motion vectors; the boxes scene has two animated boxes
- LOD: per-mesh chains generated with meshoptimizer (+ `MSFT_lod` when
  present), screen-size selection with hysteresis, disk-cached
  (`SR_LOD=0` opts out); GPU occlusion culling against the previous frame's
  Hi-Z pyramid feeding SSBO instancing + indirect draws (`SR_OCCLUSION=0`
  opts out, GUI checkbox)
- Textures: BC7 KTX2 preferred over PNG/JPG siblings (offline transcode with
  `scripts/transcode_textures.py`, AMD Compressonator), fine-mip streaming in
  the interactive viewer/GUI (bench/compare/screenshot runs upload fully for
  determinism; `SR_TEX_STREAM=0/1` overrides)
- Infrastructure: VMA allocation, persistent pipeline cache, sync2 barriers, a
  dedicated transfer queue for one-shot uploads, and a lightweight render
  graph with transient aliasing driving the viewer frame recording
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

Common viewer/compare switches (see `sr_compare <mode>` with no args for the
full text):

```
--env-map <hdr>      static equirect HDR for IBL/skybox (default: sky atmosphere)
--sun-elev / --sun-az <deg>   override the preset sun direction
--exposure <f>       manual display exposure (disables auto exposure)
--no-shadows / --shadow-debug / --no-contact-shadows
--ssr / --ssr-strength <0..1>
                     enable opaque screen-space reflections (default off in every
                     mode; --no-ssr accepted for compatibility) and scale their
                     weight (default 0.6)
--no-volfog          disable froxel volumetric fog
--no-bloom / --no-lens-fx
--motion-blur / --dof
                     enable motion blur / depth of field (default off in every
                     mode; --no-motion-blur / --no-dof accepted for compatibility)
--dof-focus <m> / --dof-fstop <f> / --dof-max-blur <px>
                     DOF tuning (viewer + compare; focus 0 = screen-centre auto-focus)
--bake-probes        bake reflection probes to the scene's .probes file, then exit
--hdr                HDR swapchain output (HDR10 PQ or scRGB probe, SDR fallback)
--lut <file.cube>    3D LUT (17^3/33^3), log domain pre-ACES
--temperature <K> / --tint <-1..1> / --contrast <f> / --saturation <f>
```

compare additionally takes `--upscalers a,b,...`, `--gt-ssaa`, `--zoom <f>`,
`--zoom-center <u,v>`, `--metric-interval <N>`; bench takes `--upscalers`,
`--frames`, `--warmup`, `--out <csv>`.

## Runtime config (`engine.toml`)

Every mode reads an optional `engine.toml` next to the exe at startup (copy
`engine.toml.example`, which is built next to the exe and documents all keys).
Precedence: **explicit CLI flag > engine.toml > code default** — the CLI
parsers track which flags were given explicitly and toml only fills the rest.
A missing file is a silent no-op; a malformed file prints the parse error to
stderr and falls back to defaults; applied keys are echoed to stderr as
`[engine.toml] <mode>: key=value ...` so scripted runs can verify them.

Sections: `[renderer]` (render_scale, hdr, env_map), `[exposure]` (manual
exposure + auto-exposure EV range), `[effects]` (ssr/ssr_strength, shadows,
contact_shadows, volfog, bloom, motion_blur, lens_fx), `[lens_fx]` (GUI
sub-items), `[culling]` (occlusion, lod — GUI), `[dof]`, `[grading]`
(temperature/tint/contrast/saturation/lut), `[sun]` (elevation/azimuth).
Bench spawns viewer children that read the same file, so bench runs stay
deterministic under a given engine.toml.

The **GUI hot-reloads** the file (~1 s mtime poll): the per-frame options —
effects toggles, DOF/grading/sun sliders, exposure, occlusion/lod, the HDR
checkbox — apply immediately; render_scale, output resolution, env_map, scene
and lut changes require an Apply rebuild or a restart and are ignored by the
reload.

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
`scripts/fetch_sdks.py` and `metrics/`). The SDL3 windowing library comes
from `python scripts/fetch_sdks.py --only sdl3` (prebuilt VC x64 package
staged into `third_party/sdl3/`; the build copies `SDL3.dll` next to the exe).

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

- `sr_compare.exe` + all runtime DLLs (SDL3, XeSS / Streamline / DLSS / NSS, CRT)
- `sr_run_gui.bat` (one-click GUI)
- `shaders/`, `assets/` (Bistro full-PBR ≈ 1.3 GB, trim as needed), `nss-emu/`

Assets and shaders resolve relative to the exe, so the package works from any
working directory. Target machine: a Vulkan 1.3 capable driver; DLSS requires
an RTX 20+ GPU (other algorithms are unaffected without one).

## Directory layout

- `renderer/` — Vulkan renderer core (context/frame loop/glTF scene/deferred
  GBuffer/lighting/transparency/GTAO/IBL/camera paths)
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

- Frame generation in the GUI only: FSR3 optical-flow + interpolation, or Arm
  NFRU, after the upscaler (display rate ×2 of the locked render FPS). Bench
  and CLI `compare` stay super-resolution only. DLSS-G and XeSS-FG are not
  included (Present-bound / no Vulkan). NFRU on PC uses Arm's ML emulation
  layer; its times are not representative of hardware. Compare PSNR for an
  interpolator column is against a midpoint-time GT; columns without FG show `--`.
- SGSR targets mobile Adreno; PC performance is indicative only.
- Transparent surfaces are rendered at render resolution and upscaled with the
  frame (the industry-standard structure); per-pixel refraction/transmission
  is intentionally not implemented.
