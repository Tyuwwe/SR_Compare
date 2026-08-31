# DLSS 5 Neural Rendering (DLSSNR) spike

Technical-validation spike for the leaked DLSS 5 neural-rendering NGX feature
(feature id 18, `NVSDK_NGX_Feature_Reserved18`, runtime `nvngx_dlssnr.dll`
310.8.0.0) on an RTX 4070 SUPER, Windows. Tested on drivers 610.62 and
616.56 (see "Results" below — still blocked on both, but the failure moved
one level deeper with 616.56).

This is a research spike, not a product integration. Nothing here links the
renderer; the exe is self-contained and loads all NGX/Streamline DLLs at
runtime. **No NVIDIA DLL is committed to the repo** (see `.gitignore`).

## Build

The `dlss5_spike` target is part of the normal CMake build; the exe lands in
`build/spike/dlss5/Release/`. Copy the runtime DLLs next to it before running
(the exe dir needs `nvngx_dlssnr.dll`, `nvngx_dlss.dll` and, for `--api
slprobe`, the `sl.*.dll` set):

```bat
copy "D:\Code\SL 2.13\*.dll" build\spike\dlss5\Release\
```

`--api glue` additionally requires `nvsdk_ngx_d.lib` from the NVIDIA DLSS
SDK (not committed). Point CMake at it with
`-DDLSS5_NGX_SDK_LIB_DIR=<sdk>/lib/Windows_x86_64/x64`; if the lib is not
found the other modes still build and `--api glue` reports it is disabled.

## Run

```bat
dlss5_spike --api vulkan  --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --out out_vk
dlss5_spike --api d3d12   --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --out out_d3d12
dlss5_spike --api glue    --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --core "D:\Code\SL 2.13\nvngx_dlss.dll" --out out_glue
dlss5_spike --api slprobe --dll <nvngx_dlssnr.dll to stage>          --out out_sl
```

Flags: `--core <nvngx_dlss.dll>` (glue mode: stage a specific core next to
the exe), `--no-iat-patch` (skip the GetModuleFileNameW IAT patch), `--spy`
(IAT-trace Win32 calls; in glue mode this traces the NGX core itself during
Init/CreateFeature1).

## Results on driver 616.56 (2026-08-31): still blocked, one level deeper

The 616.56 driver update changed the core-route answers — the driver NGX
runtime now *partially* knows feature 18 — but feature creation still
fails, and the runtime still never loads `nvngx_dlssnr.dll`. Outcomes are
identical for the stock signed DLL and the community `310.8.SF` build.

| Route | 610.62 | 616.56 |
|---|---|---|
| glue: `GetFeatureDeviceExtensionRequirements(18)` | `0xBAD0000C` OutOfDate | **Success** (returns the 4 NGX Vulkan device exts) |
| glue: `GetFeatureRequirements(18)` | `0xBAD0000C` OutOfDate | `0xBAD00012` NotImplemented |
| glue: `SuperSampling.Available` capability | 0 | 1 |
| glue: `CreateFeature1(18)`, all param sets | `0xBAD0000C` OutOfDate | `0xBAD0000B` UnableToInitializeFeature |
| Direct plugin, Vulkan / D3D12 init | `0xBAD00002` / `0xBAD00001` | unchanged (`0xBAD00002` / `0xBAD00001`) |
| Streamline `sl.dlss_nr` requirements query | `0xBAD0000C` → "not supported" | "ngxResult not implemented" → still "DLSS-NR feature is not supported" |
| NGX OTA `nvngx_update.exe -feature dlssnr` | `curl status=-1` (unreachable) | `-api update` → `curl status=35` SSL connect error (still unreachable) |

### Why it is still blocked

1. **The runtime never probes `nvngx_dlssnr.dll`.** Module enumeration after
   glue Init shows the driver's `_nvngx.dll` (loaded from the 616.56
   DriverStore) plus the app-dir `nvngx_dlss.dll`/`nvngx_dlssg.dll` plugins
   — but never `nvngx_dlssnr.dll`. An IAT spy on both core modules
   (`LoadLibrary*`, `CreateFileW`, registry) records **zero** Win32 activity
   during all four `CreateFeature1(18)` attempts: the feature-18
   initializer fails purely internally. The DLSSNR payload (plugin +
   models) is presumably expected through NGX OTA into the managed store —
   `C:\ProgramData\NVIDIA\NGX\models` has `dlss`/`dlssd`/`dlssg`/`dlisp`
   payloads but nothing for dlssnr, and OTA is unreachable from this
   network (SSL connect error), same as on 610.62.
2. **The plugin still cannot init standalone** (`0xBAD00002`, unchanged) —
   it expects in-process runtime state from its host loader.
3. The `310.8.SF` patch does not help: the core never reaches plugin code,
   so a plugin-side patch cannot matter.
4. `nvngx.log` could not be enabled on 616.56: NGX file logging needs
   `HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore` values (`LogLevel`,
   …), the driver reinstall wiped the ones used for the 610.62 logs, and
   this shell has no admin rights (HKCU values are ignored). The
   module-enumeration + IAT-spy evidence above substitutes for the log.

### What would unblock it

- Working NGX OTA connectivity (so the runtime can fetch the dlssnr payload
  the 616.56 runtime now has a table entry for), or a driver/runtime drop
  that ships the DLSSNR initializer itself, or
- the full renodx-dlss5 inline-hook stack (the add-on patches the plugin
  in-memory beyond a single IAT entry) — out of scope for this spike.

## Results on driver 610.62 (2026-08-31, historical): blocked

Every route below was run against both the stock signed DLL (SHA-256
`E16BCF15…`, Authenticode valid) and the community `310.8.SF` patched build
(RankFTW/rhi-repo `nvngx_dlssnr_310.8.SF-v2.zip`), with identical outcomes.

| Route | Result |
|---|---|
| Driver NGX core (`nvsdk_ngx_d` glue → `nvngx_dlss.dll` → plugin) | Init OK; `CreateFeature1(18)` → `0xBAD0000C` (OutOfDate) |
| Direct plugin, Vulkan (`Init_Ext2` etc.) | `0xBAD00002` (PlatformError) |
| Direct plugin, D3D12 (renodx-dlss5 recipe + IAT patch) | `0xBAD00002` (PlatformError) |
| Streamline (`sl.interposer` + `sl.dlss_nr.dll`) | `slInit` OK; NR requirements query → `0xBAD0000C`; `slSetFeatureLoaded(1004)` → `eErrorFeatureFailedToLoad` |
| NGX OTA (`nvngx_update.exe -feature dlssnr`, also `-feature dlss`) | `curl status=-1` — server unreachable |

1. **The 610.62 driver's NGX runtime had no feature-18 entry.** The NGX
   core log (`nvngx.log`) showed it probing a fixed list of known feature
   modules (`nvngx_dlss`, `nvngx_dlssg`, `nvngx_dlssd`, …) — `nvngx_dlssnr`
   was never probed, and `CreateFeature_Validate` logged *"required feature
   is not supported by NGX runtime, please update display driver"*.
   `dlss`/`dlssg` snippets validated and loaded fine through the same core.
2. **The plugin cannot init standalone** (same as 616.56, see above).
3. **Streamline knows the feature but defers to the same wall.** sl.log
   identifies DLSS NR as `kFeatureDLSS_NR` (**sl::Feature id 1004**), but
   `sl.dlss_nr.dll` still asks the NGX runtime for feature-18 requirements
   and reports it unsupported.

## Implementation notes

- `spike/dlss5/main.cpp` contains all four modes plus the Vulkan/D3D12
  test-image pipeline (1280×720 R16G16B16A16 color, R16G16 motion,
  D32/R32 depth), stats + PNG readback via stb, SEH-wrapped NGX calls, and
  detailed hex logging of every NGX result.
- `--api glue` links `nvsdk_ngx_d.lib` (the NGX SDK client glue — a real
  static lib with the client-side init the 310.x snippet ABI expects, not a
  plain import lib). Note the decisive runtime component is the **driver's**
  `_nvngx.dll` (NGXCore registry → DriverStore), which the glue loads
  in-process; the app-dir `nvngx_dlss.dll` is only the DLSS feature plugin.
  That is why the 610.62→616.56 driver update changed the answers while the
  staged DLLs stayed the same.
- DLSSNR parameter keys used (confirmed against `renodx-dlss5.addon64`
  strings): `DLSSNR.{Color,Backbuffer,MVec,Depth,Output,Intensity,Reset,
  Enabled,MVecScaleX,MVecScaleY,DepthInverted,Width,Height,InputWidth,
  InputHeight,OutputWidth,OutputHeight,ScalingRatio}`.
- NGX feature-18 Vulkan requirements: instance
  `VK_KHR_get_physical_device_properties2`; device `VK_NVX_binary_import`,
  `VK_NVX_image_view_handle`, `VK_KHR_buffer_device_address`,
  `VK_KHR_push_descriptor` (the device list query fails on 610.62 and
  succeeds on 616.56).
- NGX headers are vendored under `third_party/ngx/include` (NVIDIA DLSS SDK,
  same license posture as the vendored Streamline SDK).
