# DLSS 5 Neural Rendering (DLSSNR) spike

Technical-validation spike for the leaked DLSS 5 neural-rendering NGX feature
(feature id 18, `NVSDK_NGX_Feature_Reserved18`, runtime `nvngx_dlssnr.dll`
310.8.0.0) on an RTX 4070 SUPER, driver 610.62, Windows.

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

## Run

```bat
dlss5_spike --api vulkan  --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --out out_vk
dlss5_spike --api d3d12   --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --out out_d3d12
dlss5_spike --api slprobe --dll <nvngx_dlssnr.dll to stage>          --out out_sl
```

Flags: `--no-iat-patch` (skip the GetModuleFileNameW IAT patch), `--spy`
(IAT-trace the plugin's Win32 calls during Init).

## Results on this machine (2026-08-31): blocked

Every route below was implemented and run against both the stock signed DLL
(SHA-256 `E16BCF15…`, Authenticode valid) and the community `310.8.SF`
patched build (RankFTW/rhi-repo `nvngx_dlssnr_310.8.SF-v2.zip`). Outcomes are
identical for both builds.

| Route | Result |
|---|---|
| Driver NGX core (`nvsdk_ngx_d` glue → `nvngx_dlss.dll` → plugin) | Init OK; `CreateFeature1(18)` → `0xBAD0000C` (OutOfDate) |
| Direct plugin, Vulkan (`Init_Ext2` etc.) | `0xBAD00002` (PlatformError) |
| Direct plugin, D3D12 (renodx-dlss5 recipe + IAT patch) | `0xBAD00002` (PlatformError) |
| Streamline (`sl.interposer` + `sl.dlss_nr.dll`) | `slInit` OK; NR requirements query → `0xBAD0000C`; `slSetFeatureLoaded(1004)` → `eErrorFeatureFailedToLoad` |
| NGX OTA (`nvngx_update.exe -feature dlssnr`, also `-feature dlss`) | `curl status=-1` — server unreachable |

### Why

1. **The shipping driver's NGX runtime has no feature-18 entry.** The NGX
   core log (`nvngx.log`) shows it probing a fixed list of known feature
   modules (`nvngx_dlss`, `nvngx_dlssg`, `nvngx_dlssd`, …) — `nvngx_dlssnr`
   is never probed, and `CreateFeature_Validate` logs *"required feature is
   not supported by NGX runtime, please update display driver"*. `dlss`/
   `dlssg` snippets validate and load fine through the same core, so the
   runtime itself works — feature 18 is simply absent from its table. New
   features are meant to arrive via NGX OTA, which is unreachable here
   (`curl status=-1` even for `dlss`).
2. **The plugin cannot init standalone.** `nvngx_dlssnr.dll` (like all 310.x
   NGX DLLs) exports only the "snippet" ABI — no `AllocateParameters` /
   `NVSDK_NGX_Parameter_Set*` (the spike implements the
   `NVSDK_NGX_Parameter` vtable itself, class `SpikeParameterMap`). An IAT
   spy run shows its `Init` fails after zero registry/file/module calls: it
   expects in-process runtime state that only a real host loader (NGX core
   or Streamline) sets up. The renodx-dlss5 add-on's `GetModuleFileNameW`
   IAT patch alone does not change the result.
3. **Streamline knows the feature but defers to the same wall.** sl.log
   identifies DLSS NR as `kFeatureDLSS_NR` (**sl::Feature id 1004**), but
   `sl.dlss_nr.dll` still asks the NGX runtime for feature-18 requirements,
   gets `0xBAD0000C`, and reports *"DLSS-NR feature is not supported. Please
   check if you have a valid nvngx_dlssnr.dll or your driver supports
   DLSS-NR."*
4. The `310.8.SF` patch does not help here: whatever it patches is
   downstream of the driver-runtime feature check that fails first.

### What would unblock it

- A driver whose NGX runtime advertises `dlssnr` (the DLSS 5 launch driver
  branch), or working NGX OTA connectivity so the runtime can download the
  feature definition, or
- the full renodx-dlss5 inline-hook stack (the add-on patches the plugin
  in-memory beyond a single IAT entry) — out of scope for this spike.

## Implementation notes

- `spike/dlss5/main.cpp` contains all three modes plus the Vulkan/D3D12
  test-image pipeline (1280×720 R16G16B16A16 color, R16G16 motion,
  D32/R32 depth), stats + PNG readback via stb, SEH-wrapped NGX calls, and
  detailed hex logging of every NGX result.
- DLSSNR parameter keys used (confirmed against `renodx-dlss5.addon64`
  strings): `DLSSNR.{Color,Backbuffer,MVec,Depth,Output,Intensity,Reset,
  Enabled,MVecScaleX,MVecScaleY,DepthInverted,Width,Height,InputWidth,
  InputHeight,OutputWidth,OutputHeight,ScalingRatio}`.
- NGX feature-18 Vulkan requirements (when queried directly from the
  plugin): instance `VK_KHR_get_physical_device_properties2`; device
  `VK_NVX_binary_import`, `VK_NVX_image_view_handle`,
  `VK_KHR_buffer_device_address`, `VK_KHR_push_descriptor`.
- NGX headers are vendored under `third_party/ngx/include` (NVIDIA DLSS SDK,
  same license posture as the vendored Streamline SDK).
