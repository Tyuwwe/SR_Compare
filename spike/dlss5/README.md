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

`--api glue`/`--api glue12` additionally require `nvsdk_ngx_d.lib` from the
NVIDIA DLSS SDK (not committed). Point CMake at it with
`-DDLSS5_NGX_SDK_LIB_DIR=<sdk>/lib/Windows_x86_64/x64`; if the lib is not
found the other modes still build and the glue modes report they are
disabled.

## Run

```bat
dlss5_spike --api vulkan  --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --out out_vk
dlss5_spike --api d3d12   --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --out out_d3d12
dlss5_spike --api glue    --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --core "D:\Code\SL 2.13\nvngx_dlss.dll" --out out_glue
dlss5_spike --api glue12  --dll "D:\Code\SL 2.13\nvngx_dlssnr.dll" --core "D:\Code\SL 2.13\nvngx_dlss.dll" --out out_glue12
dlss5_spike --api slprobe --dll <nvngx_dlssnr.dll to stage>          --out out_sl
```

Flags: `--core <nvngx_dlss.dll>` (glue/glue12: stage a specific core next to
the exe), `--feature <id>` (glue12: sanity-create another feature, e.g. 1 =
DLSS SR), `--appid <id>` (glue12: NGX application id, hex ok),
`--direct-after-glue` (call the plugin's direct `Init_Ext` after the core
Init succeeded, same process), `--no-iat-patch` (skip the GetModuleFileNameW
IAT patch), `--spy` (IAT-trace Win32 calls; in glue modes this traces the NGX
core itself during Init/CreateFeature).

## Results on driver 616.56 (2026-08-31, round 2): renodx mechanism reversed, D3D12 core route tested

**New lead:** ReShade + `renodx-dlss5.addon64` successfully enables DLSS 5 NR
in Cyberpunk 2077 (D3D12) on this exact machine/driver, so the add-on's
mechanism was reverse-engineered from its binary and every newly discovered
route was tested standalone. Outcome: **still blocked** — the add-on's
success depends on in-process host state (a live game NGX/Streamline context
whose host has registered `nvngx_dlssnr.dll` into the shared NGX snippet
runtime) that no standalone route below can reproduce.

### The add-on's mechanism (static reversing of `renodx-dlss5.addon64`)

- It Detours the game's `NVSDK_NGX_D3D12_CreateFeature` / `EvaluateFeature` /
  `ReleaseFeature` / `EvaluateFeature_C` and `slEvaluateFeature`, captures the
  game's DLSS contract, and inlines feature 18.
- Feature-18 creation tries two function pointers in order, both called as
  `CreateFeature(cmdList, 0x12, params, &handle)`:
  1. **"the NGX core"** — resolved via `GetProcAddress` from the first loaded
     module of **`{_nvngx.dll, nvngx.dll, nvngx_dlss.dll, nvngx_dlssd.dll}`**;
  2. **"the signed snippet"** — resolved from `nvngx_dlssnr.dll`, which the
     add-on `LoadLibrary`s itself first (*"nvngx_dlssnr.dll was not found
     beside the addon"*).
- Its create-parameter set (wider than what we tried before): base
  `Width/Height/OutWidth/OutHeight` + `DLSS.Feature.Create.Flags` +
  `CreationNodeMask`/`VisibilityNodeMask` + `DLSSNR.{Width,Height,InputWidth,
  InputHeight,OutputWidth,OutputHeight,Output.Width,Output.Height,Scale,
  ScalingRatio,Upscaling,Hint.Render.Preset,Enabled}`.
- Its eval key set (exact names from the binary): `DLSSNR.{Color,Output,MVec,
  Depth}` + `<res>SubrectBaseX/BaseY/Width/Height` per resource +
  `MVecScaleX/Y,DepthInverted,Enabled,Reset,Intensity,LocalToneStrength,
  LocalStructureStrength,SkinStructureStrength,UseAutoMask,Style,UICorrection`.
  (`nvngx_dlssnr.dll` additionally knows optional `Backbuffer`, `ControlMask`,
  `UI`, `UIAlpha`, `BidirectionalDistortionField` inputs.)

### New mode `--api glue12` (D3D12 NGX core route)

D3D12 device → `NVSDK_NGX_D3D12_Init` (app id `0x1000000`, fallback
`Init_with_ProjectID` `a0f57b54-…`) → requirements/capability queries →
`CreateFeature(18)` with the full renodx key set → eval + PNG readback.
Extra flags: `--feature <id>` (sanity-create another feature),
`--appid <id>`, `--direct-after-glue` (direct plugin `Init_Ext` after core
Init, in the same process).

| Experiment | Result |
|---|---|
| glue12 `D3D12_Init` | **Success** |
| glue12 `GetFeatureRequirements(18)` | `0xBAD00012` NotImplemented |
| glue12 `CreateFeature(18)`, all 4 param groups incl. full renodx set | `0xBAD0000B` UnableToInitializeFeature |
| glue12 `--feature 1` (DLSS SR sanity create) | **Success** — the D3D12 core route itself works; the block is feature-18-specific |
| glue12 `--appid 0xE658700` (Cyberpunk's NGX app id from `ProgramData\NVIDIA\NGX\models\nvngx_config.txt`) | no change (`0xBAD0000B`) |
| exe renamed to `Cyberpunk2077.exe` (process-name/driver-profile gate test) | no change (`0xBAD0000B`) |
| `--direct-after-glue`: direct plugin `D3D12_Init_Ext` after core Init (stock + SF) | `0xBAD00002` PlatformError (unchanged) |
| Vulkan glue retest with the full renodx create set (task: maybe `0xBAD0000B` was missing params) | `0xBAD0000B` for all groups — params were never the issue |
| Direct `_nvngx.dll` (no glue): `D3D12_Init_Ext` | **Success** — the glue is not even required for Init |
| Direct `nvngx_dlss.dll` snippet `Init_Ext` (core-as-snippet), with/without IAT patch | `0xBAD00002` / plain Init `0xBAD00001` |
| Snippet route: core-initialized app-dir `nvngx_dlss.dll` → its own `CreateFeature(18)` | returns **Success — but hollow**: bogus id 77 "creates" identically (the id is never validated), the DLL contains zero DLSSNR code/strings, and evaluate → `0xBAD00005` InvalidParameter |
| Pre-loading `nvngx_dlssnr.dll` before the snippet create (the add-on's own step) | no change — still a hollow handle |
| Plugin `Init_Ext` with parameters populated by the initialized runtime (`nvngx_dlss.dll` `PopulateParameters_Impl`) — "host registration via parameter map" hypothesis | `0xBAD00002` (unchanged) |
| Streamline without OTA flags: `slSetFeatureLoaded(kFeatureDLSS)` | `eErrorFeatureFailedToLoad` (34) even for plain DLSS — the SL host path does not come up in our minimal harness, so `sl.dlss_nr`'s host-initialization of `nvngx_dlssnr.dll` cannot be reproduced this way |

### Interpretation

- The 616.56 driver's `_nvngx.dll` has a **hardcoded feature→plugin map that
  excludes dlssnr**: an IAT spy during Init/CreateFeature shows it probing
  only `nvngx_dlss.dll`/`nvngx_dlssg.dll` (app dir first, signature-verified,
  DriverStore fallback), reading `nvngx_config.txt`/`nvngx_deny_list.txt` —
  and never `nvngx_dlssnr.dll`. `CreateFeature(18)` fails before any plugin
  lookup on both the Vulkan and D3D12 API surfaces, identically.
- `nvngx_dlss.dll` 310.8.0.0 is *not* a dlssnr-aware core: its
  `CreateFeature` accepts any feature id (a lazy stub), and it contains no
  DLSSNR strings at all. A "successful" create through it is not a real
  feature.
- Therefore, in the working Cyberpunk setup, `nvngx_dlssnr.dll` must have
  been **registered into the shared NGX snippet runtime by a real host** —
  most plausibly `sl.dlss_nr.dll` (Cyberpunk uses Streamline), after which
  the add-on's *"signed snippet"* create (its own `CreateFeature` export)
  yields a real feature 18. Our standalone harness cannot reach that state:
  sl.dlss itself fails to load (`eErrorFeatureFailedToLoad`) outside a real
  frame/swapchain environment.
- App identity is not the gate: Cyberpunk's real NGX app id and process name
  change nothing.

### What this means for integration

The only proven end-to-end recipe is the **DLSS5-Feeder pattern**: run a
standard DLSS **DLAA** evaluate (feature 1 — works today in `glue12`) on a
D3D12 device inside a process that also hosts ReShade + the
`renodx-dlss5.addon64` detour stack; the add-on intercepts the evaluate and
inserts feature 18. For the Vulkan-based sr_compare renderer that implies a
Vulkan↔D3D12 bridge (`VK_KHR_external_memory_win32` +
`VK_KHR_external_semaphore_win32`, shared textures + shared fences, DLSS
evaluate on a private D3D12 device — exactly DLSS5-Feeder's Vulkan path) plus
shipping ReShade (dxgi.dll, add-on build) with the compare app. Estimated
cost: the bridge is the small part (~1–2 days: three shared textures, fence
ping-pong); the fragile part is the ReShade/add-on runtime dependency.
Alternative: wait for a launch driver whose `_nvngx.dll` carries the
feature-18 initializer, at which point the plain `glue`/`glue12` route in
this spike should work unmodified.

## Results on driver 616.56 (2026-08-31, round 1): still blocked, one level deeper

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
5. **Manually planting the payload does not help.** As an experiment we
   created `C:\ProgramData\NVIDIA\NGX\models\dlssnr\versions\20318208\files\`
   mimicking the OTA layout (`nvngx_dlssnr.dll` plus a `160_<appid>.bin`
   copy) and added a `[dlssnr]` section to `nvngx_config.txt`. The runtime
   still made **zero** file/registry accesses during `CreateFeature1(18)`
   and the result stayed `0xBAD0000B` — the in-memory initializer stub
   fails before any payload lookup. Combined with
   `GetFeatureRequirements(18)` → `NotImplemented`, this says the 616.56
   runtime has the feature-18 *table entry* but not its *initializer*. The
   experiment was fully reverted (config restored from backup, folder
   removed).

### NGX OTA investigation (2026-08-31): route definitively closed

The updater was coaxed into revealing its endpoint and server layout:

- `nvngx_update.exe` honors `http_proxy`/`https_proxy`; a CONNECT capture
  shows the OTA endpoint is **`ngx.download.nvidia.com:443`** (reachable
  from this network, HTTP 200). TLS is pinned to an embedded CA, so the
  request path could not be MITM'd directly; instead, the updater's
  `-test`/`-testroot` mode revealed that the remote layout mirrors the
  local `C:\ProgramData\NVIDIA\NGX\models` tree (passing that directory as
  `-testroot` satisfies its config-file lookups).
- The production server config was fetched directly:
  `https://ngx.download.nvidia.com/models/org/nvidia/team/ngx/models/config/versions/2/files/nvngx_server_config.txt`
  (plus the `dev-models/...` variant). It lists every published feature
  section (`[dlss]`, `[dlssd]`, `[dlssg]`, `[dlisp]`, broadcast/AI
  features, …) — **there is no `[dlssnr]` section** (nor any
  `cg2r`/`neural` alias), and `[dlss]` tops out below 310.8. The dev
  config only adds internal sections (`[unsigned_dlss]`, `[high_driver]`).
- Conclusion: **NGX OTA does not distribute DLSSNR at all** (pre-launch it
  ships only inside game packages, e.g. NBA 2K27). The earlier curl
  failures were network noise; even with perfect connectivity there is no
  dlssnr payload to fetch. The `nvngx_config.txt` edits made during this
  investigation were reverted.

### What would unblock it

- A driver/runtime drop that ships the DLSSNR initializer itself (the DLSS
  5 launch driver branch; possibly accompanied by a first OTA publication
  of the dlssnr payload), or
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
- DLSSNR parameter keys used (exact names extracted from
  `renodx-dlss5.addon64` and `nvngx_dlssnr.dll` — see the round-2 section for
  the full create/eval key sets): create `DLSSNR.{Width,Height,ScalingRatio,
  Hint.Render.Preset,Enabled}` + base `Width/Height/OutWidth/OutHeight`;
  evaluate `DLSSNR.{Color,Output,MVec,Depth[,Backbuffer]}` +
  `<res>SubrectBaseX/BaseY/Width/Height` + `MVecScaleX/Y,DepthInverted,
  Enabled,Reset,Intensity,LocalToneStrength,LocalStructureStrength,
  SkinStructureStrength,UseAutoMask,Style,UICorrection`.
- NGX feature-18 Vulkan requirements: instance
  `VK_KHR_get_physical_device_properties2`; device `VK_NVX_binary_import`,
  `VK_NVX_image_view_handle`, `VK_KHR_buffer_device_address`,
  `VK_KHR_push_descriptor` (the device list query fails on 610.62 and
  succeeds on 616.56).
- NGX headers are vendored under `third_party/ngx/include` (NVIDIA DLSS SDK,
  same license posture as the vendored Streamline SDK).
