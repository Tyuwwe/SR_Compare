# Patches required by the NSS integration

These patches were developed while bringing up Arm NSS on the PC emulation
layer. They are applied to sources that live outside `upscalers/nss/`; the
patch files are kept here so the changes can be re-applied after a fresh
dependency fetch. Do not commit them anywhere else.

## emulation-layer-diagnostics.patch

Target: `third_party/arm-nss/emulation-layer` (git revision 00e9d81).
Apply with `git -C third_party/arm-nss/emulation-layer apply <this file>`.

- `graph/graph_layer.cpp`
  - `vkCreateDataGraphPipelinesARM` is wrapped in try/catch so a TOSA/SPIR-V
    translation failure logs `e.what()` and returns `VK_ERROR_UNKNOWN`
    instead of fail-fast (0xC0000409) killing the host process.
  - `vkCreateDataGraphPipelineSessionARM` guards `allocateObject` the same
    way.
  - New `VMEL_DUMP_GRAPH_SPV` env var: when set, the incoming graph SPIR-V is
    written to the given file before the optimizer runs (use together with
    `spirv-dis` for offline debugging).
- `graph/spirv_pass_tosaspv_v100.cpp`
  - `handleGraph` logs every graph op at `Severity::Info`
    (`VMEL_GRAPH_SEVERITY=Info` to see it; silent by default).

Keep these applied: the try/catch wrappers are load-bearing for stability,
the logging is opt-in via env vars.

## spirv-tools-tensorarm-constants.patch

Target: `build-nss-emu/deps/SPIRV-Tools` (KhronosGroup/SPIRV-Tools @0539c81).
Apply with `git -C build-nss-emu/deps/SPIRV-Tools apply <this file>` after
re-fetching dependencies, then rebuild the emulation layer
(`scripts/nss_emu_configure.bat` + `scripts/nss_emu_build.bat`).

Backports the TensorARM constant handling from the Arm fork of SPIRV-Tools
(`github.com/arm/SPIRV-Tools`, branch `staging`), which the NSS SDK expects
but the emulation layer does not fetch:

- `constants.h`: new `TensorConstant` class (+ `AsTensorConstant` probes).
- `constants.cpp`:
  - `CreateConstant` registers `OpConstantComposite` with a TensorARM result
    type as `TensorConstant` instead of dropping it (this was the root cause
    of `Missing declared constant for id` on the first CONV2D pad tensor).
  - `GetConstantFromInst` no longer expands
    `OpConstantCompositeReplicateEXT` itself (the emulation layer expands
    replicated patterns via `tryExpandReplicatedPattern`).
  - Defensive TensorARM branches in `CreateCompositeInstruction` /
    `GetNullCompositeConstant`, matching the fork.

Without this patch `ffxCreateContext` fails during data graph pipeline
creation (NSS context creation aborts and the plugin disables itself).
