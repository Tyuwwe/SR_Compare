# third_party/ — SDK 说明

本目录由 `scripts/fetch_sdks.py` 生成（不入 git，见仓库根 `.gitignore` 的
`third_party/*/`）。下表为每个 SDK 的版本、来源与许可证。

| 子目录 | 名称 | 实际版本 | 许可证 |
|---|---|---|---|
| `fidelityfx/` | AMD FidelityFX SDK（FSR 1/2/3 全源码） | **v1.1.4** | MIT |
| `xess/` | Intel XeSS SDK（闭源 DLL） | **v3.0.2**（latest） | Intel Simplified Software License (Oct 2022) |
| `streamline/` | NVIDIA Streamline SDK（框架 + DLSS DLL） | **v2.12.0**（latest） | 框架 MIT + `nvngx_dlss.dll` NVIDIA RTX SDKs LICENSE |
| `snapdragon-gsr/` | Snapdragon SGSR 1/2（shader 源码） | git `d926f07`（main，浅克隆） | BSD-3-Clause |
| `arm-nss/` | Arm Neural Super Sampling（SDK + 模型 + Vulkan ML 模拟层） | 见下 | SDK MIT + 模型 Arm AI Model Community License |
| `../assets/sponza/` | Sponza glTF 测试场景 | Khronos glTF-Sample-Assets `main` | 模型文件 Cryengine Limited License Agreement；文档 CC-BY-4.0 |
| `../assets/bistro/` | Amazon Lumberyard Bistro | NVIDIA ORCA 2017（FBX 转 glTF） | CC-BY 4.0 |

> 注：`../assets/sponza/` 与 `../assets/bistro/` 在 `assets/` 下，不属于
> `third_party/`，为方便对照一并列出。场景获取/转换步骤见仓库根 `README.md`。

---

## FidelityFX SDK — `fidelityfx/`

- 来源：https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
- release 资产：`FidelityFX-SDK-v1.1.4.zip`（tag `v1.1.4`）
- 许可证：MIT（`fidelityfx/LICENSE.txt`）
- 只抽取了 `sdk/include` 与 `sdk/src` 源码（FSR1/2/3 源码集成用），其余
  samples/prebuilt/tools 已丢弃以省空间。

关键路径：

- 头文件（host API）：
  - `include/FidelityFX/host/ffx_fsr1.h`
  - `include/FidelityFX/host/ffx_fsr2.h`
  - `include/FidelityFX/host/ffx_fsr3.h`
  - `include/FidelityFX/host/ffx_fsr3upscaler.h`
- shader 头文件（GPU）：`include/FidelityFX/gpu/fsr1/`、`fsr2/`、`fsr3/`、`fsr3upscaler/`
- 源码：`src/components/fsr1/`、`fsr2/`、`fsr3/`、`fsr3upscaler/`，公共部分在
  `src/shared/` 与 `src/backends/`

---

## Intel XeSS SDK — `xess/`

- 来源：https://github.com/intel/xess
- release 资产：`XeSS_SDK_3.0.2.zip`（latest = `v3.0.2`）
- 许可证：Intel Simplified Software License (Version October 2022)，见 `xess/LICENSE.txt`
- 只保留了 `inc/`、`lib/`、`bin/`（`doc/`、`samples/` 已丢弃）。

关键路径：

- 头文件：`inc/xess/xess.h`、`xess_vk.h`、`xess_debug.h`、`xess_d3d12.h`（以及
  `inc/xell/`、`inc/xess_fg/`）
- 库：`lib/libxess.lib`
- DLL：`bin/libxess.dll`（约 74 MB）、`bin/libxess_fg.dll`、`bin/libxell.dll`、
  `bin/libxess_dx11.dll`

---

## NVIDIA Streamline SDK — `streamline/`

- 来源：https://github.com/NVIDIA-RTX/Streamline
- release 资产：`streamline-sdk-v2.12.0.zip`（latest = `v2.12.0`，production 打包，
  内含 dev/production 两套 DLL）
- 许可证：
  - 框架（`sl.*` 头文件与 `sl.*.dll`）：MIT，见 `streamline/license.txt`
  - `nvngx_dlss.dll` 等 NGX feature DLL：NVIDIA RTX SDKs LICENSE，见
    `streamline/bin/x64/nvngx_dlss.license.txt`
- 只保留了 `include/`、`lib/`、`bin/`。

关键路径：

- 头文件：`include/sl.h`、`include/sl_dlss.h`、`include/sl_core_api.h` 等（24 个）
- 导入库：`lib/x64/sl.interposer.lib`
- production DLL：`bin/x64/`
  - `sl.interposer.dll`、`sl.common.dll`、`sl.dlss.dll`
  - `nvngx_dlss.dll`（约 59 MB）、`nvngx_dlssd.dll`、`nvngx_dlssg.dll` 等
- development DLL：`bin/x64/development/`
  - `sl.interposer.dll`、`sl.common.dll`、`sl.dlss.dll`、`sl.imgui.dll`
  - `nvngx_dlss.dll`（约 70 MB，dev 版，用于开发期；出报告用 production 版）

---

## Snapdragon SGSR — `snapdragon-gsr/`

- 来源：https://github.com/SnapdragonGameStudios/snapdragon-gsr（浅克隆 `--depth 1`）
- 版本：git commit `d926f07`（main）
- 许可证：BSD-3-Clause，见 `snapdragon-gsr/LICENSE`
- 纯 shader 源码（含 `media/` 演示图/视频，非运行必需，可自行删除省空间）。

关键路径（shader 目录）：

- `sgsr/v1/include/hlsl/`、`sgsr/v1/include/glsl/`
- `sgsr/v2/include/glsl_2_pass_cs/`、`sgsr/v2/include/glsl_2_pass_fs/`、
  `sgsr/v2/include/glsl_3_pass_cs/`

---

## Arm Neural Super Sampling — `arm-nss/`

由三部分构成：

### 1. SDK 源码（`arm-nss/` 根，即该目录本身）

- 来源：https://github.com/arm/neural-graphics-sdk-for-game-engines（浅克隆）
- 版本：git commit `aba0d10`（main）
- 许可证：MIT（`LICENSES/MIT.txt`），另有部分内容 CC-BY-4.0（`LICENSES/CC-BY-4.0.txt`）
- 关键路径：
  - `sdk/include/`（含 `FidelityFX/`、`vulkan-headers/`、`tinyexr/`）
  - `sdk/src/{backends,components,shared}/`
  - `ffx-api/`（vendored FFX API）
- 注意：该仓库含 3 个 git submodule（`sdk/libs/SPIRV-Headers`、
  `sdk/libs/SPIRV-Tools`、`sdk/tools/ffx_shader_compiler/libs/SPIRV-Cross`），
  本次浅克隆**未初始化** submodule（如需本地编译 SDK 自身工具链再补）。

### 2. 预编译 VGF 模型（`arm-nss/models/`）

- 来源：https://huggingface.co/Arm/neural-super-sampling（`git clone --depth 1`，
  自动拉取 Git LFS 大文件）
- 版本：git commit `3dd6c4f`
- 许可证：
  - 模型源程序：Apache-2.0（见 `models/LICENSE` 指向）
  - 仓库内容（模型权重等）：Arm AI Model Community License v1.0 PRE-1154，
    见 `models/Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf`
- 关键文件（VGF 预编译模型）：
  - `models/nss_v1_0_1_high_int8.vgf`（high 档，约 161 KB）
  - `models/nss_v1_0_1_mid_low_int8.vgf`（mid/low 共用，约 155 KB）
  - 对应 metadata：`nss_v1_0_1_*_int8_metadata.json`
  - 场景 shader/配置/资产：`models/scenario/`（约 22 MB）
  - 说明：HuggingFace 仓库只有 high 与 mid_low 两个 VGF 模型文件，mid 与 low
    两档共用 `mid_low` 模型、以不同 push constant 区分（见 `scenario/configs/`）。

### 3. Vulkan ML 模拟层（`arm-nss/emulation-layer/`）

- 来源：https://github.com/arm/ai-ml-emulation-layer-for-vulkan（浅克隆）
- 版本：git commit `00e9d81`
- 头文件以 `.hpp` 为主：`common/include/mlel/*.hpp` 等
- 用于在 PC 上经 Vulkan 软件模拟推理（性能不代表真机）。

---

## Sponza 场景 — `assets/sponza/`

- 来源：https://github.com/KhronosGroup/glTF-Sample-Assets 的
  `Models/Sponza/glTF/`（sparse clone 后拷贝）
- 许可证：
  - 模型文件（含几何/贴图）：Cryengine Limited License Agreement
    （见 `assets/sponza/LicenseRef-CRYENGINE-Agreement.txt`，指向
    https://www.cryengine.com/ce-terms）
  - 文档/metadata：CC-BY-4.0（`assets/sponza/LICENSE.md`）
- 直接可加载入口：`assets/sponza/Sponza.gltf`（配套 `Sponza.bin` + 若干
  `.jpg`/`.png` 贴图）
- 获取：`python scripts/fetch_sdks.py --only sponza`（无需转换）

---

## Amazon Lumberyard Bistro — `assets/bistro/`

- 来源：https://developer.nvidia.com/orca/amazon-lumberyard-bistro
  （NVIDIA Open Research Content Archive）
- 许可证：Creative Commons CC-BY 4.0
- 引用（ORCA 要求）：

  ```
  Amazon Lumberyard Bistro, Open Research Content Archive (ORCA)
  Amazon Lumberyard, July 2017
  http://developer.nvidia.com/orca/amazon-lumberyard-bistro
  ```

- 原始素材为 FBX + DDS。本仓库用 Blender 5.1+ 经
  `scripts/convert_bistro_v2.py` 转成全 PBR glTF，再用
  `scripts/fixup_bistro_alpha.py` 修正 alphaMode。步骤见仓库根 `README.md`
  的 Scenes / Bistro 一节。
- 可加载入口：`assets/bistro/BistroExterior.gltf`、
  `assets/bistro/BistroInterior.gltf`

---

## 生成与重新下载

```bat
python scripts/fetch_sdks.py                 :: 全部
python scripts/fetch_sdks.py --only fsr,xess :: 只取部分
python scripts/fetch_sdks.py --force         :: 强制重新下载
```
