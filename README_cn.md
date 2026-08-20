# sr_compare — 超分方案横向对比工具

Vulkan 单后端，在同一场景/同一相机路径下对比多种超分算法的画质与性能。

## 覆盖算法

| 算法 | 类型 | 来源/许可证 |
|---|---|---|
| TAA（自研基线） | 时域 | 本仓库 |
| FSR 1 / 2 / 3.1 | 空间/时域 | AMD FidelityFX SDK 1.1.4（MIT） |
| Intel XeSS | ML（DP4a） | intel/xess（闭源 DLL，Intel Simplified License） |
| NVIDIA DLSS 4.5（Preset K / M） | ML（Transformer） | Streamline SDK（框架 MIT；nvngx_dlss.dll 闭源，需 NVIDIA Developer applicationId） |
| Arm NSS v1.0.1（high/mid_low int8） | ML（移动端） | arm/neural-graphics-sdk-for-game-engines（MIT）+ HF 模型（Arm AI Model Community License） |
| SGSR 1 / 2 | 空间/时域（移动端） | SnapdragonGameStudios/snapdragon-gsr（BSD-3-Clause） |

Arm NSS 在 PC 上经 Vulkan ML Emulation Layer 软件模拟推理，性能数字不代表真机。

## 渲染器

- Deferred PBR：Cook-Torrance GGX、双点光、完整 IBL（irradiance + prefiltered
  specular + BRDF LUT）、法线/ORM/AO/自发光贴图、MASK 镂空、mipmap + 16x 各向异性、
  视锥剔除、合并场景缓冲
- SSAO（16 采样法线半球 + 十字模糊，约 0.5ms @1080p）
- 前向透明 pass（lighting 之后、upscale 之前）：电介质玻璃模型（菲涅尔驱动不透明度）、
  后往前排序、写相机运动矢量与半透明覆盖度 mask
- 覆盖度 mask 接入了所有官方支持的接口：TAA（历史权重）、FSR2/FSR3（reactive + T&C
  mask）、DLSS（bias-current-color / reactive / transparency hint）、XeSS（responsive
  pixel mask）。NSS 与 SGSR2 无此输入。
- 参考真值：原生分辨率，可选 200% SSAA（4K→1080p）

## 运行模式

- **gui** — ImGui 界面（Viewer / Compare / Bench 三个标签页）。切场景/算法在工作
  线程异步加载并显示进度浮层，旧画面持续渲染不卡死；分级重建——只改算法不再
  重载场景。
- **viewer** — 自由视角单算法全屏预览。
- **compare** — 分栏实时对比：最左栏原生 GT，右侧各算法列，实时 PSNR/SSIM；等比
  裁切、滚轮缩放（指标按可视区域重算）、可选 GT 200% SSAA。
- **bench** — 固定相机路径固定帧数；超分 pass GPU 耗时、整帧耗时/FPS（均值/中位数
  /1% low）、显存占用（VK_EXT_memory_budget），导出 CSV/JSON。

![Viewer](docs/screenshot_viewer.png)

![Compare](docs/screenshot_compare.png)

```bat
sr_compare gui
sr_compare viewer --scene sponza --upscaler fsr2
sr_compare compare --scene boxes --upscalers taa,fsr2,xess,dlss-m --render-scale 0.5
sr_compare bench --scene sponza --frames 300 --warmup 60
sr_compare viewer --list-upscalers
```

## 场景（`--scene`）

- `boxes`（别名 `procedural`）：内置程序化盒子房间，无需外部资产
- `sponza`：Sponza 中庭 — Crytek Sponza 的 Khronos glTF 样例
- `bistro_exterior` / `bistro_interior`：Amazon Lumberyard Bistro（NVIDIA ORCA）
- `<任意 glTF 路径>`：加载自定义场景

`assets/` 不入库。新克隆后按下面获取 Sponza，需要 Bistro 时再转换。

### Sponza

来源：[KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
的 `Models/Sponza/glTF/`。模型文件为 Cryengine Limited License Agreement；
文档/metadata 为 CC-BY-4.0。

已经是可加载 glTF，无需转换。在仓库根目录：

```bat
python scripts/fetch_sdks.py --only sponza
```

脚本会浅克隆 Khronos 仓库，并把 glTF 目录拷到 `assets/sponza/`
（`Sponza.gltf` + `Sponza.bin` + 贴图 + 许可证文件）。

### Bistro

来源：[Amazon Lumberyard Bistro，NVIDIA ORCA](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)
（CC-BY 4.0）。从该页下载 FBX 包并解压，本仓库假定的目录结构为：

```
Bistro_v5_2/
  BistroExterior.fbx
  BistroInterior.fbx
  Textures/          :: DDS：*_BaseColor、*_Normal（DirectX Y-）、
                     ::       *_Specular（实际是 ORM：R=AO，G=Roughness，B=Metalness）、
                     ::       *_Emissive
```

需要 **Blender 5.1+**（原生 C++ FBX 导入器；旧 Python 插件遇到该文件里的灯会崩溃）
及其自带的 NumPy。在仓库根目录执行（Git Bash 示例，按本机 Blender / 解压路径改）：

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

`convert_bistro_v2.py` 导出 `GLTF_SEPARATE`（`.gltf` + `.bin` + PNG），带全 PBR
通道：baseColor+alpha、法线 G 通道翻转（DirectX → glTF）、ORM 写成
metallicRoughness + occlusion、自发光、MASK 镂空。随后
`fixup_bistro_alpha.py`（普通 Python，不需要 Blender）修正残留的 alphaMode：
全零 MASK 改回 OPAQUE、真正的玻璃改成带常数 alpha 的 BLEND、被 FBX 标成
BLEND 的不透明材质改回 OPAQUE。结果是 `assets/bistro/BistroExterior.gltf` 与
`BistroInterior.gltf`。

## 构建

```bat
build_all.bat          :: 一键 configure + Release 构建
```

依赖：Windows 11、Visual Studio（C++ 工作负载，vswhere 自动发现）、CMake ≥ 3.24
（PATH 或 VS 自带均可）、Vulkan SDK 1.4+（`VULKAN_SDK` 提供 glslangValidator）、
Python 3.10+（仅 `scripts/fetch_sdks.py` 与 `metrics/` 用）。

DLSS 需要 NVIDIA Developer 账号申请 applicationId（免费），环境变量 `SR_DLSS_APP_ID`。

### 换机器/整目录拷贝

源码树完全自包含（third_party、assets、着色器全部在树内，CMake 无硬编码路径），
整目录拷贝后装好上述依赖即可 `build_all.bat`。`build*` 目录均为本机缓存可删，
唯一例外：`build-nss-emu/` 内有 NSS 模拟层的依赖（含已打补丁的 git 克隆）。若已
删除，运行 `scripts\nss_emu_fetch_deps.bat`（git + 网络），再
`scripts\nss_emu_configure.bat` + `scripts\nss_emu_build.bat`（另需 Ninja +
Python 3）。缺失时主工程照常编译，NSS 插件自动禁用。

## 打包分发

`build/app/Release/` 构建后即为自包含目录，整体复制即可运行（无需源码与构建环境）：

- `sr_compare.exe` + 全部运行时 DLL（XeSS/Streamline/DLSS/NSS、CRT）
- `run_gui.bat`（一键启动 GUI）
- `shaders/`、`assets/`（Bistro 全 PBR 约 1.3GB 可裁剪）、`nss-emu/`

着色器/资产按 exe 相对路径解析，打包目录从任何工作目录启动都能工作。目标机器
要求：支持 Vulkan 1.3 的显卡驱动；DLSS 需要 RTX 20+（无 RTX 时 DLSS 自动不可用，
其余算法不受影响）。

## 目录结构

- `renderer/` — Vulkan 渲染器核心（设备/帧循环/glTF 场景/deferred GBuffer/lighting/
  透明 pass/SSAO/IBL/相机路径）
- `upscalers/` — 统一超分插件层（`IUpscaler.h` 为契约，各算法一个子目录）
- `compare/`、`bench/`、`gui/`、`app/` — 各前端与入口
- `metrics/` — Python 离线指标复核与 Markdown 报告生成
- `scripts/` — SDK 下载、Bistro 转换、NSS 模拟层构建
- `third_party/` — 各 SDK（脚本下载，不入库）

## 许可证

本仓库原创源码为 MIT（见 `LICENSE`）。`third_party/` 下的 SDK 与 `assets/`
下的场景资源仍按各自原许可，详见 `third_party/README.md` 与上文场景说明。

## 注意

- 帧生成（FSR3-FG / XeSS-FG / DLSS-FG）不在本项目范围。
- SGSR 面向移动 Adreno 调优，PC 上性能仅供参考。
- 透明表面在低分辨率渲染后随整帧一起超分（业界标准结构）；逐像素折射/transmission
  有意未实现。
