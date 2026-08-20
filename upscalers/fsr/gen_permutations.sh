#!/usr/bin/env bash
# Generate FidelityFX shader permutation headers for the VK backend.
# Uses the prebuilt FidelityFX Shader Compiler bundled with the Arm NSS SDK
# (third_party/arm-nss/sdk/tools/binary_store/FidelityFX_SC.exe) and the
# Vulkan SDK glslangValidator, replicating the SDK's own CMake shader build
# (see src/backends/vk/CMakeShadersFSR*.txt and
# include/FidelityFX/gpu/*/CMakeCompile*Shaders.txt).
# Output: upscalers/fsr/ffx_permutations/ (checked in; regeneration only
# needed when the SDK or shaders change).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FFX="$ROOT/third_party/fidelityfx"
export GPU="$FFX/include/FidelityFX/gpu"
export VK_SHADERS="$FFX/src/backends/vk/shaders"
export SC="$ROOT/third_party/arm-nss/sdk/tools/binary_store/FidelityFX_SC.exe"
export OUT="$ROOT/upscalers/fsr/ffx_permutations"
mkdir -p "$OUT"

if [[ -n "${VULKAN_SDK:-}" ]]; then
    export PATH="$VULKAN_SDK/Bin:$VULKAN_SDK/bin:$PATH"
fi

FSR1_SHADERS=(fsr1/ffx_fsr1_easu_pass fsr1/ffx_fsr1_rcas_pass)
FSR2_SHADERS=(fsr2/ffx_fsr2_accumulate_pass fsr2/ffx_fsr2_autogen_reactive_pass
              fsr2/ffx_fsr2_compute_luminance_pyramid_pass fsr2/ffx_fsr2_depth_clip_pass
              fsr2/ffx_fsr2_lock_pass fsr2/ffx_fsr2_rcas_pass
              fsr2/ffx_fsr2_reconstruct_previous_depth_pass fsr2/ffx_fsr2_tcr_autogen_pass)
FSR3_SHADERS=(fsr3upscaler/ffx_fsr3upscaler_accumulate_pass
              fsr3upscaler/ffx_fsr3upscaler_autogen_reactive_pass
              fsr3upscaler/ffx_fsr3upscaler_debug_view_pass
              fsr3upscaler/ffx_fsr3upscaler_luma_instability_pass
              fsr3upscaler/ffx_fsr3upscaler_luma_pyramid_pass
              fsr3upscaler/ffx_fsr3upscaler_prepare_inputs_pass
              fsr3upscaler/ffx_fsr3upscaler_prepare_reactivity_pass
              fsr3upscaler/ffx_fsr3upscaler_rcas_pass
              fsr3upscaler/ffx_fsr3upscaler_shading_change_pass
              fsr3upscaler/ffx_fsr3upscaler_shading_change_pyramid_pass)

# Emit one "comp|shader|name|half" job per (shader, variant) combo, then run
# them in parallel via xargs.  run_job re-derives all argument arrays itself
# because bash arrays cannot be exported to xargs child processes.
JOBS="$(mktemp)"
trap 'rm -f "$JOBS"' EXIT

emit_jobs() {
    local -n shaders_ref=$1
    local comp=$2
    for s in "${shaders_ref[@]}"; do
        local base
        base="$(basename "$s")"
        for variant in 0 1; do          # 0 = fp32, 1 = 16bit
            local half_name=""
            [[ "$variant" == "1" ]] && half_name="_16bit"
            echo "$comp|$s|$base$half_name|$variant" >> "$JOBS"
            echo "$comp|$s|${base}_wave64$half_name|$variant" >> "$JOBS"
        done
    done
}

emit_jobs FSR1_SHADERS fsr1
emit_jobs FSR2_SHADERS fsr2
emit_jobs FSR3_SHADERS fsr3upscaler

run_job() {
    local comp shader name half
    IFS='|' read -r comp shader name half <<< "$1"

    local api=(-compiler=glslang -e CS --target-env vulkan1.2 -S comp -Os -DFFX_GLSL=1)
    local extra inc
    case "$comp" in
        fsr1)
            extra=("-DFFX_FSR1_OPTION_APPLY_RCAS={0,1}"
                   "-DFFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA={0,1}"
                   "-DFFX_FSR1_OPTION_SRGB_CONVERSIONS={0,1}")
            inc=("-I$GPU" "-I$GPU/fsr1") ;;
        fsr2)
            extra=(-DFFX_FSR2_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
                   -DFFX_FSR2_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
                   -DFFX_FSR2_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
                   -DFFX_FSR2_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
                   -DFFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
                   "-DFFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1}"
                   "-DFFX_FSR2_OPTION_HDR_COLOR_INPUT={0,1}"
                   "-DFFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1}"
                   "-DFFX_FSR2_OPTION_JITTERED_MOTION_VECTORS={0,1}"
                   "-DFFX_FSR2_OPTION_INVERTED_DEPTH={0,1}"
                   "-DFFX_FSR2_OPTION_APPLY_SHARPENING={0,1}")
            inc=("-I$GPU" "-I$GPU/fsr2") ;;
        fsr3upscaler)
            extra=(-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
                   -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
                   -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
                   -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
                   -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
                   "-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1}"
                   "-DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT={0,1}"
                   "-DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1}"
                   "-DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS={0,1}"
                   "-DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH={0,1}"
                   "-DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={0,1}")
            inc=("-I$GPU" "-I$GPU/fsr3upscaler") ;;
    esac

    "$SC" -reflection -deps=gcc -DFFX_GPU=1 "${api[@]}" "${extra[@]}" \
        -name="$name" "-DFFX_HALF=$half" "${inc[@]}" -output="$OUT" \
        "$VK_SHADERS/$shader.glsl" > /dev/null
}
export -f run_job

xargs -a "$JOBS" -d '\n' -P 8 -I {} bash -c 'run_job "$@"' _ {}

# The bundled SC tool (Arm NSS fork) reflects uniform-block TYPE names
# (cbFSR2_t, ...) while the SDK host code matches on GLSL instance names
# (cbFSR2, ...).  Rewrite the CBV resource names to the instance names.
sed -i 's/"cbFSR1_t"/"cbFSR1"/g; s/"cbFSR2_t"/"cbFSR2"/g; s/"cbFSR3UPSCALER_t"/"cbFSR3Upscaler"/g; s/"cbRCAS_t"/"cbRCAS"/g; s/"cbSPD_t"/"cbSPD"/g; s/"cbGenerateReactive_t"/"cbGenerateReactive"/g' "$OUT"/*.h

# Depfiles are only used by the SDK CMake dependency tracking; drop them.
rm -f "$OUT"/*.d
echo "Generated $(ls "$OUT" | wc -l) files in $OUT"
