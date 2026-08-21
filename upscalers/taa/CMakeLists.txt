# TAA baseline upscaler (implements sr::IUpscaler).
# Convention: plugin targets are named `upscaler_<dirname>` and are linked
# whole-archive into the umbrella `upscalers` lib by ../CMakeLists.txt.
add_library(upscaler_taa STATIC TaaUpscaler.cpp)
sr_common_setup(upscaler_taa)
target_compile_definitions(upscaler_taa PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")

sr_compile_shader(TAA_SPV taa.comp)

# RCAS sharpen pass: includes the FidelityFX SDK GPU headers (CAS), so it gets
# its own include root plus explicit header deps for rebuild tracking.
set(FFX_GPU_INCLUDE ${SR_THIRD_PARTY_DIR}/fidelityfx/include/FidelityFX/gpu)
sr_compile_shader(TAA_CAS_SPV taa_cas.comp
    INCLUDE_DIR ${FFX_GPU_INCLUDE}
    ${FFX_GPU_INCLUDE}/cas/ffx_cas.h
    ${FFX_GPU_INCLUDE}/cas/ffx_cas_sharpen.h
    ${FFX_GPU_INCLUDE}/ffx_core.h)

add_custom_target(taa_shaders DEPENDS ${TAA_SPV} ${TAA_CAS_SPV})
add_dependencies(upscaler_taa taa_shaders)
