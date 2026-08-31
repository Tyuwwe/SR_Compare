# DLSS 5 Neural Rendering (DLSSNR) technology-validation spike.
# Standalone exe: does NOT link renderer/upscalers. nvngx_dlssnr.dll is
# loaded at runtime via LoadLibrary + GetProcAddress (no NGX import lib -
# 310.x NGX DLLs only export the "snippet" ABI, so the spike implements the
# NVSDK_NGX_Parameter vtable itself). See the investigation notes at the top
# of main.cpp for why the NGX-core route cannot create feature 18.
add_executable(dlss5_spike main.cpp)
sr_common_setup(dlss5_spike)
target_include_directories(dlss5_spike SYSTEM PRIVATE
    ${SR_THIRD_PARTY_DIR}/ngx/include
    ${SR_THIRD_PARTY_DIR}/streamline/include)
target_link_libraries(dlss5_spike PRIVATE d3d12 dxgi)
