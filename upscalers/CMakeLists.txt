# Unified upscaler layer.  IUpscaler.h (contract) and InputAdapter.h are
# header-only; UpscalerFactory implements a self-registration registry.
#
# Plugin convention (no edits to this file needed to add a module):
#   1. create upscalers/<name>/ with a CMakeLists.txt that defines a STATIC
#      target named `upscaler_<name>`;
#   2. in one of the module's .cpp files call
#      SR_REGISTER_UPSCALER("<name>", &createFn)  (see UpscalerFactory.h);
#   3. if <name> has a matching -DSR_WITH_<NAME> option (declared in the
#      top-level CMakeLists), it gates the module; otherwise it is always on.
# Modules are linked WHOLE_ARCHIVE so their static registrars are not dropped
# by the linker.
add_library(upscalers STATIC UpscalerFactory.cpp)
sr_common_setup(upscalers)

file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${children})
    set(dir ${CMAKE_CURRENT_SOURCE_DIR}/${child})
    if(NOT IS_DIRECTORY ${dir})
        continue()
    endif()
    if(NOT EXISTS ${dir}/CMakeLists.txt)
        continue()
    endif()
    string(TOUPPER ${child} upper)
    if(child STREQUAL "taa")
        set(enabled ON)  # baseline, always built
    elseif(DEFINED SR_WITH_${upper})
        set(enabled ${SR_WITH_${upper}})
    else()
        set(enabled ON)
    endif()
    if(enabled)
        add_subdirectory(${child})
        target_link_libraries(upscalers PUBLIC $<LINK_LIBRARY:WHOLE_ARCHIVE,upscaler_${child}>)
    endif()
endforeach()
