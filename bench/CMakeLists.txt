# Performance benchmark mode.  Links `upscalers` so the default algorithm set
# matches `viewer --list-upscalers`.
add_library(bench STATIC BenchMode.cpp)
sr_common_setup(bench)
target_link_libraries(bench PUBLIC upscalers)
