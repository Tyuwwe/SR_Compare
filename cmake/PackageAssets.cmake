# Incremental mirror of the scene assets next to the exe (package target).
# Invoked as: cmake -D SRC=<repo>/assets -D DST=<exe-dir>/assets -P PackageAssets.cmake
# robocopy skips files that already match (size + timestamp), so re-runs only
# copy what changed; /MIR also removes stale files in DST.  Exit codes 0-7 are
# success (0 = nothing to do, 1 = files copied, >=8 = failure).
if(NOT SRC OR NOT DST)
    message(FATAL_ERROR "PackageAssets.cmake needs -D SRC= and -D DST=")
endif()
execute_process(
    COMMAND robocopy "${SRC}" "${DST}" /MIR /NFL /NDL /NJH /NJS /NP
    RESULT_VARIABLE rc)
if(rc GREATER 7)
    message(FATAL_ERROR "robocopy failed with exit code ${rc}")
endif()
message(STATUS "assets staged: ${DST}")
