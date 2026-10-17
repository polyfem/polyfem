# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

# Patch polysolve's Backtracking line search so that, when it is comparing
# gradient norms, it also accepts a step that decreases the energy. The
# gradient-only test can fail to accept any step near convergence -- the
# gradient plateaus while the energy is still descending -- which aborts the
# solve outright. The patch adds an opt-in flag
# (line_search/accept_energy_decrease_in_grad_mode, default false), so every
# scene that does not set it is unaffected.
#
# TODO: drop this once the change is upstreamed in polysolve.
find_package(Patch REQUIRED)

file(GLOB_RECURSE patches_for_polysolve CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/polysolve_*.patch"
)

set(PATCH_COMMAND_FOR_POLYSOLVE "")
foreach(patch_filename IN LISTS patches_for_polysolve)
    list(APPEND PATCH_COMMAND_FOR_POLYSOLVE "${Patch_EXECUTABLE}" -rnN -p1 < ${patch_filename} &&)
endforeach()
if(PATCH_COMMAND_FOR_POLYSOLVE)
    list(POP_BACK PATCH_COMMAND_FOR_POLYSOLVE)
endif()

include(CPM)
CPMAddPackage(
    NAME polysolve
    GIT_REPOSITORY https://github.com/polyfem/polysolve.git
    GIT_TAG a7727e33398ff4ab703bb593e1f06041c237c600
    PATCH_COMMAND ${PATCH_COMMAND_FOR_POLYSOLVE}
)
