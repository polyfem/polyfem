# tight (https://gitlab.com/minimize-solve/tight)
# License: see tight repo
#
# polyfem does not depend on tight directly: it arrives via
# hocgv-miso -> misolib -> tight. We fetch it here, ahead of hocgv_miso, only
# so that the patch below is applied -- misolib's own recipe starts with
# if(TARGET tight::tight) return(), so whoever defines the target first wins.
#
# TODO: drop this once the MSVC flags are fixed upstream in tight, then bump
# the misolib/hocgv-miso pins.

if(TARGET tight::tight)
    return()
endif()

message(STATUS "Third-party: creating target 'tight::tight'")

find_package(Patch REQUIRED)

file(GLOB_RECURSE patches_for_tight CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/tight_*.patch"
)

set(PATCH_COMMAND_FOR_TIGHT "")
foreach(patch_filename IN LISTS patches_for_tight)
    list(APPEND PATCH_COMMAND_FOR_TIGHT "${Patch_EXECUTABLE}" -rnN -p1 < ${patch_filename} &&)
endforeach()
if(PATCH_COMMAND_FOR_TIGHT)
    list(POP_BACK PATCH_COMMAND_FOR_TIGHT)
endif()

include(CPM)
CPMAddPackage(
    NAME tight
    GIT_REPOSITORY https://gitlab.com/minimize-solve/tight.git
    GIT_TAG ce69ac487c3c75b57f20c1a6e7f40eab531db6c7
    PATCH_COMMAND ${PATCH_COMMAND_FOR_TIGHT}
    # CPM's single-argument shorthand -- which misolib uses, and which we have
    # to spell out here to attach PATCH_COMMAND -- implies both of these. Without
    # them tight's own targets, and NFG's test executable, join the default build
    # target instead of being built only on demand.
    EXCLUDE_FROM_ALL YES
    SYSTEM YES
)
