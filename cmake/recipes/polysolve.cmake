# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
find_package(Patch REQUIRED)
set(PATCH_COMMAND_ARGS "-rnN")

file(GLOB_RECURSE patches_for_polysolve CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/polysolve.patch"
)

set(PATCH_COMMAND_FOR_CPM_BASE "${Patch_EXECUTABLE}" ${PATCH_COMMAND_ARGS} -p1 < )

set(PATCH_COMMAND_FOR_CPM "")
foreach(patch_filename IN LISTS patches_for_polysolve)
    list(APPEND PATCH_COMMAND_FOR_CPM ${PATCH_COMMAND_FOR_CPM_BASE})
    list(APPEND PATCH_COMMAND_FOR_CPM ${patch_filename})
    list(APPEND PATCH_COMMAND_FOR_CPM &&)
endforeach()
list(POP_BACK PATCH_COMMAND_FOR_CPM)

message(DEBUG "Patch command: ${PATCH_COMMAND_FOR_CPM}")

CPMAddPackage(
        NAME polysolve
        GITHUB_REPOSITORY "polyfem/polysolve"
        GIT_TAG "71271fb4e231e42fa2cf764c5bf354c81856b105"
        PATCH_COMMAND ${PATCH_COMMAND_FOR_CPM})