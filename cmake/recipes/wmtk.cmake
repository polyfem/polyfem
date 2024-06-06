# WMTK (https://github.com/wildmeshing/wildmeshing-toolkit)
# License: MIT

if(TARGET wmtk::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'wmtk::toolkit'")

include(CPM)

if(POLYSOLVE_WITH_ACCELERATE)
    find_package(Patch REQUIRED)
    set(PATCH_COMMAND_ARGS "-rnN")

    file(GLOB_RECURSE patches_for_wmtk CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/wmtk_*.patch"
    )

    set(PATCH_COMMAND_FOR_CPM_BASE "${Patch_EXECUTABLE}" ${PATCH_COMMAND_ARGS} -p1 < )

    set(PATCH_COMMAND_FOR_CPM "")
    foreach(patch_filename IN LISTS patches_for_wmtk)
        list(APPEND PATCH_COMMAND_FOR_CPM ${PATCH_COMMAND_FOR_CPM_BASE})
        list(APPEND PATCH_COMMAND_FOR_CPM ${patch_filename})
        list(APPEND PATCH_COMMAND_FOR_CPM &&)
    endforeach()
    list(POP_BACK PATCH_COMMAND_FOR_CPM)

    message(DEBUG "Patch command: ${PATCH_COMMAND_FOR_CPM}")

    CPMAddPackage(
        NAME wildmeshing-toolkit
        GITHUB_REPOSITORY "zfergus/wildmeshing-toolkit"
        GIT_TAG "1ced6df948a29cd3f3028279c163ae582f44d37e"
        PATCH_COMMAND ${PATCH_COMMAND_FOR_CPM})
else()
    CPMAddPackage("gh:zfergus/wildmeshing-toolkit#1ced6df948a29cd3f3028279c163ae582f44d37e")
endif()