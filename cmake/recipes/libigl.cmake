# libigl (https://github.com/libigl/libigl)
# License: MPL

#
# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#

if(TARGET igl::core)
    return()
endif()

message(STATUS "Third-party: creating target 'igl::core'")

set(LIBIGL_PREDICATES ON CACHE BOOL "Use exact predicates" FORCE)

include(CMakeDependentOption)
cmake_dependent_option(LIBIGL_RESTRICTED_TRIANGLE "Build target igl_restricted::triangle" ON "POLYFEM_WITH_TRIANGLE" ON)

include(eigen)

include(CPM)
if(POLYSOLVE_WITH_ACCELERATE)
    find_package(Patch REQUIRED)
    set(PATCH_COMMAND_ARGS "-rnN")

    file(GLOB_RECURSE patches_for_libigl CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/igl_*.patch"
    )

    set(PATCH_COMMAND_FOR_CPM_BASE "${Patch_EXECUTABLE}" ${PATCH_COMMAND_ARGS} -p1 < )

    set(PATCH_COMMAND_FOR_CPM "")
    foreach(patch_filename IN LISTS patches_for_libigl)
        list(APPEND PATCH_COMMAND_FOR_CPM ${PATCH_COMMAND_FOR_CPM_BASE})
        list(APPEND PATCH_COMMAND_FOR_CPM ${patch_filename})
        list(APPEND PATCH_COMMAND_FOR_CPM &&)
    endforeach()
    list(POP_BACK PATCH_COMMAND_FOR_CPM)

    message(DEBUG "Patch command: ${PATCH_COMMAND_FOR_CPM}")

    CPMAddPackage(
        NAME libigl
        GITHUB_REPOSITORY "libigl/libigl"
        GIT_TAG "v2.5.0"
        PATCH_COMMAND ${PATCH_COMMAND_FOR_CPM})
else()
    CPMAddPackage("gh:libigl/libigl@2.5.0")
endif()