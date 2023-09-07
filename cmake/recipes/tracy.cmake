#
# Copyright 2021 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#

# Tracy (https://github.com/wolfpld/tracy)
# License: BSD-3-Clause

if(TARGET Tracy::TracyClient)
    return()
endif()

message(STATUS "Third-party: creating target 'Tracy::TracyClient'")

option(TRACY_ENABLE "Enable profiling with Tracy" OFF)

include(CPM)
CPMAddPackage("gh:wolfpld/tracy@0.7.8")

################################################################################
# Tracy lib
################################################################################

find_package(Threads REQUIRED)

add_library(TracyClient SHARED
    ${tracy_SOURCE_DIR}/TracyClient.cpp
    ${tracy_SOURCE_DIR}/Tracy.hpp
)
add_library(Tracy::TracyClient ALIAS TracyClient)

include(GNUInstallDirs)
target_include_directories(TracyClient SYSTEM PUBLIC
    $<BUILD_INTERFACE:${tracy_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(TracyClient PUBLIC cxx_std_11)

target_link_libraries(TracyClient PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

macro(set_option option help value)
    option(${option} ${help} ${value})
    if(${option})
        message(STATUS "${option}: ON")
        target_compile_definitions(TracyClient PUBLIC ${option})
    else()
        message(STATUS "${option}: OFF")
    endif()
endmacro()

set_option(TRACY_ENABLE "Enable profiling" ON)
set_option(TRACY_ON_DEMAND "On-demand profiling" OFF)
set_option(TRACY_CALLSTACK "Collect call stacks" OFF)
set_option(TRACY_ONLY_LOCALHOST "Only listen on the localhost interface" OFF)
set_option(TRACY_NO_BROADCAST "Disable client discovery by broadcast to local network" OFF)
set_option(TRACY_NO_CODE_TRANSFER "Disable collection of source code" OFF)
set_option(TRACY_NO_CONTEXT_SWITCH "Disable capture of context switches" OFF)
set_option(TRACY_NO_EXIT "Client executable does not exit until all profile data is sent to server" OFF)
set_option(TRACY_NO_FRAME_IMAGE "Disable capture of frame images" OFF)
set_option(TRACY_NO_SAMPLING "Disable call stack sampling" OFF)
set_option(TRACY_NO_VERIFY "Disable zone validation for C API" OFF)
set_option(TRACY_NO_VSYNC_CAPTURE "Disable capture of hardware Vsync events" OFF)

# if(BUILD_SHARED_LIBS)
    target_compile_definitions(TracyClient PRIVATE TRACY_EXPORTS)
    target_compile_definitions(TracyClient PUBLIC TRACY_IMPORTS)
# endif()

################################################################################
# Global flags
################################################################################

function(tracy_filter_flags flags)
    include(CheckCXXCompilerFlag)
    set(output_flags)
    foreach(FLAG IN ITEMS ${${flags}})
        string(REPLACE "=" "-" FLAG_VAR "${FLAG}")
        if(NOT DEFINED IS_SUPPORTED_${FLAG_VAR})
            check_cxx_compiler_flag("${FLAG}" IS_SUPPORTED_${FLAG_VAR})
        endif()
        if(IS_SUPPORTED_${FLAG_VAR})
            list(APPEND output_flags ${FLAG})
        endif()
    endforeach()
    set(${flags} ${output_flags} PARENT_SCOPE)
endfunction()

if(TRACED_ENABLE)
    set(TRACY_GLOBAL_FLAGS
        "-fno-omit-frame-pointer"
    )
    tracy_filter_flags(TRACY_GLOBAL_FLAGS)
    message(STATUS "Adding global flags: ${TRACY_GLOBAL_FLAGS}")
    # add_compile_options(${TRACY_GLOBAL_FLAGS})
    target_compile_options(TracyClient PUBLIC ${TRACY_GLOBAL_FLAGS})
endif()