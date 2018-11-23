# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.

### Configuration
set(POLYFEM_ROOT     "${CMAKE_CURRENT_LIST_DIR}/..")
set(POLYFEM_EXTERNAL "${POLYFEM_ROOT}/3rdparty")

# Download and update 3rdparty libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(PolyfemDownloadExternal)

################################################################################
# Required libraries
################################################################################

# Sanitizers
if(POLYFEM_WITH_SANITIZERS)
    polyfem_download_sanitizers()
    find_package(Sanitizers)
endif()

# spdlog
if(NOT TARGET spdlog::spdlog)
	polyfem_download_spdlog()
	add_subdirectory(${POLYFEM_EXTERNAL}/spdlog)
endif()

# CL11
if(NOT TARGET CLI11::CLI11)
    polyfem_download_cli11()
    add_subdirectory(${POLYFEM_EXTERNAL}/cli11)
endif()

# Clipper
if(NOT TARGET clipper::clipper)
	polyfem_download_clipper()
	add_library(clipper_clipper ${POLYFEM_EXTERNAL}/clipper/cpp/clipper.cpp)
	add_library(clipper::clipper ALIAS clipper_clipper)
	target_include_directories(clipper_clipper PUBLIC ${POLYFEM_EXTERNAL}/clipper/cpp)
endif()

# Nanosvg
if(NOT TARGET nanosvg::nanosvg)
	polyfem_download_nanosvg()
	add_library(nanosvg_nanosvg INTERFACE)
	add_library(nanosvg::nanosvg ALIAS nanosvg_nanosvg)
	target_include_directories(nanosvg_nanosvg INTERFACE ${POLYFEM_EXTERNAL}/nanosvg)
endif()

# Tiny file dialogs
if(NOT TARGET tinyfiledialogs::tinyfiledialogs)
	polyfem_download_tinyfiledialogs()
	set(TINYFILEDIALOGS_DIR "${POLYFEM_EXTERNAL}/tinyfiledialogs")
	add_library(tinyfiledialogs_tinyfiledialogs ${TINYFILEDIALOGS_DIR}/tinyfiledialogs.c)
	add_library(tinyfiledialogs::tinyfiledialogs ALIAS tinyfiledialogs_tinyfiledialogs)
	target_include_directories(tinyfiledialogs_tinyfiledialogs SYSTEM INTERFACE ${TINYFILEDIALOGS_DIR})
	set_target_properties(tinyfiledialogs_tinyfiledialogs PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
