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

# ...

################################################################################
# Optional libraries
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
