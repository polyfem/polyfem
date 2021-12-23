################################################################################
include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(POLYFEM_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(POLYFEM_EXTRA_OPTIONS "")
endif()

# Shortcut function
function(polyfem_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${POLYFEM_EXTERNAL}/${name}
        DOWNLOAD_DIR ${POLYFEM_EXTERNAL}/.cache/${name}
        QUIET
        ${POLYFEM_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################


## Geogram BSD
function(polyfem_download_geogram)
    polyfem_download_project(geogram
        GIT_REPOSITORY https://github.com/polyfem/geogram.git
        GIT_TAG        14001ad54398a478681c6bba7eaf10729068b5ce
    )
endfunction()


## CppNumericalSolvers MIT
function(polyfem_download_CppNumericalSolvers)
    polyfem_download_project(CppNumericalSolvers
        GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
        GIT_TAG        7eddf28fa5a8872a956d3c8666055cac2f5a535d
    )
endfunction()



## tinyxml zlib
function(polyfem_download_tinyxml)
    polyfem_download_project(tinyxml
        GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
        GIT_TAG        9.0.0
    )
endfunction()


## data
function(polyfem_download_polyfem_data)
    polyfem_download_project(data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG        c810c5547bd238f68558433f1829b0fad39ac2f2
    )
endfunction()
