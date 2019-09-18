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

## libigl MPL
function(polyfem_download_libigl)
    polyfem_download_project(libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG        c7c06e3735cdf6188bd17507403362065c4ae9dc
    )
endfunction()

## Geogram BSD
function(polyfem_download_geogram)
    polyfem_download_project(geogram
        GIT_REPOSITORY https://github.com/alicevision/geogram.git
        GIT_TAG        v1.6.8
    )
endfunction()

## Json MIT
function(polyfem_download_json)
    polyfem_download_project(json
        GIT_REPOSITORY https://github.com/jdumas/json
        GIT_TAG        0901d33bf6e7dfe6f70fd9d142c8f5c6695c6c5b
    )
endfunction()

## Catch2 BSL 1.0 optional
function(polyfem_download_catch2)
    polyfem_download_project(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v2.4.2
    )
endfunction()

## CLI11 3-Clause BSD license optional
function(polyfem_download_cli11)
    polyfem_download_project(cli11
        URL     https://github.com/CLIUtils/CLI11/archive/v1.6.1.tar.gz
        URL_MD5 48ef97262adb0b47a2f0a7edbda6e2aa
    )
endfunction()

# Clipper (BSL1.0)
function(polyfem_download_clipper)
    polyfem_download_project(clipper
        URL     https://sourceforge.net/projects/polyclipping/files/clipper_ver6.4.2.zip
        URL_MD5 100b4ec56c5308bac2d10f3966e35e11
    )
endfunction()

## CppNumericalSolvers MIT
function(polyfem_download_CppNumericalSolvers)
    polyfem_download_project(CppNumericalSolvers
        GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
        GIT_TAG        7eddf28fa5a8872a956d3c8666055cac2f5a535d
    )
endfunction()

## spectra MPL 2.0 optional
function(polyfem_download_spectra)
    polyfem_download_project(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG        v0.6.2
    )
endfunction()

## tbb Apache-2.0
function(polyfem_download_tbb)
    polyfem_download_project(tbb
        GIT_REPOSITORY https://github.com/wjakob/tbb.git
        GIT_TAG        344fa84f34089681732a54f5def93a30a3056ab9
    )
endfunction()

## hypre GNU Lesser General Public License
function(polyfem_download_hypre)
    polyfem_download_project(hypre
        GIT_REPOSITORY https://github.com/LLNL/hypre.git
        GIT_TAG        v2.15.1
    )
endfunction()


## Sanitizers MIT optional
function(polyfem_download_sanitizers)
    polyfem_download_project(sanitizers-cmake
        GIT_REPOSITORY https://github.com/arsenm/sanitizers-cmake.git
        GIT_TAG        6947cff3a9c9305eb9c16135dd81da3feb4bf87f
    )
endfunction()

## spdlog MIT
function(polyfem_download_spdlog)
    polyfem_download_project(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG         v1.3.1
    )
endfunction()

## tinyexpr zlib
function(polyfem_download_tinyexpr)
    polyfem_download_project(tinyexpr
        GIT_REPOSITORY https://github.com/polyfem/tinyexpr.git
        GIT_TAG        eb73c7e4005195bf5c0f1fa28dee3b489d59f821
    )
endfunction()


## data
function(polyfem_download_polyfem_data)
    polyfem_download_project(data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG        6b39f58be033d5cca57d5d3fd8ed206ae041e25d
    )
endfunction()
