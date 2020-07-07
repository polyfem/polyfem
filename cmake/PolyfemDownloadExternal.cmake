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
function(polyfem_download_solvers)
    polyfem_download_project(solvers
        GIT_REPOSITORY https://github.com/Huangzizhou/polysolve.git
        GIT_TAG        97250e9f36ff566041423ed13c87c3cc1dc1f6c0
    )
endfunction()


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
        URL     https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
        URL_MD5 5e5470abcb76422360409297bfc446ac
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
        GIT_REPOSITORY https://github.com/nTopology/tbb.git
        GIT_TAG        41adc7a7fbe4e6d37fe57186bd85dde99fa61e66
    )
endfunction()

## hypre GNU Lesser General Public License
function(polyfem_download_hypre)
    polyfem_download_project(hypre
        GIT_REPOSITORY https://github.com/LLNL/hypre.git
        GIT_TAG        v2.15.1
    )

    file(REMOVE ${POLYFEM_EXTERNAL}/hypre/src/utilities/version)
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


## amgcl mit
function(polyfem_download_amgcl)
    polyfem_download_project(amgcl
        GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
        GIT_TAG        a2fab1037946de87e448e5fc7539277cd6fb9ec3
    )
endfunction()


## data
function(polyfem_download_polyfem_data)
    polyfem_download_project(data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG        6b39f58be033d5cca57d5d3fd8ed206ae041e25d
    )
endfunction()
