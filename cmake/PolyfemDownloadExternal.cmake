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

## PolySolvers MIT
function(polyfem_download_solvers)
    polyfem_download_project(solvers
        GIT_REPOSITORY https://github.com/polyfem/polysolve.git
        GIT_TAG        4ade461c30f6e80e43aa504f94f6d2d4c3bfbdd6
    )
endfunction()

## libigl MPL
function(polyfem_download_libigl)
    polyfem_download_project(libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG        v2.3.0
    )
endfunction()

## Geogram BSD
function(polyfem_download_geogram)
    polyfem_download_project(geogram
        GIT_REPOSITORY https://github.com/polyfem/geogram.git
        GIT_TAG        e6b9612f1146370e40deaa341b4dd7ef90502102
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

## Clipper (BSL1.0)
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

## tbb Apache-2.0
function(polyfem_download_tbb)
    polyfem_download_project(tbb
        GIT_REPOSITORY https://github.com/wjakob/tbb.git
        GIT_TAG        141b0e310e1fb552bdca887542c9c1a8544d6503
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
        GIT_TAG        v1.3.1
    )
endfunction()

## tinyexpr zlib
function(polyfem_download_tinyexpr)
    polyfem_download_project(tinyexpr
        GIT_REPOSITORY https://github.com/polyfem/tinyexpr.git
        GIT_TAG        eb73c7e4005195bf5c0f1fa28dee3b489d59f821
    )
endfunction()

## tinyxml zlib
function(polyfem_download_tinyxml)
    polyfem_download_project(tinyxml
        GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
        GIT_TAG        2c5a6bfdd42ab919e55a613d33c83eb53de71af4
    )
endfunction()

## ipc MIT
function(polyfem_download_ipc)
    polyfem_download_project(ipc
        GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
        GIT_TAG        3342227c61703f83a262c1c7ff8af838f267e2d7
    )
endfunction()

## BVH MIT
function(polyfem_download_BVH)
    polyfem_download_project(BVH
        GIT_REPOSITORY https://github.com/geometryprocessing/SimpleBVH.git
        GIT_TAG        15574502f6cb8039b0bfa4a85ccad04e09deaf05
    )
endfunction()

## MshIO Apache-2.0
function(polyfem_download_mshio)
    polyfem_download_project(mshio
        GIT_REPOSITORY https://github.com/qnzhou/MshIO.git
        GIT_TAG        201eeba436e38043b7e716be82ec5e218cbae74d
    )
endfunction()

## data
function(polyfem_download_polyfem_data)
    polyfem_download_project(data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG        6b39f58be033d5cca57d5d3fd8ed206ae041e25d
    )
endfunction()

## filesystem library for C++11 and C++14 (MIT)
function(polyfem_download_filesystem)
    polyfem_download_project(filesystem
        GIT_REPOSITORY https://github.com/gulrak/filesystem.git
        GIT_TAG        v1.5.4
    )
endfunction()
