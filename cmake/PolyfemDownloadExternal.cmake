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
        GIT_REPOSITORY https://github.com/Huangzizhou/polysolve.git
        GIT_TAG        c0af5c
    )
endfunction()

## libigl MPL
function(polyfem_download_libigl)
    polyfem_download_project(libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG        45cfc79fede992ea3923ded9de3c21d1c4faced1
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

## tbb Apache-2.0
function(polyfem_download_tbb)
    polyfem_download_project(tbb
        GIT_REPOSITORY https://github.com/nTopology/tbb.git
        GIT_TAG        41adc7a7fbe4e6d37fe57186bd85dde99fa61e66
    )
endfunction()

## OpenVDB
function(polyfem_download_openvdb)
    polyfem_download_project(openvdb
        GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openvdb.git
        GIT_TAG        v7.1.0
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
        GIT_TAG        57e21cf9c33d1b855d526fa896c2e1d656da7809
    )
endfunction()


# MshIO Apache-2.0
function(polyfem_download_mshio)
    polyfem_download_project(mshio
        GIT_REPOSITORY https://github.com/qnzhou/MshIO.git
        GIT_TAG        a500f107c1ca97bdcc9e53118e2d5964df11f539
    )
endfunction()




## data
function(polyfem_download_polyfem_data)
    polyfem_download_project(data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG        6b39f58be033d5cca57d5d3fd8ed206ae041e25d
    )
endfunction()
