################################################################################
include(DownloadProject)

# Shortcut function
function(polyfem_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${POLYFEM_EXTERNAL}/${name}
        DOWNLOAD_DIR ${POLYFEM_EXTERNAL}/.cache/${name}
        ${ARGN}
    )
endfunction()

################################################################################

## libigl
function(polyfem_download_libigl)
    polyfem_download_project(libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG        8d5533a3088b2d471dbfd56f0a5bdaafe30d013a
    )
endfunction()

## Geogram
function(polyfem_download_geogram)
    polyfem_download_project(geogram
        GIT_REPOSITORY https://github.com/alicevision/geogram.git
        GIT_TAG        v1.6.8
    )
endfunction()

## Json
function(polyfem_download_json)
    polyfem_download_project(json
        GIT_REPOSITORY https://github.com/jdumas/json
        GIT_TAG        0901d33bf6e7dfe6f70fd9d142c8f5c6695c6c5b
    )
endfunction()

## Catch2
function(polyfem_download_catch2)
    polyfem_download_project(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v2.4.1
    )
endfunction()


# [submodule "cpp/3rdparty/tinyformat"]
#     path = cpp/3rdparty/tinyformat
#     url = https://github.com/c42f/tinyformat.git

## CppNumericalSolvers
function(polyfem_download_CppNumericalSolvers)
    polyfem_download_project(CppNumericalSolvers
        GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
        GIT_TAG        7eddf28fa5a8872a956d3c8666055cac2f5a535d
    )
endfunction()

## spectra
function(polyfem_download_spectra)
    polyfem_download_project(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG        v0.6.2
    )
endfunction()

## tbb
function(polyfem_download_tbb)
    polyfem_download_project(tbb
        GIT_REPOSITORY https://github.com/wjakob/tbb.git
        GIT_TAG        08b4341a1893a72656467e96137f1f99d0112547
    )
endfunction()

## hypre
function(polyfem_download_hypre)
    polyfem_download_project(hypre
        GIT_REPOSITORY https://github.com/LLNL/hypre.git
        GIT_TAG        v2.15.1
    )
endfunction()


## rbf
function(polyfem_download_rbf)
    polyfem_download_project(rbf
        GIT_REPOSITORY https://bitbucket.org/zulianp/opencl-rbf-pum.git
        GIT_TAG        master
    )
endfunction()

## tinyexpr
function(polyfem_download_tinyexpr)
    polyfem_download_project(tinyexpr
        GIT_REPOSITORY https://github.com/codeplea/tinyexpr.git
        GIT_TAG        ffb0d41b13e5f8d318db95feb071c220c134fe70
    )
endfunction()



## Sanitizers
function(polyfem_download_sanitizers)
    polyfem_download_project(sanitizers-cmake
        GIT_REPOSITORY https://github.com/arsenm/sanitizers-cmake.git
        GIT_TAG        6947cff3a9c9305eb9c16135dd81da3feb4bf87f
    )
endfunction()


## spdlog
function(polyfem_download_spdlog)
    polyfem_download_project(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        188cff7d6567b80c6b99bc15899fef9637a8fe52
    )
endfunction()
