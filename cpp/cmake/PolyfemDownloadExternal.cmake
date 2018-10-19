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

## Sanitizers
function(polyfem_download_libigl)
    polyfem_download_project(libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG        e6116bf9ddca2a98cc1ee8cf0a8f99d70c501119
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
