# Eigen (https://gitlab.com/libeigen/eigen)
# License: MPL2

if(TARGET Eigen3::Eigen)
    return()
endif()

message(STATUS "Third-party: creating target 'Eigen3::Eigen'")

include(CPM)
CPMAddPackage(
    NAME eigen
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 5.0.1
    DOWNLOAD_ONLY TRUE
)

add_library(Eigen3_Eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)

target_include_directories(Eigen3_Eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
)
