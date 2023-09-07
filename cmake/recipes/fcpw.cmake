# FCPW (https://github.com/rohan-sawhney/fcpw)
# MIT license
if(TARGET fcpw::fcpw)
    return()
endif()

message(STATUS "Third-party: creating target 'fcpw::fcpw'")

option(FCPW_USE_ENOKI "Build enoki" OFF)
option(FCPW_USE_EIGHT_WIDE_BRANCHING "Use 8 wide branching (default 4)" OFF)

include(CPM)
CPMAddPackage(
    NAME fcpw
    GITHUB_REPOSITORY rohan-sawhney/fcpw
    GIT_TAG a9a03107b96f64fca4fe4349d06435f0ff2ec800
    DOWNLOAD_ONLY TRUE
)

add_library(fcpw INTERFACE)
add_library(fcpw::fcpw ALIAS fcpw)
target_include_directories(fcpw SYSTEM INTERFACE ${fcpw_SOURCE_DIR}/include)
target_compile_features(fcpw INTERFACE cxx_std_17)

include(eigen)
target_link_libraries(fcpw INTERFACE Eigen3::Eigen)

if(FCPW_USE_ENOKI)
    include(enoki)

    # Update the compilation flags
    enoki_set_compile_flags()
    enoki_set_native_flags()

    target_compile_definitions(fcpw INTERFACE -DFCPW_USE_ENOKI)

    # define SIMD width
    string(TOUPPER "${ENOKI_ARCH_FLAGS}" ENOKI_ARCH_FLAGS_UPPER)
    message(STATUS "Enoki Max ISA: " ${ENOKI_ARCH_FLAGS_UPPER})
    if(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SSE")
        target_compile_definitions(fcpw INTERFACE -DFCPW_SIMD_WIDTH=4)
    elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX2")
        target_compile_definitions(fcpw INTERFACE -DFCPW_SIMD_WIDTH=8)
    elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX")
        target_compile_definitions(fcpw INTERFACE -DFCPW_SIMD_WIDTH=8)
    elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "KNL")
        target_compile_definitions(fcpw INTERFACE -DFCPW_SIMD_WIDTH=16)
    elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SKX")
        target_compile_definitions(fcpw INTERFACE -DFCPW_SIMD_WIDTH=16)
    else()
        target_compile_definitions(fcpw INTERFACE -DFCPW_SIMD_WIDTH=4)
    endif()

    if(FCPW_USE_EIGHT_WIDE_BRANCHING)
        target_compile_definitions(fcpw INTERFACE -DFCPW_USE_EIGHT_WIDE_BRANCHING)
    endif()
endif()