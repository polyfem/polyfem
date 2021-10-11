# Gulrak's Filesystem Library for C++11 and C++14 (https://github.com/gulrak/filesystem)
# License: MIT

if(TARGET ghc::filesystem)
    return()
endif()

message(STATUS "Third-party: creating target 'ghc::filesystem'")

include(FetchContent)
FetchContent_Declare(
    filesystem
    GIT_REPOSITORY https://github.com/gulrak/filesystem.git
    GIT_TAG v1.5.8
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(filesystem)
add_library(ghc::filesystem ALIAS ghc_filesystem)

# Check if we need to link against any special libraries (e.g., stdc++fs for GCC < 9)
set(SAVED_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
set(SAVED_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})

set(GHC_FILESYSTEM_TEST_CODE "\
    #include <ghc/fs_std.hpp>
    int main() {
        auto cwd = fs::current_path();
        return static_cast<int>(cwd.string().size());
    }")

# Try to compile a simple filesystem program without any linker flags
list(APPEND CMAKE_REQUIRED_INCLUDES "${filesystem_SOURCE_DIR}/include")
# TODO: I can't figure out how to specify check_cxx_source_compiles should use C++11,
# so hard code this flag here.
# check_cxx_source_compiles("${GHC_FILESYSTEM_TEST_CODE}" GHC_FILESYSTEM_NO_LINK_NEEDED)
set(GHC_FILESYSTEM_NO_LINK_NEEDED TRUE)

if(NOT GHC_FILESYSTEM_NO_LINK_NEEDED)
    # Add the libstdc++ flag
    set(CMAKE_REQUIRED_LIBRARIES ${SAVED_CMAKE_REQUIRED_LIBRARIES} -lstdc++fs)
    check_cxx_source_compiles("${GHC_FILESYSTEM_TEST_CODE}" GHC_FILESYSTEM_STDCPPFS_NEEDED)

    if(NOT GHC_FILESYSTEM_STDCPPFS_NEEDED)
        # Try the libc++ flag
        set(CMAKE_REQUIRED_LIBRARIES ${SAVED_CMAKE_REQUIRED_LIBRARIES} -lc++fs)
        check_cxx_source_compiles("${GHC_FILESYSTEM_TEST_CODE}" GHC_FILESYSTEM_CPPFS_NEEDED)
    endif()
endif()

if(GHC_FILESYSTEM_NO_LINK_NEEDED)
    # on certain linux distros we have a version of libstdc++ which has the final code for c++17 fs in the
    # libstdc++.so.*. BUT when compiling with g++ < 9, we MUST still link with libstdc++fs.a
    # libc++ should not suffer from this issue, so, in theory we should be fine with only checking for
    # GCC's libstdc++
    if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.0"))
        target_link_libraries(ghc_filesystem INTERFACE -lstdc++fs)
    endif()
elseif(GHC_FILESYSTEM_STDCPPFS_NEEDED)
    target_link_libraries(ghc_filesystem INTERFACE -lstdc++fs)
elseif(GHC_FILESYSTEM_CPPFS_NEEDED)
    target_link_libraries(ghc_filesystem INTERFACE -lc++fs)
else()
    message(FATAL_ERROR "Unable to determine correct linking options to compile GHC filesystem!")
endif()

set(CMAKE_REQUIRED_INCLUDES ${SAVED_CMAKE_REQUIRED_INCLUDES})
set(CMAKE_REQUIRED_LIBRARIES ${SAVED_CMAKE_REQUIRED_LIBRARIES})
