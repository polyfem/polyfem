if(TARGET finitediff::finitediff)
    return()
endif()

message(STATUS "Third-party: creating target 'finitediff::finitediff'")

include(FetchContent)
FetchContent_Declare(
    finite-diff
    GIT_REPOSITORY https://github.com/zfergus/finite-diff.git
    GIT_TAG v1.0.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(finite-diff)
