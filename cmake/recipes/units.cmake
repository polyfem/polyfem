# LLNL units
# License: BSD-3

if(TARGET units::units)
    return()
endif()

message(STATUS "Third-party: creating target 'units::units'")

SET(UNITS_ENABLE_TESTS OFF)
SET(UNITS_INSTALL OFF)

include(FetchContent)
FetchContent_Declare(
    units
    GIT_REPOSITORY https://github.com/LLNL/units.git
    GIT_TAG v0.7.0
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(units)
