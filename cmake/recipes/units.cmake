# LLNL units
# License: BSD-3

if(TARGET units::units)
    return()
endif()

message(STATUS "Third-party: creating target 'units::units'")

SET(UNITS_ENABLE_TESTS OFF)
SET(UNITS_INSTALL OFF)

include(CPM)
CPMAddPackage("gh:LLNL/units@0.7.0")

