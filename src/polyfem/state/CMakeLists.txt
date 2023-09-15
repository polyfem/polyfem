set(SOURCES
	StateDiff.cpp
	StateInit.cpp
	StateLoad.cpp
	StateOutput.cpp
	StateRemesh.cpp
	StateSolve.cpp
	StateSolveLinear.cpp
	StateSolveNavierStokes.cpp
	StateSolveNonlinear.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
