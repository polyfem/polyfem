set(SOURCES
	StateInit.cpp
	StateLoad.cpp
	StateSolve.cpp
	StateDiff.cpp
	StateSolveLinear.cpp
	StateSolveNavierStokes.cpp
	StateSolveNonlinear.cpp
	StateOutput.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
