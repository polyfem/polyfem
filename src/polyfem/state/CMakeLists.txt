set(SOURCES
	StateInit.cpp
	StateLoad.cpp
	StateHomogenization.cpp
	StateOutput.cpp
	StateSolve.cpp
	StateSolveLinear.cpp
	StateSolveNavierStokes.cpp
	StateSolveNonlinear.cpp
)

if(POLYFEM_WITH_ITR)
	list(APPEND SOURCES StateRemesh.cpp)
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
