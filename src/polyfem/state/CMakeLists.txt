set(SOURCES
	StateInit.cpp
	StateLoad.cpp
	StateSolve.cpp
	StateSolveLinear.cpp
	StateSolveNavierStokes.cpp
	StateSolveNonlinear.cpp
	StateOutput.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
