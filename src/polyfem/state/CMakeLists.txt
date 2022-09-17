set(SOURCES
	StateInit.cpp
	StateLoad.cpp
	StateOutput.cpp
	StatePref.cpp
	StateSolve.cpp
	StateSolveLinear.cpp
	StateSolveNavierStokes.cpp
	StateSolveNonlinear.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
