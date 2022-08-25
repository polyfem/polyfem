set(SOURCES
	StateInit.cpp
	StateInterpolation.cpp
	StateLoad.cpp
	StateOutput.cpp
	StatePref.cpp
	StateHomogenization.cpp
	StateSolve.cpp
	StateDiff.cpp
	StateSolveLinear.cpp
	StateSolveNavierStokes.cpp
	StateSolveNonlinear.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
