set(SOURCES
	StateInit.cpp
	StateInterpolation.cpp
	StateLoad.cpp
	StateOutput.cpp
	StatePref.cpp
	StateSolve.cpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
