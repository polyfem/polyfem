set(SOURCES
	AMIPSForm.cpp
	AMIPSForm.hpp
	Collapse.cpp
	OperationCache.hpp
	OperationCache.cpp
	Execute.cpp
	LocalMesh.cpp
	LocalMesh.hpp
	LocalRelaxation.cpp
	ProjectQuantities.cpp
	Smooth.cpp
	Split.cpp
	Swap.cpp
	Timings.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
