set(SOURCES
	Parametrization.hpp
	Parametrization.cpp
	Parametrizations.hpp
	Parametrizations.cpp
	PeriodicMeshToMesh.hpp
	PeriodicMeshToMesh.cpp
	SplineParametrizations.hpp
	SplineParametrizations.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
