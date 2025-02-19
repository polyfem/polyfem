set(SOURCES
	Collapse.cpp
	OperationCache.hpp
	OperationCache.cpp
	Execute.cpp
	LocalMesh.cpp
	LocalMesh.hpp
	LocalRelaxation.cpp
	LocalRelaxationData.cpp
	LocalRelaxationData.hpp
	Smooth.cpp
	Split.cpp
	Swap.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
