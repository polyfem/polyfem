set(SOURCES
	CollisionProxy.cpp
	CollisionProxy.hpp
	UpsampleMesh.cpp
	UpsampleMesh.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})