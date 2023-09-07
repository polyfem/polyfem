set(SOURCES
	GeometryReader.cpp
	GeometryReader.hpp
	LocalBoundary.cpp
	LocalBoundary.hpp
	Mesh.cpp
	Mesh.hpp
	MeshNodes.cpp
	MeshNodes.hpp
	MeshUtils.cpp
	MeshUtils.hpp
	Obstacle.cpp
	Obstacle.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})


add_subdirectory(mesh2D)
add_subdirectory(mesh3D)
add_subdirectory(collision_proxy)