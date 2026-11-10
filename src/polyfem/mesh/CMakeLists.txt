set(SOURCES
	GeometryLoader.cpp
	GeometryLoader.hpp
	LocalBoundary.cpp
	LocalBoundary.hpp
	Mesh.cpp
	Mesh.hpp
	MeshData.cpp
	MeshData.hpp
	MeshLoader.cpp
	MeshLoader.hpp
	MeshReader.cpp
	MeshReader.hpp
	MeshNodes.cpp
	MeshNodes.hpp
	MeshUtils.cpp
	MeshUtils.hpp
	MMGRemesh.cpp
	MMGRemesh.hpp
	Obstacle.cpp
	Obstacle.hpp
	SlimSmooth.cpp
	SlimSmooth.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(mesh2D)
add_subdirectory(mesh3D)
add_subdirectory(collision_proxy)
