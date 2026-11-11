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
	PointTriangleDistance.cpp
	PointTriangleDistance.hpp
	SlimSmooth.cpp
	SlimSmooth.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# Reproduce the floating-point semantics geogram was compiled with (no FMA
# contraction), so that distances match the ones geogram's AABB computed.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	set_source_files_properties(PointTriangleDistance.cpp
		TARGET_DIRECTORY polyfem
		PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-frounding-math")
endif()

add_subdirectory(mesh2D)
add_subdirectory(mesh3D)
add_subdirectory(collision_proxy)
