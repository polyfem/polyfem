set(SOURCES
	GeometryReader.cpp
	GeometryReader.hpp
	LocalBoundary.hpp
	Mesh.cpp
	Mesh.hpp
	MeshNodes.cpp
	MeshNodes.hpp
	MeshUtils.cpp
	MeshUtils.hpp
	Obstacle.cpp
	Obstacle.hpp
	VTUWriter.cpp
	VTUWriter.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})


add_subdirectory(mesh2D)
add_subdirectory(mesh3D)