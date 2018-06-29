set(SOURCES
	Mesh3D.cpp
	Mesh3D.hpp
	Mesh3DStorage.hpp
	MeshProcessing3D.cpp
	MeshProcessing3D.hpp
	Navigation3D.cpp
	Navigation3D.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
