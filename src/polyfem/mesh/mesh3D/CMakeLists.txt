set(SOURCES
	Mesh3D.cpp
	Mesh3D.hpp
	CMesh3D.cpp
	CMesh3D.hpp
	NCMesh3D.cpp
	NCMesh3D.hpp
	Mesh3DStorage.hpp
	MeshProcessing3D.cpp
	MeshProcessing3D.hpp
	Navigation3D.cpp
	Navigation3D.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
