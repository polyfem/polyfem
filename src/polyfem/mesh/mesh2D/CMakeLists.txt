set(SOURCES
	Mesh2D.cpp
	Mesh2D.hpp
	NCMesh2D.cpp
	NCMesh2D.hpp
	CMesh2D.cpp
	CMesh2D.hpp
	Navigation.cpp
	Navigation.hpp
	PolygonUtils.cpp
	PolygonUtils.hpp
	Refinement.cpp
	Refinement.hpp
	Singularities.cpp
	Singularities.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
