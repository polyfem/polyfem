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

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
