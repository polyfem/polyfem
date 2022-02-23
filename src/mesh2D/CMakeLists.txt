set(SOURCES
	Mesh2D.cpp
	Mesh2D.hpp
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
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
