set(SOURCES
	L2Projection.cpp
	L2Projection.hpp
	MMGRemesh.cpp
	MMGRemesh.hpp
	Remesh.hpp
	Remesh.cpp
	WildRemesh2D.cpp
	WildRemesh2D.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(wild_remesh)