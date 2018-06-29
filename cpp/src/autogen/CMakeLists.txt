set(SOURCES
	auto_rhs.cpp
	auto_rhs.hpp
	auto_eigs.cpp
	auto_eigs.hpp
	auto_bases.cpp
	auto_bases.hpp
	auto_tetrahedron.ipp
	auto_triangle.ipp
)

polyfem_autogen(polyfem eigs.py auto_eigs)
polyfem_autogen(polyfem FEBases.py auto_bases)
polyfem_autogen(polyfem elasticity_rhs.py auto_rhs)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
polyfem_copy_headers(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
