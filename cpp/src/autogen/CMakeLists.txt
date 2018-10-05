set(AUTOGEN
	auto_rhs.cpp
	auto_rhs.hpp
	auto_eigs.cpp
	auto_eigs.hpp
	auto_p_bases.cpp
	auto_p_bases.hpp
	auto_q_bases.cpp
	auto_q_bases.hpp
)

set(SOURCES
	auto_tetrahedron.ipp
	auto_triangle.ipp
)

prepend_current_path(SOURCES)
prepend_current_path(AUTOGEN)
polyfem_set_source_group(${SOURCES} ${AUTOGEN})
polyfem_copy_headers(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES} ${AUTOGEN})
