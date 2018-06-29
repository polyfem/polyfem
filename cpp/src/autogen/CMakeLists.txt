set(SOURCES
	auto_rhs.cpp
	auto_rhs.hpp
	auto_eigs.cpp
	auto_eigs.hpp
	auto_bases.cpp
	auto_bases.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
