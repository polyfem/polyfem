set(N_BASES
	p_n_bases.hpp
	p_n_bases.cpp
)

add_library(polyfem_n_bases ${N_BASES})
add_library(polyfem::n_bases ALIAS polyfem_n_bases)
target_link_libraries(polyfem_n_bases PRIVATE Eigen3::Eigen)

# -----------------------------------------------------------------------------

set(AUTOGEN_BASES
	auto_p_bases.cpp
	auto_p_bases.hpp
	auto_q_bases.cpp
	auto_q_bases.hpp
)

polyfem_set_source_group(${AUTOGEN_BASES})
if(WIN32)
	SET_SOURCE_FILES_PROPERTIES(${AUTOGEN_BASES} PROPERTIES COMPILE_FLAGS -Od)
endif()

add_library(polyfem_autogen ${AUTOGEN_BASES})
add_library(polyfem::autogen ALIAS polyfem_autogen)
target_link_libraries(polyfem_autogen PRIVATE Eigen3::Eigen polyfem::n_bases)
target_compile_features(polyfem_autogen PRIVATE cxx_std_11)

target_link_libraries(polyfem PRIVATE polyfem::autogen)

# -----------------------------------------------------------------------------

set(AUTOGEN
	auto_elasticity_rhs.cpp
	auto_elasticity_rhs.hpp
	auto_eigs.cpp
	auto_eigs.hpp
)

prepend_current_path(AUTOGEN)
polyfem_set_source_group(${AUTOGEN})
target_sources(polyfem PRIVATE ${AUTOGEN})

# -----------------------------------------------------------------------------

set(SOURCES
	auto_tetrahedron.ipp
	auto_triangle.ipp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})