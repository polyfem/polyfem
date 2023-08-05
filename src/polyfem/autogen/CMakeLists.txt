set(AUTOGEN_BASES
	auto_p_bases.cpp
	auto_p_bases.hpp
	auto_q_bases.cpp
	auto_q_bases.hpp
)

set(AUTOGEN
	auto_elasticity_rhs.cpp
	auto_elasticity_rhs.hpp
	auto_eigs.cpp
	auto_eigs.hpp
)

set(N_BASES
	p_n_bases.hpp
	p_n_bases.cpp
)

set(SOURCES
	auto_tetrahedron.ipp
	auto_triangle.ipp
)
add_library(lib_n_bases ${N_BASES})
target_include_directories(lib_n_bases PRIVATE ${PROJECT_BINARY_DIR}/include)

target_link_libraries(lib_n_bases PRIVATE Eigen3::Eigen)

prepend_current_path(SOURCES)
prepend_current_path(AUTOGEN)
polyfem_set_source_group(${SOURCES} ${AUTOGEN} ${AUTOGEN_BASES})
if(WIN32)
	SET_SOURCE_FILES_PROPERTIES(${AUTOGEN_BASES} PROPERTIES COMPILE_FLAGS -Od)
endif()

add_library(polyfem_autogen ${AUTOGEN_BASES})
target_include_directories(polyfem_autogen PRIVATE ${PROJECT_BINARY_DIR}/include)
target_link_libraries(polyfem_autogen PRIVATE Eigen3::Eigen lib_n_bases)
# Use C++14
target_compile_features(polyfem_autogen PRIVATE ${CXX11_FEATURES})


target_link_libraries(polyfem PRIVATE polyfem_autogen)
target_sources(polyfem PRIVATE ${SOURCES} ${AUTOGEN})

