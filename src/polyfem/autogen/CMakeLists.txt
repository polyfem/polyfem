set(AUTOGEN_BASES
	auto_p_bases.cpp
	auto_p_bases.hpp
	auto_p_bases_nodes.cpp
	auto_p_bases_nodes.hpp
	legacy_auto_p_bases.cpp
	legacy_auto_p_bases.hpp
	auto_q_bases_1d_val.cpp
	auto_q_bases_1d_nodes.cpp
	auto_q_bases_1d_grad.cpp

	auto_b_bases.cpp
	auto_b_bases.hpp
	legacy_auto_b_bases.cpp
	legacy_auto_b_bases.hpp

	auto_pyramid_bases.cpp
	auto_pyramid_bases.hpp
	auto_pyramid_bases_nodes.cpp
	auto_pyramid_bases_nodes.hpp
	legacy_auto_pyramid_bases.cpp
	legacy_auto_pyramid_bases.hpp

	auto_q_bases_2d_val.cpp
	auto_q_bases_2d_nodes.cpp
	auto_q_bases_2d_grad.cpp


	auto_q_bases_3d_val.cpp
	auto_q_bases_3d_nodes.cpp
	auto_q_bases_3d_grad.cpp
	auto_q_bases.hpp
	legacy_auto_q_bases.cpp
	legacy_auto_q_bases.hpp

	auto_q_bases_3d_grad_0.cpp
	auto_q_bases_3d_grad_1.cpp
	auto_q_bases_3d_grad_2.cpp
	auto_q_bases_3d_grad_3.cpp
	auto_q_bases_3d_grad_-2.cpp

	prism_bases.cpp
	prism_bases.hpp
)

set(AUTOGEN_CUDA_BASES
	auto_p_bases.cpp
	auto_b_bases.cpp
	auto_q_bases_1d_val.cpp
	auto_q_bases_1d_grad.cpp
	auto_q_bases_2d_val.cpp
	auto_q_bases_2d_grad.cpp
	auto_q_bases_3d_val.cpp
	auto_q_bases_3d_grad.cpp
	auto_pyramid_bases.cpp
)

set(AUTOGEN
	auto_elasticity_rhs.cpp
	auto_elasticity_rhs.hpp
	auto_eigs.cpp
	auto_eigs.hpp
	auto_mooney_rivlin_gradient_hessian.cpp
	auto_mooney_rivlin_gradient_hessian.hpp
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
target_include_directories(lib_n_bases PRIVATE ${PROJECT_BINARY_DIR}/include ${POLYFEM_INCLUDE_DIR})

target_link_libraries(lib_n_bases PRIVATE Eigen3::Eigen)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${AUTOGEN})



add_library(polyfem_autogen ${AUTOGEN_BASES})
target_include_directories(polyfem_autogen PRIVATE ${PROJECT_BINARY_DIR}/include ${POLYFEM_INCLUDE_DIR})
target_link_libraries(polyfem_autogen PRIVATE Eigen3::Eigen lib_n_bases)

if(POLYFEM_WITH_CUDA)
	target_compile_definitions(lib_n_bases PRIVATE POLYFEM_WITH_CUDA)
	target_link_libraries(lib_n_bases PRIVATE CCCL::CCCL)

	set_source_files_properties(${AUTOGEN_CUDA_BASES} PROPERTIES LANGUAGE CUDA)
	set_target_properties(polyfem_autogen PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
	)
	target_compile_definitions(polyfem_autogen PUBLIC POLYFEM_WITH_CUDA)
	target_compile_features(polyfem_autogen PRIVATE cuda_std_17)
	target_compile_options(polyfem_autogen PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
		$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
	)
	target_link_libraries(polyfem_autogen PUBLIC CCCL::CCCL)
endif()

target_compile_features(polyfem_autogen PRIVATE cxx_std_17)

if(MSVC)
	cmake_policy(SET CMP0141 NEW)

	set_property(TARGET polyfem_autogen PROPERTY MSVC_DEBUG_INFORMATION_FORMAT "")

	target_compile_options(polyfem_autogen PRIVATE 
	/Oy-
	/bigobj)

	SET_SOURCE_FILES_PROPERTIES(${AUTOGEN_BASES} PROPERTIES COMPILE_OPTIONS "/Od;/Ob0;/GL-")
endif()

target_link_libraries(polyfem PRIVATE polyfem_autogen)
target_sources(polyfem PRIVATE ${SOURCES} ${AUTOGEN})
