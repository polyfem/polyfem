cmake_minimum_required(VERSION 3.1)
################################################################################

set(test_sources
	main.cpp
	test_bases.cpp
	test_matrix.cpp
	test_normal.cpp
	test_problem.cpp
	test_quadrature.cpp
	test_solver.cpp
	test_tbb.cpp
	test_utils.cpp
)

polyfem_download_polyfem_data()

add_executable(unit_tests ${test_sources})

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()

target_link_libraries(unit_tests PUBLIC polyfem catch warnings::all)

set(DATA_DIR "${THIRD_PARTY_DIR}/data/")
target_compile_definitions(unit_tests PUBLIC -DPOLYFEM_DATA_DIR=\"${DATA_DIR}\")


if(POLYFEM_WITH_SANITIZERS)
	add_sanitizers(unit_tests)
endif()

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
include(Catch)
catch_discover_tests(unit_tests)
