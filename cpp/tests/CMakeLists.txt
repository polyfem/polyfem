cmake_minimum_required(VERSION 3.1)
################################################################################

set(test_sources
	main.cpp
	test_solver.cpp
	test_bases.cpp
	test_problem.cpp
	test_matrix.cpp
	test_utils.cpp
	test_tbb.cpp
)

add_executable(unit_tests ${test_sources})

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()

target_link_libraries(unit_tests PUBLIC polyfem catch)

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(unit_tests)
