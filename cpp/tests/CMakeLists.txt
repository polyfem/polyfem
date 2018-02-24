cmake_minimum_required(VERSION 3.1)
################################################################################

add_executable(unit_tests
	main.cpp
	test_foo.cpp
)

target_link_libraries(unit_tests PUBLIC polyfem catch)

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(unit_tests)
