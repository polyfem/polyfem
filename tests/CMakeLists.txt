################################################################################
# Tests
################################################################################

set(test_sources
	main.cpp
	test_assembler.cpp
	test_bases.cpp
	test_matrix.cpp
	test_normal.cpp
	test_problem.cpp
	test_quadrature.cpp
	test_rbf.cpp
	test_solver.cpp
	test_tbb.cpp
	test_utils.cpp
)

add_executable(unit_tests ${test_sources})

################################################################################
# Required Libraries
################################################################################

target_link_libraries(unit_tests PUBLIC polyfem)

target_link_libraries(unit_tests PUBLIC warnings::all)

include(catch2)
target_link_libraries(unit_tests PUBLIC Catch2::Catch2)

################################################################################
# Compiler options
################################################################################

include(polyfem_data)
target_compile_definitions(unit_tests PUBLIC -DPOLYFEM_DATA_DIR=\"${POLYFEM_DATA_DIR}\")

target_compile_definitions(unit_tests PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

################################################################################
# Register tests
################################################################################

foreach(source IN ITEMS ${test_sources})
    source_group("tests" FILES "${source}")
endforeach()

set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)

# Include Catch2 and provide function `catch_discover_tests` to register tests.
FetchContent_GetProperties(catch2)
include("${catch2_SOURCE_DIR}/contrib/Catch.cmake")

catch_discover_tests(unit_tests)

################################################################################
# CUDA
################################################################################

if(true)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "No CUDA support found!")
  endif()

  # Nvidia 1080/1080Ti -> compute_61
  # Nvidia V100 -> compute_70
  # Nvidia RTX8000 -> compute_75
  # Nvidia 3080Ti -> compute_86
  set_target_properties(unit_tests PROPERTIES CUDA_ARCHITECTURES "70;75;86")
endif()
