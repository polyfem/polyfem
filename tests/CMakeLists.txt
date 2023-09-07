# ###############################################################################
# Tests
# ###############################################################################

set(test_sources
  test_assembler.cpp
  test_bases.cpp
  test_diff.cpp
  test_opt.cpp
  test_cmesh.cpp
  test_collision_proxy.cpp
  test_form_derivatives.cpp
  test_geometry_utils.cpp
  test_hdf5.cpp
  test_interpolation.cpp
  test_matrix.cpp
  test_ncmesh.cpp
  test_normal.cpp
  test_output.cpp
  test_parametrizations.cpp
  test_problem.cpp
  test_quadrature.cpp
  test_rbf.cpp
  test_restart.cpp
  test_solver.cpp
  test_tbb.cpp
  test_time_integrators.cpp
  test_utils.cpp
  verify_run.cpp
)

add_executable(unit_tests ${test_sources})
# set_property(TARGET unit_tests PROPERTY COMPILE_WARNING_AS_ERROR ON)

################################################################################
# Required Libraries
################################################################################

target_link_libraries(unit_tests PUBLIC polyfem::polyfem)

include(polyfem_warnings)
target_link_libraries(unit_tests PUBLIC polyfem::warnings)

include(catch2)
target_link_libraries(unit_tests PUBLIC Catch2::Catch2WithMain)

include(finite_diff)
target_link_libraries(unit_tests PUBLIC finitediff::finitediff)

include(polyfem_data)
target_link_libraries(unit_tests PUBLIC polyfem::data)

################################################################################
# Compiler options
################################################################################

target_compile_definitions(unit_tests PUBLIC -DPOLYFEM_TEST_DIR=\"${CMAKE_SOURCE_DIR}/tests\")

target_compile_definitions(unit_tests PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

################################################################################
# Register tests
################################################################################

FetchContent_GetProperties(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(unit_tests)

################################################################################
# CUDA
################################################################################

if(IPC_TOOLKIT_WITH_CUDA)
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
