cmake_minimum_required(VERSION 3.1)

polyfem_add_application(navigation_test.cpp polyfem geogram_gfx)
polyfem_add_application(refinement_test.cpp polyfem CLI11::CLI11)
polyfem_add_application(solver_test.cpp polyfem CLI11::CLI11)
polyfem_add_application(extract_polyhedra.cpp polyfem)
polyfem_add_application(dmat_to_geogram.cpp igl::core geogram CLI11::CLI11)

set_target_properties(navigation_test PROPERTIES FOLDER misc)
set_target_properties(refinement_test PROPERTIES FOLDER misc)
set_target_properties(solver_test PROPERTIES FOLDER misc)
set_target_properties(extract_polyhedra PROPERTIES FOLDER misc)
set_target_properties(dmat_to_geogram PROPERTIES FOLDER misc)
