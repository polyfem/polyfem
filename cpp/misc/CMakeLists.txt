cmake_minimum_required(VERSION 3.1)

################################################################################

function(add_application APP_SOURCE)
	# Add executable
	get_filename_component(APP_NAME ${APP_SOURCE} NAME_WE)
	add_executable(${APP_NAME} ${APP_SOURCE})
	message(STATUS "Compiling single-source application: ${APP_NAME}")

	# Dependencies
	target_link_libraries(${APP_NAME} PRIVATE ${ARGN})

	# Output directory for binaries
	set_target_properties(${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

	if(POLYFEM_WITH_SANITIZERS)
		add_sanitizers(${APP_NAME})
	endif()
endfunction()

################################################################################

add_application(navigation_test.cpp polyfem geogram_gfx)
add_application(refinement_test.cpp polyfem cli11)
add_application(solver_test.cpp polyfem cli11)
add_application(extrude_mesh.cpp polyfem cli11)
add_application(square_beam.cpp polyfem cli11)
# add_application(svg_to_mesh.cpp cli11 nanosvg polyclipping igl::core igl::triangle)
add_application(extract_polyhedra.cpp polyfem)
add_application(tetrahedralize.cpp igl::core igl::tetgen geogram)
add_application(tetrahedralize_hybrid.cpp polyfem igl::core igl::tetgen geogram cli11)
add_application(bc_setter.cpp polyfem igl::core igl::tetgen geogram cli11)
add_application(b_extractor.cpp polyfem igl::core geogram cli11)
add_application(param.cpp igl::core igl::tetgen igl::opengl_glfw_imgui igl::png)
add_application(dmat_to_geogram.cpp igl::core geogram cli11)

set_target_properties(navigation_test PROPERTIES FOLDER misc)
set_target_properties(refinement_test PROPERTIES FOLDER misc)
set_target_properties(solver_test PROPERTIES FOLDER misc)
set_target_properties(extrude_mesh PROPERTIES FOLDER misc)
# set_target_properties(svg_to_mesh PROPERTIES FOLDER misc)
set_target_properties(extract_polyhedra PROPERTIES FOLDER misc)
set_target_properties(tetrahedralize PROPERTIES FOLDER misc)
set_target_properties(tetrahedralize_hybrid PROPERTIES FOLDER misc)
set_target_properties(bc_setter PROPERTIES FOLDER misc)
set_target_properties(b_extractor PROPERTIES FOLDER misc)
