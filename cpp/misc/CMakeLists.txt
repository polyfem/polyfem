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
endfunction()

################################################################################

add_application(navigation_test.cpp polyfem geogram_gfx)
add_application(refinement_test.cpp polyfem cli11)
add_application(solver_test.cpp polyfem cli11)
add_application(extrude_mesh.cpp polyfem cli11)
add_application(svg_to_mesh.cpp cli11 nanosvg clipper igl::core igl::triangle)
add_application(extract_polyhedra.cpp polyfem)
add_application(tetrahedralize.cpp igl::core igl::tetgen geogram)
add_application(tetrahedralize_hybrid.cpp polyfem igl::core igl::tetgen geogram cli11)
