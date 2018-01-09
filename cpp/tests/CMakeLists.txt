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
add_application(mesh2d_2_mesh3d.cpp polyfem cli11)
