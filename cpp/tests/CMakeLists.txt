cmake_minimum_required(VERSION 3.1)

################################################################################

function(add_application APP_SOURCE)
	# Add executable
	get_filename_component(APP_NAME ${APP_SOURCE} NAME_WE)
	add_executable(${APP_NAME} ${APP_SOURCE})
	message(STATUS "Compiling single-source application: ${APP_NAME}")

	# Dependencies
	target_link_libraries(${APP_NAME} PRIVATE polyfem geogram_gfx)

	# Output directory for binaries
	set_target_properties(${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

################################################################################

add_application(test_navigation.cpp polyfem geogram_gfx)
