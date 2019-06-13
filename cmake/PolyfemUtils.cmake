# Copy PolyFEM header files into the build directory
function(polyfem_copy_headers)
	foreach(filepath IN ITEMS ${ARGN})
		get_filename_component(filename "${filepath}" NAME)
		if(${filename} MATCHES ".*\.(hpp|h|ipp)$")
			configure_file(${filepath} ${PROJECT_BINARY_DIR}/include/polyfem/${filename})
		endif()
	endforeach()
endfunction()

# Set source group for IDE like Visual Studio or XCode
function(polyfem_set_source_group)
	foreach(filepath IN ITEMS ${ARGN})
		get_filename_component(folderpath "${filepath}" DIRECTORY)
		get_filename_component(foldername "${folderpath}" NAME)
		source_group(foldername FILES "${filepath}")
	endforeach()
endfunction()

# Autogen helper function
function(polyfem_autogen MAIN_TARGET PYTHON_SCRIPT OUTPUT_BASE)
	if(NOT POLYFEM_REGENERATE_AUTOGEN)
		polyfem_copy_headers(${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp)
		return()
	endif()
	find_package(PythonInterp 3 QUIET)

	if(NOT ${PYTHONINTERP_FOUND})
		execute_process(
			COMMAND
			python -c "import sys; sys.stdout.write(sys.version)"
			OUTPUT_VARIABLE PYTHON_VERSION)

		if(NOT PYTHON_VERSION)
			MESSAGE(WARNING "Unable to run python, ${PYTHON_SCRIPT} not running")
			polyfem_copy_headers(${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp)
			return()
		endif()

		STRING(REGEX MATCH "^3\.*" IS_PYTHON3 ${PYTHON_VERSION})

		if(NOT IS_PYTHON3)
			MESSAGE(WARNING "Unable to find python 3, ${PYTHON_SCRIPT} not running")
			polyfem_copy_headers(${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp)
			return()
		else()
			SET(PYTHON_EXECUTABLE "python")
		endif()
	endif()

	execute_process(
			COMMAND
			${PYTHON_EXECUTABLE} -c "import sympy; import sys; sys.stdout.write('ok')"
			OUTPUT_VARIABLE PYTHON_HAS_LIBS)

	if(NOT PYTHON_HAS_LIBS)
		MESSAGE(WARNING "Unable to find sympy and/or numpy, ${PYTHON_SCRIPT} not running")
		polyfem_copy_headers(${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp)
		return()
	endif()


	add_custom_command(
		OUTPUT
			${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.cpp
			${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp
		COMMAND
			${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/autogen/${PYTHON_SCRIPT} ${PROJECT_SOURCE_DIR}/src/autogen/
		DEPENDS
			${PROJECT_SOURCE_DIR}/src/autogen/${PYTHON_SCRIPT}
	)

	add_custom_target(
		autogen_${OUTPUT_BASE}
		COMMAND
			${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp ${PROJECT_BINARY_DIR}/include/polyfem/
		DEPENDS
			${PROJECT_SOURCE_DIR}/src/autogen/${OUTPUT_BASE}.hpp
	)

	add_dependencies(${MAIN_TARGET} autogen_${OUTPUT_BASE})
endfunction()

# Add an application from one source
function(polyfem_add_application APP_SOURCE)
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
