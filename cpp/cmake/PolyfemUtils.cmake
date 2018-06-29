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
	find_package(PythonInterp 3 REQUIRED)

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
